package com.navx.inference

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * TFLite Runtime wrapper for on-device inference.
 * Handles model loading, delegate selection, and buffer management.
 */
object TFLiteRuntime {
    private const val TAG = "TFLiteRuntime"
    
    /** Supported hardware delegates */
    enum class Delegate {
        CPU,
        NNAPI,
        GPU
    }
    
    /**
     * Creates an interpreter with the specified delegate.
     */
    @Throws(Exception::class)
    fun createInterpreter(
        context: Context,
        modelPath: String,
        delegate: Delegate = Delegate.CPU,
        numThreads: Int = 4
    ): Interpreter {
        val options = Interpreter.Options().apply {
            setNumThreads(numThreads)
            
            when (delegate) {
                Delegate.NNAPI -> {
                    val nnApiDelegate = NnApiDelegate()
                    addDelegate(nnApiDelegate)
                    Log.i(TAG, "Using NNAPI delegate")
                }
                Delegate.GPU -> {
                    val gpuDelegate = GpuDelegate()
                    addDelegate(gpuDelegate)
                    Log.i(TAG, "Using GPU delegate")
                }
                else -> {
                    Log.i(TAG, "Using CPU delegate")
                }
            }
            
            setAllowBufferHandleOutput(true)
            setAllowFp16PrecisionForFp32(true)
        }
        
        val modelFile = java.io.File(context.filesDir, modelPath)
        if (!modelFile.exists()) {
            // Copy from assets if not present
            copyFromAssets(context, modelPath)
        }
        
        return Interpreter(modelFile, options)
    }
    
    private fun copyFromAssets(context: Context, modelPath: String) {
        val inputStream = context.assets.open(modelPath)
        val outputFile = java.io.File(context.filesDir, modelPath)
        outputFile.parentFile?.mkdirs()
        val outputStream = java.io.FileOutputStream(outputFile)
        inputStream.copyTo(outputStream)
        inputStream.close()
        outputStream.close()
    }
    
    companion object {
        private const val TAG = "TFLiteRuntime"
        
        /** Model file names in assets */
        const val MODEL_VELOCITY = "velocity.tflite"
        const val MODEL_PSEUDO_ODO = "pseudo_odo.tflite"
        const val MODEL_SLIP = "slip.tflite"
        const val MODEL_VISUAL_ODO = "visual_odo.tflite"
        const val MODEL_TEXTURE_GATE = "texture_gate.tflite"
        const val MODEL_FORECASTER = "forecaster.tflite"
    }
}

/**
 * Base class for TFLite model inference with buffer reuse.
 */
abstract class TFLiteModel(
    protected val interpreter: Interpreter,
    protected val inputShape: IntArray,
    protected val outputShape: IntArray,
    protected val inputType: Class<*>,
    protected val outputType: Class<*>
) {
    // Reusable buffers to avoid GC pressure
    protected val inputBuffer = ByteBuffer.allocateDirect(
        inputShape.fold(1) { acc, dim -> acc * dim } * 4
    ).order(ByteOrder.nativeOrder())
    
    protected val outputBuffer = ByteBuffer.allocateDirect(
        outputShape.fold(1) { acc, dim -> acc * dim } * 4
    ).order(ByteOrder.nativeOrder())
    
    /**
     * Run inference with buffer reuse.
     */
    fun run(inputArray: FloatArray): FloatArray {
        // Fill input buffer
        inputBuffer.rewind()
        inputBuffer.asFloatBuffer().put(inputArray)
        
        // Run inference
        interpreter.run(inputBuffer, outputBuffer)
        
        // Read output
        val outputArray = FloatArray(outputShape.reduce { acc, dim -> acc * dim })
        outputBuffer.rewind()
        outputBuffer.asFloatBuffer().get(outputArray)
        return outputArray
    }
    
    /**
     * Run inference with multiple inputs.
     */
    fun runMulti(inputs: Array<FloatArray>): FloatArray {
        val inputBuffers = inputs.mapIndexed { i, input ->
            ByteBuffer.allocateDirect(inputShape[i].fold(1) { acc, dim -> acc * dim } * 4)
                .order(ByteOrder.nativeOrder()).apply { put(input[i]).rewind() }
        }.toArray()
        
        interpreter.runForMultipleInputsOutputs(inputBuffers, outputBuffer)
        
        val outputArray = FloatArray(outputShape.reduce { acc, dim -> acc * dim })
        outputBuffer.rewind()
        outputBuffer.asFloatBuffer().get(outputArray)
        return outputArray
    }
    
    fun close() {
        interpreter.close()
    }
}

/**
 * Velocity estimation model (TCN + LSTM)
 */
class VelocityModel(
    context: Context,
    delegate: TFLiteRuntime.Delegate = TFLiteRuntime.Delegate.CPU
) : TFLiteModel(
    interpreter = TFLiteRuntime.createInterpreter(
        context, TFLiteRuntime.MODEL_VELOCITY, delegate
    ),
    inputShape = intArrayOf(1, 200, 6),  // (1, 200, 6) - 2s window @ 100Hz
    outputShape = intArrayOf(1, 1),      // (1, 1) velocity scalar
    inputType = Float::class.java,
    outputType = Float::class.java
) {
    companion object {
        const val INPUT_WINDOW = 200  // 2 seconds @ 100Hz
        const val INPUT_CHANNELS = 6   // acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
    }
    
    /**
     * Run velocity inference on IMU window.
     * @param imuWindow FloatArray of shape [200, 6] - [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
     * @param att0 Initial attitude matrix (3x3) as flat array of 9 floats
     * @return VelocityResult(velocity_mps, confidence)
     */
    fun run(imuWindow: FloatArray, att0: FloatArray): Pair<Float, Float> {
        // Prepare input: [imu_window(200x6), att0(9)] -> flatten to single input tensor
        val input = FloatArray(INPUT_WINDOW * INPUT_CHANNELS + 9)
        System.arraycopy(imuWindow, 0, input, 0, INPUT_WINDOW * INPUT_CHANNELS)
        System.arraycopy(att0, 0, input, INPUT_WINDOW * INPUT_CHANNELS, 9)
        
        val output = run(input)
        val velocity = output[0]
        val confidence = 1.0f // Could add uncertainty head later
        return Pair(velocity, 1.0f)
    }
}

/**
 * Pseudo wheel odometry model (DeepOdo-style)
 */
class PseudoOdoModel(
    context: Context,
    delegate: TFLiteRuntime.Delegate = TFLiteRuntime.Delegate.CPU
) : TFLiteModel(
    interpreter = TFLiteRuntime.createInterpreter(
        context, TFLiteRuntime.MODEL_PSEUDO_ODO, TFLiteRuntime.Delegate.CPU
    ),
    inputShape = intArrayOf(1, 200, 6),  // IMU window
    outputShape = intArrayOf(1, 2),      // [wheel_speed, reliability]
    inputType = Float::class.java,
    outputType = Float::class.java
) {
    companion object {
        const val INPUT_WINDOW = 200
        const val INPUT_CHANNELS = 6
    }
    
    /**
     * Run pseudo-odometry inference.
     * @param imuWindow FloatArray[200, 6]
     * @param baroDelta Pressure change over window (1,)
     * @return Pair(wheel_speed_mps, reliability)
     */
    fun run(imuWindow: FloatArray, baroDelta: FloatArray): Pair<Float, Float> {
        // Input: [imuWindow(200x6), baroDelta(1)]
        val input = FloatArray(INPUT_WINDOW * INPUT_CHANNELS + 1)
        System.arraycopy(imuWindow, 0, input, 0, INPUT_WINDOW * INPUT_CHANNELS)
        input[INPUT_WINDOW * INPUT_CHANNELS] = baroDelta[0]
        
        val output = run(input)
        return Pair(output[0], output[1])
    }
}

/**
 * Slip classifier (Grip/Slip/Stationary)
 */
class SlipClassifier(
    context: Context,
    delegate: TFLiteRuntime.Delegate = TFLiteRuntime.Delegate.CPU
) : TFLiteModel(
    interpreter = TFLiteRuntime.createInterpreter(
        context, TFLiteRuntime.MODEL_SLIP, TFLiteRuntime.Delegate.CPU
    ),
    inputShape = intArrayOf(1, 200, 6),
    outputShape = intArrayOf(1, 3),  // [grip, slip, stationary]
    inputType = Float::class.java,
    outputType = Float::class.java
) {
    companion object {
        const val INPUT_WINDOW = 200
        const val INPUT_CHANNELS = 6
    }
    
    enum class SlipState {
        GRIP, SLIP_BRAKE, STATIONARY
    }
    
    fun run(imuWindow: FloatArray): Int {  // Returns class index
        val output = run(imuWindow)
        return output.maxIndexOrNull() ?: 0
    }
    
    fun runWithProbs(imuWindow: FloatArray): FloatArray {
        return run(imuWindow)
    }
}

/**
 * Texture gate classifier
 */
class TextureGate(
    context: Context,
    delegate: TFLiteRuntime.Delegate = TFLiteRuntime.Delegate.CPU
) : TFLiteModel(
    interpreter = TFLiteRuntime.createInterpreter(
        context, TFLiteRuntime.MODEL_TEXTURE_GATE, TFLiteRuntime.Delegate.CPU
    ),
    inputShape = intArrayOf(1, 96, 96, 1),  // 96x96 grayscale
    outputShape = intArrayOf(1, 3),         // [rich, medium, poor]
    inputType = Float::class.java,
    outputType = Float::class.java
) {
    companion object {
        const val INPUT_SIZE = 96
    }
    
    enum class TextureClass {
        RICH, MEDIUM, POOR
    }
    
    fun run(image96x96: ByteArray): Pair<Int, Float> {  // Returns (class, confidence)
        val input = FloatArray(INPUT_SIZE * INPUT_SIZE)
        for (i in image.indices) {
            input[i] = image[i].toFloat() / 255.0f
        }
        val output = run(image.map { it.toFloat() / 255.0f }.toFloatArray())
        val classIdx = output.maxIndexOrNull() ?: 0
        val confidence = output.maxOrNull() ?: 0f
        return Pair(classIdx, confidence)
    }
    
    companion object {
        const val INPUT_SIZE = 96
    }
}

/**
 * Visual odometry model (UL-VIO style)
 */
class VisualOdoModel(
    context: Context,
    delegate: TFLiteRuntime.Delegate = TFLiteRuntime.Delegate.CPU
) : TFLiteModel(
    interpreter = TFLiteRuntime.createInterpreter(
        context, TFLiteRuntime.MODEL_VISUAL_ODO, TFLiteRuntime.Delegate.CPU
    ),
    inputShape = intArrayOf(1, 2, 224, 224),  // 2-channel 224x224
    outputShape = intArrayOf(1, 6),            // 6-DoF delta
    inputType = Float::class.java,
    outputType = Float::class.java
) {
    companion object {
        const val INPUT_HEIGHT = 224
        const val INPUT_WIDTH = 224
    }
    
    fun run(prevFrame: ByteArray, currFrame: ByteArray): FloatArray {
        // Input: [prev_frame, curr_frame] each 224x224 grayscale
        val input = FloatArray(2 * INPUT_HEIGHT * INPUT_WIDTH)
        for (i in 0 until INPUT_HEIGHT * INPUT_WIDTH) {
            input[i] = prevFrame[i].toFloat() / 255.0f
            input[i + INPUT_HEIGHT * INPUT_WIDTH] = currFrame[i].toFloat() / 255.0f
        }
        return run(input)
    }
    
    companion object {
        const val INPUT_HEIGHT = 224
        const val INPUT_WIDTH = 224
    }
}

/**
 * GNSS outage forecaster
 */
class ForecasterModel(
    context: Context,
    delegate: TFLiteRuntime.Delegate = TFLiteRuntime.Delegate.CPU
) : TFLiteModel(
    interpreter = TFLiteRuntime.createInterpreter(
        context, TFLiteRuntime.MODEL_FORECASTER, TFLiteRuntime.Delegate.CPU
    ),
    inputShape = intArrayOf(1, 30, 104),  // 30 steps of 104 features
    outputShape = intArrayOf(1, 3),        // 3 horizons: 5s, 10s, 15s
    inputType = Float::class.java,
    outputType = Float::class.java
) {
    fun run(features: FloatArray): FloatArray = run(features)
}