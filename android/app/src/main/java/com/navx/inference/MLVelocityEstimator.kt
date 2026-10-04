package com.navx.inference

import android.content.Context
import android.util.Log
import com.navx.sensors.NavDataFrame
import java.util.ArrayDeque

/**
 * ML-based velocity estimator wrapping TFLite VelocityModel.
 * Handles variable-rate IMU input, resamples to 100Hz/200-sample window,
 * computes initial attitude from gravity, runs inference.
 * Gracefully falls back if model not available.
 */
class MLVelocityEstimator(
    private val context: Context,
    private val delegate: TFLiteRuntime.Delegate = TFLiteRuntime.Delegate.CPU
) {
    private val TAG = "MLVelocityEstimator"
    
    companion object {
        private const val TARGET_WINDOW_MS = 2000L  // 2 seconds
        private const val TARGET_SAMPLES = 200       // 100Hz * 2s
        private const val TARGET_DT_MS = 10L         // 100Hz = 10ms
        private const val GRAVITY_CALIB_SAMPLES = 50
        private const val INFERENCE_INTERVAL_MS = 1000L
    }
    
    private var velocityModel: VelocityModel? = null
    private var modelLoaded = false
    private var modelLoadAttempted = false
    
    // Ring buffer for raw IMU samples (device frame) with timestamps
    private val imuBuffer = ArrayDeque<ImuSample>()
    
    // Initial attitude matrix (device -> canonical gravity-aligned frame)
    private var att0: FloatArray? = null
    private var gravityCalibrated = false
    private val gravityAccumulator = FloatArray(3)
    private var gravitySampleCount = 0
    
    // Last inference result
    private var lastVelocity = 0.0f
    private var lastInferenceTimeMs = 0L
    private var inferenceCount = 0L
    private var lastInferenceLatencyMs = 0L
    
    data class ImuSample(
        val timestampMs: Long,
        val accX: Float, val accY: Float, val accZ: Float,
        val gyrX: Float, val gyrY: Float, val gyrZ: Float
    )
    
    /**
     * Add IMU sample from SensorHub (device frame).
     * Call at sensor rate (typically 50-100Hz).
     */
    fun addSample(frame: NavDataFrame): Pair<Float, Float>? {
        // Lazy-load model on first sample
        if (!modelLoadAttempted) {
            tryLoadModel()
        }
        if (!modelLoaded) return null
        
        val now = System.currentTimeMillis()
        imuBuffer.add(ImuSample(
            now,
            frame.accel[0], frame.accel[1], frame.accel[2],
            frame.gyro[0], frame.gyro[1], frame.gyro[2]
        ))
        
        // Calibrate gravity from first ~50 samples (stationary assumption)
        if (!gravityCalibrated) {
            gravityAccumulator[0] += frame.accel[0]
            gravityAccumulator[1] += frame.accel[1]
            gravityAccumulator[2] += frame.accel[2]
            gravitySampleCount++
            
            if (gravitySampleCount >= GRAVITY_CALIB_SAMPLES) {
                val gx = gravityAccumulator[0] / gravitySampleCount
                val gy = gravityAccumulator[1] / gravitySampleCount
                val gz = gravityAccumulator[2] / gravitySampleCount
                val norm = Math.sqrt((gx*gx + gy*gy + gz*gz).toDouble()).toFloat()
                if (norm > 1e-6f) {
                    att0 = computeAtt0FromGravity(gx / norm, gy / norm, gz / norm)
                    gravityCalibrated = true
                    Log.d(TAG, "Gravity calibrated: g=($gx,$gy,$gz) norm=$norm")
                }
            }
        }
        
        // Trim buffer to 2.5s (keep margin for resampling)
        val cutoff = now - 2500
        while (imuBuffer.isNotEmpty() && imuBuffer.first().timestampMs < cutoff) {
            imuBuffer.removeFirst()
        }
        
        // Run inference every ~1 second if we have enough samples
        if (gravityCalibrated && now - lastInferenceTimeMs >= INFERENCE_INTERVAL_MS && imuBuffer.size >= 100) {
            return runInference()
        }
        
        return null
    }
    
    private fun tryLoadModel() {
        modelLoadAttempted = true
        try {
            velocityModel = VelocityModel(context, delegate)
            modelLoaded = true
            Log.d(TAG, "ML velocity model loaded successfully")
        } catch (e: Exception) {
            Log.w(TAG, "ML velocity model not available (no velocity.tflite in assets), using fallback: ${e.message}")
            modelLoaded = false
        }
    }
    
    /**
     * Resample buffered IMU to fixed 200 samples @ 100Hz over last 2s,
     * run TFLite inference, return velocity.
     */
    private fun runInference(): Pair<Float, Float>? {
        val now = System.currentTimeMillis()
        val windowStart = now - TARGET_WINDOW_MS
        
        // Collect samples in time range
        val samples = imuBuffer.filter { it.timestampMs >= windowStart && it.timestampMs <= now }
            .toList()
        
        if (samples.size < 100) {
            Log.w(TAG, "Insufficient samples for inference: ${samples.size}")
            return null
        }
        
        // Resample to 200 evenly-spaced points over 2s using linear interpolation
        val resampledAcc = FloatArray(TARGET_SAMPLES * 3)
        val resampledGyr = FloatArray(TARGET_SAMPLES * 3)
        
        for (i in 0 until TARGET_SAMPLES) {
            val targetTime = windowStart + i * TARGET_DT_MS
            val interp = interpolateAtTime(samples, targetTime)
            resampledAcc[i * 3] = interp[0]
            resampledAcc[i * 3 + 1] = interp[1]
            resampledAcc[i * 3 + 2] = interp[2]
            resampledGyr[i * 3] = interp[3]
            resampledGyr[i * 3 + 1] = interp[4]
            resampledGyr[i * 3 + 2] = interp[5]
        }
        
        // Run TFLite inference
        val startInfer = System.nanoTime()
        try {
            val result = velocityModel?.run(resampledAcc, resampledGyr, att0!!) ?: return null
            val latencyMs = (System.nanoTime() - startInfer) / 1_000_000
            
            lastVelocity = result.first
            lastInferenceTimeMs = now
            inferenceCount++
            lastInferenceLatencyMs = latencyMs
            
            Log.d(TAG, String.format("ML Inference #%d: vel=%.3f m/s latency=%dms", inferenceCount, lastVelocity, latencyMs))
            
            return result
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed: ${e.message}")
            return null
        }
    }
    
    /**
     * Linear interpolation of IMU values at target timestamp.
     */
    private fun interpolateAtTime(samples: List<ImuSample>, targetMs: Long): FloatArray {
        // Find bracketing samples
        var before: ImuSample? = null
        var after: ImuSample? = null
        
        for (s in samples) {
            if (s.timestampMs <= targetMs) {
                before = s
            } else if (after == null) {
                after = s
                break
            }
        }
        
        if (before == null) before = samples.first()
        if (after == null) after = samples.last()
        
        if (before === after) {
            return floatArrayOf(before.accX, before.accY, before.accZ, before.gyrX, before.gyrY, before.gyrZ)
        }
        
        val t1 = before.timestampMs.toFloat()
        val t2 = after.timestampMs.toFloat()
        val alpha = if (t2 > t1) (targetMs - t1) / (t2 - t1) else 0f
        
        return floatArrayOf(
            lerp(before.accX, after.accX, alpha),
            lerp(before.accY, after.accY, alpha),
            lerp(before.accZ, after.accZ, alpha),
            lerp(before.gyrX, after.gyrX, alpha),
            lerp(before.gyrY, after.gyrY, alpha),
            lerp(before.gyrZ, after.gyrZ, alpha)
        )
    }
    
    private fun lerp(a: Float, b: Float, t: Float): Float = a + (b - a) * t
    
    /**
     * Compute att0 (device -> canonical gravity-aligned frame) from gravity vector.
     * Canonical frame: z-down (gravity), x-forward (projected heading), y-right.
     * This matches the training canonicalize() in velocity.py.
     */
    private fun computeAtt0FromGravity(gx: Float, gy: Float, gz: Float): FloatArray {
        // Gravity in device frame (normalized, pointing down)
        val g = floatArrayOf(gx, gy, gz)
        
        // Canonical z = -g (gravity points down in canonical)
        val zx = -gx
        val zy = -gy
        val zz = -gz
        
        // Choose reference x in horizontal plane (orthogonal to z)
        val xRef = if (Math.abs(zx) < 0.9) floatArrayOf(1f, 0f, 0f) else floatArrayOf(0f, 1f, 0f)
        
        // Project xRef onto horizontal plane: xCanon = xRef - (xRef·z)z
        val dot = xRef[0]*zx + xRef[1]*zy + xRef[2]*zz
        var xx = xRef[0] - dot * zx
        var xy = xRef[1] - dot * zy
        var xz = xRef[2] - dot * zz
        val xNorm = Math.sqrt((xx*xx + xy*xy + xz*xz).toDouble()).toFloat()
        if (xNorm < 1e-6f) {
            xx = if (Math.abs(zx) < 0.9) 0f else 1f
            xy = if (Math.abs(zy) < 0.9) 0f else 1f
            xz = 0f
        } else {
            xx /= xNorm
            xy /= xNorm
            xz /= xNorm
        }
        
        // yCanon = z × x
        val yx = zy * xz - zz * xy
        val yy = zz * xx - zx * xz
        val yz = zx * xy - zy * xx
        
        // R_canonical_from_device: columns are canonical axes in device frame
        // [x_canon, y_canon, z_canon] as column vectors = 3x3 matrix
        // Flattened row-major for TFLite input
        return floatArrayOf(
            xx, xy, xz,   // row 0: x_canon components
            yx, yy, yz,   // row 1: y_canon components
            zx, zy, zz    // row 2: z_canon components
        )
    }
    
    fun getLastVelocity(): Float = lastVelocity
    fun isReady(): Boolean = gravityCalibrated && modelLoaded
    fun getInferenceCount(): Long = inferenceCount
    fun getLastInferenceLatencyMs(): Long = lastInferenceLatencyMs
    fun isModelLoaded(): Boolean = modelLoaded
    
    fun reset() {
        imuBuffer.clear()
        gravityCalibrated = false
        gravitySampleCount = 0
        gravityAccumulator[0] = 0f; gravityAccumulator[1] = 0f; gravityAccumulator[2] = 0f
        att0 = null
        lastVelocity = 0f
        lastInferenceTimeMs = 0
        inferenceCount = 0
        lastInferenceLatencyMs = 0
    }
    
    fun close() {
        velocityModel?.close()
        velocityModel = null
        modelLoaded = false
    }
}