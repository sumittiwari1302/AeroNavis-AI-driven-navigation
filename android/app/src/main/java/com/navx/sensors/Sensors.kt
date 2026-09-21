package com.navx.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ReceiveChannel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Central sensor hub that coordinates all sensor inputs.
 * Provides a unified NavDataFrame at 100Hz for the fusion pipeline.
 */
class SensorHub(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    
    // Ring buffers for each sensor (100Hz = 100 samples per second)
    private val ACCEL_BUFFER_SIZE = 1000  // 10 seconds
    private val GYRO_BUFFER_SIZE = 1000
    private val MAG_BUFFER_SIZE = 500     // 50Hz
    private val BARO_BUFFER_SIZE = 250    // 25Hz
    private val GNSS_BUFFER_SIZE = 100    // 1Hz
    
    // Ring buffers (pre-allocated arrays)
    private val accelBuffer = Array(ACCEL_BUFFER_SIZE) { FloatArray(3) }
    private val gyroBuffer = Array(GYRO_BUFFER_SIZE) { FloatArray(3) }
    private val magBuffer = Array(MAG_BUFFER_SIZE) { FloatArray(3) }
    private val baroBuffer = Array(BARO_BUFFER_SIZE) { FloatArray(1) }
    private val gnssBuffer = Array(GNSS_BUFFER_SIZE) { FloatArray(7) } // lat, lon, alt, speed, heading, pdop, cno
    
    private var accelHead = 0
    private var gyroHead = 0
    private var magHead = 0
    private var baroHead = 0
    private var gnssHead = 0
    
    // Camera image analysis
    private var cameraExecutor: ImageAnalysis? = null
    private var latestFrame: ByteArray? = null
    
    // Output channel for fused sensor data at 100Hz
    private val dataChannel = Channel<NavDataFrame>(100)
    val dataFlow = dataChannel.receiveAsFlow()
    
    // Sensor listeners
    private val accelListener = SensorEventListener { event ->
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            accelBuffer[accelHead] = event.values.map { it }.toFloatArray()
            accelHead = (accelHead + 1) % ACCEL_BUFFER_SIZE
        }
    }
    
    private val gyroListener = SensorEventListener { event ->
        if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
            gyroBuffer[gyroHead] = event.values.map { it }.toFloatArray()
            gyroHead = (gyroHead + 1) % GYRO_BUFFER_SIZE
        }
    }
    
    private val magListener = SensorEventListener { event ->
        if (event.sensor.type == Sensor.TYPE_MAGNETIC_FIELD) {
            magBuffer[magHead] = event.values.map { it }.toFloatArray()
            magHead = (magHead + 1) % MAG_BUFFER_SIZE
        }
    }
    
    private val baroListener = SensorEventListener { event ->
        if (event.sensor.type == Sensor.TYPE_PRESSURE) {
            baroBuffer[baroHead][0] = event.values[0]
            baroHead = (baroHead + 1) % BARO_BUFFER_SIZE
        }
    }
    
    // Camera image analysis
    private val imageAnalyzer = ImageAnalysis.Builder()
        .setTargetResolution(android.util.Size(96, 96))
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .build()
        .also { it.setAnalyzer(lifecycleOwner, ImageAnalysis.Analyzer { imageProxy ->
            val buffer = ByteArray(imageProxy.width * imageProxy.height)
            imageProxy.getPlanes()[0].buffer.get(buffer)
            latestFrame = buffer
            imageProxy.close()
        })

/**
 * Unified sensor data frame at 100Hz
 */
data class NavDataFrame(
    val timestamp: Long,
    val accel: FloatArray,      // 3 elements
    val gyro: FloatArray,       // 3 elements
    val mag: FloatArray,        // 3 elements (optional)
    val baro: FloatArray,       // 1 element
    val gnss: FloatArray?,      // [lat, lon, alt, speed, heading, pdop, cno]
    val cameraFrame: ByteArray? // 96x96 grayscale
)

/**
 * Texture quality gate output
 */
data class TextureGateResult(
    val gate: Int,              // 0=rich, 1=medium, 2=poor
    val confidence: Float       // softmax probability
)

/**
 * Velocity estimation result
 */
data class VelocityResult(
    val velocity: Float,        // m/s forward
    val confidence: Float,      // 0-1
    val features: FloatArray    // 128-dim features for fusion
)

/**
 * Wheel odometry result
 */
data class WheelResult(
    val speed: Float,           // m/s
    val reliability: Float      // 0-1
)

/**
 * Visual odometry result
 */
data class VisualOdoResult(
    val deltaPose: FloatArray?, // [dx, dy, dz, rx, ry, rz] in meters/radians
    val confidence: Float       // 0-1, null if gated
)

/**
 * GNSS outage forecast
 */
data class OutageForecast(
    val horizons: FloatArray,   // [5s, 10s, 15s] probabilities
    val threshold: Float        // alert threshold
)

/**
 * Fused navigation state
 */
data class NavState(
    val timestamp: Long,
    val position: DoubleArray,      // [x, y, z] ENU meters
    val velocity: FloatArray,       // [vx, vy, vz] body frame m/s
    val attitude: FloatArray,       // [roll, pitch, yaw] radians
    val covariance: FloatArray,     // 9x9 covariance matrix (flattened)
    val sourceHealth: Map<String, Boolean>, // IMU/Wheel/VO/GNSS health
    val adaptationState: AdaptationState
)

/**
 * Adaptation engine state
 */
data class AdaptationState(
    val isActive: Boolean,
    val loraParams: ByteArray?,  // Serialized LoRA adapter weights
    val velocityBias: Float,      // Learned velocity bias correction
    val slipState: Int            // 0=grip, 1=slip, 2=stationary
)

/**
 * Map matching result
 */
data class MatchResult(
    val poseSnapped: Boolean,
    val x: Double,
    val y: Double,
    val yaw: Double,
    val roadSegmentId: String?,
    val confidence: Float
)

/**
 * Main sensor interface for the pipeline
 */
interface SensorHubInterface {
    val dataFlow: kotlinx.coroutines.flow.Flow<NavDataFrame>
    val textureGateFlow: kotlinx.coroutines.flow.Flow<TextureGateResult>
    val velocityFlow: kotlinx.coroutines.flow.Flow<VelocityResult>
    val wheelFlow: kotlinx.coroutines.flow.Flow<WheelResult>
    val visualOdoFlow: kotlinx.coroutines.flow.Flow<VisualOdoResult?>
    val outageForecastFlow: kotlinx.coroutines.flow.Flow<OutageForecast>
    val fusedStateFlow: kotlinx.coroutines.flow.Flow<NavState>
    val matchResultFlow: kotlinx.coroutines.flow.Flow<MatchResult>
    
    fun start()
    fun stop()
}

object SensorConstants {
    const val IMU_HZ = 100
    const val GNSS_HZ = 1
    const val CAM_HZ = 10
    const val WINDOW_SIZE = 200  // 2 seconds at 100Hz
    const val STRIDE = 10
}