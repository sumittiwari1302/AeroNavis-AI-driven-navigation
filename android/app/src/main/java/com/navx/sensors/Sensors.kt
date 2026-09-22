package com.navx.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.Flow

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

    // Output flow for fused sensor data at 100Hz
    private val dataFlowMutable = MutableStateFlow(emptyFrame())
    val dataFlow: Flow<NavDataFrame> = dataFlowMutable.asStateFlow()

    private val accelListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
                accelBuffer[accelHead] = event.values.copyOf()
                accelHead = (accelHead + 1) % ACCEL_BUFFER_SIZE
                emitFrameIfReady()
            }
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    private val gyroListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
                gyroBuffer[gyroHead] = event.values.copyOf()
                gyroHead = (gyroHead + 1) % GYRO_BUFFER_SIZE
            }
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    private val magListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            if (event.sensor.type == Sensor.TYPE_MAGNETIC_FIELD) {
                magBuffer[magHead] = event.values.copyOf()
                magHead = (magHead + 1) % MAG_BUFFER_SIZE
            }
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    private val baroListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            if (event.sensor.type == Sensor.TYPE_PRESSURE) {
                baroBuffer[baroHead][0] = event.values[0]
                baroHead = (baroHead + 1) % BARO_BUFFER_SIZE
            }
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    private var accelStore = accelBuffer[0].copyOf()
    private var gyroStore = gyroBuffer[0].copyOf()

    private fun emitFrameIfReady() {
        if (gyroHead == 0) return
        accelStore = accelBuffer[(accelHead - 1 + ACCEL_BUFFER_SIZE) % ACCEL_BUFFER_SIZE].copyOf()
        gyroStore = gyroBuffer[(gyroHead - 1 + GYRO_BUFFER_SIZE) % GYRO_BUFFER_SIZE].copyOf()
        dataFlowMutable.value = NavDataFrame(
            timestamp = System.currentTimeMillis(),
            accel = accelStore,
            gyro = gyroStore,
            mag = (magBuffer[(magHead - 1 + MAG_BUFFER_SIZE) % MAG_BUFFER_SIZE]).copyOf(),
            baro = (baroBuffer[(baroHead - 1 + BARO_BUFFER_SIZE) % BARO_BUFFER_SIZE]).copyOf(),
            gnss = if (gnssHead > 0) gnssBuffer[(gnssHead - 1 + GNSS_BUFFER_SIZE) % GNSS_BUFFER_SIZE].copyOf() else null,
            cameraFrame = null
        )
    }

    private fun emptyFrame(): NavDataFrame = NavDataFrame(
        timestamp = 0L,
        accel = FloatArray(3),
        gyro = FloatArray(3),
        mag = FloatArray(3),
        baro = FloatArray(1),
        gnss = null,
        cameraFrame = null
    )

    fun start() {
        sensorManager.registerListener(accelListener, sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_GAME)
        sensorManager.registerListener(gyroListener, sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE), SensorManager.SENSOR_DELAY_GAME)
        sensorManager.registerListener(magListener, sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD), SensorManager.SENSOR_DELAY_UI)
        sensorManager.registerListener(baroListener, sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE), SensorManager.SENSOR_DELAY_UI)
    }

    fun stop() {
        sensorManager.unregisterListener(accelListener)
        sensorManager.unregisterListener(gyroListener)
        sensorManager.unregisterListener(magListener)
        sensorManager.unregisterListener(baroListener)
    }
}

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
    val dataFlow: Flow<NavDataFrame>
    val textureGateFlow: Flow<TextureGateResult>
    val velocityFlow: Flow<VelocityResult>
    val wheelFlow: Flow<WheelResult>
    val visualOdoFlow: Flow<VisualOdoResult?>
    val outageForecastFlow: Flow<OutageForecast>
    val fusedStateFlow: Flow<NavState>
    val matchResultFlow: Flow<MatchResult>

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