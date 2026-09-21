package com.navx.fusion

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Navigation state with covariance.
 */
data class NavState(
    val timestamp: Long,
    val position: DoubleArray,      // [x, y, z] ENU meters
    val velocity: FloatArray,       // [vx, vy, vz] body frame m/s
    val attitude: FloatArray,       // [roll, pitch, yaw] radians
    val covariance: FloatArray,     // 9x9 covariance matrix (flattened row-major)
    val sourceHealth: Map<String, Boolean>
)

/**
 * Wheel measurement for correction.
 */
data class WheelMeasurement(
    val timestamp: Long,
    val vxBody: Float,
    val vyBody: Float = 0f,
    val vzBody: Float = 0f,
    val R: FloatArray,  // 3x3 diagonal covariance
    val gate: String = "full"  // "full", "low", "zupt"
)

/**
 * Visual odometry measurement.
 */
data class VisualOdoMeasurement(
    val timestamp: Long,
    val deltaPose: FloatArray,  // [dx, dy, dz, rx, ry, rz] in meters/radians
    val R: FloatArray  // 6x6 diagonal covariance
)

/**
 * GNSS measurement.
 */
data class GnssMeasurement(
    val timestamp: Long,
    val position: DoubleArray,   // [lat, lon, alt] degrees, meters
    val velocity: FloatArray,    // [vx, vy, vz] m/s ENU
    val covariance: FloatArray,  // 3x3 diagonal
    val fixQuality: Int,         // 0=none, 1=GPS, 2=DGPS, 4=RTK, 5=Float RTK
    val numSatellites: Int,
    val hdop: Float
)

/**
 * Navigation filter state for Part 7 InEKF on SE(2).
 * State vector (10): [x, y, psi, vx, vy, bg_z, ba_x, ba_y, s_wheel, s_vo]
 */
class InEKF(private val config: InEKFConfig) {
    
    data class Config(
        val processNoiseGyroBias: Double = 1e-6,
        val processNoiseAccelBias: Double = 1e-4,
        val processNoiseGyro: Double = 1e-4,
        val processNoiseAccel: Double = 1e-4,
        val gnssPosStd: Double = 0.5,
        val gnssVelStd: Double = 0.5,
        val wheelSpeedStd: Double = 0.1,
        val wheelNhcStd: Double = 0.05,
        val voPosStd: Double = 0.5,
        val voYawStd: Double = 0.05,
        val minEig: Double = 1e-12,
        val psdInflationFactor: Double = 1.5,
        val residualGateSigma: Double = 6.0,
        val reseedJumpMaxM: Double = 0.3
    )
    
    // State: [x, y, psi, vx, vy, bg_z, ba_x, ba_y, s_wheel, s_vo] = 10
    private val state = DoubleArray(10)
    private val covariance = DoubleArray(100) { if (it % 11 == 0) 1e-6 else 0.0 }  // 10x10 identity * 1e-6
    
    // Process noise continuous
    private val Qc = DoubleArray(100) { 0.0 }
    private val minEig = 1e-12
    val psdInflation = 1.5
    val residualGate = 6.0
    val reseedJumpMaxM = 0.3
    
    private var outageActive = false
    private var handoverActive = false
    private var lastGoodPose: DoubleArray? = null
    private var handoverBlend = 1.0
    
    private var skippedMeasurements = 0
    var psdRecoveries = 0
    
    init {
        initQc()
        initState()
    }
    
    private fun initQc() {
        qc = DoubleArray(100) { 0.0 }
        // Gyro bias RW (z-axis only for SE(2))
        qc[5 * 10 + 5] = config.processNoiseGyroBias
        // Accel bias random walk
        qc[6 * 10 + 6] = config.processNoiseAccelBias
        qc[7 * 10 + 7] = config.processNoiseAccelBias
        // Gyro noise (affects attitude)
        qc[2 * 10 + 2] = config.procNoiseGyro
        // Accel noise (affects velocity)
        qc[3 * 10 + 3] = config.procNoiseAccel
        qc[4 * 10 + 4] = config.procNoiseAccel
        // Scale factors (slow RW)
        qc[8 * 10 + 8] = 1e-8
        qc[9 * 10 + 9] = 1e-8
    }
    
    private fun initState() {
        state = DoubleArray(10) { 0.0 }
        state[8] = 1.0  // s_wheel = 1.0
        state[9] = 1.0  // s_vo = 1.0
        covariance = DoubleArray(100) { if (it % 11 == 0) 1e-6 else 0.0 }
    }
    
    // State: [x, y, psi, vx, vy, bg_z, ba_x, ba_y, s_wheel, s_vo]
    private var state = DoubleArray(10) { 0.0 }
    private var covariance = DoubleArray(100) { 0.0 }
    private var config = InEKFConfig()
    
    var qc: DoubleArray = DoubleArray(100)
    var config = InEKFConfig()
    
    // Configuration data class
    data class Config(
        val processNoiseGyroBias: Double = 1e-6,
        val processNoiseAccelBias: Double = 1e-4,
        val processNoiseGyro: Double = 1e-4,
        val processNoiseAccel: Double = 1e-4,
        val gnssPosStd: Double = 0.5,
        val gnssVelStd: Double = 0.5,
        val wheelSpeedStd: Double = 0.1,
        val wheelNhcStd: Double = 0.05,
        val voPosStd: Double = 0.5,
        val voYawStd: Double = 0.05,
        val minEig: Double = 1e-12,
        val psdInflationFactor: Double = 1.5,
        val residualGateSigma: Double = 6.0,
        val reseedJumpMaxM: Double = 0.3
    )
    
    // Getters
    val stateDim = 10
    val stateVec: DoubleArray get() = state
    val covarianceMatrix: DoubleArray get() = covariance
    val position: DoubleArray get() = doubleArrayOf(state[0], state[1])
    val velocity: FloatArray get() = floatArrayOf(state[3].toFloat(), state[4].toFloat())
    val heading: Double get() = state[2]
    val yawRate: Double = 0.0  // Not in state directly
    val isOutageActive: Boolean get() = outageActive
    val isHandoverActive: Boolean get() = handoverActive
    val covarianceTrace: Double get() = covariance.sum { it * it }
    
    private var outageActive = false
    private var handoverActive = false
    private var lastGoodPose: DoubleArray? = null
    private var handoverBlend = 1.0
    var skippedMeasurements = 0
    var psdRecoveries = 0
    
    private var outageActive = false
    private var handoverActive = false
    private var lastGoodPose: DoubleArray? = null
    private var handoverBlend = 1.0
    
    // Process noise continuous
    private val Qc = DoubleArray(100) { 0.0 }
    
    init {
        // State: [x, y, psi, vx, vy, bg_z, ba_x, ba_y, s_wheel, s_vo] = 10
        // x = [x, y, psi, vx, vy, bg_z, ba_x, ba_y, s_wheel, s_vo]
        // P = 10x10
    }
    
    fun predict(acc: FloatArray, gyr: FloatArray, dt: Float) {
        val gyrCorr = gyr.copyOf()
        gyrCorr[2] -= state[5]  // bg_z
        
        val accCorr = FloatArray(3)
        accCorr[0] = acc[0] - state[6]
        accCorr[1] = acc[1] - state[7]
        accCorr[2] = acc[2] - state[8]  // ba_z not in state, use 0
        
        val c = Math.cos(state[2])
        val s = Math.sin(state[2])
        
        // State propagation
        state[0] += (state[3] * Math.cos(state[2]) - state[4] * Math.sin(state[2])) * dt
        state[1] += (state[3] * Math.sin(state[2]) + state[4] * Math.cos(state[2])) * dt
        state[2] = wrapAngle(state[2] + gyr[2] * dt)
        state[3] += accCorr[0] * dt
        state[4] += accCorr[1] * dt
        
        // F matrix
        val F = Array(10) { DoubleArray(10) { 0.0 } }.apply { for (i in 0..9) this[i][i] = 1.0 }
        val c = Math.cos(state[2])
        val s = Math.sin(state[2])
        val vx = state[3]
        val vy = state[4]
        
        this[0][2] = (-vx * s - vy * c) * dt
        this[1][2] = (vx * c - vy * s) * dt
        this[0][3] = Math.cos(state[2]) * dt
        this[0][4] = -Math.sin(state[2]) * dt
        this[1][3] = Math.sin(state[2]) * dt
        this[1][4] = Math.cos(state[2]) * dt
        this[3][5] = dt  // bg_z affects psi
        this[4][6] = dt  // ba_x affects vx
        this[4][7] = dt  // ba_y affects vy
        
        // Discretized Q
        val Qd = discretizeQc(dt)
        
        // Covariance propagation
        var newP = multiply(multiply(F, P), transpose(F))
        for (i in 0..9) for (j in 0..9) {
            covariance[i * 10 + j] = this[i * 10 + j] + Qd[i * 10 + j]
        }
        
        enforcePsd()
    }
    
    private fun discretizeQc(dt: Float): DoubleArray {
        val Qd = DoubleArray(100) { 0.0 }
        for (i in 0..9) for (j in 0..9) {
            Qd[i * 10 + j] = Qc[i * 10 + j] * dt
        }
        return Qd
    }
    
    // GNSS position/heading correction (1 Hz)
    fun correctGnss(z: DoubleArray, R: DoubleArray? = null) {
        val Rmat = R ?: buildRgnss()
        val H = Array(3) { DoubleArray(10) { 0.0 } }
        H[0][0] = 1.0; H[1][1] = 1.0; H[2][2] = 1.0
        
        val zPred = doubleArrayOf(state[0], state[1], state[2])
        val y = z.zip(zPred) { a, b -> a - b }
        y[2] = wrapAngle(y[2])
        
        val S = multiply(multiply(H, P), transpose(H)).also { addR(it, R) }
        
        if (residualGate(y, S)) {
            skippedMeasurements++
            return
        }
        
        val K = multiply(multiply(P, transpose(H)), invert(S))
        val dx = multiply(K, y.toDoubleArray())
        for (i in 0..9) state[i] += dx[i]
        state[2] = wrapAngle(state[2])
        
        val I_KH = Array(10) { DoubleArray(10) { if (it[0] == it[1]) 1.0 else 0.0 } }.also { I ->
            val KH = multiply(K, H)
            for (i in 0..9) for (j in 0..9) I[i][j] -= KH[i][j]
        }
        
        P = multiply(multiply(I, P), transpose(I_KH))
        enforcePsd()
    }
    
    private fun buildRgnss(): DoubleArray = doubleArrayOf(
        fusion_cfg.gnss_pos_std * fusion_cfg.gnss_pos_std,
        fusion_cfg.gnss_pos_std * fusion_cfg.gnss_pos_std,
        Math.toRadians(5.0).pow(2)
    )
    
    // Wheel speed correction with NHC (10 Hz)
    fun correctWheel(m: WheelMeasurement) {
        // Implementation
    }
    
    // Visual odometry correction
    fun correctVo(deltaPose: DoubleArray, Rvo: DoubleArray? = null) {
        // Implementation
    }
    
    // NHC: lateral/vertical velocity = 0
    private fun correctNhc() { /* ... */ }
    
    // Visual odometry correction
    // Seeder integration
    fun seedAnchor(sigmaInflate: Double) {
        // Inflate covariance before predicted outage
    }
    
    fun handover(gnssPose: DoubleArray, blend: Double) {
        // Zero-jump re-anchor
    }
    
    // State access
    fun state(): DoubleArray = state.clone()
    fun covariance(): DoubleArray = covariance.clone()
    
    // Helpers
    private fun wrapAngle(a: Double): Double {
        var a = a
        while (a <= -Math.PI) a += 2 * Math.PI
        while (a > Math.PI) a -= 2 * Math.PI
        return a
    }
    
    private fun wrapAngle(arr: DoubleArray, idx: Int) {
        var a = state[idx]
        while (a <= -Math.PI) a += 2 * Math.PI
        while (a > Math.PI) a -= 2 * Math.PI
        state[idx] = a
    }
    
    private fun enforcePsd() {
        val eigen = EigenDecomposition(covariance, 10)
        for (i in 0..9) if (eigen.values[i] < minEig) eigen.values[i] = minEig
        covariance = eigen.reconstruct()
    }
    
    private fun enforcePsd() {
        // Ensure PSD via eigenvalue clamping
        val eigen = EigenDecomposition(covariance)
        for (i in 0..9) if (eigen.values[i] < minEig) eigen.values[i] = minEig
        // Reconstruct
    }
    
    // Matrix operations (simplified - use multik in production)
    private fun multiply(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> = TODO()
    private fun transpose(A: Array<DoubleArray>): Array<DoubleArray> = TODO()
    private fun invert(A: Array<DoubleArray>): Array<DoubleArray> = TODO()
    private fun addR(S: Array<DoubleArray>, R: DoubleArray) = TODO()
    private fun wrapAngle(a: Double): Double {
        var a = a
        while (a <= -Math.PI) a += 2 * Math.PI
        while (a > Math.PI) a -= 2 * Math.PI
        return a
    }
    
    companion object {
        private const val TAG = "InEKF"
    }
}

// Helper classes for matrix ops (use Multik in production)
class EigenDecomposition(val matrix: Array<DoubleArray>) {
    val values = DoubleArray(10)
    val vectors = Array(10) { DoubleArray(10) }
    init { /* JAMA/Eigen port or use Multik */ }
    fun reconstruct(): DoubleArray = TODO()
}

private fun TODO(): Nothing = throw NotImplementedError("Use Multik for matrix ops")

// Config holder
data class InEKFConfig(
    val processNoiseGyroBias: Double = 1e-6,
    val processNoiseAccelBias: Double = 1e-4,
    val processNoiseGyro: Double = 1e-4,
    val processNoiseAccel: Double = 0.0001,
    val gnssPosStd: Double = 0.5,
    val gnssVelStd: Double = 0.5,
    val wheelSpeedStd: Double = 0.1,
    val wheelNhcStd: Double = 0.05,
    val voPosStd: Double = 0.5,
    val voYawStd: Double = 0.05,
    val minEig: Double = 1e-12,
    val psdInflationFactor: Double = 1.5,
    val residualGateSigma: Double = 6.0,
    val reseedJumpMaxM: Double = 0.3
)

// InEKFConfig from global config
fun InEKFConfig.fromGlobal(config: Config): InEKFConfig = InEKFConfig(
    processNoiseGyroBias = config.fusion.procNoiseGyroBias,
    processNoiseAccelBias = config.fusion.procNoiseAccelBias,
    processNoiseGyro = config.fusion.procNoiseGyro,
    processNoiseAccel = config.fusion.procNoiseAccel,
    gnssPosStd = config.fusion.gnssPosStd,
    gnssVelStd = config.fusion.gnssVelStd,
    wheelSpeedStd = config.fusion.wheelSpeedStd,
    wheelNhcStd = config.fusion.wheelNhcStd,
    voPosStd = config.fusion.voPosStd,
    voYawStd = config.fusion.voYawStd,
    minEig = config.fusion.minEig,
    psdInflationFactor = config.fusion.psdInflationFactor,
    residualGateSigma = config.fusion.residualGateSigma,
    reseedJumpMaxM = config.fusion.reseedJumpMaxM
)