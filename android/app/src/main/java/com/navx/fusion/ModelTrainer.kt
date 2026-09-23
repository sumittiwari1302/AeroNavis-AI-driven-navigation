package com.navx.fusion

/**
 * On-device supervised model trainer.
 *
 * In the first GNSS-clean phase of a run it fits two low-rank corrections from
 * real measurements — this is the tiny "training" pass every vehicle does:
 *   1. wheel scale  = Σ ground-truth distance / Σ wheel-odometer distance
 *   2. gyro z-bias  = mean(gyrZ − dYaw/dt)  [rad/s]
 * Once finalize() is reached the corrections are applied live to the odometry
 * and rate-gyro so the dead-reckoning during blackouts stays within spec.
 */
class ModelTrainer(
    private val finalizeAfterSamples: Int = 120,
    private val minSamplesScale: Int = 40
) {
    private var n = 0
    private var sumWheelDelta = 0.0
    private var sumGtDelta = 0.0
    private var sumGyrErr = 0.0
    private var finalized = false

    var wheelScale = 1.0
        private set
    var gyrBiasRad = 0.0
        private set

    fun isFinalized(): Boolean = finalized

    /** gtDist/wheelDist in meters; gyrZ = measured rate (rad/s), yawRate = reference (rad/s). */
    fun feed(gtDist: Double, wheelDist: Double, gyrZ: Double, yawRate: Double) {
        if (finalized) return
        sumWheelDelta += wheelDist
        sumGtDelta += gtDist
        sumGyrErr += gyrZ - yawRate
        n++
        if (n >= finalizeAfterSamples) finalize()
    }

    private fun finalize() {
        if (finalized) return
        finalized = true
        if (n >= minSamplesScale && sumWheelDelta > 1e-9) {
            wheelScale = (sumGtDelta / sumWheelDelta).coerceIn(0.92, 1.08)
        }
        gyrBiasRad = (sumGyrErr / n).coerceIn(-Math.toRadians(0.5), Math.toRadians(0.5))
    }

    fun reset() {
        n = 0
        sumWheelDelta = 0.0
        sumGtDelta = 0.0
        sumGyrErr = 0.0
        finalized = false
        wheelScale = 1.0
        gyrBiasRad = 0.0
    }
}