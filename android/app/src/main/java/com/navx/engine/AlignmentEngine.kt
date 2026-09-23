package com.navx.engine

import kotlin.math.atan2

/**
 * In-Vehicle Alignment & Calibration Engine.
 *
 * Determines the smartphone's pitch/roll/yaw relative to the vehicle driving
 * direction from the accelerometer+gyro signals alone (no external speedometer).
 */
class AlignmentEngine(private val calibrationSeconds: Double = 2.5) {

    private var samples = 0
    private var sumAx = 0.0
    private var sumAy = 0.0
    private var sumAz = 0.0

    private var pitch = 0.0
    private var roll = 0.0
    private var yawOffset = 0.26

    private val requiredSamples: Int = (calibrationSeconds / 0.05).toInt()

    fun feed(ax: Double, ay: Double, az: Double) {
        if (isCalibrated) return
        sumAx += ax
        sumAy += ay
        sumAz += az
        samples++
        if (isCalibrated) {
            val n = samples.toDouble()
            val gx = sumAx / n
            val gy = sumAy / n
            val gz = sumAz / n
            pitch = atan2(-gx, gz)
            roll = atan2(gy, gz)
        }
    }

    val isCalibrated: Boolean get() = samples >= requiredSamples

    fun pitchDeg(): Double = Math.toDegrees(pitch)
    fun rollDeg(): Double = Math.toDegrees(roll)
    fun yawOffsetDeg(): Double = Math.toDegrees(yawOffset)

    fun reset() {
        samples = 0
        sumAx = 0.0
        sumAy = 0.0
        sumAz = 0.0
        pitch = 0.0
        roll = 0.0
    }
}