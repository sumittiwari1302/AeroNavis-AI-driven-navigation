package com.navx.engine

import kotlin.math.abs

/**
 * AI Speed & Vibration Filter.
 *
 * Statistical signal-processing filter that removes high-frequency road noise
 * (potholes, engine harmonics, bumps) and estimates forward velocity from the
 * smartphone IMU, nudged by GNSS speed when the fix is alive.
 */
class SpeedFilter {

    private var aFiltered = 0.0
    private var speed = 0.0
    private var vibrationEvents = 0L

    fun estimate(accLongitudinal: Float, dt: Float, satSpeedHint: Double?): Double {
        aFiltered = aFiltered * 0.6 + accLongitudinal * 0.4
        if (abs(accLongitudinal - aFiltered) > 2.5) vibrationEvents++
        speed += aFiltered * dt
        if (satSpeedHint != null) {
            speed = speed * 0.75 + satSpeedHint * 0.25
        }
        if (speed < 0.0) speed = 0.0
        return speed
    }

    fun vibrationCount(): Long = vibrationEvents

    fun reset() {
        aFiltered = 0.0
        speed = 0.0
        vibrationEvents = 0L
    }
}