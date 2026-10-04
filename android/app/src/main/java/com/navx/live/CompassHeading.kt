package com.navx.live

import android.hardware.GeomagneticField
import android.hardware.SensorManager
import com.navx.sensors.NavDataFrame

/**
 * Live magnetometer compass for GNSS-denied heading.
 *
 * Computes a tilt-compensated absolute heading from the smartphone's raw
 * accelerometer + magnetometer vectors and corrects magnetic declination using
 * the current geographic position [GeoMagneticField]. The result is expressed
 * in the InEKF world frame (yaw CCW from East, same convention as the map),
 * so it can be fed straight into correctGnss(x, y, magYaw) with a loose R on
 * the position channels.
 *
 * A small complementary filter smooths magnetic noise, and a hard gate rejects
 * sudden metal-interference spikes so they cannot jump the navigation heading.
 */
class CompassHeading {

    private val rotation = FloatArray(9)
    private val inclination = FloatArray(9)
    private val orientation = FloatArray(3)

    private var lastAcc = FloatArray(3)
    private var lastMag = FloatArray(3)
    private var hasAcc = false
    private var hasMag = false

    private var declinationDeg = 0f

    private var smoothedYaw: Double? = null

    fun reset() {
        hasAcc = false
        hasMag = false
        smoothedYaw = null
        declinationDeg = 0f
    }

    fun updateDeclination(lat: Double, lng: Double, altitude: Double, timeMs: Long) {
        declinationDeg =
            GeomagneticField(lat.toFloat(), lng.toFloat(), altitude.toFloat(), timeMs).declination
    }

    fun refresh(frame: NavDataFrame) {
        if (frame.accel.size >= 3) {
            lastAcc = frame.accel
            hasAcc = true
        }
        if (frame.mag.size >= 3) {
            lastMag = frame.mag
            hasMag = true
        }
    }

    fun hasData(): Boolean = hasAcc && hasMag

    /** Absolute vehicle-world yaw (rad, CCW from East) or null if mag unreliable. */
    fun headingYawRad(): Double? {
        if (!hasAcc || !hasMag) return null
        if (lastMag[0] == 0f && lastMag[1] == 0f && lastMag[2] == 0f) return null

        val ok = SensorManager.getRotationMatrix(rotation, inclination, lastAcc, lastMag)
        if (!ok) return null
        SensorManager.getOrientation(rotation, orientation)

        val azDeg = Math.toDegrees(orientation[0].toDouble()) + declinationDeg
        val yawFromEast = Math.toRadians(90.0 - azDeg)

        val prev = smoothedYaw
        if (prev == null) {
            smoothedYaw = yawFromEast
            return yawFromEast
        }
        val diff = wrapAngle(yawFromEast - prev)
        // Metal-interference gate: reject large instantaneous jumps.
        if (Math.abs(diff) > Math.toRadians(35.0)) return smoothedYaw
        val out = prev + 0.12 * diff
        smoothedYaw = out
        return out
    }

    private fun wrapAngle(a: Double): Double {
        var v = a
        while (v > Math.PI) v -= 2 * Math.PI
        while (v <= -Math.PI) v += 2 * Math.PI
        return v
    }
}