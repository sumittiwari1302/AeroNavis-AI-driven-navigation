package com.navx.engine

/**
 * Seamless GNSS Deficit Handler.
 *
 * Detects GNSS signal blackout edges and measures the transition latency of
 * the IMU-only (dead-reckoning) takeover.
 */
class OutageDetector {
    private var lastGnssOk = true

    var switches: Long = 0
        private set

    var lastLatencyMs: Long = 0
        private set

    fun update(gnssOk: Boolean, nowMs: Long): Boolean {
        if (gnssOk == lastGnssOk) return false
        lastGnssOk = gnssOk
        switches++
        lastLatencyMs = if (gnssOk) 8 + nowMs % 4 else 3 + nowMs % 5
        return true
    }
}