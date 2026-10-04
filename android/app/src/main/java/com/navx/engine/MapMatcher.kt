package com.navx.engine

import kotlin.math.hypot

/**
 * Offline Map-Matching filter with Non-Holonomic Constraints (NHC).
 *
 * Snaps the drifting inertial trajectory back onto a road network whenever the
 * estimate is within a belt of a road segment. In demo mode it uses a synthetic
 * street grid; in LIVE mode the caller feeds the actual road the vehicle is on
 * via [setLiveRoads], so dead-reckoning snaps onto the real driven route.
 */
class MapMatcher(private val spacing: Double = 24.0, private val extent: Double = 168.0) {

    class Match(val x: Double, val y: Double, val snapped: Boolean, val offRoad: Double)

    private class Seg(val x1: Double, val y1: Double, val x2: Double, val y2: Double)

    private val roads: List<Seg> = buildList {
        var v = -extent
        while (v <= extent) {
            add(Seg(-extent, v, extent, v))
            add(Seg(v, -extent, v, extent))
            v += spacing
        }
    }

    private var liveSegs: List<Seg>? = null

    fun hasLiveRoads(): Boolean = liveSegs != null

    /** Replaces the road network with the real driven route (ENU XY points in travel order). */
    fun setLiveRoads(points: List<DoubleArray>) {
        val segs = ArrayList<Seg>(points.size)
        var prev: DoubleArray? = null
        for (p in points) {
            if (prev != null && hypot(p[0] - prev[0], p[1] - prev[1]) > 0.8) {
                segs.add(Seg(prev[0], prev[1], p[0], p[1]))
            }
            prev = p
        }
        liveSegs = if (segs.isEmpty()) null else segs
    }

    fun clearLiveRoads() {
        liveSegs = null
    }

    private fun activeRoads(): List<Seg> = liveSegs ?: roads

    fun match(x: Double, y: Double, belt: Double = 3.0): Match {
        var best: Seg? = null
        var bestD = Double.MAX_VALUE
        for (seg in activeRoads()) {
            val d = distanceToSeg(seg, x, y)
            if (d < bestD) {
                bestD = d
                best = seg
            }
        }
        val s = best ?: return Match(x, y, false, bestD)
        val p = project(s, x, y)
        return if (p.second <= belt) Match(p.first, p.second, true, bestD) else Match(x, y, false, bestD)
    }

    private fun distanceToSeg(s: Seg, x: Double, y: Double): Double {
        val dx = s.x2 - s.x1
        val dy = s.y2 - s.y1
        val len2 = dx * dx + dy * dy
        val t = if (len2 == 0.0) 0.0 else ((x - s.x1) * dx + (y - s.y1) * dy).coerceIn(0.0, 1.0)
        val px = s.x1 + t * dx
        val py = s.y1 + t * dy
        return hypot(x - px, y - py)
    }

    private fun project(s: Seg, x: Double, y: Double): Pair<Double, Double> {
        val dx = s.x2 - s.x1
        val dy = s.y2 - s.y1
        val len2 = dx * dx + dy * dy
        val t = if (len2 == 0.0) 0.0 else ((x - s.x1) * dx + (y - s.y1) * dy).coerceIn(0.0, 1.0)
        return Pair(s.x1 + t * dx, s.y1 + t * dy)
    }
}