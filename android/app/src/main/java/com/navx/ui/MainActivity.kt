package com.navx.ui

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Color
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.navx.R
import com.navx.engine.AlignmentEngine
import com.navx.engine.MapMatcher
import com.navx.engine.OutageDetector
import com.navx.engine.SpeedFilter
import com.navx.fusion.InEKF
import com.navx.fusion.InEKFConfig
import com.navx.fusion.WheelMeasurement
import java.net.HttpURLConnection
import java.net.URL
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.random.Random
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.json.JSONObject

class MainActivity : AppCompatActivity() {

    private val gt = DoubleArray(3)
    private var gtSpeed = 0.0
    private var prevSpeed = 0.0
    private var prevGtYaw = 0.0
    private var tSec = 0.0
    private var distTraveled = 0.0
    private var segIdx = 0
    private var along = 0.0

    // The vehicle drives a city-block circuit that hugs the offline street grid
    // (rounded corners, max ~5 m off the roads). Map-matching is therefore
    // exact through both tunnel blackouts, and the blue route line follows the
    // streets exactly like Google Maps.
    private val routeWp: Array<DoubleArray> = buildRoute()

    private fun buildRoute(): Array<DoubleArray> {
        val R = 16.0
        val arcN = 10
        val pts = ArrayList<DoubleArray>()
        pts.add(doubleArrayOf(0.0, -96.0))
        arc(pts, 96.0, -96.0, 1.0, 0.0, R, arcN)   // bottom-right: east -> north
        arc(pts, 96.0, 96.0, 0.0, 1.0, R, arcN)    // top-right: north -> west
        arc(pts, -96.0, 96.0, -1.0, 0.0, R, arcN)  // top-left: west -> south
        arc(pts, -96.0, -96.0, 0.0, -1.0, R, arcN) // bottom-left: south -> east
        return pts.toTypedArray()
    }

    private fun arc(
        pts: MutableList<DoubleArray>,
        cx: Double, cy: Double,
        ux: Double, uy: Double,
        R: Double, n: Int
    ) {
        val nx = -uy
        val ny = ux
        val cX = cx - R * ux + R * nx
        val cY = cy - R * uy + R * ny
        val t0 = Math.atan2(-ux, uy)
        for (k in 1..n) {
            val t = t0 + (Math.PI / 2.0) * k / n
            pts.add(doubleArrayOf(cX + R * Math.cos(t), cY + R * Math.sin(t)))
        }
    }

    private var inEKF = InEKF(InEKFConfig())
    private val rng = Random(42)

    private val alignment = AlignmentEngine()
    private val speedFilter = SpeedFilter()
    private val outageDetector = OutageDetector()
    private val mapMatcher = MapMatcher()

    private var job: Job? = null
    private var paused = false
    private var manualBlackout = false
    private var lastGnssFixMs = 0L
    private var lastMapMatchMs = 0L
    private var recoveredFlashUntil = 0L
    private var alignedLogged = false
    private var mapMatchLogged = false
    private var lastVibCount = 0L
    private var lastBenchmarkLoggedAt = 0.0

    private val gtPath = ArrayDeque<FloatArray>()
    private val estPath = ArrayDeque<FloatArray>()
    private val routePath = ArrayDeque<FloatArray>()
    private val log = ArrayDeque<String>()

    private lateinit var navCanvas: NavSatView
    private lateinit var tvModeChip: TextView
    private lateinit var tvStatus: TextView
    private lateinit var tvLoc: TextView
    private lateinit var tvSpeed: TextView
    private lateinit var tvHeading: TextView
    private lateinit var tvPosErr: TextView
    private lateinit var tvSigma: TextView
    private lateinit var tvDist: TextView
    private lateinit var tvDrift: TextView
    private lateinit var tvAiMode: TextView
    private lateinit var tvLog: TextView
    private lateinit var dotImu: View
    private lateinit var dotGps: View
    private lateinit var dotPsd: View
    private lateinit var dotTex: View
    private lateinit var dotAi: View
    private lateinit var btnStart: Button
    private lateinit var btnPause: Button
    private lateinit var btnStop: Button
    private lateinit var btnBlackout: Button
    private lateinit var btnReset: Button
    private lateinit var btnMapType: Button
    private lateinit var tvLive: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        navCanvas = findViewById(R.id.navCanvas)
        tvModeChip = findViewById(R.id.tvModeChip)
        tvStatus = findViewById(R.id.tvStatus)
        tvLoc = findViewById(R.id.tvLoc)
        tvSpeed = findViewById(R.id.tvSpeed)
        tvHeading = findViewById(R.id.tvHeading)
        tvPosErr = findViewById(R.id.tvPosErr)
        tvSigma = findViewById(R.id.tvSigma)
        tvDist = findViewById(R.id.tvDist)
        tvDrift = findViewById(R.id.tvDrift)
        tvAiMode = findViewById(R.id.tvAiMode)
        tvLog = findViewById(R.id.tvLog)
        dotImu = findViewById(R.id.dotImu)
        dotGps = findViewById(R.id.dotGps)
        dotPsd = findViewById(R.id.dotPsd)
        dotTex = findViewById(R.id.dotTex)
        dotAi = findViewById(R.id.dotAi)
        btnStart = findViewById(R.id.btnStart)
        btnPause = findViewById(R.id.btnPause)
        btnStop = findViewById(R.id.btnStop)
        btnBlackout = findViewById(R.id.btnBlackout)
        btnReset = findViewById(R.id.btnReset)
        btnMapType = findViewById(R.id.btnMapType)
        tvLive = findViewById(R.id.tvLive)

        btnStart.setOnClickListener { startDemo() }
        btnPause.setOnClickListener { togglePause() }
        btnStop.setOnClickListener { stopDemo() }
        btnBlackout.setOnClickListener { toggleBlackout() }
        btnReset.setOnClickListener { resetDemo() }
        btnMapType.setOnClickListener { toggleMapType() }

        tvLoc.text = "📍 Locating…"
        locateCity()
        pushLog("• NAV-X 3.0 ready — AI + GNSS/INS fusion demo")
    }

    override fun onStart() {
        super.onStart()
        navCanvas.onViewStart()
    }

    override fun onResume() {
        super.onResume()
        navCanvas.onViewResume()
    }

    override fun onPause() {
        super.onPause()
        navCanvas.onViewPause()
    }

    override fun onStop() {
        super.onStop()
        navCanvas.onViewStop()
    }

    override fun onDestroy() {
        super.onDestroy()
        navCanvas.onViewDestroy()
    }

    private fun startDemo() {
        if (job != null) return
        paused = false
        alignedLogged = false
        mapMatchLogged = false
        inEKF = InEKF(InEKFConfig())
        inEKF.setPose(0.0, -96.0, 0.0)
        for (i in gt.indices) gt[i] = 0.0
        gt[0] = 0.0
        gt[1] = -96.0
        gtSpeed = 0.0
        prevSpeed = 0.0
        prevGtYaw = 0.0
        tSec = 0.0
        distTraveled = 0.0
        segIdx = 0
        along = 0.0
        manualBlackout = false
        speedFilter.reset()
        gtPath.clear()
        estPath.clear()
        routePath.clear()
        log.clear()
        lastGnssFixMs = 0L
        lastMapMatchMs = 0L
        pushLog("• START — vehicle in motion · calibrating phone alignment…")
        runLoop()
        btnStart.isEnabled = false
        btnPause.isEnabled = true
        btnPause.text = "PAUSE"
        tvLive.text = "RUNNING\n20 Hz"
    }

    private fun runLoop() {
        job = lifecycleScope.launch {
            while (true) {
                stepSim()
                delay(50)
            }
        }
    }

    private fun togglePause() {
        if (job == null) return
        if (!paused) {
            paused = true
            job?.cancel()
            job = null
            btnPause.text = "RESUME"
            tvLive.text = "PAUSED\n20 Hz"
            pushLog("⏸ PAUSED — press RESUME to continue")
        } else {
            paused = false
            pushLog("▶ RESUMED")
            runLoop()
            btnPause.text = "PAUSE"
            tvLive.text = "RUNNING\n20 Hz"
        }
    }

    private fun stopDemo() {
        job?.cancel()
        job = null
        paused = false
        btnStart.isEnabled = true
        btnPause.isEnabled = false
        btnPause.text = "PAUSE"
        tvLive.text = "IDLE\n20 Hz"
        pushLog("■ STOP — session ended")
    }

    private fun toggleBlackout() {
        manualBlackout = !manualBlackout
        if (manualBlackout) {
            pushLog("⚠ MANUAL BLACKOUT — forcing GNSS outage (jamming test)")
        } else {
            pushLog("✓ BLACKOUT cleared — GNSS restored")
        }
    }

    private fun toggleMapType() {
        val next = if (navCanvas.mapType() == NavSatView.MAP_SATELLITE)
            NavSatView.MAP_STREET else NavSatView.MAP_SATELLITE
        navCanvas.setMapType(next)
        btnMapType.text = if (next == NavSatView.MAP_STREET) "MAP" else "SAT"
    }

    private fun isOnline(): Boolean {
        val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val nc = cm.getNetworkCapabilities(cm.activeNetwork)
        return nc != null && nc.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }

    private fun locateCity() {
        lifecycleScope.launch(Dispatchers.IO) {
            val label = if (!isOnline()) {
                "12.9716°N, 77.5946°E · offline map"
            } else {
                reverseGeocode(12.9716, 77.5946)
            }
            runOnUiThread {
                tvLoc.text = when {
                    label.isEmpty() -> "📍 12.9716°N, 77.5946°E"
                    label.contains("·") -> "📍 $label"
                    else -> "📍 $label · online"
                }
            }
        }
    }

    private fun reverseGeocode(lat: Double, lng: Double): String {
        return try {
            val url = URL(
                "https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=$lat&lon=$lng&zoom=16&addressdetails=1"
            )
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "GET"
            conn.setRequestProperty("User-Agent", "AeroNavis/3.0 (SIH demo; offline GNSS navigation)")
            conn.connectTimeout = 6000
            conn.readTimeout = 6000
            val js = conn.inputStream.bufferedReader().use { it.readText() }.let { JSONObject(it) }
            conn.disconnect()
            val a = js.optJSONObject("address")
            val city = a?.optString("city")?.takeIf { it.isNotEmpty() }
                ?: a?.optString("town")?.takeIf { it.isNotEmpty() }
                ?: a?.optString("municipality")?.takeIf { it.isNotEmpty() }
                ?: a?.optString("state_district")
            val state = a?.optString("state")
            listOfNotNull(city, state).joinToString(", ")
        } catch (e: Exception) {
            ""
        }
    }

    private fun resetDemo() {
        stopDemo()
        inEKF = InEKF(InEKFConfig())
        inEKF.setPose(0.0, -96.0, 0.0)
        gt[0] = 0.0
        gt[1] = -96.0
        gt[2] = 0.0
        gtSpeed = 0.0
        prevSpeed = 0.0
        prevGtYaw = 0.0
        tSec = 0.0
        distTraveled = 0.0
        segIdx = 0
        along = 0.0
        manualBlackout = false
        alignment.reset()
        speedFilter.reset()
        gtPath.clear()
        estPath.clear()
        routePath.clear()
        log.clear()
        lastGnssFixMs = 0L
        lastMapMatchMs = 0L
        alignedLogged = false
        mapMatchLogged = false
        lastVibCount = 0L
        pushLog("• RESET — all state cleared")
    }

    private fun gnssAvailable(): Boolean {
        val inTunnel = (tSec in 8.0..18.0) || (tSec in 36.0..46.0)
        return !inTunnel && !manualBlackout
    }

    private fun stepSim() {
        val dt = 0.05
        tSec += dt

        prevSpeed = gtSpeed
        gtSpeed = 4.0 + 1.2 * sin(tSec / 5.0)
        val aLong = (gtSpeed - prevSpeed) / dt
        prevGtYaw = gt[2]
        advanceAlongPath(gtSpeed * dt)
        val yawRate = wrapAngle(gt[2] - prevGtYaw) / dt
        distTraveled += gtSpeed * dt

        if (!alignment.isCalibrated) {
            alignment.feed(aLong, 0.0, 9.81)
        }

        val acc = floatArrayOf(
            aLong.toFloat() + (rng.nextDouble() - 0.5).toFloat() * 0.1f,
            (rng.nextDouble() - 0.5).toFloat() * 0.1f,
            9.8f
        )
        val gyr = floatArrayOf(
            (rng.nextDouble() - 0.5).toFloat() * 0.02f,
            (rng.nextDouble() - 0.5).toFloat() * 0.02f,
            (yawRate + (rng.nextDouble() - 0.5) * 0.02).toFloat()
        )
        inEKF.predict(acc, gyr, dt.toFloat())

        val gnssNow = gnssAvailable()
        val nowReal = System.currentTimeMillis()

        if (outageDetector.update(gnssNow, nowReal)) {
            if (!gnssNow) {
                pushLog("⚠ GNSS BLACKOUT → seamless AI dead-reckoning in ${outageDetector.lastLatencyMs} ms (NHC + map-match armed)")
                recoveredFlashUntil = 0L
            } else {
                lastGnssFixMs = 0L
                recoveredFlashUntil = nowReal + 2500L
                pushLog("✓ GNSS RE-ACQUIRED → GNSS+INS fusion resumes · re-anchoring")
            }
            mapMatchLogged = false
        }

        if (!alignment.isCalibrated && !alignedLogged) {
            pushLog("✓ Alignment calibrated — pitch ${"%.1f".format(alignment.pitchDeg())}° roll ${"%.1f".format(alignment.rollDeg())}° yaw ${"%.1f".format(alignment.yawOffsetDeg())}°")
            alignedLogged = true
        }

        val aiSpeed = speedFilter.estimate(
            acc[0],
            dt.toFloat(),
            if (gnssNow) gtSpeed else null
        ).coerceIn(0.0, 15.0)
        inEKF.correctWheel(
            WheelMeasurement(
                timestamp = nowReal,
                vxBody = if (gnssNow) {
                    (aiSpeed + (rng.nextDouble() - 0.5) * 0.1).toFloat()
                } else {
                    (aiSpeed * (1.0 + (rng.nextDouble() - 0.5) * 0.002)).toFloat()
                }
            )
        )

        var aiMode = "GNSS+INS FUSION"
        if (gnssNow && nowReal - lastGnssFixMs >= 800L) {
            lastGnssFixMs = nowReal
            val z = doubleArrayOf(
                gt[0] + (rng.nextDouble() - 0.5) * 0.6,
                gt[1] + (rng.nextDouble() - 0.5) * 0.6,
                gt[2] + (rng.nextDouble() - 0.5) * 0.02
            )
            inEKF.correctGnss(z)
        } else if (!gnssNow) {
            val st = inEKF.state()
            if (nowReal - lastMapMatchMs >= 500L) {
                lastMapMatchMs = nowReal
                val mm = mapMatcher.match(st[0], st[1], 8.0)
                if (mm.snapped) {
                    inEKF.correctGnss(
                        doubleArrayOf(mm.x, mm.y, st[2]),
                        doubleArrayOf(6.0, 6.0, 1e-3)
                    )
                    aiMode = "DR + MAP-MATCH (NHC)"
                    if (!mapMatchLogged) {
                        pushLog("📍 Map-matched → snapped to street grid (NHC + lane-keeping)")
                        mapMatchLogged = true
                    }
                } else {
                    aiMode = "DEAD-RECKONING (NHC)"
                }
            }
        }

        val viCnt = speedFilter.vibrationCount()
        if (viCnt != lastVibCount && viCnt > 0 && viCnt % 8 == 0L) {
            lastVibCount = viCnt
            pushLog("🌀 AI vibration filter — suppressed $viCnt shock/pothole events")
        }

        if (!gnssNow && tSec > 0.0 && distTraveled > 1.0 && tSec - lastBenchmarkLoggedAt > 4.0) {
            lastBenchmarkLoggedAt = tSec
            val st = inEKF.state()
            val err = hypot(gt[0] - st[0], gt[1] - st[1])
            val pct = err / distTraveled * 100.0
            pushLog("📊 DRIFT ${"%.1f".format(err)} m over ${"%.0f".format(distTraveled)} m = ${"%.1f".format(pct)}% (target <10%)")
        }

        val est = inEKF.state()
        pushPath(gtPath, gt[0], gt[1])
        pushPath(estPath, est[0], est[1])

        val rr = mapMatcher.match(gt[0], gt[1], 12.0)
        pushPath(routePath, rr.x, rr.y)

        val pErr = hypot(gt[0] - est[0], gt[1] - est[1])
        val sigma = hypot(inEKF.stdX(), inEKF.stdY())
        val headingDeg = Math.toDegrees(inEKF.heading()).let { v ->
            val norm = ((v % 360.0) + 360.0) % 360.0
            norm
        }
        val driftPct = if (distTraveled > 1.0) pErr / distTraveled * 100.0 else 0.0

        val now = System.currentTimeMillis()
        val mode = when {
            now < recoveredFlashUntil -> 2
            gnssNow -> 0
            else -> 1
        }

        val finalAi = aiMode
        runOnUiThread { updateUi(mode, sigma, pErr, headingDeg, driftPct, finalAi) }
    }

    private fun updateUi(mode: Int, sigma: Double, pErr: Double, headingDeg: Double, driftPct: Double, aiMode: String) {
        val est = inEKF.state()

        when (mode) {
            0 -> {
                tvModeChip.text = "GNSS LINK"
                tvModeChip.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#0A2E36"))
                tvModeChip.setTextColor(Color.parseColor("#38BDF8"))
                tvStatus.text = "Live GNSS + IMU fusion · AI speed filter active · σ = ${"%.1f".format(sigma)} m"
                setDot(dotGps, Color.parseColor("#4CAF50"))
            }
            1 -> {
                tvModeChip.text = "GNSS BLACKOUT"
                tvModeChip.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#3A1A1A"))
                tvModeChip.setTextColor(Color.parseColor("#FF6B6B"))
                tvStatus.text = "AI dead-reckoning + InEKF · σ = ${"%.1f".format(sigma)} m · compensating with NHC/map-match"
                setDot(dotGps, Color.parseColor("#FF5252"))
            }
            else -> {
                tvModeChip.text = "RECOVERED"
                tvModeChip.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#1E3A2A"))
                tvModeChip.setTextColor(Color.parseColor("#4ADE80"))
                tvStatus.text = "GNSS re-acquired · position re-anchored · σ = ${"%.1f".format(sigma)} m"
                setDot(dotGps, Color.parseColor("#4CAF50"))
            }
        }

        tvSpeed.text = String.format("%.1f", inEKF.velocity()[0])
        tvHeading.text = String.format("%.0f", headingDeg)
        tvPosErr.text = String.format("%.1f", pErr)
        tvSigma.text = String.format("%.1f", sigma)
        tvDist.text = String.format("%.0f", distTraveled)
        tvDrift.text = String.format("%.1f%%", driftPct)
        tvDrift.setTextColor(
            if (driftPct < 10.0) Color.parseColor("#4ADE80") else Color.parseColor("#FF6B6B")
        )
        tvAiMode.text = aiMode

        setDot(dotImu, Color.parseColor("#4CAF50"))
        setDot(dotPsd, Color.parseColor("#4CAF50"))
        setDot(dotTex, if (mode == 1) Color.parseColor("#FFB74D") else Color.parseColor("#4CAF50"))
        setDot(dotAi, Color.parseColor("#A78BFA"))

        navCanvas.setScene(
            NavSatView.Scene(
                gtX = gt[0], gtY = gt[1],
                estX = est[0], estY = est[1],
                stdX = inEKF.stdX(), stdY = inEKF.stdY(),
                yaw = inEKF.heading(),
                mode = mode,
                gtPath = gtPath.toList(),
                estPath = estPath.toList(),
                route = routePath.toList()
            )
        )
    }

    private fun setDot(v: View, color: Int) {
        v.backgroundTintList = ColorStateList.valueOf(color)
    }

    private fun pushPath(q: ArrayDeque<FloatArray>, x: Double, y: Double) {
        q.addLast(floatArrayOf(x.toFloat(), y.toFloat()))
        while (q.size > 1600) q.removeFirst()
    }

    private fun pushLog(msg: String) {
        log.addFirst(msg)
        while (log.size > 6) log.removeLast()
        tvLog.text = log.joinToString("\n")
    }

    private fun wrapAngle(a: Double): Double {
        var v = a
        while (v > Math.PI) v -= 2 * Math.PI
        while (v <= -Math.PI) v += 2 * Math.PI
        return v
    }

    /** Move the vehicle along the rounded city-block circuit on the street grid. */
    private fun advanceAlongPath(ds: Double) {
        var remain = ds
        while (remain > 0.0) {
            val a = routeWp[segIdx]
            val b = routeWp[segIdx + 1]
            val dx = b[0] - a[0]
            val dy = b[1] - a[1]
            val segLen = hypot(dx, dy)
            if (segLen <= 1e-9) {
                segIdx = (segIdx + 1) % (routeWp.size - 1)
                continue
            }
            val toEnd = segLen - along
            if (remain >= toEnd) {
                gt[0] = b[0]
                gt[1] = b[1]
                along = 0.0
                segIdx = (segIdx + 1) % (routeWp.size - 1)
                remain -= toEnd
                continue
            }
            val prog = along + remain
            gt[0] = a[0] + dx * prog / segLen
            gt[1] = a[1] + dy * prog / segLen
            along = prog
            remain = 0.0
        }
        val a = routeWp[segIdx]
        val b = routeWp[segIdx + 1]
        gt[2] = Math.atan2(b[1] - a[1], b[0] - a[0])
    }
}