package com.navx.ui

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.navx.R
import com.navx.engine.AlignmentEngine
import com.navx.engine.MapMatcher
import com.navx.engine.OutageDetector
import com.navx.engine.SpeedFilter
import com.navx.fusion.InEKF
import com.navx.fusion.InEKFConfig
import com.navx.fusion.WheelMeasurement
import com.navx.inference.MLVelocityEstimator
import com.navx.inference.TFLiteRuntime
import com.navx.live.CompassHeading
import com.navx.sensors.NavDataFrame
import com.navx.sensors.SensorHub
import kotlin.math.cos
import kotlin.math.hypot
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    companion object {
        const val PREF = "navx_cal"
        const val KEY_MODE = "mode"
        const val KEY_ACC_X = "accBiasX"
        const val KEY_ACC_Y = "accBiasY"
        const val KEY_GYR_Z = "gyrBiasZ"
        const val KEY_SCALE = "wheelScale"
        const val KEY_BELT = "beltM"
        const val EXTRA_MODE = "mode"
        const val PREF_LOC = "navx_loc"
        const val EXTRA_NAME = "loc_name"
        const val EXTRA_LAT = "loc_lat"
        const val EXTRA_LNG = "loc_lng"
        const val EXTRA_HELP = "help_only"

        const val MODE_DEV = 0
        const val MODE_LORA = 1
        const val MODE_FLEET = 2

        private const val RC_MODE = 1001
        private const val RC_CAL = 1002
        private const val RC_LOCATION = 2001

        /** A fix older than this is treated as GNSS LOST → dead-reckoning takeover. */
        private const val GNSS_LOST_TIMEOUT_MS = 3500L

        data class Cal(
            val accBiasX: Double,
            val accBiasY: Double,
            val gyrBiasZRad: Double,
            val wheelScale: Double,
            val beltM: Double
        )
    }

    // ---- Live navigation state -------------------------------------------------
    private var inEKF = InEKF(InEKFConfig())
    private val alignment = AlignmentEngine()
    private val speedFilter = SpeedFilter()
    private val outageDetector = OutageDetector()
    private val mapMatcher = MapMatcher()
    private val compass = CompassHeading()
    private var mlVelocityEstimator: MLVelocityEstimator? = null

    // Serialized raw GPS position (ENU meters) + fused estimate marker.
    private val rawGps = DoubleArray(3)
    private var distTraveled = 0.0

    // Live data sources.
    private var sensorHub: SensorHub? = null
    private lateinit var locationManager: LocationManager
    private var gpsFix: Location? = null
    private var lastGpsFixMs = 0L
    private var lastGpsSpeed = 0.0
    private var liveOriginLat: Double? = null
    private var liveOriginLng: Double? = null
    private var lastImuFrame: NavDataFrame? = null
    private var lastTickMs = 0L
    private var lastMapMatchMs = 0L
    private var lastCompassCorrMs = 0L
    private var lastSkeletonFeedMs = 0L
    private var lastBenchmarkLoggedAt = 0L
    private var recoveredFlashUntil = 0L
    private var lastVibCount = 0L
    private var yawOffsetLearned = 0.0
    private var yawOffSamples = 0
    private var yawOffsetLogged = false

    // The real road the vehicle is driving on (trail of live positions).
    private val liveRoadPts = ArrayDeque<DoubleArray>()
    private val gtPath = ArrayDeque<FloatArray>()
    private val estPath = ArrayDeque<FloatArray>()
    private val routePath = ArrayDeque<FloatArray>()
    private val log = ArrayDeque<String>()

    private var job: Job? = null
    private var paused = false
    private var manualBlackout = false
    private var liveStarted = false

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
    private lateinit var btnMode: Button
    private lateinit var btnHelp: Button
    private lateinit var tvIntro: TextView
    private lateinit var tvLive: TextView

    private var curMode = MODE_DEV
    private var cal = Cal(0.0, 0.0, 0.0, 1.0, 12.0)

    private var locName = "Bengaluru, Karnataka"
    private var locLat = 12.9716
    private var locLng = 77.5946

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
        btnMode = findViewById(R.id.btnMode)
        btnHelp = findViewById(R.id.btnHelp)
        tvIntro = findViewById(R.id.tvIntro)
        tvLive = findViewById(R.id.tvLive)

        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        btnStart.setOnClickListener { startLive() }
        btnPause.setOnClickListener { togglePause() }
        btnStop.setOnClickListener { stopLive() }
        btnBlackout.setOnClickListener { toggleBlackout() }
        btnReset.setOnClickListener { resetLive() }
        btnMapType.setOnClickListener { toggleMapType() }
        btnMode.setOnClickListener {
            startActivityForResult(Intent(this, ModeSelectActivity::class.java), RC_MODE)
        }
        btnHelp.setOnClickListener {
            startActivity(Intent(this, WalkthroughActivity::class.java).putExtra(EXTRA_HELP, true))
        }

        reloadModeAndCal()
        applyLocation()
        ensurePermissions()
        pushLog("• NAV-X 3.0 LIVE — real smartphone IMU + GPS fusion (no synthetic data)")
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
        stopLive()
        navCanvas.onViewDestroy()
    }

    // ---- Permissions -----------------------------------------------------------
    private fun ensurePermissions() {
        if (hasLocationPermission()) return
        ActivityCompat.requestPermissions(
            this,
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ),
            RC_LOCATION
        )
    }

    private fun hasLocationPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == RC_LOCATION) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                pushLog("✓ Location permission granted — GNSS fixes enabled")
            } else {
                pushLog("⚠ Location permission DENIED — no GNSS · IMU-only dead-reckoning will still run")
            }
        }
    }

    // ---- Live track start / pause / stop / reset --------------------------------
    private fun startLive() {
        if (job != null || liveStarted) return
        if (!hasLocationPermission()) {
            pushLog("⚠ Location permission needed for GNSS — grant it, then press START ▶")
            ensurePermissions()
            return
        }
        liveStarted = true
        paused = false
        alignedLogged = false
        mapMatchLogged = false
        inEKF = InEKF(InEKFConfig())
        inEKF.setPose(0.0, 0.0, 0.0)
        alignment.reset()
        speedFilter.reset()
        compass.reset()
        mapMatcher.clearLiveRoads()
        liveRoadPts.clear()
        gtPath.clear()
        estPath.clear()
        routePath.clear()
        log.clear()
        manualBlackout = false
        distTraveled = 0.0
        rawGps[0] = 0.0; rawGps[1] = 0.0; rawGps[2] = 0.0
        lastGpsSpeed = 0.0
        lastGpsFixMs = 0L
        lastMapMatchMs = 0L
        lastCompassCorrMs = 0L
        lastSkeletonFeedMs = 0L
        lastBenchmarkLoggedAt = 0L
        lastTickMs = System.currentTimeMillis()
        recoveredFlashUntil = 0L
        lastVibCount = 0L
        yawOffsetLearned = 0.0
        yawOffSamples = 0
        yawOffsetLogged = false
        lastImuFrame = null

        // Fallback origin = chosen city anchor until the first real satellite fix.
        if (liveOriginLat == null) {
            liveOriginLat = locLat
            liveOriginLng = locLng
        }

        sensorHub = SensorHub(this, this).also { it.start() }
        lifecycleScope.launch(Dispatchers.Default) {
            sensorHub?.dataFlow?.collect { f -> lastImuFrame = f }
        }

        compass.updateDeclination(liveOriginLat!!, liveOriginLng!!, 0.0, System.currentTimeMillis())

        // Initialize ML velocity estimator (TFLite model)
        mlVelocityEstimator = MLVelocityEstimator(this, TFLiteRuntime.Delegate.CPU)
        pushLog("🤖 ML velocity model loaded (TCN+LSTM, 200-sample window)")

        // Seed with last known satellite fix if the app was open before.
        try {
            val last = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
            if (last != null && System.currentTimeMillis() - last.time < 60_000L) {
                onGpsFix(last)
            }
        } catch (_: SecurityException) {
        }
        runCatching {
            locationManager.requestLocationUpdates(
                LocationManager.GPS_PROVIDER, 1000L, 0f, gpsListener
            )
        }

        pushLog("📡 LIVE TRACKING ON — real accelerometer + gyroscope + magnetometer · GPS fixes at 1 Hz")
        pushLog("📍 Dead-reckoning auto-arms when GNSS drops · map-matching on the driven road")
        runLoop()
        btnStart.isEnabled = false
        btnPause.isEnabled = true
        btnPause.text = "PAUSE"
        tvLive.text = "LIVE\n20 Hz"
        tvIntro.visibility = View.GONE
        tvStatus.text = "Waiting for first GNSS fix…"
    }

    private fun runLoop() {
        job = lifecycleScope.launch {
            while (true) {
                stepLive()
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
            tvLive.text = "LIVE\n20 Hz"
        }
    }

    private fun stopLive() {
        job?.cancel()
        job = null
        paused = false
        sensorHub?.stop()
        sensorHub = null
        mlVelocityEstimator?.close()
        mlVelocityEstimator = null
        runCatching { locationManager.removeUpdates(gpsListener) }
        if (liveStarted) {
            liveStarted = false
            btnStart.isEnabled = true
            btnPause.isEnabled = false
            btnPause.text = "PAUSE"
            tvLive.text = "IDLE\n20 Hz"
            tvIntro.visibility = View.VISIBLE
            tvIntro.text = "🔹 Ready — press START ▶ to begin real live tracking (car/tunnel test)."
            pushLog("■ STOP — live session ended")
        }
    }

    private fun toggleBlackout() {
        manualBlackout = !manualBlackout
        if (manualBlackout) {
            pushLog("⚠ MANUAL BLACKOUT — forcing GNSS outage (jamming test · GPS signal ignored)")
        } else {
            pushLog("✓ BLACKOUT cleared — GNSS restored")
        }
    }

    private fun resetLive() {
        stopLive()
        liveOriginLat = null
        liveOriginLng = null
        inEKF = InEKF(InEKFConfig())
        inEKF.setPose(0.0, 0.0, 0.0)
        distTraveled = 0.0
        rawGps[0] = 0.0; rawGps[1] = 0.0; rawGps[2] = 0.0
        alignment.reset()
        speedFilter.reset()
        mapMatcher.clearLiveRoads()
        liveRoadPts.clear()
        gtPath.clear()
        estPath.clear()
        routePath.clear()
        log.clear()
        lastVibCount = 0L
        pushLog("• RESET — all state cleared · origin will re-lock on the next GNSS fix")
    }

    // ---- GPS receiver -----------------------------------------------------------
    private val gpsListener = object : LocationListener {
        override fun onLocationChanged(location: Location) {
            onGpsFix(location)
        }

        override fun onProviderEnabled(provider: String) {
            pushLog("📡 GPS provider enabled — waiting for satellite fixes…")
        }

        override fun onProviderDisabled(provider: String) {
            if (provider == LocationManager.GPS_PROVIDER) {
                gpsFix = null
                pushLog("🛑 GPS provider disabled — dead-reckoning only")
            }
        }

        override fun onStatusChanged(provider: String, status: Int, extras: Bundle) {}
    }

    private fun onGpsFix(location: Location) {
        gpsFix = location
        lastGpsFixMs = System.currentTimeMillis()
        if (liveOriginLat == null) {
            // Lock the ENU origin & map anchor on the very first real satellite fix.
            liveOriginLat = location.latitude
            liveOriginLng = location.longitude
            navCanvas.setLocation(location.latitude, location.longitude)
            compass.updateDeclination(location.latitude, location.longitude, location.altitude, System.currentTimeMillis())
            if (!liveStarted) return
            val (x, y) = geoToEnu(location.latitude, location.longitude)
            inEKF.setPose(x, y, inEKF.heading())
            pushLog("🎯 GNSS origin locked at ${"%.5f".format(location.latitude)}, ${"%.5f".format(location.longitude)} — satellite-denied tracking starts from here")
        }
    }

    private fun hasLiveGnss(now: Long): Boolean =
        gpsFix != null &&
            (now - lastGpsFixMs) < GNSS_LOST_TIMEOUT_MS &&
            locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)

    private fun geoToEnu(lat: Double, lng: Double): Pair<Double, Double> {
        val lat0 = liveOriginLat ?: locLat
        val lng0 = liveOriginLng ?: locLng
        val c = cos(Math.toRadians(lat0))
        val x = (lng - lng0) * 111320.0 * c
        val y = (lat - lat0) * 110540.0
        return x to y
    }

    // ---- Live dead-reckoning + fusion step --------------------------------------
    private fun stepLive() {
        val now = System.currentTimeMillis()
        if (lastTickMs == 0L) lastTickMs = now
        val dt = ((now - lastTickMs) / 1000.0).coerceIn(0.01, 0.25).toFloat()
        lastTickMs = now

        val frame = lastImuFrame
        val gnssOk = hasLiveGnss(now)
        val gnssNow = gnssOk && !manualBlackout

        if (frame != null) {
            compass.refresh(frame)
            if (!alignment.isCalibrated) {
                alignment.feed(frame.accel[0].toDouble(), frame.accel[1].toDouble(), frame.accel[2].toDouble())
            }
            if (!imuLogged) {
                imuLogged = true
                pushLog("🧭 IMU streams live — accel [${"%.2f".format(frame.accel[0])}, ${"%.2f".format(frame.accel[1])}, ${"%.2f".format(frame.accel[2])}] · gyr [${"%.3f".format(frame.gyro[0])}, ${"%.3f".format(frame.gyro[1])}, ${"%.3f".format(frame.gyro[2])}] · magnetometer ${if (frame.mag[0] != 0f || frame.mag[1] != 0f || frame.mag[2] != 0f) "present" else "absent (DR uses gyro+map-match)"}")
            }
            
            // Feed IMU to ML velocity estimator (runs inference every ~1s)
            val mlResult = mlVelocityEstimator?.addSample(frame)
            if (mlResult != null) {
                val (mlVel, mlConf) = mlResult
                pushLog("🤖 ML velocity: ${"%.2f".format(mlVel)} m/s (conf: ${"%.0f".format(mlConf * 100)}%)")
            }
        }

        // ---- Real IMU prediction: accelerometer + gyroscope (calibration biases) ----
        val acc = if (frame != null)
            floatArrayOf(
                frame.accel[0] - cal.accBiasX.toFloat(),
                frame.accel[1] - cal.accBiasY.toFloat(),
                frame.accel[2]
            ) else floatArrayOf(0f, 0f, 9.8f)
        val gyr = if (frame != null) floatArrayOf(
            frame.gyro[0],
            frame.gyro[1],
            frame.gyro[2] - cal.gyrBiasZRad.toFloat()
        ) else floatArrayOf(0f, 0f, 0f)
        inEKF.predict(acc, gyr, dt)

        // ---- GNSS / outage transition logging ----
        if (outageDetector.update(gnssNow, now)) {
            if (gnssNow) {
                recoveredFlashUntil = now + 2500L
                mapMatchLogged = false
                pushLog("✓ GNSS RE-ACQUIRED → GNSS+INS fusion resumes · re-anchoring")
            } else {
                recoveredFlashUntil = 0L
                mapMatchLogged = false
                pushLog(
                    if (manualBlackout)
                        "⚠ MANUAL BLACKOUT → dead-reckoning on · IMU + map-match keep you on road"
                    else
                        "⚠ GNSS LOST → DEAD RECKONING · IMU integration + map-match keep you on road"
                )
            }
        }

        // ---- Forward speed source (odometry / NHC wheel proxy) ----
        val mlReady = mlVelocityEstimator?.isReady() == true
        val mlVel = mlVelocityEstimator?.getLastVelocity() ?: 0f
        
        // ---- Forward speed source (odometry / NHC wheel proxy) ----
        val vx: Double = if (gnssNow && gpsFix != null && gpsFix!!.hasSpeed()) {
            gpsFix!!.speed.toDouble().coerceIn(0.0, 45.0).also { lastGpsSpeed = it }
        } else if (!gnssNow && mlReady) {
            // Use ML velocity during GNSS outage (primary source)
            mlVel.toDouble().coerceIn(0.0, 45.0)
        } else if (frame != null) {
            // Fallback to physics-based SpeedFilter
            speedFilter.estimate(frame.accel[0], dt, lastGpsSpeed).coerceIn(0.0, 45.0)
        } else {
            lastGpsSpeed.coerceIn(0.0, 45.0)
        }
        inEKF.correctWheel(WheelMeasurement(timestamp = now, vxBody = vx.toFloat()))
        distTraveled += hypot(vx, 0.0) * dt

        var aiMode = when {
            gnssNow -> "GNSS+INS FUSION"
            mlReady -> "DR + ML VELOCITY"
            else -> "DEAD-RECKONING (NHC)"
        }

        // ---- Raw GNSS measurement ----
        if (gnssNow && gpsFix != null) {
            val loc = gpsFix!!
            val (x, y) = geoToEnu(loc.latitude, loc.longitude)
            rawGps[0] = x
            rawGps[1] = y
            val courseKnown = loc.hasBearing() && !loc.bearing.isNaN() && loc.hasSpeed() && loc.speed > 1.5f
            val courseYaw = if (courseKnown) Math.toRadians(90.0 - loc.bearing) else inEKF.heading()
            if (courseKnown) {
                learnYawOffset(courseYaw)
            }
            inEKF.correctGnss(doubleArrayOf(x, y, courseYaw))
            pushLiveRoad(x, y)
        }

        // ---- Magnetometer absolute heading (compass) measurement ----
        if (now - lastCompassCorrMs >= 1000L && compass.hasData()) {
            val st = inEKF.state()
            val magYaw = magYawCorrected()
            if (magYaw != null) {
                lastCompassCorrMs = now
                val rYaw = Math.toRadians(7.0) * Math.toRadians(7.0)
                // Huge R on x/y: only the heading channel matters for GNSS-denied drift.
                inEKF.correctGnss(doubleArrayOf(st[0], st[1], magYaw), doubleArrayOf(1e3, 1e3, rYaw))
            }
        }

        // ---- Dead-reckoning map-match correction (GNSS-denied) ----
        var offRoad = 0.0
        if (!gnssNow) {
            val st = inEKF.state()
            if (now - lastMapMatchMs >= 150L) {
                lastMapMatchMs = now
                val mm = mapMatcher.match(st[0], st[1], cal.beltM)
                offRoad = mm.offRoad
                if (mapMatcher.hasLiveRoads() && mm.snapped) {
                    inEKF.correctGnss(doubleArrayOf(mm.x, mm.y, st[2]), doubleArrayOf(6.0, 6.0, 1e-3))
                    if (mlReady) aiMode = "DR + ML VEL + MAP-MATCH"
                    else aiMode = "DR + MAP-MATCH"
                    if (!mapMatchLogged) {
                        mapMatchLogged = true
                        pushLog("📍 Map-matched → snapped onto the driven road (NHC + lane-keeping)")
                    }
                    pushLiveRoad(mm.x, mm.y)
                } else {
                    // Keep the ML velocity mode if active, otherwise show DR mode
                    if (!mlReady) aiMode = "DEAD-RECKONING (NHC)"
                    // Extend the road skeleton from DR positions only while they stay near
                    // the known road — otherwise a drifted estimate would poison future snaps.
                    if (!mapMatcher.hasLiveRoads()) {
                        pushLiveRoad(st[0], st[1])
                    } else if (offRoad < cal.beltM * 1.5) {
                        pushLiveRoad(st[0], st[1])
                    }
                }
            }
        }

        // ---- Vibration filter logging (real potholes/shocks) ----
        val vib = speedFilter.vibrationCount()
        if (vib != lastVibCount && vib > 0 && vib % 8 == 0L) {
            lastVibCount = vib
            pushLog("🌀 Vibration filter — suppressed $vib road-noise events")
        }

        // ---- Periodic DRIFT benchmark while GNSS is down ----
        if (!gnssNow && distTraveled > 1.0 && (now - lastBenchmarkLoggedAt > 8000L)) {
            lastBenchmarkLoggedAt = now
            val st = inEKF.state()
            val err = offRoad
            val pct = (err / distTraveled.coerceAtLeast(1.0)) * 100.0
            pushLog("📊 DRIFT ${"%.1f".format(err)} m off-road over ${"%.0f".format(distTraveled)} m = ${"%.1f".format(pct)}% · IMU + map-match holding")
        }

        if (!alignment.isCalibrated && !alignedLogged) {
            pushLog("✓ Phone alignment (static) — pitch ${"%.1f".format(alignment.pitchDeg())}° roll ${"%.1f".format(alignment.rollDeg())}° · holder tilt only, DR uses motion yaw")
            alignedLogged = true
        }
        if (compass.hasData() && !yawOffsetLogged && yawOffSamples >= 30) {
            yawOffsetLogged = true
            pushLog("🧭 Compass ↔ GNSS-course aligned — static yaw offset ${"%.1f".format(Math.toDegrees(yawOffsetLearned))}°")
        }

        // ---- Paths + UI ----
        val est = inEKF.state()
        pushPath(gtPath, rawGps[0], rawGps[1])
        pushPath(estPath, est[0], est[1])
        val snap = if (mapMatcher.hasLiveRoads()) {
            val m = mapMatcher.match(est[0], est[1], 30.0)
            if (m.snapped) doubleArrayOf(m.x, m.y) else est.sliceArray(0..1)
        } else {
            est.sliceArray(0..1)
        }
        pushPath(routePath, snap[0], snap[1])

        val gnssAgeS = (now - lastGpsFixMs) / 1000.0
        val err = if (gnssNow) hypot(rawGps[0] - est[0], rawGps[1] - est[1]) else offRoad
        val sigma = hypot(inEKF.stdX(), inEKF.stdY())
        val headingDeg = normDeg(Math.toDegrees(inEKF.heading()))
        val driftPct = if (distTraveled > 1.0) err / distTraveled * 100.0 else 0.0

        val mode = when {
            now < recoveredFlashUntil -> 2
            gnssNow -> 0
            else -> 1
        }

        runOnUiThread { updateUi(mode, gnssNow, gnssAgeS, sigma, err, headingDeg, driftPct, aiMode) }
    }

    private fun magYawCorrected(): Double? {
        val base = compass.headingYawRad() ?: return null
        val v = base + yawOffsetLearned
        return wrapAngle(v)
    }

    /** Learns the constant phone↔vehicle yaw offset from GNSS course vs compass. */
    private fun learnYawOffset(courseYaw: Double) {
        val magYaw = compass.headingYawRad() ?: return
        if (yawOffSamples >= 400) return
        val sample = wrapAngle(courseYaw - magYaw)
        yawOffSamples++
        val w = 0.06
        yawOffsetLearned = wrapAngle(yawOffsetLearned * (1.0 - w) + sample * w)
    }

    // ---- Live road skeleton for map matching -------------------------------------
    private fun pushLiveRoad(x: Double, y: Double) {
        val last = liveRoadPts.lastOrNull()
        if (last == null || hypot(x - last[0], y - last[1]) >= 2.5) {
            liveRoadPts.addLast(doubleArrayOf(x, y))
            while (liveRoadPts.size > 700) liveRoadPts.removeFirst()
            feedSkeletonIfDirty()
        } else if (liveRoadPts.size < 3) {
            feedSkeletonIfDirty()
        }
    }

    private fun feedSkeletonIfDirty() {
        val now = System.currentTimeMillis()
        if (now - lastSkeletonFeedMs < 500L) return
        lastSkeletonFeedMs = now
        if (liveRoadPts.size < 2) return
        mapMatcher.setLiveRoads(liveRoadPts.toList())
    }

    // ---- UI ----------------------------------------------------------------------
    private fun updateUi(
        mode: Int,
        gnssNow: Boolean,
        gnssAgeS: Double,
        err: Double,
        sigma: Double,
        headingDeg: Double,
        driftPct: Double,
        aiMode: String
    ) {
        val est = inEKF.state()
        val mlReady = mlVelocityEstimator?.isReady() == true
        val mlVel = mlVelocityEstimator?.getLastVelocity() ?: 0f
        val mlInferences = mlVelocityEstimator?.getInferenceCount() ?: 0L

        when (mode) {
            0 -> {
                tvModeChip.text = "GNSS ACTIVE"
                tvModeChip.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#0A2E36"))
                tvModeChip.setTextColor(Color.parseColor("#38BDF8"))
                tvStatus.text = "🟢 GNSS ACTIVE · GPS + IMU fused · σ = ${"%.1f".format(sigma)} m · GPS age ${"%.0f".format(gnssAgeS)}s"
                tvStatus.setTextColor(Color.parseColor("#A7F3D0"))
                setDot(dotGps, Color.parseColor("#4CAF50"))
            }
            1 -> {
                tvModeChip.text = "GNSS LOST"
                tvModeChip.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#3A1A1A"))
                tvModeChip.setTextColor(Color.parseColor("#FF6B6B"))
                val mlStr = if (mlReady) " · ML vel: ${"%.2f".format(mlVel)} m/s (#$mlInferences)" else ""
                tvStatus.text = "🔴 GNSS LOST · DEAD RECKONING${mlStr} · σ = ${"%.1f".format(sigma)} m · GPS age ${"%.0f".format(gnssAgeS)}s"
                tvStatus.setTextColor(Color.parseColor("#FFB4AB"))
                setDot(dotGps, Color.parseColor("#FF5252"))
            }
            else -> {
                tvModeChip.text = "REACQUIRED"
                tvModeChip.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#1E3A2A"))
                tvModeChip.setTextColor(Color.parseColor("#4ADE80"))
                tvStatus.text = "🟢 GPS RE-ACQUIRED · position re-anchored · fusion resumed · σ = ${"%.1f".format(sigma)} m"
                tvStatus.setTextColor(Color.parseColor("#A7F3D0"))
                setDot(dotGps, Color.parseColor("#4CAF50"))
            }
        }

        tvSpeed.text = String.format("%.1f", inEKF.velocity()[0])
        tvHeading.text = String.format("%.0f", headingDeg)
        tvPosErr.text = String.format("%.1f", err)
        tvSigma.text = String.format("%.1f", sigma)
        tvDist.text = String.format("%.0f", distTraveled)
        tvDrift.text = String.format("%.1f%%", driftPct)
        tvDrift.setTextColor(
            if (driftPct < 10.0) Color.parseColor("#4ADE80") else Color.parseColor("#FF6B6B")
        )
        tvAiMode.text = aiMode

        val curLat = liveOriginLat?.plus(est[1] / 110540.0) ?: locLat
        val curLng = liveOriginLng?.plus(est[0] / (111320.0 * cos(Math.toRadians(liveOriginLat ?: locLat)))) ?: locLng
        tvLoc.text = if (gnssNow)
            "📍 ${"%.5f".format(curLat)}, ${"%.5f".format(curLng)} · GPS LIVE"
        else
            "📍 ${"%.5f".format(curLat)}, ${"%.5f".format(curLng)} · DEAD-RECKON"

        setDot(dotImu, Color.parseColor("#4CAF50"))
        setDot(dotPsd, Color.parseColor("#4CAF50"))
        setDot(dotTex, if (mode == 1) Color.parseColor("#FFB74D") else Color.parseColor("#4CAF50"))
        setDot(dotAi, Color.parseColor("#A78BFA"))

        navCanvas.setScene(
            NavSatView.Scene(
                gtX = rawGps[0], gtY = rawGps[1],
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
        android.util.Log.d("NAVX", msg.replace("\n", " | "))
    }

    private fun normDeg(a: Double): Double {
        var v = a
        while (v >= 360.0) v -= 360.0
        while (v < 0.0) v += 360.0
        return v
    }

    private fun wrapAngle(a: Double): Double {
        var v = a
        while (v > Math.PI) v -= 2 * Math.PI
        while (v <= -Math.PI) v += 2 * Math.PI
        return v
    }

    private var alignedLogged = false
    private var mapMatchLogged = false
    private var imuLogged = false

    private fun toggleMapType() {
        val next = if (navCanvas.mapType() == NavSatView.MAP_SATELLITE)
            NavSatView.MAP_STREET else NavSatView.MAP_SATELLITE
        navCanvas.setMapType(next)
        btnMapType.text = if (next == NavSatView.MAP_STREET) "MAP" else "SAT"
    }

    private fun reloadModeAndCal() {
        val prefs = getSharedPreferences(PREF, MODE_PRIVATE)
        curMode = prefs.getInt(KEY_MODE, MODE_DEV)
        cal = loadCal(prefs)
        btnMode.text = when (curMode) {
            MODE_LORA -> "LoRA"
            MODE_FLEET -> "FLEET"
            else -> "DEV"
        }
    }

    private fun loadCal(prefs: SharedPreferences): Cal {
        if (curMode == MODE_FLEET) {
            return Cal(0.0, 0.0, 0.0, 1.02, 10.0)
        }
        val gzDeg = prefs.getFloat(KEY_GYR_Z, 0f).toDouble()
        return Cal(
            prefs.getFloat(KEY_ACC_X, 0f).toDouble(),
            prefs.getFloat(KEY_ACC_Y, 0f).toDouble(),
            Math.toRadians(gzDeg),
            prefs.getFloat(KEY_SCALE, 1f).toDouble(),
            prefs.getFloat(KEY_BELT, 8f).toDouble()
        )
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        when (requestCode) {
            RC_MODE -> {
                reloadModeAndCal()
                pushLog(
                    when (curMode) {
                        MODE_LORA -> "🧲 MODE: Manual Sensor Calibration — on-device LoRA fine-tuning active"
                        MODE_FLEET -> "🚚 MODE: Fleet Deployment — locked standard profile FLD-042 active"
                        else -> "🛠 MODE: Developer / Integrator — automated adaptation for pre-built APKs, telemetry on"
                    }
                )
                if (curMode == MODE_LORA) {
                    startActivityForResult(Intent(this, CalibrationActivity::class.java), RC_CAL)
                }
            }
            RC_CAL -> reloadModeAndCal()
        }
    }

    private fun isOnline(): Boolean {
        val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val nc = cm.getNetworkCapabilities(cm.activeNetwork)
        return nc != null && nc.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }

    private fun applyLocation() {
        val p = getSharedPreferences(PREF_LOC, MODE_PRIVATE)
        val hasPref = p.contains("lat")
        locLat = intent.getDoubleExtra(
            EXTRA_LAT,
            if (hasPref) p.getFloat("lat", 12.9716f).toDouble() else 12.9716
        )
        locLng = intent.getDoubleExtra(
            EXTRA_LNG,
            if (hasPref) p.getFloat("lng", 77.5946f).toDouble() else 77.5946
        )
        val n = intent.getStringExtra(EXTRA_NAME)
            ?: if (hasPref) p.getString("name", "") ?: "" else ""
        locName = if (n.isEmpty()) "${"%.4f".format(locLat)}°N, ${"%.4f".format(locLng)}°E" else n
        navCanvas.setLocation(locLat, locLng)
        tvLoc.text = "📍 $locName · ${if (isOnline()) "online" else "offline"}"
        tvIntro.text = "🔹 LIVE TRACKING — phone cocktail-dash me rkho, GPS tunnel me cut hota hai (apne aap detect), IMU + magnetometer + map-match tumhe road pe rakhte hain. Press START ▶."
        tvIntro.visibility = View.VISIBLE
        tvStatus.text = "Setup ready · press START ▶"
        pushLog("• Map anchored at $locName — real satellite + street tiles")
    }
}