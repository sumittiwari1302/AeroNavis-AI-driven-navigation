package com.navx.ui

import android.os.Bundle
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.navx.R
import com.navx.fusion.InEKF
import com.navx.fusion.InEKFConfig
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.sin

class MainActivity : AppCompatActivity() {

    private val inEKF = InEKF(InEKFConfig())
    private var running = false
    private var blackout = false
    private var tMs = 0L

    private lateinit var tvVelocity: TextView
    private lateinit var tvHeading: TextView
    private lateinit var tvStdX: TextView
    private lateinit var tvStdY: TextView
    private lateinit var tvTexture: TextView
    private lateinit var pbTextureConfidence: ProgressBar
    private lateinit var tvOutage5s: TextView
    private lateinit var tvOutage10s: TextView
    private lateinit var tvOutage15s: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvVelocity = findViewById(R.id.tvVelocity)
        tvHeading = findViewById(R.id.tvHeading)
        tvStdX = findViewById(R.id.tvStdX)
        tvStdY = findViewById(R.id.tvStdY)
        tvTexture = findViewById(R.id.tvTexture)
        pbTextureConfidence = findViewById(R.id.pbTextureConfidence)
        tvOutage5s = findViewById(R.id.tvOutage5s)
        tvOutage10s = findViewById(R.id.tvOutage10s)
        tvOutage15s = findViewById(R.id.tvOutage15s)

        findViewById<Button>(R.id.btnStartDemo).setOnClickListener { startDemo() }
        findViewById<Button>(R.id.btnSimulateBlackout).setOnClickListener {
            blackout = !blackout
        }
    }

    private fun startDemo() {
        if (running) return
        running = true
        lifecycleScope.launch {
            while (true) {
                delay(50)
                step()
            }
        }
    }

    private fun step() {
        tMs += 50
        val dt = 0.05f
        val tSec = tMs / 1000.0

        // Simulated ground truth: robot drives forward, gently turning.
        val headingRate = 0.6 * sin(tSec / 3.0) + 0.2
        val speed = 3.0 + 1.0 * sin(tSec / 2.0)

        // Simulated IMU readings.
        val acc = if (blackout) floatArrayOf(0f, 0f, 0f) else floatArrayOf(0.1f, 0.02f, 9.8f)
        val gyr = floatArrayOf(0.02f, -0.01f, headingRate.toFloat())

        inEKF.predict(acc, gyr, dt)

        if (!blackout) {
            inEKF.correctWheel(com.navx.fusion.WheelMeasurement(
                timestamp = tMs,
                vxBody = speed.toFloat()
            ))
        }

        runOnUiThread { updateUi() }
    }

    private fun updateUi() {
        val v = inEKF.velocity()[0]
        val headingDeg = Math.toDegrees(inEKF.heading())
        tvVelocity.text = String.format("Velocity: %.1f m/s", v)
        tvHeading.text = String.format("Heading: %.1f°", headingDeg)
        tvStdX.text = String.format("σx: %.2f m", inEKF.stdX())
        tvStdY.text = String.format("σy: %.2f m", inEKF.stdY())

        when {
            blackout -> tvTexture.text = "POOR"
            else -> tvTexture.text = "RICH"
        }
        pbTextureConfidence.progress = (if (blackout) 35 else 92)

        val risk = if (blackout) 0.9f else 0.15f
        tvOutage5s.text = String.format("5s: %.0f%%", risk * 100)
        tvOutage10s.text = String.format("10s: %.0f%%", risk * 100)
        tvOutage15s.text = String.format("15s: %.0f%%", risk * 100)
    }
}