package com.navx.ui

import android.content.Intent
import android.content.SharedPreferences
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.navx.R

/**
 * Three input modes for USER/VEHICLE flexibility (per the SIH problem):
 *  0. DEVELOPER / INTEGRATOR — automated adaptation for pre-built APKs.
 *  1. MANUAL SENSOR CALIBRATION — on-device LoRA fine-tuning (CalibrationActivity).
 *  2. FLEET DEPLOYMENT — automated adaptation for pre-built APKs, locked fleet profile.
 */
class ModeSelectActivity : AppCompatActivity() {

    private data class M(
        val id: Int,
        val code: String,
        val title: String,
        val desc: String
    )

    private val prefs: SharedPreferences
        get() = getSharedPreferences(MainActivity.PREF, MODE_PRIVATE)

    private val modes = listOf(
        M(
            MainActivity.MODE_DEV,
            "DEV",
            "Developer / Integrator Mode",
            "Automated adaptation for pre-built APKs. Full telemetry, live sensor diagnostics and the integration API surface stay exposed for testing hooks."
        ),
        M(
            MainActivity.MODE_LORA,
            "LoRA",
            "Manual Sensor Calibration",
            "On-device LoRA fine-tuning. Tune the low-rank weights (IMU biases, wheel scale, map band) for THIS phone mount and THIS vehicle."
        ),
        M(
            MainActivity.MODE_FLEET,
            "FLEET",
            "Fleet Deployment Mode",
            "Automated adaptation for pre-built APKs. A locked, standardized deployment profile pushes to a fleet of vehicles with zero per-vehicle setup."
        )
    )

    private var selected = MainActivity.MODE_DEV
    private val rows = ArrayList<Pair<LinearLayout, M>>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        selected = prefs.getInt(MainActivity.KEY_MODE, MainActivity.MODE_DEV)
        render()
    }

    private fun render() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xFF0B1220.toInt())
            setPadding(24, 48, 24, 24)
        }

        root.addView(
            TextView(this).apply {
                text = "NAV-X 3.0 · INPUT MODES"
                setTextColor(0xFF38BDF8.toInt())
                textSize = 22f
                setTypeface(typeface, android.graphics.Typeface.BOLD)
            }
        )
        root.addView(
            TextView(this).apply {
                text = "Pick how NAV-X adapts to this user / vehicle. Select then APPLY."
                setTextColor(0xFF94A3B8.toInt())
                textSize = 13f
            }.also { it.layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = 8 } }
        )

        for (m in modes) {
            val card = modeCard(m)
            root.addView(card).also { card.layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = 20 } }
            rows.add(card to m)
        }

        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }

        bar.addView(Button(this).apply {
            text = "CANCEL"
            setTextColor(0xFF94A3B8.toInt())
            isAllCaps = false
            setOnClickListener { finish() }
        }.also { it.layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f) })

        bar.addView(Button(this).apply {
            text = "APPLY"
            setBackgroundColor(0xFF1E8E5A.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            isAllCaps = false
            setOnClickListener {
                prefs.edit().putInt(MainActivity.KEY_MODE, selected).apply()
                setResult(RESULT_OK, Intent().putExtra(MainActivity.EXTRA_MODE, selected))
                finish()
            }
        }.also { it.layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f) })

        root.addView(bar).also { bar.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = 28 } }

        setContentView(root)
        rows.firstOrNull { it.second.id == selected }?.let { highlight(it.first, it.second) }
    }

    private fun modeCard(m: M): LinearLayout =
        LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(20, 16, 20, 16)
            background = getDrawable(R.drawable.card_option)
            isClickable = true
            setOnClickListener {
                for ((card, mm) in rows) highlight(card, mm)
                selected = m.id
            }
            val head = LinearLayout(this@ModeSelectActivity).apply {
                orientation = LinearLayout.HORIZONTAL
            }
            head.addView(
                TextView(this@ModeSelectActivity).apply {
                    text = m.code
                    setTextColor(0xFF38BDF8.toInt())
                    textSize = 15f
                    setTypeface(typeface, android.graphics.Typeface.BOLD)
                }
            )
            head.addView(
                TextView(this@ModeSelectActivity).apply {
                    text = "  ${m.title}"
                    setTextColor(0xFFE2E8F0.toInt())
                    textSize = 16f
                    setTypeface(typeface, android.graphics.Typeface.BOLD)
                }
            )
            addView(head)
            addView(
                TextView(this@ModeSelectActivity).apply {
                    text = m.desc
                    setTextColor(0xFF94A3B8.toInt())
                    textSize = 13f
                }.also { it.layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { topMargin = 6 } }
            )
        }

    private fun highlight(card: View, m: M) {
        card.background = getDrawable(
            if (selected == m.id) R.drawable.card_option_sel else R.drawable.card_option
        )
    }
}