package com.navx.ui

import android.content.Context
import android.content.SharedPreferences
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.navx.R

/**
 * Manual Sensor Calibration — on-device LoRA fine-tuning.
 *
 * Tunes the low-rank "adapter" parameters for the specific phone mount and
 * vehicle: accel biases, gyro-bias, wheel-scale factor and the map-matching
 * road belt. Values are stored on-device and applied live by the fusion
 * engine — they never leave the vehicle.
 */
class CalibrationActivity : AppCompatActivity() {

    private data class P(
        val key: String,
        val label: String,
        val unit: String,
        val min: Double,
        val max: Double,
        val step: Double,
        val digits: Int,
        val init: Double
    )

    private val prefs: SharedPreferences
        get() = getSharedPreferences(MainActivity.PREF, MODE_PRIVATE)

    private val params = listOf(
        P(MainActivity.KEY_ACC_X, "Accelerometer Bias X", "m/s²", -2.0, 2.0, 0.05, 2, 0.0),
        P(MainActivity.KEY_ACC_Y, "Accelerometer Bias Y", "m/s²", -2.0, 2.0, 0.05, 2, 0.0),
        P(MainActivity.KEY_GYR_Z, "Gyro Bias Z", "°/s", -10.0, 10.0, 0.2, 1, 0.0),
        P(MainActivity.KEY_SCALE, "Wheel Scale Factor", "1.00x", 0.90, 1.10, 0.005, 3, 1.0),
        P(MainActivity.KEY_BELT, "Map Band (road belt)", "m", 2.0, 20.0, 0.5, 1, 12.0)
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xFF0B1220.toInt())
        }

        val body = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 48, 24, 24)
        }

        body.addView(
            TextView(this).apply {
                text = "MANUAL SENSOR CALIBRATION"
                setTextColor(0xFF38BDF8.toInt())
                textSize = 20f
                setTypeface(typeface, android.graphics.Typeface.BOLD)
            }
        )
        body.addView(
            TextView(this).apply {
                text = "On-device LoRA fine-tuning · tune the low-rank weights for THIS phone mount + vehicle. " +
                    "Applied live to the EKF/filters · stays on-device."
                setTextColor(0xFF94A3B8.toInt())
                textSize = 13f
            }.also { it.layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = 8 } }
        )

        val values = HashMap<String, Double>()
        for (p in params) values[p.key] = prefs.getFloat(p.key, p.init.toFloat()).toDouble()

        val rowViews = ArrayList<LinearLayout>()
        for (p in params) {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = android.view.Gravity.CENTER_VERTICAL
                setBackgroundColor(0xFF12202E.toInt())
                setPadding(16, 12, 16, 12)
            }
            val left = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
            }
            left.addView(
                TextView(this@CalibrationActivity).apply {
                    text = p.label
                    setTextColor(0xFFE2E8F0.toInt())
                    textSize = 14f
                }
            )
            left.addView(
                TextView(this@CalibrationActivity).apply {
                    text = p.unit
                    setTextColor(0xFF64748B.toInt())
                    textSize = 11f
                }
            )
            val valueTv = TextView(this@CalibrationActivity).apply {
                setTextColor(0xFF38BDF8.toInt())
                textSize = 16f
                setTypeface(typeface, android.graphics.Typeface.BOLD)
                gravity = android.view.Gravity.CENTER
            }
            valueTv.layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.5f)

            fun fmt(v: Double) = "%.${p.digits}f".format(v)

            valueTv.text = fmt(values.getValue(p.key))
            val minus = Button(this@CalibrationActivity).apply {
                text = "−"
                minWidth = 0
                minHeight = 0
                setTextColor(0xFFE2E8F0.toInt())
            }
            val plus = Button(this@CalibrationActivity).apply {
                text = "+"
                minWidth = 0
                minHeight = 0
                setTextColor(0xFFE2E8F0.toInt())
            }
            minus.setOnClickListener {
                values[p.key] = (values.getValue(p.key) - p.step).coerceIn(p.min, p.max)
                valueTv.text = fmt(values.getValue(p.key))
            }
            plus.setOnClickListener {
                values[p.key] = (values.getValue(p.key) + p.step).coerceIn(p.min, p.max)
                valueTv.text = fmt(values.getValue(p.key))
            }

            row.addView(left, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 2f))
            row.addView(minus, LinearLayout.LayoutParams(120, LinearLayout.LayoutParams.WRAP_CONTENT))
            row.addView(valueTv, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.2f))
            row.addView(plus, LinearLayout.LayoutParams(120, LinearLayout.LayoutParams.WRAP_CONTENT))
            body.addView(row).also { row.layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = 14
                this@apply.height = 130
            } }
            rowViews.add(row)
        }

        val reset = Button(this).apply {
            text = "FACTORY RESET"
            setBackgroundColor(0xFF1E3A5A.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            isAllCaps = false
            setOnClickListener {
                val e = prefs.edit()
                for (p in params) {
                    values[p.key] = p.init
                    e.putFloat(p.key, p.init.toFloat())
                }
                e.apply()
                for (i in params.indices) {
                    val vt = (rowViews[i].getChildAt(2) as TextView)
                    vt.text = "%.${params[i].digits}f".format(params[i].init)
                }
            }
        }
        reset.layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        body.addView(reset).also { reset.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = 18 } }

        val save = Button(this).apply {
            text = "SAVE & APPLY (live to EKF)"
            setBackgroundColor(0xFF1E8E5A.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            isAllCaps = false
            setOnClickListener {
                val e = prefs.edit()
                for (p in params) e.putFloat(p.key, values.getValue(p.key).toFloat())
                e.apply()
                setResult(RESULT_OK)
                finish()
            }
        }
        body.addView(save).also { save.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = 10 } }

        root.addView(
            ScrollView(this).apply { addView(body, LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            )) },
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT)
        )
        setContentView(root)
    }
}