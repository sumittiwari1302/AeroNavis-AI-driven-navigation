package com.navx.ui

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.navx.R

/**
 * 4-slide onboarding shown the first time a city is loaded (and reopenable from
 * the HELP button). Teaches the user how the satellite screen works: one tap
 * Start, GPS drops in tunnels, AI + map-matching keeps navigation exact,
 * and every run self-trains a tiny sensor model.
 */
class WalkthroughActivity : AppCompatActivity() {

    private data class Slide(val icon: String, val title: String, val desc: String)

    private val slides = listOf(
        Slide(
            "🗺️",
            "Your city, live on the map",
            "You picked a real city. NAV-X spun up its live satellite + street map right there — no static backdrop, real roads under the vehicle."
        ),
        Slide(
            "▶️",
            "One tap starts the drive",
            "Tap START and the demo vehicle starts driving the city blocks. The blue line is the AI path snapped onto the real streets, exactly like Google Maps."
        ),
        Slide(
            "🛰️🚫",
            "GPS drops — AI takes over",
            "In tunnels (and BLACKOUT) GNSS cuts out. The InEKF + map-matching keeps navigating seamlessly, drift stays under 10%, then GPS auto-reconnects. Watch the HUD go red, then green."
        ),
        Slide(
            "🧠",
            "Self-training every run",
            "NAV-X trains a tiny on-device model from the live sensors — wheel scale and gyro bias — so dead-reckoning stays exact even with no GPS."
        )
    )

    private var idx = 0
    private var helpOnly = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_walkthrough)

        helpOnly = intent.getBooleanExtra(MainActivity.EXTRA_HELP, false)

        findViewById<TextView>(R.id.btnSkip).setOnClickListener { leave() }
        findViewById<Button>(R.id.btnNext).setOnClickListener { next() }
        findViewById<Button>(R.id.btnGo).setOnClickListener { leave() }

        draw()
    }

    private fun next() {
        if (idx < slides.size - 1) {
            idx++
            draw()
        } else {
            leave()
        }
    }

    private fun leave() {
        if (helpOnly) {
            finish()
            return
        }
        getSharedPreferences(MainActivity.PREF_LOC, MODE_PRIVATE)
            .edit().putBoolean("seen_walkthrough", true).apply()
        val i = Intent(this, MainActivity::class.java)
        i.putExtra(MainActivity.EXTRA_NAME, intent.getStringExtra(MainActivity.EXTRA_NAME))
        i.putExtra(MainActivity.EXTRA_LAT, intent.getDoubleExtra(MainActivity.EXTRA_LAT, 0.0))
        i.putExtra(MainActivity.EXTRA_LNG, intent.getDoubleExtra(MainActivity.EXTRA_LNG, 0.0))
        startActivity(i)
        finish()
    }

    private fun draw() {
        val s = slides[idx]
        findViewById<TextView>(R.id.tvIcon).text = s.icon
        findViewById<TextView>(R.id.tvTitle).text = s.title
        findViewById<TextView>(R.id.tvDesc).text = s.desc

        val dots = findViewById<LinearLayout>(R.id.dotRow)
        dots.removeAllViews()
        for (k in slides.indices) {
            val d = View(this)
            val size = if (k == idx) 12 else 8
            d.background = getDrawable(R.drawable.dot)
            d.backgroundTintList = android.content.res.ColorStateList.valueOf(
                if (k == idx) 0xFF38BDF8.toInt() else 0xFF334155.toInt()
            )
            dots.addView(
                d,
                LinearLayout.LayoutParams(size, size).apply {
                    marginStart = 6
                    marginEnd = 6
                }
            )
        }

        findViewById<Button>(R.id.btnNext).text =
            if (idx == slides.size - 1) "I GOT IT ✓" else "NEXT →"
    }
}