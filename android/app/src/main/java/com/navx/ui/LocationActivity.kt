package com.navx.ui

import android.content.Intent
import android.content.SharedPreferences
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.navx.R
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * State + City entry. Forward-geocodes the entered city (Nominatim) and hands
 * the real coordinates to MainActivity, which anchors its 3D satellite/street
 * map there — replacing the old static Bengaluru anchor.
 */
class LocationActivity : AppCompatActivity() {

    private lateinit var etState: EditText
    private lateinit var etCity: EditText
    private lateinit var tvResult: TextView
    private lateinit var btnSearch: Button
    private lateinit var btnLoad: Button

    private var foundLat = 0.0
    private var foundLng = 0.0
    private var foundName = ""
    private var foundState = ""
    private var foundCity = ""

    private val prefs: SharedPreferences
        get() = getSharedPreferences(MainActivity.PREF_LOC, MODE_PRIVATE)

    private val quick = listOf(
        "Bengaluru" to "Karnataka",
        "Pune" to "Maharashtra",
        "Mumbai" to "Maharashtra",
        "Hyderabad" to "Telangana",
        "Chennai" to "Tamil Nadu",
        "New Delhi" to "Delhi"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_location)

        etState = findViewById(R.id.etState)
        etCity = findViewById(R.id.etCity)
        tvResult = findViewById(R.id.tvResult)
        btnSearch = findViewById(R.id.btnSearch)
        btnLoad = findViewById(R.id.btnLoad)

        etCity.setText(prefs.getString("city", ""))
        etState.setText(prefs.getString("state", ""))

        val chipRow = findViewById<LinearLayout>(R.id.chipRow)
        for ((c, s) in quick) {
            val chip = Button(this).apply {
                text = c
                background = getDrawable(R.drawable.btn_round)
                setTextColor(0xFFE2E8F0.toInt())
                isAllCaps = false
                minWidth = 0
                minHeight = 0
                textSize = 11f
                setOnClickListener {
                    etCity.setText(c)
                    etState.setText(s)
                    tvResult.text = "📍 $c, $s — tap SEARCH."
                    tvResult.setTextColor(0xFF94A3B8.toInt())
                    btnLoad.isEnabled = false
                }
            }
            chipRow.addView(chip, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = 6 })
        }

        btnSearch.setOnClickListener { doSearch() }
        btnLoad.setOnClickListener {
            val e = prefs.edit()
            e.putString("city", foundCity)
            e.putString("state", foundState)
            e.putString("name", foundName)
            e.putFloat("lat", foundLat.toFloat())
            e.putFloat("lng", foundLng.toFloat())
            e.apply()

            val i = Intent(this, MainActivity::class.java)
            i.putExtra(MainActivity.EXTRA_NAME, foundName)
            i.putExtra(MainActivity.EXTRA_LAT, foundLat)
            i.putExtra(MainActivity.EXTRA_LNG, foundLng)
            val seen = prefs.getBoolean("seen_walkthrough", false)
            if (seen) {
                startActivity(i)
            } else {
                startActivity(
                    Intent(this, WalkthroughActivity::class.java)
                        .putExtra(MainActivity.EXTRA_NAME, foundName)
                        .putExtra(MainActivity.EXTRA_LAT, foundLat)
                        .putExtra(MainActivity.EXTRA_LNG, foundLng)
                )
            }
            finish()
        }
    }

    private data class Hit(val name: String, val lat: Double, val lng: Double)

    private fun doSearch() {
        val state = etState.text.toString().trim()
        val city = etCity.text.toString().trim()
        if (city.isEmpty() || state.isEmpty()) {
            tvResult.text = "❌ Enter both State and City."
            tvResult.setTextColor(0xFFFF6B6B.toInt())
            return
        }
        tvResult.text = "⏳ Locating $city, $state…"
        tvResult.setTextColor(0xFFE2E8F0.toInt())
        btnLoad.isEnabled = false

        lifecycleScope.launch(Dispatchers.IO) {
            val hit = forwardGeocode(city, state)
            runOnUiThread {
                if (hit == null) {
                    tvResult.text = "❌ Not found — check the spelling and try again."
                    tvResult.setTextColor(0xFFFF6B6B.toInt())
                } else {
                    foundState = state
                    foundCity = city
                    foundName = hit.name
                    foundLat = hit.lat
                    foundLng = hit.lng
                    tvResult.text = "📍 $foundName\n   Lat ${"%.4f".format(foundLat)} · Lon ${"%.4f".format(foundLng)} — found on real map."
                    tvResult.setTextColor(0xFF4ADE80.toInt())
                    btnLoad.isEnabled = true
                    btnLoad.text = "LOAD REAL MAP → $foundName"
                }
            }
        }
    }

    /** Forward-Geocodes city,state via Nominatim. */
    private fun forwardGeocode(city: String, state: String): Hit? {
        return try {
            val q = URLEncoder.encode("$city,$state", "UTF-8")
            val url = URL("https://nominatim.openstreetmap.org/search?format=jsonv2&q=$q&limit=1&addressdetails=1")
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "GET"
            conn.setRequestProperty("User-Agent", "AeroNavis/3.0 (SIH demo; GNSS-denied navigation)")
            conn.connectTimeout = 8000
            conn.readTimeout = 8000
            val raw = conn.inputStream.bufferedReader().use { it.readText() }
            conn.disconnect()
            val arr = org.json.JSONArray(raw)
            if (arr.length() == 0) return null
            val o = arr.getJSONObject(0)
            Hit(
                o.optString("display_name", "$city, $state"),
                o.optDouble("lat"),
                o.optDouble("lon")
            )
        } catch (e: Exception) {
            null
        }
    }
}