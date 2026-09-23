package com.navx.ui

import android.content.Context
import android.graphics.Color
import android.util.AttributeSet
import android.widget.FrameLayout
import org.maplibre.android.MapLibre
import org.maplibre.android.WellKnownTileServer
import org.maplibre.android.camera.CameraPosition
import org.maplibre.android.camera.CameraUpdateFactory
import org.maplibre.android.geometry.LatLng
import org.maplibre.android.maps.MapLibreMap
import org.maplibre.android.maps.MapView
import org.maplibre.android.maps.Style
import org.maplibre.android.style.layers.CircleLayer
import org.maplibre.android.style.layers.FillExtrusionLayer
import org.maplibre.android.style.layers.FillLayer
import org.maplibre.android.style.layers.LineLayer
import org.maplibre.android.style.layers.PropertyFactory
import org.maplibre.android.style.sources.GeoJsonSource
import org.maplibre.geojson.Feature
import org.maplibre.geojson.FeatureCollection
import org.maplibre.geojson.LineString
import org.maplibre.geojson.Point
import org.maplibre.geojson.Polygon
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/**
 * Live 3D satellite map backed by MapLibre.
 *
 * Uses free Esri World Imagery satellite tiles (no API key) and renders the
 * real ground, then overlays:
 *   - ground-truth path (green line)
 *   - InEKF estimated path  (cyan line)
 *   - the running 3-sigma uncertainty as an extruded 3D tower that grows on a
 *     GNSS blackout and collapses when the fix is re-acquired
 *   - a white dot at the current estimated position
 *
 * The camera starts tilted (60°) for a Google-Maps-bird's-eye 3D look; users
 * can rotate/tilt/zoom with the standard map gestures.
 */
class NavSatView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : FrameLayout(context, attrs, defStyleAttr) {

    data class Scene(
        val gtX: Double = 0.0,
        val gtY: Double = 0.0,
        val estX: Double = 0.0,
        val estY: Double = 0.0,
        val stdX: Double = 1.0,
        val stdY: Double = 1.0,
        val yaw: Double = 0.0,
        val mode: Int = 0, // 0=link, 1=outage, 2=recovered
        val gtPath: List<FloatArray> = emptyList(),
        val estPath: List<FloatArray> = emptyList()
    )

    // Bengaluru (ISRO HQ) — sensor-fusion demo lives in local meters around here.
    private val anchorLat = 12.9716
    private val anchorLng = 77.5946

    private val mapView: MapView
    private var map: MapLibreMap? = null
    private var styleReady = false
    private var firstCamera = true

    private var gtSrc: GeoJsonSource? = null
    private var estSrc: GeoJsonSource? = null
    private var uncSrc: GeoJsonSource? = null
    private var posSrc: GeoJsonSource? = null
    private var gtPosSrc: GeoJsonSource? = null
    private var uncExtrusion: FillExtrusionLayer? = null
    private var uncFill: FillLayer? = null

    init {
        // MapLibre 11 requires an instance before the MapView is created.
        // Tiles come from Esri World Imagery (asset style), so this dummy key
        // is only needed to satisfy the SDK's configuration check.
        MapLibre.getInstance(context, "aeronavis-satellite-demo", WellKnownTileServer.MapLibre)
        mapView = MapView(context)
        mapView.layoutParams = LayoutParams(
            LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT
        )
        addView(mapView)
        mapView.getMapAsync { m ->
            map = m
            m.setStyle(Style.Builder().fromUri("asset://satellite.json")) { style ->
                setupLayers(style)
                styleReady = true
                m.moveCamera(
                    CameraUpdateFactory.newCameraPosition(
                        CameraPosition.Builder()
                            .target(LatLng(anchorLat, anchorLng))
                            .zoom(17.2)
                            .tilt(60.0)
                            .build()
                    )
                )
            }
        }
    }

    private fun setupLayers(style: Style) {
        val empty = FeatureCollection.fromFeatures(emptyList())

        gtSrc = GeoJsonSource("gt-src", empty)
        estSrc = GeoJsonSource("est-src", empty)
        uncSrc = GeoJsonSource("unc-src", empty)
        posSrc = GeoJsonSource("pos-src", empty)
        gtPosSrc = GeoJsonSource("gt-pos-src", empty)

        val g = gtSrc as GeoJsonSource
        val e = estSrc as GeoJsonSource
        val u = uncSrc as GeoJsonSource
        val p = posSrc as GeoJsonSource
        val gp = gtPosSrc as GeoJsonSource
        style.addSource(g)
        style.addSource(e)
        style.addSource(u)
        style.addSource(p)
        style.addSource(gp)

        style.addLayer(
            LineLayer("gt-line", "gt-src").withProperties(
                PropertyFactory.lineColor(Color.parseColor("#4CAF50")),
                PropertyFactory.lineWidth(3.5f),
                PropertyFactory.lineOpacity(0.95f)
            )
        )
        style.addLayer(
            LineLayer("est-line", "est-src").withProperties(
                PropertyFactory.lineColor(Color.parseColor("#38BDF8")),
                PropertyFactory.lineWidth(5f),
                PropertyFactory.lineOpacity(0.95f)
            )
        )
        style.addLayer(
            FillLayer("unc-fill", "unc-src").withProperties(
                PropertyFactory.fillColor(Color.parseColor("#EAB308")),
                PropertyFactory.fillOpacity(0.16f),
                PropertyFactory.fillOutlineColor(Color.parseColor("#FACC15"))
            )
        )
        val extrusion = FillExtrusionLayer("unc-ext", "unc-src").withProperties(
            PropertyFactory.fillExtrusionColor(Color.parseColor("#EAB308")),
            PropertyFactory.fillExtrusionOpacity(0.35f),
            PropertyFactory.fillExtrusionHeight(6f)
        )
        uncExtrusion = extrusion
        style.addLayer(extrusion)
        style.addLayer(
            CircleLayer("pos", "pos-src").withProperties(
                PropertyFactory.circleColor(Color.parseColor("#FFFFFF")),
                PropertyFactory.circleRadius(6f),
                PropertyFactory.circleStrokeColor(Color.parseColor("#38BDF8")),
                PropertyFactory.circleStrokeWidth(2f)
            )
        )
        style.addLayer(
            CircleLayer("gt-pos", "gt-pos-src").withProperties(
                PropertyFactory.circleColor(Color.parseColor("#A7F3D0")),
                PropertyFactory.circleRadius(4f),
                PropertyFactory.circleStrokeColor(Color.parseColor("#15803D")),
                PropertyFactory.circleStrokeWidth(1.5f)
            )
        )
    }

    fun setScene(scene: Scene) {
        val m = map ?: return
        if (!styleReady) return

        followCamera(m, scene.estX, scene.estY)

        val outage = scene.mode == 1

        val tgts = scene.gtPath.map { f ->
            Point.fromLngLat(lng(f[0].toDouble()), lat(f[1].toDouble()))
        }
        val tests = scene.estPath.map { f ->
            Point.fromLngLat(lng(f[0].toDouble()), lat(f[1].toDouble()))
        }
        val g = gtSrc
        val e = estSrc
        val u = uncSrc
        val p = posSrc
        val gp = gtPosSrc
        g?.setGeoJson(LineString.fromLngLats(tgts))
        e?.setGeoJson(LineString.fromLngLats(tests))

        val sigmaX = scene.stdX * 3.0
        val sigmaY = scene.stdY * 3.0
        val ring = (0..36).map { i ->
            val a = 2.0 * PI * i / 36
            Point.fromLngLat(
                lng(scene.estX + sigmaX * cos(a)),
                lat(scene.estY + sigmaY * sin(a))
            )
        }
        u?.setGeoJson(Polygon.fromLngLats(listOf(ring)))

        val height = max(3.0f, ((sigmaX + sigmaY) / 2.0).toFloat())
        val wallColor = if (outage) Color.parseColor("#EF4444") else Color.parseColor("#EAB308")
        uncExtrusion?.let { ext ->
            ext.setProperties(
                PropertyFactory.fillExtrusionHeight(height),
                PropertyFactory.fillExtrusionColor(wallColor),
                PropertyFactory.fillExtrusionOpacity(if (outage) 0.5f else 0.35f)
            )
            uncFill?.setProperties(
                PropertyFactory.fillColor(if (outage) Color.parseColor("#EF4444") else Color.parseColor("#EAB308")),
                PropertyFactory.fillOpacity(if (outage) 0.22f else 0.16f)
            )
        }

        p?.setGeoJson(
            FeatureCollection.fromFeatures(
                listOf(Feature.fromGeometry(Point.fromLngLat(lng(scene.estX), lat(scene.estY))))
            )
        )
        gp?.setGeoJson(
            FeatureCollection.fromFeatures(
                listOf(Feature.fromGeometry(Point.fromLngLat(lng(scene.gtX), lat(scene.gtY))))
            )
        )
    }

    private fun followCamera(m: MapLibreMap, x: Double, y: Double) {
        val newLat = lat(y)
        val newLng = lng(x)
        if (firstCamera) {
            firstCamera = false
            m.moveCamera(
                CameraUpdateFactory.newCameraPosition(
                    CameraPosition.Builder()
                        .target(LatLng(newLat, newLng))
                        .zoom(17.2)
                        .tilt(60.0)
                        .build()
                )
            )
            return
        }
        val cur = m.cameraPosition
        val curLat = cur.target?.latitude ?: newLat
        val curLng = cur.target?.longitude ?: newLng
        val latScale = cos(Math.toRadians(anchorLat))
        val dxMeters = (newLng - curLng) * 111320.0 * latScale
        val dyMeters = (newLat - curLat) * 110540.0
        if (Math.hypot(dxMeters, dyMeters) > 7.0) {
            m.moveCamera(
                CameraUpdateFactory.newCameraPosition(
                    CameraPosition.Builder()
                        .target(LatLng(newLat, newLng))
                        .zoom(cur.zoom)
                        .tilt(cur.tilt)
                        .bearing(cur.bearing)
                        .build()
                )
            )
        }
    }

    private fun lat(yMeters: Double): Double =
        anchorLat + yMeters / 110540.0

    private fun lng(xMeters: Double): Double =
        anchorLng + xMeters / (111320.0 * cos(Math.toRadians(anchorLat)))

    /** Lifecycle forwarding for the embedded MapView. */
    fun onViewStart() = mapView.onStart()
    fun onViewResume() = mapView.onResume()
    fun onViewPause() = mapView.onPause()
    fun onViewStop() = mapView.onStop()
    fun onViewDestroy() = mapView.onDestroy()
}