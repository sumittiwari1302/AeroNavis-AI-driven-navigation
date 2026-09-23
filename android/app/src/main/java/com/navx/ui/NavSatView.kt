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
import org.maplibre.android.style.layers.Property
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
 * Live 3D map backed by MapLibre, Google-Maps style.
 *
 * Basemap: real satellite imagery (Esri World Imagery) with a switchable
 * street map (OpenStreetMap) layer — both free, no API key. Overlays:
 *   - offline synthetic road network (thin gray grid streets)
 *   - Google-Maps-like blue navigation route (map-matched to roads)
 *   - ground-truth path (green line)
 *   - InEKF estimated path (cyan line)
 *   - running 3-sigma uncertainty as an extruded 3D tower (amber, red on outage)
 *   - white dot = estimated position, light-green dot = ground truth
 *
 * The camera starts tilted 60° for a 3D bird's-eye look; users can pan,
 * rotate, tilt and zoom with the standard map gestures.
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
        val estPath: List<FloatArray> = emptyList(),
        val route: List<FloatArray> = emptyList()
    )

    companion object {
        const val MAP_SATELLITE = 0
        const val MAP_STREET = 1
    }

    // Bengaluru (ISRO HQ) — sensor-fusion demo lives in local meters around here.
    private val anchorLat = 12.9716
    private val anchorLng = 77.5946

    private val mapView: MapView
    private var map: MapLibreMap? = null
    private var style: Style? = null
    private var styleReady = false
    private var firstCamera = true
    private var mapType = MAP_SATELLITE

    private var gtSrc: GeoJsonSource? = null
    private var estSrc: GeoJsonSource? = null
    private var uncSrc: GeoJsonSource? = null
    private var posSrc: GeoJsonSource? = null
    private var gtPosSrc: GeoJsonSource? = null
    private var routeSrc: GeoJsonSource? = null
    private var roadSrc: GeoJsonSource? = null
    private var uncExtrusion: FillExtrusionLayer? = null
    private var uncFill: FillLayer? = null

    init {
        // MapLibre 11 requires an instance before the MapView is created.
        // Tiles are free (Esri / OSM), so this dummy key only satisfies the SDK check.
        MapLibre.getInstance(context, "aeronavis-satellite-demo", WellKnownTileServer.MapLibre)
        mapView = MapView(context)
        mapView.layoutParams = LayoutParams(
            LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT
        )
        addView(mapView)
        mapView.getMapAsync { m ->
            map = m
            m.setStyle(Style.Builder().fromUri("asset://satellite.json")) { s ->
                style = s
                setupLayers(s)
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
        routeSrc = GeoJsonSource("route-src", empty)
        roadSrc = GeoJsonSource("road-src", FeatureCollection.fromFeatures(buildRoadFeatures()))

        val g = gtSrc as GeoJsonSource
        val e = estSrc as GeoJsonSource
        val u = uncSrc as GeoJsonSource
        val p = posSrc as GeoJsonSource
        val gp = gtPosSrc as GeoJsonSource
        val rt = routeSrc as GeoJsonSource
        val rd = roadSrc as GeoJsonSource
        style.addSource(rd)
        style.addSource(rt)
        style.addSource(g)
        style.addSource(e)
        style.addSource(u)
        style.addSource(p)
        style.addSource(gp)

        style.addLayer(
            LineLayer("road-line", "road-src").withProperties(
                PropertyFactory.lineColor(Color.parseColor("#FFFFFF")),
                PropertyFactory.lineWidth(1f),
                PropertyFactory.lineOpacity(0.30f)
            )
        )
        style.addLayer(
            LineLayer("route-line", "route-src").withProperties(
                PropertyFactory.lineColor(Color.parseColor("#3B82F6")),
                PropertyFactory.lineWidth(5.5f),
                PropertyFactory.lineOpacity(0.95f),
                PropertyFactory.lineCap(Property.LINE_CAP_ROUND),
                PropertyFactory.lineJoin(Property.LINE_JOIN_ROUND)
            )
        )
        style.addLayer(
            LineLayer("gt-line", "gt-src").withProperties(
                PropertyFactory.lineColor(Color.parseColor("#4CAF50")),
                PropertyFactory.lineWidth(3f),
                PropertyFactory.lineOpacity(0.6f)
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

    /** Synthetic offline street layout (matches the map-matching road network). */
    private fun buildRoadFeatures(): List<Feature> {
        val spacing = 24.0
        val extent = 168.0
        val feats = ArrayList<Feature>()
        var v = -extent
        while (v <= extent) {
            feats.add(
                Feature.fromGeometry(
                    LineString.fromLngLats(
                        listOf(
                            Point.fromLngLat(lng(-extent), lat(v)),
                            Point.fromLngLat(lng(extent), lat(v))
                        )
                    )
                )
            )
            feats.add(
                Feature.fromGeometry(
                    LineString.fromLngLats(
                        listOf(
                            Point.fromLngLat(lng(v), lat(-extent)),
                            Point.fromLngLat(lng(v), lat(extent))
                        )
                    )
                )
            )
            v += spacing
        }
        return feats
    }

    /** 0 = satellite imagery, 1 = street map. */
    fun setMapType(type: Int) {
        mapType = type
        val s = style ?: return
        val sat = s.getLayer("aeronavis-satellite-layer") ?: return
        val street = s.getLayer("aeronavis-street-layer") ?: return
        if (type == MAP_STREET) {
            sat.setProperties(PropertyFactory.visibility(Property.NONE))
            street.setProperties(PropertyFactory.visibility(Property.VISIBLE))
        } else {
            sat.setProperties(PropertyFactory.visibility(Property.VISIBLE))
            street.setProperties(PropertyFactory.visibility(Property.NONE))
        }
    }

    fun mapType(): Int = mapType

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
        val rt = scene.route.map { f ->
            Point.fromLngLat(lng(f[0].toDouble()), lat(f[1].toDouble()))
        }
        val g = gtSrc
        val e = estSrc
        val u = uncSrc
        val p = posSrc
        val gp = gtPosSrc
        routeSrc?.setGeoJson(LineString.fromLngLats(rt))
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