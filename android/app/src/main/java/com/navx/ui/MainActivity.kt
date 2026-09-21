package com.navx.ui

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.collect
import com.navx.R
import com.navx.databinding.ActivityMainBinding
import com.navx.fusion.InEKF
import com.navx.fusion.AdaptationEngine
import com.navx.inference.TFLiteRuntime
import com.navx.inference.VelocityModel
import com.navx.inference.PseudoOdoModel
import com.navx.inference.SlipClassifier
import com.navx.inference.TextureGate
import com.navx.inference.VisualOdoModel
import com.navx.inference.ForecasterModel
import com.navx.sensors.SensorHub
import com.navx.sensors.NavDataFrame
import com.navx.sensors.TextureGateResult
import com.navx.sensors.VelocityResult
import com.navx.sensors.WheelResult
import com.navx.sensors.VisualOdoResult
import com.navx.sensors.OutageForecast
import com.navx.fusion.InEKF
import com.navx.fusion.AdaptationEngine
import com.navx.sensors.SensorHub
import com.navx.app.mapmatch.MapMatcher
import com.navx.app.ranchor.RanchorManager
import org.maplibre.gl.MapView
import org.maplibre.gl.style.layers.LineLayer
import org.maplibre.gl.style.layers.SymbolLayer
import org.maplibre.gl.style.sources.GeoJsonSource
import org.maplibre.gl.geometry.LatLng
import org.maplibre.gl.annotations.MarkerOptions
import org.maplibre.gl.style.expressions.Expression
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.collect

class MainActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainBinding
    private val sensorHub by lazy { SensorHub(this, this) }
    private val velocityModel by lazy { VelocityModel(this) }
    private val pseudoOdoModel by lazy { PseudoOdoModel(this) }
    private val slipClassifier by lazy { SlipClassifier(this) }
    private val textureGate by lazy { TextureGate(this) }
    private val visualOdoModel by lazy { VisualOdoModel(this) }
    private val forecasterModel by lazy { ForecasterModel(this) }
    private val sensorHub by lazy { SensorHub(this, this) }
    private val inEKF by lazy { InEKF(config) }
    private val adaptationEngine by lazy { AdaptationEngine(velocityModel, config) }
    private val mapMatcher by lazy { MapMatcher(config, mapStore) }
    private val ranchorManager by lazy { RanchorManager(config, inEKF) }
    
    private val config = com.navx.config.get_config()
    private lateinit var mapView: org.maplibre.gl.MapView
    private var isRecording = false
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        initMap()
        initPipeline()
        initUI()
    }
    
    private fun initMap() {
        mapView = findViewById(R.id.mapView)
        mapView.onCreate(savedInstanceState)
        mapView.getMapAsync { mapboxMap ->
            mapboxMap.loadStyleUri("asset://maplibre/style.json") {
                setupMapLayers(it)
            }
        }
    }
    
    private fun setupMapLayers(mapboxMap: org.maplibre.gl.MapboxMap) {
        // Add road layer from offline tiles
        val source = GeoJsonSource("roads", "")
        mapboxMap.addSource(source)
        
        val roadLayer = LineLayer("road-layer", "roads")
        roadLayer.setSourceLayer("roads")
        roadLayer.setProperties(
            org.maplibre.gl.style.properties.PropertyFactory.lineColor("#4285F4"),
            org.maplibre.gl.style.properties.PropertyFactory.lineWidth(2f)
        )
        mapboxMap.addLayer(roadLayer)
        
        // Add pose marker
        val marker = MarkerOptions()
            .position(LatLng(0.0, 0.0))
            .title("Current Position")
        mapboxMap.addMarker(marker)
    }
    
    private fun initPipeline() {
        // Start sensor hub
        sensorHub.start()
        
        // Subscribe to sensor data flow
        lifecycleScope.launch {
            sensorHub.dataFlow.collect { frame ->
                processSensorFrame(frame)
            }
        }
        
        lifecycleScope.launch {
            sensorHub.textureGateFlow.collect { gate ->
                runOnUiThread { updateTextureGateUI(gate) }
            }
        }
        
        lifecycleScope.launch {
            sensorHub.velocityFlow.collect { vel ->
                // Process velocity
            }
        }
        
        lifecycleScope.launch {
            sensorHub.wheelFlow.collect { wheel ->
                // Process wheel
            }
        }
        
        lifecycleScope.launch {
            sensorHub.visualOdoFlow.collect { vo ->
                vo?.let { voResult ->
                    // Process visual odometry
                }
            }
        }
        
        lifecycleScope.launch {
            sensorHub.outageForecastFlow.collect { forecast ->
                updateOutageUI(forecast)
            }
        }
        
        lifecycleScope.launch {
            sensorHub.fusedStateFlow.collect { state ->
                updateNavStateUI(state)
            }
        }
        
        lifecycleScope.launch {
            sensorHub.matchResultFlow.collect { match ->
                updateMapMatch(match)
            }
        }
    }
    
    private fun processSensorFrame(frame: NavDataFrame) {
        // 1. Run velocity inference
        val velocityResult = velocityModel.run(frame.imu, frame.att0)
        
        // 2. Run pseudo-odo
        val wheelResult = pseudoOdoModel.run(frame.imu, frame.baro)
        
        // 3. Run slip classifier
        val slipState = slipClassifier.run(frame.imu)
        
        // 4. Run texture gate
        val gate = textureGate.run(frame.cameraFrame)
        
        // 5. Visual odometry (if gate allows)
        val voResult = if (gate.gate == 0) {
            visualOdoModel.run(frame.prevFrame, frame.cameraFrame)
        } else null
        
        // 5. Forecaster
        val forecast = forecasterModel.run(extractFeatures())
        
        // 5. Run InEKF predict
        inEKF.predict(frame.imu, frame.gyro, 0.01f)
        
        // 6. Wheel correction
        if (wheelResult.reliability > 0.7) {
            inEKF.correctWheel(WheelMeasurement(
                timestamp = System.currentTimeMillis(),
                vxBody = wheelResult.speed,
                gate = if (wheelResult.reliability > 0.7) "full" else "low"
            ))
        }
        
        // 6. Visual odometry correction
        if (voResult != null) {
            inEKF.correctVo(VisualOdoMeasurement(
                timestamp = System.currentTimeMillis(),
                deltaPose = voResult.deltaPose,
                R = floatArrayOf(0.5f, 0.5f, 0.5f, 0.1f, 0.1f, 0.1f)
            ))
        }
        
        // 5. GNSS correction (handled by seeder)
        
        // 6. Adaptation step
        adaptationEngine.step(
            acc = frame.imu,
            gyr = frame.gyro,
            pseudoSpeed = wheelResult.speed,
            pseudoRel = wheelResult.reliability,
            gnssSpeed = gnssSpeed,
            gnssAvailable = gnssOk
        )
        
        // Update UI
        updateDashboard()
    }
    
    private fun updateDashboard() {
        runOnUiThread {
            // Update UI elements
            binding.tvVelocity.text = String.format("%.1f m/s", inEKF.velocity[0])
            binding.tvHeading.text = String.format("%.1f°", Math.toDegrees(inEKF.heading))
            binding.tvPosStd.text = String.format("%.2f m", inEKF.state().std_x)
        }
    }
    
    private fun updateTextureGateUI(gate: TextureGateResult) {
        runOnUiThread {
            when (gate.gate) {
                0 -> binding.tvTexture.text = "RICH"
                1 -> binding.tvTexture.text = "MEDIUM"
                2 -> binding.tvTexture.text = "POOR"
            }
            binding.pbTextureConfidence.progress = (gate.confidence * 100).toInt()
        }
    }
    
    private fun updateOutageUI(forecast: OutageForecast) {
        runOnUiThread {
            binding.tvOutage5s.text = String.format("%.0f%%", forecast.horizons[0] * 100)
            binding.tvOutage10s.text = String.format("%.0f%%", forecast.horizons[1] * 100)
            binding.tvOutage15s.text = String.format("%.0f%%", forecast.horizons[2] * 100)
        }
    }
    
    private fun updateNavStateUI(state: NavState) {
        runOnUiThread {
            binding.tvPosX.text = String.format("%.2f", state.position[0])
            binding.tvPosY.text = String.format("%.2f", state.position[1])
            binding.tvStdX.text = String.format("%.2f", state.std_x)
            binding.tvStdY.text = String.format("%.2f", state.std_y)
        }
    }
    
    private fun updateMapMatch(match: MatchResult) {
        runOnUiThread {
            if (match.poseSnapped) {
                // Update map marker
            }
        }
    }
    
    private fun extractFeatures(): FloatArray {
        // Extract 104 features for forecaster
        return FloatArray(104)
    }
    
    private fun updateOutageUI(forecast: OutageForecast) {
        // Update UI
    }
    
    override fun onStart() {
        super.onStart()
        mapView.onStart()
    }
    
    override fun onResume() {
        super.onResume()
        mapView.onResume()
    }
    
    override fun onPause() {
        super.onPause()
        mapView.onPause()
    }
    
    override fun onStop() {
        super.onStop()
        mapView.onStop()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        mapView.onDestroy()
    }
    
    override fun onLowMemory() {
        super.onLowMemory()
        mapView.onLowMemory()
    }
    
    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        mapView.onSaveInstanceState(outState)
    }
}