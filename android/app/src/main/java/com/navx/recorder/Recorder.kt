package com.navx.recorder

import android.content.Context
import android.os.Environment
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.receiveAsFlow

/**
 * Recorder for Part 1 self-collected data.
 * Writes per-sensor CSVs matching the RECORDING_GUIDE.md schema.
 */
class Recorder(
    private val context: Context
) {
    private val dataChannel = Channel<RecordEvent>(1000)
    val dataFlow = dataChannel.receiveAsFlow()
    
    private var imuWriter: BufferedWriter? = null
    private var gyroWriter: BufferedWriter? = null
    private var magWriter: BufferedWriter? = null
    private var baroWriter: BufferedWriter? = null
    private var gnssWriter: BufferedWriter? = null
    private var eventsWriter: BufferedWriter? = null
    private var metadataWriter: BufferedWriter? = null
    
    private var sessionDir: File? = null
    private var isRecording = false
    private val sessionId = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    
    sealed class RecordEvent {
        data class Imu(val tsMs: Long, val accX: Float, val accY: Float, val accZ: Float,
                       val gyrX: Float, val gyrY: Float, val gyrZ: Float) : RecordEvent()
        data class Mag(val tsMs: Long, val x: Float, val y: Float, val z: Float) : RecordEvent()
        data class Baro(val tsMs: Long, val pressure: Float) : RecordEvent()
        data class Gnss(val tsMs: Long, val lat: Double, val lon: Double, val alt: Double,
                        val speed: Float, val heading: Float, val pdop: Float, val cno: Float) : RecordEvent()
        data class Event(val tsMs: Long, val type: String, val details: String)
    }
    
    fun start() {
        val baseDir = File(Environment.getExternalStorageDirectory(), "navx/recordings/$sessionId")
        baseDir.mkdirs()
        
        val imuFile = File(baseDir, "imu.csv")
        val magFile = File(baseDir, "mag.csv")
        val baroFile = File(baseDir, "baro.csv")
        val gnssFile = File(baseDir, "gnss.csv")
        val eventsFile = File(baseDir, "events.csv")
        val metaFile = File(baseDir, "metadata.json")
        
        imuWriter = BufferedWriter(FileWriter(imuFile)).apply { write("ts_ms,acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z\n") }
        magWriter = BufferedWriter(FileWriter(File(baseDir, "mag.csv"))).apply { write("ts_ms,mag_x,mag_y,mag_z\n") }
        baroWriter = BufferedWriter(FileWriter(File(baseDir, "baro.csv")).apply { write("ts_ms,pressure_hpa\n") })
        gnssWriter = BufferedWriter(FileWriter(File(baseDir, "gnss.csv")).apply { 
            write("ts_ms,lat,lon,alt_m,speed_mps,heading_deg,pdop,cno_db_hz\n") 
        })
        eventsWriter = BufferedWriter(FileWriter(File(baseDir, "events.csv")).apply { 
            write("ts_ms,type,details\n") 
        })
        
        // Write metadata
        File(baseDir, "metadata.json").writeText("""
            {
              "session_id": "$sessionId",
              "start_time": "${System.currentTimeMillis()}",
              "schema_version": "1.0",
              "sensor_config": {
                "imu_hz": 100,
                "gnss_hz": 1,
                "mag_hz": 50,
                "baro_hz": 25,
                "cam_hz": 10
              }
            }
        """.trimIndent())
        
        isRecording = true
    }
    
    suspend fun logImu(tsMs: Long, accX: Float, accY: Float, accZ: Float,
                       gyrX: Float, gyrY: Float, gyrZ: Float) {
        imuWriter?.write("$tsMs,$accX,$accY,$accZ,$gyrX,$gyrY,$gyrZ\n")
    }
    
    suspend fun logMag(tsMs: Long, x: Float, y: Float, z: Float) {
        magWriter?.write("$tsMs,$x,$y,$z\n")
    }
    
    suspend fun logBaro(tsMs: Long, pressure: Float) {
        baroWriter?.write("$tsMs,$pressure\n")
    }
    
    suspend fun logGnss(tsMs: Long, lat: Double, lon: Double, alt: Double,
                        speed: Float, heading: Float, pdop: Float, cno: Float) {
        gnssWriter?.write("$tsMs,$lat,$lon,$alt,$speed,$heading,$pdop,$cno\n")
    }
    
    suspend fun logEvent(type: String, details: String) {
        eventsWriter?.write("${System.currentTimeMillis()},$type,$details\n")
    }
    
    fun stop() {
        isRecording = false
        imuWriter?.close()
        magWriter?.close()
        baroWriter?.close()
        gnssWriter?.close()
        eventsWriter?.close()
    }
    
    companion object {
        fun create(context: Context): Recorder {
            return Recorder(context)
        }
    }
}