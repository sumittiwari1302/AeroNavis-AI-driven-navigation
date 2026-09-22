package com.navx.fusion

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * State vector (10): [x, y, yaw, vx, vy, gyro_bias, accel_bias_x, accel_bias_y, qw, qz]
 * The quaternion part is kept minimal for heading-only 2D navigation on device.
 */
data class InEKFConfig(
    val processNoiseGyroBias: Double = 1e-6,
    val processNoiseAccelBias: Double = 1e-4,
    val processNoiseGyro: Double = 1e-4,
    val processNoiseAccel: Double = 0.0001,
    val gnssPosStd: Double = 0.5,
    val gnssVelStd: Double = 0.5,
    val wheelSpeedStd: Double = 0.1,
    val wheelNhcStd: Double = 0.05,
    val voPosStd: Double = 0.5,
    val voYawStd: Double = 0.05,
    val minEig: Double = 1e-12,
    val psdInflationFactor: Double = 1.5,
    val residualGateSigma: Double = 6.0,
    val reseedJumpMaxM: Double = 0.3
)

data class WheelMeasurement(
    val timestamp: Long,
    val vxBody: Float,
    val vyBody: Float = 0f,
    val vzBody: Float = 0f,
    val R: FloatArray = floatArrayOf(0.1f, 0.05f, 0.05f),
    val gate: String = "full"
)

data class VisualOdoMeasurement(
    val timestamp: Long,
    val deltaPose: FloatArray,
    val R: FloatArray = floatArrayOf(0.5f, 0.5f, 0.5f, 0.1f, 0.1f, 0.1f)
)

data class GnssMeasurement(
    val timestamp: Long,
    val position: DoubleArray,
    val velocity: FloatArray,
    val covariance: FloatArray,
    val fixQuality: Int = 1,
    val numSatellites: Int = 0,
    val hdop: Float = 1.0f
)

class InEKF(private val config: InEKFConfig) {

    private val N = 10
    private var state = DoubleArray(N) { 0.0 }
    private var covariance = DoubleArray(N * N) { if (it % (N + 1) == 0) 1e-6 else 0.0 }
    private val Qc = DoubleArray(N * N) { 0.0 }

    private var outageActive = false
    private var handoverActive = false
    private var lastGoodPose: DoubleArray? = null
    private var handoverBlend = 1.0

    var skippedMeasurements = 0
    var psdRecoveries = 0

    init {
        initQc()
    }

    private fun initQc() {
        Qc[5 * N + 5] = config.processNoiseGyroBias
        Qc[6 * N + 6] = config.processNoiseAccelBias
        Qc[7 * N + 7] = config.processNoiseAccelBias
        Qc[2 * N + 2] = config.processNoiseGyro
        Qc[3 * N + 3] = config.processNoiseAccel
        Qc[4 * N + 4] = config.processNoiseAccel
        Qc[8 * N + 8] = 1e-8
        Qc[9 * N + 9] = 1e-8
    }

    fun predict(acc: FloatArray, gyr: FloatArray, dt: Float) {
        if (dt <= 0f) return
        val dtD = dt.toDouble()

        val gyrCorr = gyr.copyOf()
        gyrCorr[2] = (gyrCorr[2].toDouble() - state[5]).toFloat()

        val yaw = state[2]
        val c = cos(yaw)
        val s = sin(yaw)

        val vx = state[3]
        val vy = state[4]

        val ax = acc[0].toDouble() - state[6]
        val ay = acc[1].toDouble() - state[7]

        state[0] += (vx * c - vy * s) * dtD
        state[1] += (vx * s + vy * c) * dtD
        state[2] = wrapAngle(state[2] + gyrCorr[2].toDouble() * dtD)
        state[3] += ax * dtD
        state[4] += ay * dtD

        val F = Array(N) { DoubleArray(N) { 0.0 } }
        for (i in 0 until N) F[i][i] = 1.0
        F[0][2] = (-vx * s - vy * c) * dtD
        F[1][2] = (vx * c - vy * s) * dtD
        F[0][3] = c * dtD
        F[0][4] = -s * dtD
        F[1][3] = s * dtD
        F[1][4] = c * dtD
        F[3][2] = -ax * dtD
        F[4][2] = ay * dtD
        F[3][5] = -dtD
        F[4][5] = -dtD
        F[3][6] = -dtD
        F[4][7] = -dtD

        val Qd = DoubleArray(N * N) { Qc[it] * dtD }

        val P_new = matMul(matMul(F, covariance), matTranspose(F))
        for (i in 0 until N) for (j in 0 until N) {
            covariance[i * N + j] = P_new[i][j] + Qd[i * N + j]
        }
        enforcePsd()
    }

    fun correctGnss(z: DoubleArray, R: DoubleArray? = null) {
        val Rmat = R ?: buildRgnss()
        val H = Array(3) { DoubleArray(N) { 0.0 } }
        H[0][0] = 1.0
        H[1][1] = 1.0
        H[2][2] = 1.0

        val zPred = doubleArrayOf(state[0], state[1], state[2])
        val y = DoubleArray(3) { z[it] - zPred[it] }
        y[2] = wrapAngle(y[2])

        val S = matMul(matMul(H, covariance), matTranspose(H))
        for (i in 0..2) S[i][i] += if (Rmat.size > i) Rmat[i] else 1e-6

        if (residualGate(y, S)) {
            skippedMeasurements++
            return
        }

        val K = matMul(matMul(covariance, matTranspose(H)), invert(S))
        val dx = DoubleArray(N) { 0.0 }
        for (i in 0 until N) for (j in 0..2) dx[i] += K[i][j] * y[j]

        for (i in 0 until N) state[i] += dx[i]
        state[2] = wrapAngle(state[2])

        val I = Array(N) { DoubleArray(N) { 0.0 } }
        for (i in 0 until N) I[i][i] = 1.0
        val KH = matMul(K, H)
        for (i in 0 until N) for (j in 0 until N) I[i][j] -= KH[i][j]

        covariance = flatten(matMul(matMul(I, covariance), matTranspose(I)))
        enforcePsd()
    }

    fun correctWheel(m: WheelMeasurement) {
        val H = Array(1) { DoubleArray(N) { 0.0 } }
        val c = cos(state[2])
        val s = sin(state[2])
        H[0][3] = c
        H[0][4] = -s

        val zPred = H[0][3] * state[3] + H[0][4] * state[4]
        val y = m.vxBody.toDouble() - zPred
        val r = (config.wheelSpeedStd * config.wheelSpeedStd).coerceAtLeast(1e-4)
        val S = matMul(matMul(H, covariance), matTranspose(H))[0][0] + r

        val K = DoubleArray(N) { 0.0 }
        for (i in 0 until N) K[i] = covariance[i * N + i] * H[0][i] / S

        for (i in 0 until N) state[i] += K[i] * y
        state[2] = wrapAngle(state[2])

        val I_KH = Array(N) { DoubleArray(N) { 0.0 } }
        for (idx in 0 until N) I_KH[idx][idx] = 1.0
        for (i in 0 until N) for (j in 0 until N) {
            I_KH[i][j] -= K[i] * H[0][j]
        }
        covariance = flatten(matMul(matMul(I_KH, covariance), matTranspose(I_KH)))
        enforcePsd()
    }

    fun correctVo(deltaPose: DoubleArray, Rvo: DoubleArray? = null) {
        val c = cos(state[2])
        val s = sin(state[2])
        val dx = deltaPose[0].toDouble()
        val dy = deltaPose[1].toDouble()

        val H = Array(2) { DoubleArray(N) { 0.0 } }
        H[0][3] = c
        H[0][4] = -s
        H[1][3] = s
        H[1][4] = c

        val zPred = doubleArrayOf(
            H[0][3] * state[3] + H[0][4] * state[4],
            H[1][3] * state[3] + H[1][4] * state[4]
        )
        val y = doubleArrayOf(dx - zPred[0], dy - zPred[1])

        val r = maxOf(config.voPosStd * config.voPosStd, 1e-4)
        val S = matMul(matMul(H, covariance), matTranspose(H))
        S[0][0] += r
        S[1][1] += r

        val Sinv = invert(S)
        val K = Array(N) { DoubleArray(2) { 0.0 } }
        val PHt = matMul(covariance, matTranspose(H))
        for (i in 0 until N) for (j in 0..1) {
            for (k in 0..1) K[i][j] += PHt[i][k] * Sinv[k][j]
        }

        for (i in 0 until N) {
            state[i] += K[i][0] * y[0] + K[i][1] * y[1]
        }
        state[2] = wrapAngle(state[2])

        val I_KH = Array(N) { DoubleArray(N) { 0.0 } }
        for (idx in 0 until N) I_KH[idx][idx] = 1.0
        for (i in 0 until N) for (j in 0 until N) {
            I_KH[i][j] -= K[i][0] * H[0][j] + K[i][1] * H[1][j]
        }
        covariance = flatten(matMul(matMul(I_KH, covariance), matTranspose(I_KH)))
        enforcePsd()
    }

    fun seedAnchor(sigmaInflate: Double) {
        for (i in 0 until N) {
            covariance[i * N + i] += sigmaInflate
        }
        lastGoodPose = state.clone()
    }

    fun handover(gnssPose: DoubleArray, blend: Double) {
        val clamped = blend.coerceIn(0.0, 1.0)
        for (i in 0..2) {
            state[i] = (1.0 - clamped) * state[i] + clamped * gnssPose[i]
        }
        handoverBlend = clamped
        handoverActive = true
    }

    fun setOutageActive(active: Boolean) {
        outageActive = active
        if (!active) lastGoodPose = state.clone()
    }

    fun state(): DoubleArray = state.clone()
    fun covariance(): DoubleArray = covariance.clone()
    fun velocity(): DoubleArray = doubleArrayOf(state[3], state[4])
    fun heading(): Double = state[2]
    fun stdX(): Double = sqrt(covariance[0])
    fun stdY(): Double = sqrt(covariance[1 * N + 1])

    private fun buildRgnss(): DoubleArray = doubleArrayOf(
        config.gnssPosStd * config.gnssPosStd,
        config.gnssPosStd * config.gnssPosStd,
        Math.toRadians(5.0) * Math.toRadians(5.0)
    )

    private fun residualGate(y: DoubleArray, S: Array<DoubleArray>): Boolean {
        var mahal = 0.0
        val diag = doubleArrayOf(S[0][0].coerceAtLeast(1e-9), S[1][1].coerceAtLeast(1e-9), S[2][2].coerceAtLeast(1e-9))
        for (i in y.indices) mahal += (y[i] * y[i]) / diag[i]
        return mahal > config.residualGateSigma * config.residualGateSigma
    }

    private fun enforcePsd() {
        val eig = jacobiEigen(covariance)
        var inflated = false
        for (i in 0 until N) {
            if (eig.values[i] < config.minEig) {
                eig.values[i] = config.minEig
                inflated = true
            }
        }
        if (inflated) {
            covariance = eig.reconstruct()
            psdRecoveries++
        }
    }

    private fun wrapAngle(a: Double): Double {
        var v = a
        while (v <= -Math.PI) v += 2 * Math.PI
        while (v > Math.PI) v -= 2 * Math.PI
        return v
    }

    private fun i1(i: Int) = i % N
    private fun i2(i: Int) = i / N

    private fun flatten(M: Array<DoubleArray>): DoubleArray {
        val out = DoubleArray(M.size * M[0].size) { 0.0 }
        for (i in M.indices) for (j in M[i].indices) out[i * M[i].size + j] = M[i][j]
        return out
    }

    private fun matMul(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        val m = A.size
        val n = B[0].size
        val p = B.size
        val out = Array(m) { DoubleArray(n) { 0.0 } }
        for (i in 0 until m) {
            for (k in 0 until p) {
                val a = A[i][k]
                if (a == 0.0) continue
                for (j in 0 until n) {
                    out[i][j] += a * B[k][j]
                }
            }
        }
        return out
    }

    private fun matMul(A: DoubleArray, B: Array<DoubleArray>): Array<DoubleArray> {
        val n = B.size
        val m = B[0].size
        val out = Array(n) { DoubleArray(m) { 0.0 } }
        for (i in 0 until n) {
            for (j in 0 until m) {
                var acc = 0.0
                for (k in 0 until n) acc += A[i * n + k] * B[k][j]
                out[i][j] = acc
            }
        }
        return out
    }

    private fun matMul(A: Array<DoubleArray>, B: DoubleArray): Array<DoubleArray> {
        val m = A.size
        val n = A[0].size
        val out = Array(m) { DoubleArray(n) { 0.0 } }
        for (i in 0 until m) {
            for (j in 0 until n) {
                var acc = 0.0
                for (k in 0 until n) acc += A[i][k] * B[k * n + j]
                out[i][j] = acc
            }
        }
        return out
    }

    private fun matTranspose(A: Array<DoubleArray>): Array<DoubleArray> {
        val m = A.size
        val n = A[0].size
        val out = Array(n) { DoubleArray(m) { 0.0 } }
        for (i in 0 until m) for (j in 0 until n) out[j][i] = A[i][j]
        return out
    }

    private fun invert(A: Array<DoubleArray>): Array<DoubleArray> {
        val n = A.size
        val a = Array(n) { A[it].copyOf() }
        val inv = Array(n) { DoubleArray(n) { 0.0 } }
        for (i in 0 until n) inv[i][i] = 1.0

        for (p in 0 until n) {
            var pivot = p
            var max = Math.abs(a[p][p])
            for (r in p + 1 until n) {
                val v = Math.abs(a[r][p])
                if (v > max) { max = v; pivot = r }
            }
            if (max < 1e-12) continue
            if (pivot != p) {
                val tmp = a[p]; a[p] = a[pivot]; a[pivot] = tmp
                val tmpInv = inv[p]; inv[p] = inv[pivot]; inv[pivot] = tmpInv
            }
            val d = a[p][p]
            for (j in 0 until n) { a[p][j] /= d; inv[p][j] /= d }
            for (r in 0 until n) {
                if (r == p) continue
                val f = a[r][p]
                if (f == 0.0) continue
                for (j in 0 until n) {
                    a[r][j] -= f * a[p][j]
                    inv[r][j] -= f * inv[p][j]
                }
            }
        }
        return inv
    }

    private class EigenResult(val values: DoubleArray, val vectors: Array<DoubleArray>) {
        fun reconstruct(): DoubleArray {
            val n = values.size
            val out = DoubleArray(n * n) { 0.0 }
            for (r in 0 until n) {
                for (c in 0 until n) {
                    var acc = 0.0
                    for (k in 0 until n) acc += vectors[k][r] * values[k] * vectors[k][c]
                    out[r * n + c] = acc
                }
            }
            return out
        }
    }

    private fun jacobiEigen(mat: DoubleArray): EigenResult {
        val n = N
        val a = Array(n) { mat.slice(it * n until it * n + n).toDoubleArray() }
        val v = Array(n) { DoubleArray(n) { 0.0 } }
        for (i in 0 until n) v[i][i] = 1.0

        for (iter in 0 until 64) {
            var p = 0
            var q = 1
            var maxOff = 0.0
            for (i in 0 until n) for (j in i + 1 until n) {
                val x = Math.abs(a[i][j])
                if (x > maxOff) { maxOff = x; p = i; q = j }
            }
            if (maxOff < 1e-12) break

            val apq = a[p][q]
            val app = a[p][p]
            val aqq = a[q][q]
            val theta = 0.5 * Math.atan2(2 * apq, aqq - app)
            val c = Math.cos(theta)
            val s = Math.sin(theta)

            for (k in 0 until n) {
                val akp = a[k][p]
                val akq = a[k][q]
                a[k][p] = c * akp - s * akq
                a[k][q] = s * akp + c * akq
            }
            for (k in 0 until n) {
                val apk = a[p][k]
                val aqk = a[q][k]
                a[p][k] = c * apk - s * aqk
                a[q][k] = s * apk + c * aqk
            }
            val saved = a[p][p]
            a[p][p] = c * c * app + s * s * aqq + 2 * s * c * apq
            a[q][q] = c * c * aqq + s * s * app - 2 * s * c * apq
            a[p][q] = 0.0
            a[q][p] = 0.0

            for (k in 0 until n) {
                val vkp = v[k][p]
                val vkq = v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq
            }
        }

        val values = DoubleArray(n)
        for (i in 0 until n) values[i] = a[i][i].coerceAtLeast(0.0)

        val vectors = Array(n) { DoubleArray(n) { 0.0 } }
        for (i in 0 until n) for (j in 0 until n) vectors[j][i] = v[i][j]

        return EigenResult(values, vectors)
    }
}