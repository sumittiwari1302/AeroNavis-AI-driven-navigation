"""Right-invariant Extended Kalman Filter (InEKF) on SE(2) for NAV-X 3.0.

State vector (10-dim on SE(2) with biases):
    x = [x, y, psi, vx, vy, bg_z, ba_x, ba_y, s_wheel, s_vo]

Where:
    - x, y: ENU position (meters)
    - psi: heading angle (radians)
    - vx, vy: body-frame velocity (m/s)
    - bg_z: gyro bias (rad/s) - z-axis only for SE(2)
    - ba_x, ba_y: accelerometer bias (m/s^2)
    - s_wheel: wheel speed scale factor
    - s_vo: visual odometry scale factor

Right-invariant error state for numerical stability on SE(2).
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import numpy.linalg as la

from aeronavis.config import Config
from aeronavis.fusion.learned_noise import LearnedNoiseModel

logger = logging.getLogger(__name__)


@dataclass
class NavState:
    """Navigation state with 3-sigma confidence bands."""

    ts: float
    x: float
    y: float
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0
    std_x: float = 0.0
    std_y: float = 0.0
    std_z: float = 0.0
    std_vx: float = 0.0
    std_vy: float = 0.0
    std_vz: float = 0.0
    std_yaw: float = 0.0
    source_health: Dict[str, bool] = None

    def __post_init__(self):
        if self.source_health is None:
            self.source_health = {
                "gnss": False,
                "wheel": False,
                "vo": False,
                "velocity": False,
            }


class InEKF:
    """Right-invariant Extended Kalman Filter on SE(2) for ground vehicles."""

    def __init__(self, config: Config, learned_noise: Optional["LearnedNoiseModel"] = None):
        self.config = config
        self.fusion_cfg = config.fusion
        self.learned_noise = learned_noise

        # State dimension: [x, y, psi, vx, vy, bg_z, ba_x, ba_y, s_wheel, s_vo]
        self.n_state = 10
        self.n_meas_gnss = 3  # x, y, psi
        self.n_meas_wheel = 1  # vx (forward speed)
        self.n_meas_nhc = 2  # vy=0, vz=0 (NHC)
        self.n_meas_vo = 3  # dx, dy, dpsi

        # State vector: [x, y, psi, vx, vy, bg_x, bg_y, ba_x, ba_y, s_wheel, s_vo]
        self.x = np.zeros(self.n_state)
        self.P = np.eye(self.n_state) * 1e-6

        # Process noise (continuous-time, will be discretized)
        self.Qc = self._build_process_noise_continuous()

        # Measurement noise (base values, inflated by seeder/learned noise)
        self.R_gnss = self._build_R_gnss()
        self.R_wheel = np.diag([self.fusion_cfg.wheel_speed_std**2])
        self.R_nhc = np.diag(
            [
                self.fusion_cfg.wheel_nhc_std**2,
                self.fusion_cfg.wheel_nhc_std**2,
            ]
        )
        self.R_vo = np.diag(
            [
                self.fusion_cfg.vo_pos_std**2,
                self.fusion_cfg.vo_pos_std**2,
                self.fusion_cfg.vo_yaw_std**2,
            ]
        )

        # Numerical safety
        self.min_eig = self.fusion_cfg.min_eig
        self.psd_inflation = self.fusion_cfg.psd_inflation_factor
        self.residual_gate = self.fusion_cfg.residual_gate_sigma

        # Seeder integration
        self._cov_inflation = 1.0
        self._outage_active = False
        self._handover_active = False
        self._last_good_pose = None
        self._handover_blend = 1.0

        # Stats
        self.skipped_measurements = 0
        self.psd_recoveries = 0

        # Initialize state
        self._initialize_state()

    @property
    def state_dim(self) -> int:
        return self.n_state

    def _initialize_state(self):
        """Initialize filter state with small uncertainty."""
        self.x = np.zeros(self.n_state)
        self.x[8] = 1.0  # s_wheel = 1.0
        self.x[9] = 1.0  # s_vo = 1.0
        self.P = np.diag(
            [
                1.0,
                1.0,
                0.01,  # x, y, psi
                0.1,
                0.1,  # vx, vy
                1e-6,  # bg_z
                1e-4,
                1e-4,  # ba_x, ba_y
                0.01,
                0.01,  # s_wheel, s_vo
            ]
        )

    def _build_process_noise_continuous(self) -> np.ndarray:
        """Build continuous-time process noise matrix Qc (10x10)."""
        Qc = np.zeros((self.n_state, self.n_state))
        # Gyro bias random walk (z-axis only for SE(2))
        Qc[5, 5] = self.fusion_cfg.proc_noise_gyro_bias
        # Accel bias random walk
        Qc[6, 6] = self.fusion_cfg.proc_noise_accel_bias
        Qc[7, 7] = self.fusion_cfg.proc_noise_accel_bias
        # Gyro noise (affects attitude)
        Qc[2, 2] = self.fusion_cfg.proc_noise_gyro
        # Accel noise (affects velocity)
        Qc[3, 3] = self.fusion_cfg.proc_noise_accel
        Qc[4, 4] = self.fusion_cfg.proc_noise_accel
        # Scale factors (slow random walk)
        Qc[8, 8] = 1e-8
        Qc[9, 9] = 1e-8
        return Qc

    def _build_R_gnss(self) -> np.ndarray:
        """Build GNSS measurement noise matrix."""
        return np.diag(
            [
                self.fusion_cfg.gnss_pos_std**2,
                self.fusion_cfg.gnss_pos_std**2,
                (np.deg2rad(5.0)) ** 2,  # 5 deg heading uncertainty
            ]
        )

    def _enforce_psd(self, P: np.ndarray) -> np.ndarray:
        """Ensure P is positive semi-definite."""
        P = (P + P.T) / 2
        eigvals, eigvecs = la.eigh(P)
        eigvals = np.maximum(eigvals, self.min_eig)
        P = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return P

    def _check_and_repair_psd(self, P: np.ndarray) -> np.ndarray:
        """Check PSD and repair if needed."""
        try:
            la.cholesky(P)
            return P
        except la.LinAlgError:
            logger.warning("Covariance lost PSD, repairing...")
            eigvals, eigvecs = la.eigh(P)
            eigvals = np.maximum(eigvals, self.min_eig)
            P = eigvecs @ np.diag(eigvals) @ eigvecs.T
            self.psd_recoveries += 1
            return P

    def _inflate_covariance(self, factor: float):
        """Inflate covariance by factor (for outage prediction)."""
        self.P *= factor
        self.P = self._check_and_repair_psd(self.P)

    def _discretize_Q(self, dt: float) -> np.ndarray:
        """First-order discretization of continuous-time Qc."""
        return self.Qc * dt

    def _rodrigues(self, omega: np.ndarray, dt: float) -> np.ndarray:
        """Rodrigues formula for SO(2) attitude update."""
        theta = omega * dt
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    # =========================================================================
    # PREDICT STEP (100 Hz)
    # =========================================================================
    def predict(self, acc: np.ndarray, gyr: np.ndarray, dt: float):
        """Propagate state and covariance at 100 Hz."""
        # Unpack state: [x, y, psi, vx, vy, bg_z, ba_x, ba_y, s_wheel, s_vo]
        psi = self.x[2]
        vx, vy = self.x[3], self.x[4]
        self.x[5]
        ba = self.x[6:8]

        # Correct IMU readings with biases
        gyr_corr = gyr.copy()
        gyr_corr[2] -= self.x[5]  # bg_z
        acc_corr = acc - np.array([ba[0], ba[1], 0])  # subtract ba_x, ba_y

        # Body-frame acceleration to ENU
        c, s = np.cos(psi), np.sin(psi)
        R_b2n = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
        acc_ned = R_b2n @ acc_corr[:2]

        # State propagation (first-order)
        self.x[0] += self.x[3] * np.cos(psi) * dt - self.x[4] * np.sin(psi) * dt
        self.x[1] += self.x[3] * np.sin(psi) * dt + self.x[4] * np.cos(psi) * dt
        self.x[2] = self._wrap_angle(psi + gyr_corr[2] * dt)
        self.x[3] += acc_ned[0] * dt
        self.x[4] += acc_ned[1] * dt
        # Biases and scale factors are constant (random walk handled in Q)

        # Linearized state transition matrix F
        F = np.eye(self.n_state)
        c, s = np.cos(psi), np.sin(psi)
        vx, vy = self.x[3], self.x[4]
        F[0, 2] = (-vx * s - vy * c) * dt
        F[1, 2] = (vx * c - vy * s) * dt
        F[0, 3] = np.cos(psi) * dt
        F[0, 4] = -np.sin(psi) * dt
        F[1, 3] = np.sin(psi) * dt
        F[1, 4] = np.cos(psi) * dt
        F[2, 2] = 1.0
        F[3, 6] = dt  # ba_x affects vx
        F[4, 7] = dt  # ba_y affects vy

        # Discretized process noise
        self._discretize_Q(dt)

        # Covariance propagation
        self.P = F @ self.P @ F.T + self.Qc * dt
        self.P = self._check_and_repair_psd(self.P)

    # =========================================================================
    # CORRECTION STEPS
    # =========================================================================

    def correct_gnss(self, z: np.ndarray, R: Optional[np.ndarray] = None):
        """GNSS position/heading correction (1 Hz)."""
        if R is None:
            R = self.R_gnss * self._cov_inflation

        # Innovation
        H = np.zeros((self.n_meas_gnss, self.n_state))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # psi

        z_pred = np.array([self.x[0], self.x[1], self.x[2]])
        y = z - z_pred
        y[2] = self._wrap_angle(y[2])  # wrap angle

        S = H @ self.P @ H.T + R
        # Residual gating
        if self._residual_gate(y, S):
            self.skipped_measurements += 1
            return

        K = self.P @ H.T @ la.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.n_state) - K @ H) @ self.P
        self.P = self._check_and_repair_psd(self.P)

    def correct_wheel(self, m: "WheelMeasurement") -> None:
        """Wheel speed correction with NHC (10 Hz)."""
        R_wheel = self.R_wheel
        R_nhc = self.R_nhc

        if m.gate == "full":
            pass
        elif m.gate == "low":
            R_wheel * 10.0  # down-weight
        elif m.gate == "zupt":
            np.diag([1e-4])  # ZUPT
        else:
            return

        # Forward speed measurement
        H_wheel = np.zeros((1, self.n_state))
        H_wheel[0, 3] = 1.0  # vx
        z_wheel = np.array([m.vx_body * self.x[8]])  # scale factor applied (s_wheel at index 8)
        y_wheel = z_wheel - self.x[3] * self.x[8]

        S = self.x[8] * self.P[3, 3] * self.x[8] + R_wheel
        if self._residual_gate(y_wheel, S):
            return

        K = self.P @ H_wheel.T * (1.0 / S)
        self.x = self.x + K.ravel() * y_wheel[0]
        self.P = (np.eye(self.n_state) - K @ H_wheel) @ self.P
        self.P = self._check_and_repair_psd(self.P)

        # NHC: lateral/vertical velocity = 0
        H_nhc = np.zeros((2, self.n_state))
        H_nhc[0, 4] = 1.0  # vy = 0
        H_nhc[1, 5] = 1.0  # bg_z = 0 (approximation for vz constraint)
        z_nhc = np.zeros(2)
        y_nhc = z_nhc - np.array([self.x[4], 0.0])

        S_nhc = H_nhc @ self.P @ H_nhc.T + R_nhc
        if not self._residual_gate(y_nhc, S_nhc):
            K = self.P @ H_nhc.T @ la.inv(S_nhc)
            self.x = self.x + K @ y_nhc
            self.P = (np.eye(self.n_state) - K @ H_nhc) @ self.P
            self.P = self._check_and_repair_psd(self.P)

    def correct_vo(self, delta_pose: np.ndarray, R_vo: Optional[np.ndarray] = None):
        """Visual odometry relative pose correction (5 Hz)."""
        if R_vo is None:
            R_vo = self.R_vo

        # delta_pose: [dx, dy, dpsi] in body frame
        # Transform to ENU using current heading
        _c, _s = np.cos(self.x[2]), np.sin(self.x[2])
        R_b2n = np.array(
            [[np.cos(self.x[2]), -np.sin(self.x[2])], [np.sin(self.x[2]), np.cos(self.x[2])]]
        )
        delta_enu = R_b2n @ delta_pose[:2]
        delta_psi = delta_pose[2]

        H = np.zeros((3, self.n_state))
        H[0, 0] = 1.0  # dx
        H[1, 1] = 1.0  # dy
        H[2, 2] = 1.0  # dpsi

        z_pred = np.array([self.x[0], self.x[1], self.x[2]])
        z_meas = z_pred + np.array([delta_enu[0], delta_enu[1], delta_psi])
        y = z_meas - z_pred
        y[2] = self._wrap_angle(y[2])

        R = self.R_vo * self._cov_inflation
        S = H @ self.P @ H.T + R
        if self._residual_gate(y, S):
            self.skipped_measurements += 1
            return

        K = self.P @ H.T @ la.inv(S)
        self.x = self.x + K @ y
        self.x[2] = self._wrap_angle(self.x[2])
        self.P = (np.eye(self.n_state) - K @ H) @ self.P
        self.P = self._check_and_repair_psd(self.P)

    # =========================================================================
    # SEEDER INTEGRATION
    # =========================================================================

    def seed_anchor(self, sigma_inflate: float):
        """Part 6 hook: inflate covariance before predicted outage."""
        self._inflate_covariance(sigma_inflate)
        self._outage_active = True
        self._last_good_pose = {
            "position": self.x[:2].copy(),
            "heading": self.x[2],
            "timestamp": time.time() if "time" in globals() else 0.0,
        }

    def handover(self, gnss_pose: np.ndarray, blend_s: float):
        """Zero-jump re-anchor on GNSS reacquisition."""
        # gnss_pose: [x, y, psi]
        self._handover_active = True
        # Blend current state toward GNSS
        alpha = blend_s
        self.x[:2] = (1 - alpha) * self.x[:2] + alpha * gnss_pose[:2]
        self.x[2] = self._wrap_angle((1 - alpha) * self.x[2] + alpha * gnss_pose[2])
        # Reduce covariance
        self.P[:3, :3] *= 0.1
        self.P = self._check_and_repair_psd(self.P)
        self._handover_active = False

    # =========================================================================
    # STATE ACCESS
    # =========================================================================

    def state(self) -> "NavState":
        """Return current state with 3-sigma bounds."""
        np.sqrt(np.maximum(np.diag(self.P), 0))
        return NavState(
            ts=time.time() if "time" in globals() else 0.0,
            x=self.x[0],
            y=self.x[1],
            z=self.x[0],  # z=0 for SE(2)
            vx=self.x[3],
            vy=self.x[4],
            vz=0.0,
            yaw=self.x[2],
            std_x=np.sqrt(max(self.P[0, 0], 0)),
            std_y=np.sqrt(max(self.P[1, 1], 0)),
            std_z=0.0,
            std_vx=np.sqrt(max(self.P[3, 3], 0)),
            std_vy=np.sqrt(max(self.P[4, 4], 0)),
            std_vz=0.0,
            std_yaw=np.sqrt(max(self.P[2, 2], 0)),
            source_health={
                "gnss": not self._outage_active,
                "wheel": True,
                "vo": True,
                "velocity": True,
            },
        )

    def covariance(self) -> np.ndarray:
        return self.P.copy()

    def _residual_gate(self, y: np.ndarray, S: np.ndarray) -> bool:
        """Mahalanobis distance gate."""
        try:
            la.inv(S)
            mahal = y.T @ la.inv(S) @ y
            return mahal > self.residual_gate**2
        except la.LinAlgError:
            return True  # Skip if S is singular


# ============================================================================
# WHEEL MEASUREMENT DATACLASS (from Part 3)
# ============================================================================


@dataclass
class WheelMeasurement:
    ts: float
    vx_body: float
    vy_body: float = 0.0
    vz_body: float = 0.0
    R_diag: np.ndarray = None
    gate: str = "full"  # "full", "low", "zupt"
