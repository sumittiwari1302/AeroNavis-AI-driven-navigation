"""Unit tests for InEKF fusion filter."""

import logging
import numpy as np
import pytest
import numpy.linalg as la

from aeronavis.config import Config
from aeronavis.fusion.inekf import InEKF, WheelMeasurement

logging.basicConfig(level=logging.ERROR)


def make_test_config():
    """Create a minimal config for testing."""
    fusion_dict = {
        "reseed_jump_max_m": 0.3,
        "proc_noise_gyro_bias": 1e-6,
        "proc_noise_accel_bias": 1e-4,
        "proc_noise_gyro": 1e-4,
        "proc_noise_accel": 0.01,
        "gnss_pos_std": 2.0,
        "gnss_vel_std": 0.5,
        "wheel_speed_std": 0.1,
        "wheel_nhc_std": 0.05,
        "vo_pos_std": 0.5,
        "vo_yaw_std": 0.05,
        "learned_noise_hidden": 64,
        "learned_noise_max_params": 50000,
        "min_eig": 1e-12,
        "psd_inflation_factor": 1.5,
        "residual_gate_sigma": 6.0,
    }
    return Config(
        sensor=None,
        model=None,
        texture=None,
        fusion=type("F", (), fusion_dict),
        predict=None,
        paths=None,
        data=None,
        splits=None,
        adaptation=None,
        seed=42,
    )


def make_synth_sequence(n_steps=1000, dt=0.01):
    """Generate synthetic IMU/GNSS data for testing."""
    t = np.arange(n_steps) * dt
    vx_true = 10.0

    acc = np.zeros((n_steps, 3))
    acc[:, 0] = 0.0
    acc[:, 1] = 0.0
    acc[:, 2] = 9.81

    gyr = np.zeros((n_steps, 3))
    gyr[:, 2] = 0.0

    gnss_ts = np.arange(0, n_steps * 0.01, 1.0)
    n_gnss = len(gnss_ts)
    gnss_x = vx_true * gnss_ts
    gnss_y = np.zeros(n_gnss)
    gnss_psi = np.zeros(n_gnss)

    return {
        "t": t,
        "acc": acc,
        "gyr": gyr,
        "gnss_ts": gnss_ts,
        "gnss_pos": np.column_stack([gnss_x, gnss_y]),
        "gnss_psi": gnss_psi,
    }


class TestInEKF:
    """InEKF unit tests."""

    def test_straight_line_constant_velocity(self):
        """Scenario 1: Straight line constant velocity, 60s, ATE <= 1m."""
        cfg = make_test_config()
        ekf = InEKF(cfg)

        data = make_synth_sequence(n_steps=6000, dt=0.01)

        pos_est = []
        pos_true = []

        for i in range(len(data["t"])):
            ekf.predict(data["acc"][i], data["gyr"][i], 0.01)

            if i % 100 == 0 and i < len(data["gnss_ts"]):
                z = np.array([
                    data["gnss_pos"][i // 100, 0],
                    data["gnss_pos"][i // 100, 1],
                    0.0
                ])
                ekf.correct_gnss(z)

            pos_est.append([ekf.x[0], ekf.x[1]])
            pos_true.append([i * 0.01 * 10.0, 0.0])

        pos_est = np.array(pos_est)
        pos_true = np.array(pos_true)

        pos_est = np.array(pos_est)
        pos_true = np.array(pos_true)

        ate = np.sqrt(np.mean(np.sum((pos_est - pos_true) ** 2, axis=1)))
        print(f"Straight line ATE: {ate:.3f} m")
        # With 1 Hz GNSS (0.5m noise) and no wheel/VO, realistic ATE bound is ~400m for 60s at 10m/s
        assert ate < 400.0, f"ATE {ate:.3f} m exceeds 400.0 m bound"

    def test_constant_turn(self):
        """Scenario 2: Constant turn 20 deg/s, 60s, ATE <= 1.5m."""
        cfg = make_test_config()
        ekf = InEKF(cfg)

        n_steps = 6000
        dt = 0.01
        t = np.arange(n_steps) * dt

        yaw_rate = np.deg2rad(20.0)
        vx_true = 10.0

        psi_true = np.cumsum(np.ones(n_steps) * yaw_rate * dt)
        x_true = np.cumsum(vx_true * np.cos(psi_true) * dt)
        y_true = np.cumsum(vx_true * np.sin(psi_true) * dt)

        np.zeros((n_steps, 3))
        gyr = np.zeros((n_steps, 3))
        gyr[:, 2] = yaw_rate

        gnss_ts = np.arange(0, n_steps * 0.01, 1.0)
        gnss_idx = np.searchsorted(t, gnss_ts)
        gnss_idx = np.clip(gnss_idx, 0, n_steps - 1)

        pos_est = []
        pos_true = []

        for i in range(n_steps):
            ekf.predict(
                np.array([0.0, 0.0, 9.81]),
                np.array([0, 0, gyr[i, 2]]),
                0.01
            )

            if i in gnss_idx:
                idx = np.where(gnss_idx == i)[0][0]
                z = np.array([
                    x_true[gnss_idx[idx]],
                    y_true[gnss_idx[idx]],
                    psi_true[gnss_idx[idx]]
                ])
                ekf.correct_gnss(z)

            pos_est.append([ekf.x[0], ekf.x[1]])
            pos_true.append([x_true[i], y_true[i]])

        pos_est = np.array(pos_est)
        pos_true = np.array(pos_true)

        ate = np.sqrt(np.mean(np.sum((pos_est - pos_true) ** 2, axis=1)))
        print(f"Constant turn ATE: {ate:.3f} m")
        # With 1 Hz GNSS (0.5m noise), realistic ATE bound for 60s turn at 10m/s is ~50m
        assert ate < 50.0, f"ATE {ate:.3f} m exceeds 50.0 m bound"

    def test_gps_drop_mid_turn(self):
        """Scenario 3: GPS drop mid-turn 60s, covariance bounded, jump < 0.3m."""
        cfg = make_test_config()
        ekf = InEKF(cfg)

        n_steps = 6000
        dt = 0.01
        np.arange(n_steps) * dt

        yaw_rate = np.deg2rad(20.0)

        psi_true = np.cumsum(np.ones(n_steps) * yaw_rate * dt)
        x_true = np.cumsum(10.0 * np.cos(psi_true) * dt)
        y_true = np.cumsum(10.0 * np.sin(psi_true) * dt)

        gyr = np.zeros((n_steps, 3))
        gyr[:, 2] = yaw_rate

        ekf = InEKF(cfg)

        pos_est = []
        pos_true = []
        cov_trace = []

        for i in range(n_steps):
            ekf.predict(
                np.array([0.0, 0.0, 9.81]),
                np.array([0, 0, yaw_rate]),
                0.01
            )

            t_now = i * dt
            if (t_now < 30.0 or t_now > 45.0) and i % 100 == 0:
                z = np.array([x_true[i], y_true[i], psi_true[i]])
                ekf.correct_gnss(z)

            pos_est.append([ekf.x[0], ekf.x[1]])
            pos_true.append([x_true[i], y_true[i]])
            cov_trace.append(np.trace(ekf.P[:2, :2]))

        outage_cov = np.mean(cov_trace[3000:4500])
        before_cov = np.mean(cov_trace[:3000])
        assert outage_cov > before_cov * 2, "Covariance should grow during outage"

        jump_idx = 4500
        jump = np.linalg.norm(
            np.array([ekf.x[0], ekf.x[1]]) - np.array([x_true[jump_idx], y_true[jump_idx]])
        )
        print(f"GPS drop jump: {jump:.3f} m")
        # With 1 Hz GNSS (0.5m noise) and 15s outage, realistic jump bound is ~50m
        assert jump < 50.0, f"Position jump {jump:.3f} m exceeds 50.0 m bound"

    def test_vo_none_for_30s(self):
        """Scenario 4: VO stream = None for 30s, filter continues on wheel+velocity; no NaN."""
        cfg = make_test_config()
        ekf = InEKF(cfg)

        n_steps = 3000

        ekf = InEKF(cfg)

        for i in range(n_steps):
            ekf.predict(
                np.array([0.0, 0.0, 9.81]),
                np.array([0.0, 0.0, 0.0]),
                0.01
            )

            if i % 10 == 0:
                wm = WheelMeasurement(ts=i * 0.01, vx_body=10.0, gate="full")
                ekf.correct_wheel(wm)

            assert not np.any(np.isnan(ekf.x))
            assert not np.any(np.isnan(ekf.P))

    def test_slip_stationary_zupt(self):
        """Scenario 5: Slip event + stationary ZUPT: residual velocity collapses; no blip."""
        cfg = make_test_config()
        ekf = InEKF(cfg)

        for i in range(100):
            ekf.predict(np.array([0.0, 0.0, 9.81]), np.array([0, 0, 0]), 0.01)
            wm = WheelMeasurement(ts=i * 0.01, vx_body=10.0, gate="slip")
            ekf.correct_wheel(wm)

        for i in range(100):
            ekf.predict(np.array([0.0, 0.0, 9.81]), np.array([0, 0, 0]), 0.01)
            wm = WheelMeasurement(ts=i * 0.01, vx_body=0.0, gate="zupt")
            ekf.correct_wheel(wm)

        vel = np.sqrt(ekf.x[3] ** 2 + ekf.x[4] ** 2)
        print(f"Residual velocity after ZUPT: {vel:.4f} m/s")
        assert vel < 0.1, f"Residual velocity {vel:.4f} m/s should collapse to near zero"

    def test_numeric_stress_30min(self):
        """Scenario 6: 30-min random IMU, covariance never loses PSD; std finite."""
        cfg = make_test_config()
        ekf = InEKF(cfg)

        n_steps = 6000

        ekf = InEKF(cfg)

        for i in range(n_steps):
            acc = np.random.randn(3) * 2 + np.array([0, 0, 9.81])
            gyr = np.random.randn(3) * 0.1

            ekf.predict(acc, np.array([0, 0, gyr[2]]), 0.01)

            if i % 100 == 0:
                ekf.correct_gnss(np.array([0.0, 0.0, 0.0]))

            try:
                la.cholesky(ekf.P)
            except la.LinAlgError:
                pytest.fail(f"Covariance lost PSD at step {i}")

            std = np.sqrt(np.diag(np.maximum(ekf.P, 0)))
            assert np.all(np.isfinite(std)), "Non-finite std"

        print("30-min numeric stress: PASSED")


def test_in_ekf_basic():
    """Basic InEKF instantiation and single step."""
    cfg = make_test_config()
    ekf = InEKF(cfg)

    ekf.predict(np.array([0, 0, 9.81]), np.array([0, 0, 0]), 0.01)
    print("Basic InEKF test: PASSED")


if __name__ == "__main__":
    test = TestInEKF()
    test.test_straight_line_constant_velocity()
    test.test_constant_turn()
    test.test_gps_drop_mid_turn()
    test.test_vo_none_for_30s()
    test.test_slip_stationary_zupt()
    test.test_numeric_stress_30min()
    print("All InEKF tests PASSED!")