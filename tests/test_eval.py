"""Tests for evaluation metrics and forced-blackout protocol."""

import json
import tempfile
import numpy as np
import pytest
from pathlib import Path

from aeronavis.eval import (
    compute_ate,
    compute_rte,
    compute_drift_pct_km,
    heading_error,
    umeyama_alignment,
    latency_ms,
    ForcedBlackoutEvaluator,
    BlackoutResult,
)


def test_umeyama_alignment():
    """Test Umeyama alignment on known transforms."""
    # Identity transform
    src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    dst = src.copy()
    R, s, t = umeyama_alignment(src, dst)
    assert np.allclose(R, np.eye(3))
    assert abs(s - 1.0) < 1e-6
    assert np.allclose(t, 0)

    # Translation
    dst = src + np.array([1.0, 2.0, 3.0])
    R, s, t = umeyama_alignment(src, dst)
    assert np.allclose(R, np.eye(3))
    assert abs(s - 1.0) < 1e-6
    assert np.allclose(t, [1.0, 2.0, 3.0])

    # Scale + rotation
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    dst = (src @ rot.T) * 2.0 + np.array([5, 5, 5])
    R, s, t = umeyama_alignment(src, dst)
    assert abs(s - 2.0) < 1e-6
    assert np.allclose(R @ R.T, np.eye(3))


def test_ate():
    """Test ATE computation on known trajectories."""
    # Perfect match
    gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    est = gt.copy()
    ate = compute_ate(est, gt)
    assert ate.rmse < 1e-6

    # Constant offset (no alignment)
    est = gt + np.array([1.0, 1.0, 1.0])
    ate = compute_ate(est, gt, align="none")
    assert abs(ate.rmse - np.sqrt(3)) < 1e-6

    # Scaling (with sim3 alignment)
    est = gt * 2
    ate = compute_ate(est, gt, align="sim3")
    assert ate.rmse < 1e-6


def test_rte():
    """Test RTE on known trajectories."""
    # Perfect circle
    theta = np.linspace(0, 2 * np.pi, 100)
    gt = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    est = gt.copy()
    
    rte = compute_rte(est, gt, seg_lengths=[1.0, 2.0])
    assert len(rte.window_lengths) == 2
    assert all(abs(r) < 1e-10 for r in rte.rmse_per_window)


def test_drift_pct_km():
    """Test drift percentage per km."""
    # Straight line 1km, 10m endpoint error
    gt = np.column_stack([
        np.linspace(0, 1000, 1000),
        np.zeros(1000),
        np.zeros(1000)
    ])
    est = gt.copy()
    est[-1, 0] += 10  # 10m endpoint error
    
    drift = compute_drift_pct_km(est, gt)
    # 10m error over 1km = 1% per km
    assert abs(drift.drift_pct_per_km - 1.0) < 0.1


def test_heading_error():
    """Test heading error calculation."""
    # Perfect match
    gt = np.column_stack([np.arange(100), np.zeros(100), np.zeros(100), np.zeros(100)])
    est = gt.copy()
    hdr = heading_error(est, gt)
    assert hdr.mean_deg < 1e-6

    # 90 degree constant offset
    est = gt.copy()
    est[:, 3] += np.pi/2
    hdr = heading_error(est, gt)
    assert abs(hdr.mean_deg - 90) < 1e-6


def test_latency_ms():
    """Test latency measurement."""
    def dummy_fn():
        pass
    
    res = latency_ms(lambda: None, n=100, warmup=5)
    assert res.p50_ms >= 0
    assert res.p95_ms >= res.p50_ms
    assert res.mean_ms >= 0


def test_save_metrics():
    """Test saving metrics to JSON."""
    
    gt = np.array([[0,0,0], [1,0,0], [2,0,0]])
    est = gt.copy()
    
    ate = compute_ate(est, gt)
    rte = compute_rte(est, gt)
    drift = compute_drift_pct_km(est, gt)
    hdr = heading_error(est, est)
    
    from aeronavis.eval import MetricsResult, save_metrics
    result = MetricsResult(ate=ate, rte=rte, drift=drift, heading_error=hdr)
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        save_metrics(result, Path(f.name))
        with open(f.name) as f:
            data = json.load(f)
        assert "ate" in data
        assert "rte" in data
        assert "drift" in data


def test_blackout_schedule():
    """Test blackout schedule creation."""
    from aeronavis.data.preprocess import NavSequence
    import pandas as pd
    import numpy as np
    
    # Create a minimal mock sequence
    imu = pd.DataFrame({"ts": np.arange(0, 100, 0.01), "acc_x": 0, "acc_y": 0, "acc_z": 9.81, "gyr_x": 0, "gyr_y": 0, "gyr_z": 0})
    truth = pd.DataFrame({"ts": np.arange(0, 100, 0.1), "x_m": np.arange(0, 1000, 1), "y_m": 0, "z_m": 0, "heading": 0})
    gnss = pd.DataFrame({"ts": np.arange(0, 100, 1.0), "lat": 0, "lon": 0, "alt_m": 0, "speed_mps": 10, "heading_deg": 0})
    
    seq = NavSequence(
        seq_id="test",
        source="test",
        imu=imu,
        truth=truth,
        gnss=gnss,
    )
    
    evaluator = ForcedBlackoutEvaluator()
    schedule = evaluator.build_blackout_schedule(
        sequence=seq,
        blackout_distances=[100, 200, 300],
        durations_s=[30, 60, 30]
    )
    assert len(schedule) == 3
    assert schedule[0].distance_m == 100
    assert schedule[0].duration_s == 30


def test_blackout_result():
    """Test BlackoutResult dataclass."""
    
    result = BlackoutResult(
        start_distance_m=100.0,
        duration_s=30.0,
        ate_before=1.0,
        ate_during=2.0,
        ate_after=1.5,
        max_cov_growth=2.0,
        jump_at_reacquisition_m=0.1
    )
    assert result.start_distance_m == 100.0
    assert result.duration_s == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])