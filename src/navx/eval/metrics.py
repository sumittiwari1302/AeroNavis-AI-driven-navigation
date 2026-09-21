"""Evaluation metrics for NAV-X 3.0 navigation engine.

Implements ATE, RTE, drift, heading error, latency, and energy metrics
with exact mathematical definitions and unit tests.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ATEResult:
    """Absolute Trajectory Error result."""
    rmse: float
    mean: float
    median: float
    p95: float
    max: float
    aligned_est: np.ndarray  # Aligned estimated trajectory
    aligned_gt: np.ndarray   # Ground truth trajectory


@dataclass
class RTEResult:
    """Relative Trajectory Error result."""
    window_lengths: List[int]
    rmse_per_window: List[float]
    mean: float
    median: float
    p95: float


@dataclass
class DriftResult:
    """Drift percentage per km result."""
    drift_pct_per_km: float
    endpoint_error_m: float
    traveled_distance_m: float


@dataclass
class HeadingErrorResult:
    """Heading error statistics."""
    mean_deg: float
    median_deg: float
    p95_deg: float
    max_deg: float


@dataclass
class LatencyResult:
    """Latency measurement results."""
    p50_ms: float
    p95_ms: float
    mean_ms: float
    std_ms: float


@dataclass
class EnergyResult:
    """Energy consumption result (placeholder for Android)."""
    energy_mah: float
    duration_s: float
    device: str


@dataclass
class MetricsResult:
    """Complete metrics result container."""
    ate: "ATEResult"
    rte: "RTEResult"
    drift: "DriftResult"
    heading_error: "HeadingErrorResult"
    latency: Optional["LatencyResult"] = None
    energy: Optional["EnergyResult"] = None


def umeyama_alignment(
    src: np.ndarray,
    dst: np.ndarray,
    with_scaling: bool = True,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Umeyama alignment: find similarity transform (s, R, t) to align src to dst.
    
    Minimizes ||s * src @ R.T + t - dst||^2
    
    Returns:
        R: rotation matrix (3x3)
        s: scale factor
        t: translation vector (3,)
    """
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    
    # Scale
    var_src = np.sum(src_c ** 2) / len(src)
    if var_src < 1e-10:
        return np.eye(3), 1.0, np.zeros(3)
    
    # Rotation via SVD
    cov = src_c.T @ dst_c / len(src)
    U, _, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    
    # Scale
    s = np.trace(cov @ R.T) / var_src
    if s < 0:
        s = 1.0
    
    # Translation
    t = mu_dst - s * (mu_src @ R.T)
    
    return R, s, t


def align_trajectory(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Align predicted trajectory to ground truth using Umeyama."""
    R, s, t = umeyama_alignment(pred, dst=gt)
    return s * (pred @ R.T) + t


def compute_ate(
    traj_est: np.ndarray,
    traj_truth: np.ndarray,
    align: str = "sim3",
) -> "ATEResult":
    """
    Absolute Trajectory Error (ATE) using Umeyama alignment.
    
    Args:
        traj_est: (N, 3) or (N, 4) estimated trajectory [x, y, z] or [x, y, z, yaw]
        traj_truth: (N, 3) or (N, 4) ground truth trajectory
        align: "sim3" (similarity), "se3" (rigid), "none" (no alignment)
    
    Returns:
        ATEResult with RMSE and aligned trajectories
    """
    from dataclasses import dataclass
    
    @dataclass
    class ATEResult:
        rmse: float
        mean: float
        median: float
        p95: float
        max: float
        aligned_est: np.ndarray
        aligned_gt: np.ndarray
    
    if traj_est.shape != traj_truth.shape:
        raise ValueError(f"Shape mismatch: {traj_est.shape} vs {traj_truth.shape}")
    
    if traj_est.shape[1] >= 3:
        pred_xyz = traj_est[:, :3]
        gt_xyz = traj_truth[:, :3]
    else:
        raise ValueError("Trajectory must have at least 3 columns (x, y, z)")
    
    if align == "sim3":
        pred_aligned = align_trajectory(traj_est[:, :3], traj_truth[:, :3])
    elif align == "se3":
        R, s, t = umeyama_alignment(traj_est[:, :3], traj_truth[:, :3], with_scaling=False)
        pred_aligned = (traj_est[:, :3] @ R.T) + t
    elif align == "none":
        pred_aligned = traj_est[:, :3]
    else:
        raise ValueError(f"Unknown align mode: {align}")
    
    errors = np.linalg.norm(pred_aligned - traj_truth[:, :3], axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    p95_err = float(np.percentile(errors, 95))
    max_err = float(np.max(errors))
    
    @dataclass
    class ATEResult:
        rmse: float
        mean: float
        median: float
        p95: float
        max: float
        aligned_est: np.ndarray
        aligned_gt: np.ndarray
    
    return ATEResult(
        rmse=rmse,
        mean=mean_err,
        median=median_err,
        p95=p95_err,
        max=max_err,
        aligned_est=pred_aligned,
        aligned_gt=traj_truth[:, :3],
    )


def compute_rte(
    traj_est: np.ndarray,
    traj_truth: np.ndarray,
    seg_lengths: List[float] = None,
) -> "RTEResult":
    """
    Relative Trajectory Error (RTE) over specified segment lengths.
    
    Args:
        traj_est: (N, 3) estimated trajectory
        traj_truth: (N, 3) ground truth trajectory
        seg_lengths: list of segment lengths in meters (default: [10, 30, 60, 120, 300])
    
    Returns:
        RTEResult with per-window RMSE
    """
    from dataclasses import dataclass
    
    @dataclass
    class RTEResult:
        window_lengths: List[int]
        rmse_per_window: List[float]
        mean: float
        median: float
        p95: float
    
    if seg_lengths is None:
        seg_lengths = [10.0, 30.0, 60.0, 120.0, 300.0]
    
    # Align first
    pred_aligned = align_trajectory(traj_est, traj_truth)
    
    # Compute cumulative distances along trajectory
    diffs = np.diff(traj_truth[:, :3], axis=0)
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_dists = np.concatenate([[0.0], np.cumsum(dists)])
    
    window_results = {}
    for seg_len in seg_lengths:
        window_rmse = []
        for i in range(len(traj_truth)):
            # Find window of length seg_len
            d = seg_len
            j = np.searchsorted(cum_dists, cum_dists[i] + d, side='right')
            if j <= i or j >= len(traj_truth):
                continue
            pred_seg = traj_est[i:j]
            truth_seg = traj_truth[i:j]
            if len(pred_seg) < 2:
                continue
            # Align segment
            pred_aligned = align_trajectory(traj_est[i:j], traj_truth[i:j, :3])
            err = np.sqrt(np.mean(np.sum((pred_aligned - traj_truth[i:j, :3]) ** 2, axis=1)))
            window_results.setdefault(seg_len, []).append(err)
    
    # Aggregate
    window_lengths = []
    rmse_per_window = []
    for seg_len in seg_lengths:
        if seg_len in window_results and window_results[seg_len]:
            errors = np.array(window_results[seg_len])
            window_lengths.append(seg_len)
            rmse_per_window.append(float(np.sqrt(np.mean(np.array(window_results[seg_len]) ** 2))))
    
    if not rmse_per_window:
        @dataclass
        class RTEResult:
            window_lengths: List[int]
            rmse_per_window: List[float]
            mean: float
            median: float
            p95: float
        return RTEResult(window_lengths=[], rmse_per_window=[], mean=0.0, median=0.0, p95=0.0)
    
    arr = np.array(rmse_per_window)
    
    @dataclass
    class RTEResult:
        window_lengths: List[int]
        rmse_per_window: List[float]
        mean: float
        median: float
        p95: float
    
    return RTEResult(
        window_lengths=window_lengths,
        rmse_per_window=rmse_per_window,
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        p95=float(np.percentile(arr, 95)),
    )


def compute_drift_pct_km(
    traj_est: np.ndarray,
    traj_truth: np.ndarray,
) -> "DriftResult":
    """
    Drift percentage per km: endpoint error / traveled distance as %.
    """
    from dataclasses import dataclass
    
    @dataclass
    class DriftResult:
        drift_pct_per_km: float
        endpoint_error_m: float
        traveled_distance_m: float
    
    pred_aligned = align_trajectory(traj_est[:, :3], traj_truth[:, :3])
    endpoint_error = np.linalg.norm(pred_aligned[-1] - traj_truth[-1, :3])
    
    # Total traveled distance
    diffs = np.diff(traj_truth[:, :3], axis=0)
    traveled = np.sum(np.linalg.norm(diffs, axis=1))
    
    if traveled < 1e-6:
        return DriftResult(drift_pct_per_km=0.0, endpoint_error_m=0.0, traveled_distance_m=0.0)
    
    # Drift % = (endpoint_error / traveled_distance) * 100
    drift_pct_per_km = (endpoint_error / traveled) * 100.0
    return DriftResult(
        drift_pct_per_km=drift_pct_per_km,
        endpoint_error_m=float(endpoint_error),
        traveled_distance_m=float(traveled),
    )


def heading_error(
    traj_est: np.ndarray,
    traj_truth: np.ndarray,
) -> "HeadingErrorResult":
    """
    Heading error in degrees.
    Assumes trajectory has yaw in column 3 (index 3) or uses velocity direction.
    """
    from dataclasses import dataclass
    
    @dataclass
    class HeadingErrorResult:
        mean_deg: float
        median_deg: float
        p95_deg: float
        max_deg: float
    
    if traj_est.shape[1] >= 4 and traj_truth.shape[1] >= 4:
        est_yaw = traj_est[:, 3]
        gt_yaw = traj_truth[:, 3]
    else:
        # Estimate from velocity direction
        est_vel = np.diff(traj_est[:, :2], axis=0)
        gt_vel = np.diff(traj_truth[:, :2], axis=0)
        est_yaw = np.arctan2(est_vel[:, 1], est_vel[:, 0])
        gt_yaw = np.arctan2(gt_vel[:, 1], gt_vel[:, 0])
        est_yaw = np.concatenate([est_yaw[:1], est_yaw])
        gt_yaw = np.concatenate([gt_yaw[:1], gt_yaw])
    
    # Angular difference (wrapped to [-pi, pi])
    diff = np.abs(yaw_diff(est_yaw, gt_yaw))
    diff_deg = np.rad2deg(diff)
    
    from dataclasses import dataclass
    @dataclass
    class HeadingErrorResult:
        mean_deg: float
        median_deg: float
        p95_deg: float
        max_deg: float
    
    return HeadingErrorResult(
        mean_deg=float(np.mean(diff_deg)),
        median_deg=float(np.median(diff_deg)),
        p95_deg=float(np.percentile(diff_deg, 95)),
        max_deg=float(np.max(diff_deg)),
    )


def yaw_diff(yaw1: np.ndarray, yaw2: np.ndarray) -> np.ndarray:
    """Smallest angular difference between two yaws."""
    diff = yaw1 - yaw2
    return np.arctan2(np.sin(diff), np.cos(diff))


def latency_ms(
    inference_fn: callable,
    n: int = 100,
    warmup: int = 10,
) -> "LatencyResult":
    """
    Measure inference latency in milliseconds.
    """
    from dataclasses import dataclass
    
    @dataclass
    class LatencyResult:
        p50_ms: float
        p95_ms: float
        mean_ms: float
        std_ms: float
    
    
    # Warmup
    for _ in range(warmup):
        inference_fn()
    
    times = []
    for _ in range(n):
        start = time.perf_counter()
        inference_fn()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    
    times = np.array(times)
    return LatencyResult(
        p50_ms=float(np.median(times)),
        p95_ms=float(np.percentile(times, 95)),
        mean_ms=float(np.mean(times)),
        std_ms=float(np.std(times)),
    )


def energy_mah(profile: Dict) -> "EnergyResult":
    """Placeholder for energy measurement (Android only)."""
    from dataclasses import dataclass
    
    @dataclass
    class EnergyResult:
        energy_mah: float
        duration_s: float
        device: str
    
    return EnergyResult(
        energy_mah=profile.get("energy_mah", 0.0),
        duration_s=profile.get("duration_s", 0.0),
        device=profile.get("device", "unknown"),
    )


def compute_all_metrics(
    traj_est: np.ndarray,
    traj_truth: np.ndarray,
    inference_fn: Optional[callable] = None,
    energy_profile: Optional[Dict] = None,
) -> "MetricsResult":
    """Compute all metrics in one call."""
    from dataclasses import dataclass
    
    @dataclass
    class MetricsResult:
        ate: "ATEResult"
        rte: "RTEResult"
        drift: "DriftResult"
        heading_error: "HeadingErrorResult"
        latency: Optional["LatencyResult"] = None
        energy: Optional["EnergyResult"] = None
    
    return MetricsResult(
        ate=compute_ate(traj_est, traj_truth),
        rte=compute_rte(traj_est, traj_truth),
        drift=compute_drift_pct_km(traj_est, traj_truth),
        heading_error=heading_error(traj_est, traj_truth),
        latency=latency_ms(inference_fn) if inference_fn else None,
        energy=None,  # Placeholder
    )


def save_metrics(result: "MetricsResult", path: Union[str, Path]):
    """Save metrics to JSON."""
    data = {
        "ate": {
            "rmse": result.ate.rmse,
            "mean": result.ate.mean,
            "median": result.ate.median,
            "p95": result.ate.p95,
            "max": result.ate.max,
        },
        "rte": {
            "window_lengths": result.rte.window_lengths,
            "rmse_per_window": result.rte.rmse_per_window,
            "mean": result.rte.mean,
            "median": result.rte.median,
            "p95": result.rte.p95,
        },
        "drift": {
            "drift_pct_per_km": result.drift.drift_pct_per_km,
            "endpoint_error_m": result.drift.endpoint_error_m,
            "traveled_distance_m": result.drift.traveled_distance_m,
        },
        "heading_error": {
            "mean_deg": result.heading_error.mean_deg,
            "median_deg": result.heading_error.median_deg,
            "p95_deg": result.heading_error.p95_deg,
            "max_deg": result.heading_error.max_deg,
        },
    }
    if result.latency:
        data["latency"] = {
            "p50_ms": result.latency.p50_ms,
            "p95_ms": result.latency.p95_ms,
            "mean_ms": result.latency.mean_ms,
            "std_ms": result.latency.std_ms,
        }
    if result.energy:
        data["energy"] = {
            "energy_mah": result.energy.energy_mah,
            "duration_s": result.energy.duration_s,
            "device": result.energy.device,
        }
    
    Path(path).write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    # Quick sanity check
    np.random.seed(42)
    gt = np.cumsum(np.random.randn(100, 3) * 0.1, axis=0)
    gt[:, 2] = 0
    est = gt + np.random.randn(100, 3) * 0.1
    
    ate = compute_ate(est, gt)
    rte = compute_rte(est, gt)
    drift = compute_drift_pct_km(est, gt)
    hdr = heading_error(est, gt)
    
    print(f"ATE RMSE: {ate.rmse:.3f} m")
    print(f"RTE: {rte.rmse_per_window}")
    print(f"Drift: {drift.drift_pct_per_km:.2f}%/km")
    print(f"Heading error: {heading_error(ate.aligned_est, ate.aligned_gt).mean_deg:.3f} deg")