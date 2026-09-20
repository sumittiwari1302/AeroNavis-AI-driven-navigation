"""Velocity model evaluation: ATE, RTE, speed bias, failure analysis.

Computes:
- ATE: RMSE of trajectory aligned via Umeyama (translation+rotation+scale)
- ATE windows: at {10, 30, 60}s
- RTE: drift % per km on {0.5, 1, 2} km windows
- Speed bias: mean(pred - true) / true
- Failure annotations: aggressive turns, lean, phone-in-bag
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aeronavis.config import get_config
from aeronavis.models.velocity import VelocityModel, build_velocity_model
from aeronavis.data.velocity_dataset import velocity_collate, build_velocity_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Umeyama alignment: find s, R, t to minimize ||s * src @ R.T + t - dst||^2.

    Returns (R, s, t) where dst_aligned = s * src @ R.T + t
    """
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Scale
    var_src = np.sum(src_c**2) / len(src)
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
    R, s, t = umeyama_alignment(pred, gt)
    return s * (pred @ R.T) + t


def compute_ate(pred: np.ndarray, gt: np.ndarray) -> float:
    """RMSE after Umeyama alignment (3D)."""
    pred_aligned = align_trajectory(pred, gt)
    return float(np.sqrt(np.mean(np.sum((pred_aligned - gt) ** 2, axis=1))))


def compute_ate_windows(
    pred: np.ndarray, gt: np.ndarray, times: np.ndarray, window_secs: list[int]
) -> dict:
    """ATE computed on sliding windows of given lengths."""
    results = {}
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    for w in window_secs:
        w_samples = max(1, int(w / dt))
        if len(pred) < w_samples:
            continue
        ates = []
        for i in range(0, len(pred) - w_samples + 1, w_samples // 2):
            p_win = pred[i : i + w_samples]
            g_win = gt[i : i + w_samples]
            if len(p_win) < 2:
                continue
            ate = compute_ate(p_win, g_win)
            ates.append(ate)
        if ates:
            results[f"ATE_{w}s"] = {
                "mean": float(np.mean(ates)),
                "median": float(np.median(ates)),
                "p90": float(np.percentile(ates, 90)),
            }
    return results


def compute_rte(
    pred: np.ndarray, gt: np.ndarray, times: np.ndarray, window_kms: list[float]
) -> dict:
    """RTE: drift % per km on windows of given km lengths."""
    results = {}

    # Compute cumulative distance along GT
    gt_diffs = np.diff(gt, axis=0)
    gt_dist = np.cumsum(np.linalg.norm(gt_diffs, axis=1))
    gt_dist = np.concatenate([[0], gt_dist])

    for w_km in window_kms:
        w_m = w_km * 1000
        rtes = []
        i = 0
        while i < len(pred) - 1:
            # Find window ending at w_m distance
            j = i + 1
            while j < len(pred) and gt_dist[j] - gt_dist[i] < w_m:
                j += 1
            if j - i < 2:
                break
            p_win = pred[i:j]
            g_win = gt[i:j]
            if len(p_win) < 2:
                i = j
                continue
            ate = compute_ate(p_win, g_win)
            rte_pct = (ate / w_m) * 100
            rtes.append(rte_pct)
            i = j
        if rtes:
            results[f"RTE_{w_km}km"] = {
                "mean_pct": float(np.mean(rtes)),
                "median_pct": float(np.median(rtes)),
                "p90_pct": float(np.percentile(rtes, 90)),
            }
    return results


def compute_speed_bias(vel_pred: np.ndarray, vel_true: np.ndarray) -> dict:
    """Speed bias metrics."""
    mask = vel_true > 0.1  # Only when moving
    if not np.any(mask):
        return {"bias": 0.0, "rel_bias": 0.0, "rmse": 0.0}
    bias = float(np.mean(vel_pred[mask] - vel_true[mask]))
    rel_bias = float(np.mean((vel_pred[mask] - vel_true[mask]) / (vel_true[mask] + 1e-6)))
    rmse = float(np.sqrt(np.mean((vel_pred[mask] - vel_true[mask]) ** 2)))
    return {"bias": bias, "rel_bias": rel_bias, "rmse": rmse}


def detect_failures(seq, vel_pred, vel_true) -> list[str]:
    """Detect failure modes: aggressive turns, lean, phone-in-bag."""
    failures = []
    # This would need sequence metadata; placeholder for now
    return failures


def evaluate_model(
    model: VelocityModel,
    test_loader: DataLoader,
    device: torch.device,
    config: dict,
) -> dict:
    """Run full evaluation on test set."""
    model.eval()
    device = torch.device(device)

    all_vel_pred = []
    all_vel_true = []

    with torch.no_grad():
        for batch in test_loader:
            acc = batch["acc"].to(device)
            gyr = batch["gyr"].to(device)
            att0 = batch["att0"].to(device)
            vel_target = batch["vel_target"].to(device)

            vel_pred = model(acc, gyr, att0).squeeze(-1)

            all_vel_pred.append(vel_pred.cpu().numpy())
            all_vel_true.append(vel_target.cpu().numpy())

    vel_pred_all = np.concatenate(all_vel_pred)
    vel_true_all = np.concatenate(all_vel_true)

    # Overall metrics
    speed_bias = compute_speed_bias(vel_pred_all, vel_true_all)
    vel_rmse = float(np.sqrt(np.mean((vel_pred_all - vel_true_all) ** 2)))
    vel_mae = float(np.mean(np.abs(vel_pred_all - vel_true_all)))

    summary = {
        "velocity_rmse": vel_rmse,
        "velocity_mae": vel_mae,
        "speed_bias": speed_bias,
        "total_samples": int(len(vel_pred_all)),
    }
    return {"summary": summary}


def evaluate_full_sequences(
    model: VelocityModel,
    sequences,
    config: dict,
    device: torch.device,
) -> dict:
    """Evaluate on full sequences: integrate velocity to get trajectory, compute ATE/RTE."""
    model.eval()
    device = torch.device(device)

    results = {
        "sequences": [],
        "ate_windows": {},
        "rte_windows": {},
    }

    # For full trajectory evaluation, we need to reconstruct per-sequence
    # This requires grouping samples by sequence - simplified here
    # In practice, would iterate sequences and predict full velocity profile

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate velocity model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="models/velocity/eval_results.json")
    args = parser.parse_args()

    config = get_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_velocity_model(config)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Build test dataset
    _, _, test_ds = build_velocity_datasets(config)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=velocity_collate)

    results = evaluate_model(model, test_loader, device, config.__dict__)

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")
    logger.info(f"Summary: {json.dumps(results['summary'], indent=2)}")


if __name__ == "__main__":
    main()
