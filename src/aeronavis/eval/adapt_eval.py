"""Adaptation evaluation: ATE comparison with/without adapters under forced blackout."""

import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from aeronavis.config import get_config, Config
from aeronavis.models.velocity import build_velocity_model
from aeronavis.models.adaptation_engine import AdaptationEngine
from aeronavis.data.velocity_dataset import build_velocity_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> tuple:
    """Umeyama alignment: find s, R, t to minimize ||s * src @ R.T + t - dst||^2."""
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = np.sum(src_c ** 2) / len(src)
    if var_src < 1e-10:
        return np.eye(3), 1.0, np.zeros(3)

    cov = src_c.T @ dst_c / len(src)
    U, _, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    s = np.trace(cov @ R.T) / var_src
    if s < 0:
        s = 1.0

    t = mu_dst - s * (mu_src @ R.T)
    return R, s, t


def align_trajectory(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    R, s, t = umeyama_alignment(pred, gt)
    return s * (pred @ R.T) + t


def compute_ate(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_aligned = align_trajectory(pred, gt)
    return float(np.sqrt(np.mean(np.sum((pred_aligned - gt) ** 2, axis=1))))


def compute_ate_windows(
    pred: np.ndarray, gt: np.ndarray, times: np.ndarray, window_secs: list
) -> dict:
    results = {}
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    for w in window_secs:
        w_samples = max(1, int(w / dt))
        if len(pred) < w_samples:
            continue
        ates = []
        for i in range(0, len(pred) - w_samples + 1, w_samples // 2):
            p_win = pred[i:i + w_samples]
            g_win = gt[i:i + w_samples]
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


def compute_rte(pred: np.ndarray, gt: np.ndarray, times: np.ndarray, window_kms: list) -> dict:
    results = {}
    times[1] - times[0] if len(times) > 1 else 1.0

    gt_diffs = np.diff(gt, axis=0)
    gt_dist = np.cumsum(np.linalg.norm(gt_diffs, axis=1))
    gt_dist = np.concatenate([[0], gt_dist])

    for w_km in window_kms:
        w_m = w_km * 1000
        rtes = []
        i = 0
        while i < len(pred) - 1:
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


def evaluate_model_on_sequence(
    model,
    engine: Optional[AdaptationEngine],
    sequence,
    config: Config,
    device: torch.device,
) -> dict:
    """Evaluate on a single sequence, returning trajectory and metrics."""
    from aeronavis.data.velocity_dataset import VelocityDataset

    dataset = VelocityDataset([sequence], config, is_train=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    all_vel_pred = []
    all_vel_true = []
    all_times = []

    with torch.no_grad():
        for batch in loader:
            acc = batch["acc"].to(device)
            gyr = batch["gyr"].to(device)
            vel_target = batch["vel_target"].to(device)

            if engine is not None:
                # Use adaptation engine forward
                vel_pred = engine.infer(
                    window_acc=acc,
                    window_gyr=gyr,
                    att0=None,
                )
                if isinstance(vel_pred, torch.Tensor):
                    vel_pred = vel_pred.cpu().numpy()
            else:
                # Base model forward
                vel_pred = model(acc, gyr, None).squeeze(-1).cpu().numpy()

            all_vel_pred.append(vel_pred)
            all_vel_true.append(vel_target.cpu().numpy())
            all_times.append(batch.get("ts", np.arange(len(vel_target)) * 0.01))

    vel_pred_all = np.concatenate(all_vel_pred)
    vel_true_all = np.concatenate(all_vel_true)
    times_all = np.concatenate(all_times) if all_times else np.arange(len(vel_pred_all)) * 0.01

    # Integrate velocity to get trajectory (simple Euler)
    dt = times_all[1] - times_all[0] if len(times_all) > 1 else 0.01
    pred_traj = np.cumsum(vel_pred_all) * dt
    true_traj = np.cumsum(vel_true_all) * dt

    # 2D trajectory (x, y) assuming forward motion
    pred_2d = np.column_stack([pred_traj, np.zeros_like(pred_traj)])
    true_2d = np.column_stack([true_traj, np.zeros_like(true_traj)])

    ate = compute_ate(pred_2d, true_2d)
    ate_windows = compute_ate_windows(pred_2d, true_2d, times_all, [10, 30, 60])
    rte_windows = compute_rte(pred_2d, true_2d, times_all, [0.5, 1.0, 2.0])

    return {
        "ate": ate,
        "ate_windows": ate_windows,
        "rte_windows": rte_windows,
        "vel_rmse": float(np.sqrt(np.mean((vel_pred_all - vel_true_all) ** 2))),
        "vel_mae": float(np.mean(np.abs(vel_pred_all - vel_true_all))),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate adaptation vs base model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--base-checkpoint", required=True, help="Base velocity model checkpoint")
    parser.add_argument("--adapt-checkpoint", help="Adaptation engine checkpoint (stable.pt)")
    parser.add_argument("--out", default="models/adaptation/eval_results.json")
    parser.add_argument("--max-test-seqs", type=int, default=5)
    args = parser.parse_args()

    config = get_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load base model
    base_model = build_velocity_model(config)
    base_ckpt = torch.load(args.base_checkpoint, map_location=device)
    base_model.load_state_dict(base_ckpt["model_state_dict"])
    base_model.to(device).eval()

    # Load adaptation engine if checkpoint provided
    engine = None
    if args.adapt_checkpoint:
        base_model_adapt = build_velocity_model(config)
        engine = AdaptationEngine(base_model_adapt, get_config(args.config), device=device)
        engine.load_snapshot(Path(args.adapt_checkpoint))

    # Build test dataset
    _, _, test_ds = build_velocity_datasets(config, max_train_seqs=1, max_val_seqs=1)
    DataLoader(test_ds, batch_size=1, shuffle=False)

    logger.info("Evaluating base model...")
    base_results = evaluate_model_on_sequence(base_model, None, config, device)

    adapt_results = {}
    if engine is not None:
        logger.info("Evaluating adapted model...")
        adapt_results = evaluate_model_on_sequence(engine.lora_model, engine, config, device)

    # Compare
    improvement = {}
    if adapt_results:
        ate_improvement = (base_results["ate"] - adapt_results["ate"]) / base_results["ate"] * 100
        improvement["ATE_60s_gain_pct"] = ate_improvement

    results = {
        "base": base_results,
        "adapted": adapt_results,
        "improvement": improvement,
    }

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")

    logger.info(f"Base ATE: {base_results['ate']:.2f} m")
    if adapt_results:
        logger.info(f"Adapted ATE: {adapt_results['ate']:.2f} m")
        logger.info(f"Improvement: {improvement.get('ATE_60s_gain_pct', 0):.1f}%")


if __name__ == "__main__":
    main()