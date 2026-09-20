"""Evaluation script for GNSS outage prediction model."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from aeronavis.config import get_config
from aeronavis.models.predict.model import build_predict_model
from aeronavis.data.velocity_dataset import build_velocity_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_auc_roc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUC-ROC for binary classification."""
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return 0.0


def compute_ppv(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Positive Predictive Value at threshold."""
    pred = (y_pred >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (pred == 1))
    fp = np.sum((y_true == 0) & (pred == 1))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Recall at threshold."""
    pred = (y_pred >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (pred == 1))
    fn = np.sum((y_true == 1) & (pred == 0))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def mean_lead_time(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, dt: float = 1.0
) -> float:
    """Mean lead time in seconds."""
    (y_pred >= threshold).astype(int)
    # Find first positive prediction before each positive ground truth
    leads = []
    n = len(y_true)
    for i in range(n):
        if y_true[i] == 1:
            # Look backward for first positive prediction
            for j in range(i, max(-1, i - 30), -1):  # max 30s lookback
                if y_pred[j] >= 0.5:
                    leads.append((i - j) * dt)
                    break
    return float(np.mean(leads)) if leads else 0.0


def compute_fp_cost(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, cost_per_fp: float = 1.0
) -> float:
    """False positive cost in ATE meters."""
    pred = (y_pred >= threshold).astype(int)
    fp = np.sum((y_true == 0) & (pred == 1))
    return float(fp * cost_per_fp)


def evaluate_predict_model(
    model,
    test_loader,
    device: torch.device,
    horizons: List[int] = None,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate prediction model on test set."""
    if horizons is None:
        horizons = [5, 10, 15]

    model.eval()
    all_preds = {h: [] for h in horizons}
    all_labels = {h: [] for h in horizons}

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            preds = model(features)

            for i, h in enumerate(model.config.horizons):
                all_preds[h].extend(preds[h].cpu().numpy())
                all_labels[h].extend(labels[:, i].cpu().numpy())

    results = {}
    for h in horizons:
        np.array(all_labels[h])
        np.array(all_preds[h])

        auc = compute_auc_roc(all_labels[h], all_preds[h])
        ppv = compute_ppv(all_labels[h], all_preds[h])
        recall = compute_recall(all_labels[h], all_preds[h])
        lead = mean_lead_time(all_labels[h], all_preds[h])

        results[h] = {
            "auc": float(auc),
            "ppv": float(ppv),
            "recall": float(recall),
            "mean_lead_time": float(lead),
            "n_positive": int(np.sum(all_labels[h])),
            "n_samples": len(all_labels[h]),
        }

    return results


def compute_lead_time_on_outages(
    model,
    test_loader,
    device: torch.device,
    horizons: List[int] = None,
) -> Dict[int, float]:
    """Compute mean lead time only on outage intervals."""
    if horizons is None:
        horizons = [5, 10, 15]

    model.eval()
    results = {}

    with torch.no_grad():
        for h in horizons:
            leads = []
            for batch in test_loader:
                features = batch["features"].to(device)
                batch["labels"].to(device)

                preds = model(features)
                pred_h = preds[h].cpu().numpy()
                label_h = batch["labels"][:, list(model.config.horizons).index(h)].cpu().numpy()

                # Find outage onsets
                diff = np.diff(label_h)
                onsets = np.where(diff == 1)[0] + 1

                for onset in onsets:
                    # Look back from onset for first positive prediction
                    for j in range(onset - 1, max(-1, onset - 30), -1):
                        if pred_h[j] >= 0.5:
                            leads.append((onset - j) * 1.0)  # 1 second dt
                            break

            if leads:
                results[h] = float(np.mean(leads))
            else:
                results[h] = 0.0

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate GNSS outage prediction model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--out", default="models/predict/eval_results.json")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    config = get_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    model = build_predict_model(config)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # Build test dataset
    _, _, test_ds = build_velocity_datasets(config, max_train_seqs=1, max_val_seqs=1)
    DataLoader(test_ds, batch_size=32, shuffle=False)

    # Evaluate
    logger.info("Evaluating prediction model...")
    results = evaluate_predict_model(
        model, DataLoader(test_ds, batch_size=32, shuffle=False), device
    )

    # Lead time analysis
    lead_times = compute_lead_time_on_outages(
        model, DataLoader(test_ds, batch_size=32, shuffle=False), device
    )

    # Combine results
    results["lead_time"] = lead_times

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {out_path}")
    for h in [5, 10, 15]:
        if h in results:
            r = results[h]
            logger.info(
                f"Horizon {h}s: AUC={r['auc']:.3f}, PPV={r['ppv']:.3f}, "
                f"Recall={r['recall']:.3f}, Lead={lead_times.get(h, 0):.1f}s"
            )


if __name__ == "__main__":
    main()
