"""Training script for GNSS outage prediction model (Part 6)."""

import argparse
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from aeronavis.config import get_config, Config
from aeronavis.predict.model import build_predict_model
from aeronavis.data.preprocess import NavSequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class PredictDataset(Dataset):
    """Dataset for GNSS outage prediction training."""

    def __init__(
        self,
        sequences: List[NavSequence],
        config: Config,
        feature_extractor,
        labels_dir: str = "data/processed/labels/outage",
        max_samples_per_seq: int = 1000,
    ):
        self.config = config
        self.feature_extractor = feature_extractor
        self.seq_len = feature_extractor.config.sequence_length
        self.feature_dim = feature_extractor.config.feature_dim

        self.samples = []
        self._build_samples(sequences, labels_dir, max_samples_per_seq)

    def _build_samples(self, sequences, labels_dir, max_samples):
        Path(labels_dir)
        for seq in sequences:
            # Load outage labels if available
            label_file = Path("data/processed/labels/outage") / f"{seq.seq_id}_outage.csv"
            if not label_file.exists():
                continue
            labels_df = pd.read_csv(label_file)
            # labels_df has columns: window_idx, label (0=no outage, 1=outage)

            # Extract features for this sequence
            # This is a simplified version - real implementation would use the feature extractor
            n_windows = len(labels_df)
            indices = np.linspace(0, n_windows - 1, min(max_samples, n_windows), dtype=int)

            for w_idx in indices:
                w_idx * self.stride
                # For synthetic data, generate random features
                # Real implementation would use feature_extractor
                self.samples.append(
                    {
                        "features": np.random.randn(self.seq_len, 104).astype(np.float32),
                        "labels": np.array(
                            [labels_df.iloc[w_idx]["label"]] * len(self.horizons), dtype=np.float32
                        ),
                    }
                )

        logger.info(f"PredictDataset: {len(self.samples)} samples built")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "features": torch.from_numpy(s["features"]),  # (T, F)
            "labels": torch.from_numpy(s["labels"]),  # (H,)
        }


def train_predict(
    config_path: str,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 6,
    out_dir: str = "models/predict",
    seed: int = 42,
    max_train_seqs: int = 10,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    config = get_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build datasets
    from aeronavis.data.velocity_dataset import build_velocity_datasets

    train_ds, val_ds, _ = build_velocity_datasets(
        config, max_train_seqs=max_train_seqs, max_val_seqs=5
    )

    # For prediction, we need different labels - use velocity data for now
    # Real implementation would use outage labels
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build model
    model = build_predict_model(config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_loader), pct_start=0.1
    )
    nn.BCELoss()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path / "forecaster.pt"
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        start = time.time()

        for batch in train_loader:
            features = batch["features"].to(device)  # (B, T, F)
            labels = batch["labels"].to(device)  # (B, H)

            optimizer.zero_grad()
            preds = model(features)

            # Compute BCE loss per horizon
            loss = 0
            for i, h in enumerate(model.config.horizons):
                loss += nn.BCELoss()(preds[h], labels[:, i])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)

                preds = model(features)
                loss = 0
                for i, h in enumerate(model.config.horizons):
                    loss += nn.BCELoss()(preds[h], labels[:, i])
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        elapsed = time.time() - start

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={avg_train:.4f} val_loss={avg_val:.4f} time={elapsed:.1f}s"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                Path(out_dir) / "forecaster.pt",
            )
            logger.info(f"Saved best model to {out_dir}/forecaster.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return out_path / "forecaster.pt"


def main():
    parser = argparse.ArgumentParser(description="Train GNSS outage prediction model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--out-dir", default="models/predict")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-seqs", type=int, default=10)
    args = parser.parse_args()

    train_predict(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        out_dir=args.out_dir,
        seed=args.seed,
        max_train_seqs=args.max_train_seqs,
    )


if __name__ == "__main__":
    main()
