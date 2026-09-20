"""Training script for pseudo wheel odometry model."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from aeronavis.config import get_config
from aeronavis.data.preprocess import NavSequence
from aeronavis.models.pseudo_odo import build_pseudo_odo_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class PseudoOdoDataset(Dataset):
    """Dataset for pseudo wheel odometry training."""

    def __init__(
        self,
        sequences: list[NavSequence],
        config,
        labels_dir: str = "data/processed/labels/slip",
        max_windows_per_seq: int = 2000,
    ):
        self.config = config
        self.dt = 1.0 / config.data.imu_hz
        self.window = config.model.window
        self.stride = config.model.stride

        self.samples = []
        self._build_samples(sequences, labels_dir, max_windows_per_seq)

    def _build_samples(self, sequences, labels_dir, max_windows):
        labels_path = Path(labels_dir)
        for seq in sequences:
            # Load labels
            label_file = labels_path / f"{seq.seq_id}_slip.csv"
            if not label_file.exists():
                continue
            labels_df = pd.read_csv(label_file)
            labels = labels_df["label"].values

            # IMU data
            imu = seq.imu[["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]].values.astype(
                np.float32
            )
            if len(imu) < self.window:
                continue

            # Wheel speed targets
            if seq.wheel is None or len(seq.wheel) == 0:
                continue

            # Barometer delta
            baro_delta = 0.0
            if seq.baro is not None and len(seq.baro) > 1:
                baro_delta = float(seq.baro["pressure_hpa"].diff().mean())

            # Build windows
            n_windows = (len(imu) - self.window) // self.stride + 1
            indices = np.linspace(0, n_windows - 1, min(max_windows, n_windows), dtype=int)

            for w_idx in indices:
                i = w_idx * self.stride
                center_idx = i + self.window // 2

                # Wheel speed target at window center
                if len(seq.wheel["ts"]) > 1:
                    target_speed = np.interp(
                        center_idx * self.dt,
                        seq.wheel["ts"].values,
                        seq.wheel["wheel_speed_mps"].values,
                    )
                else:
                    target_speed = seq.wheel["wheel_speed_mps"].values[0]

                # Label
                label = labels[w_idx] if w_idx < len(labels) else 0

                self.samples.append(
                    {
                        "acc": imu[i : i + self.window, :3],
                        "gyr": imu[i : i + self.window, 3:],
                        "baro_delta": np.array([baro_delta], dtype=np.float32),
                        "target_speed": np.array([target_speed], dtype=np.float32),
                        "label": label,
                    }
                )

        logger.info(f"Built {len(self.samples)} training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "acc": torch.from_numpy(s["acc"]),
            "gyr": torch.from_numpy(s["gyr"]),
            "baro_delta": torch.from_numpy(s["baro_delta"]),
            "target_speed": torch.from_numpy(s["target_speed"]),
            "label": torch.tensor(s["label"], dtype=torch.long),
        }


def collate_fn(batch):
    return {
        "acc": torch.stack([b["acc"] for b in batch]),
        "gyr": torch.stack([b["gyr"] for b in batch]),
        "baro_delta": torch.stack([b["baro_delta"] for b in batch]),
        "target_speed": torch.stack([b["target_speed"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


def train_pseudo_odo(
    config_path: str,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
    out_dir: str = "models/pseudo_odo",
    seed: int = 42,
    max_train_seqs: int = 20,
):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    config = get_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load sequences
    from aeronavis.data.preprocess import NavSequence

    cache_root = Path(config.paths.processed) / "io_vnbd"
    all_seqs = NavSequence.load_all(cache_root)

    # Split
    from aeronavis.data.splits import make_splits

    buckets = make_splits(all_seqs, config.splits)
    train_seqs = buckets["train"][:max_train_seqs]
    val_seqs = buckets["val"][:5]

    # Build datasets
    train_ds = PseudoOdoDataset(train_seqs, config)
    val_ds = PseudoOdoDataset(val_seqs, config)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = build_pseudo_odo_model(config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_loader), pct_start=0.1
    )
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_path = out_path / "best.pt"
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        start = time.time()

        for batch in train_loader:
            acc = batch["acc"].to(device)
            gyr = batch["gyr"].to(device)
            baro = batch["baro_delta"].to(device)
            target_speed = batch["target_speed"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()
            pred_speed, pred_rel = model(acc, gyr, baro)

            speed_loss = mse_loss(pred_speed, target_speed.squeeze())
            rel_target = (label == 0).float()  # reliability = 1 for grip
            rel_loss = bce_loss(pred_rel, rel_target)

            loss = speed_loss + 0.5 * rel_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                acc = batch["acc"].to(device)
                gyr = batch["gyr"].to(device)
                baro = batch["baro_delta"].to(device)
                target_speed = batch["target_speed"].to(device)
                label = batch["label"].to(device)

                pred_speed, pred_rel = model(acc, gyr, baro)
                speed_loss = mse_loss(pred_speed, target_speed.squeeze())
                rel_target = (label == 0).float()
                rel_loss = bce_loss(pred_rel, rel_target)
                loss = speed_loss + 0.5 * rel_loss
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
                best_path,
            )
            logger.info(f"Saved best model to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train pseudo wheel odometry")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--out-dir", default="models/pseudo_odo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-seqs", type=int, default=20)
    args = parser.parse_args()

    train_pseudo_odo(
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
