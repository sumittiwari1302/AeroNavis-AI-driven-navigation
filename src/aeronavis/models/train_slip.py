"""Training script for slip/grip/stationary classifier."""

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
from aeronavis.models.slip import build_slip_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class SlipDataset(Dataset):
    """Dataset for slip classifier training."""

    def __init__(
        self,
        sequences: list,
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
            label_file = labels_path / f"{seq.seq_id}_slip.csv"
            if not label_file.exists():
                continue
            labels_df = pd.read_csv(label_file)
            labels = labels_df["label"].values

            imu = seq.imu[["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]].values.astype(
                np.float32
            )
            if len(imu) < self.window:
                continue

            baro_delta = 0.0
            if seq.baro is not None and len(seq.baro) > 1:
                baro_delta = float(seq.baro["pressure_hpa"].diff().mean())

            n_windows = (len(imu) - self.window) // self.stride + 1
            indices = np.linspace(0, n_windows - 1, min(max_windows, n_windows), dtype=int)

            for w_idx in indices:
                i = w_idx * self.stride
                if w_idx >= len(labels):
                    continue
                self.samples.append(
                    {
                        "acc": imu[i : i + self.window, :3],
                        "gyr": imu[i : i + self.window, 3:],
                        "baro_delta": np.array([baro_delta], dtype=np.float32),
                        "label": np.array([labels[w_idx]], dtype=np.int64),
                    }
                )

        logger.info(f"Built {len(self.samples)} slip training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "acc": torch.from_numpy(s["acc"]),
            "gyr": torch.from_numpy(s["gyr"]),
            "baro_delta": torch.from_numpy(s["baro_delta"]),
            "label": torch.from_numpy(s["label"]),
        }


def collate_fn(batch):
    return {
        "acc": torch.stack([b["acc"] for b in batch]),
        "gyr": torch.stack([b["gyr"] for b in batch]),
        "baro_delta": torch.stack([b["baro_delta"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


def train_slip(
    config_path: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
    out_dir: str = "models/slip",
    seed: int = 42,
    max_train_seqs: int = 20,
    share_trunk: bool = True,
):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    config = get_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load sequences
    cache_root = Path(config.paths.processed) / "io_vnbd"
    all_seqs = NavSequence.load_all(cache_root)

    # Split
    from aeronavis.data.splits import make_splits

    buckets = make_splits(all_seqs, config.splits)
    train_seqs = buckets["train"][:max_train_seqs]
    val_seqs = buckets["val"][:5]

    # Build datasets
    train_ds = SlipDataset(train_seqs, config)
    val_ds = SlipDataset(val_seqs, config)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Build shared trunk if requested
    shared_trunk = None
    if share_trunk:
        from aeronavis.models.pseudo_odo import get_pseudo_odo_config

        get_pseudo_odo_config(config)
        pseudo_model = build_pseudo_odo_model(config)
        # Extract trunk modules manually
        shared_trunk = {
            "cnn": pseudo_model.cnn,
            "gru": pseudo_model.gru,
            "spec_extractor": pseudo_model.spec_extractor,
            "prenet": pseudo_model.prenet,
        }

    model = build_slip_model(config, shared_trunk=shared_trunk).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_loader), pct_start=0.1
    )
    ce_loss = nn.CrossEntropyLoss()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        start = time.time()

        for batch in train_loader:
            acc = batch["acc"].to(device)
            gyr = batch["gyr"].to(device)
            baro = batch["baro_delta"].to(device)
            label = batch["label"].to(device).squeeze()

            optimizer.zero_grad()
            logits = model(acc, gyr, baro)
            loss = ce_loss(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            pred = logits.argmax(dim=1)
            train_correct += (pred == label).sum().item()
            train_total += label.size(0)

        # Validation
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                acc = batch["acc"].to(device)
                gyr = batch["gyr"].to(device)
                baro = batch["baro_delta"].to(device)
                label = batch["label"].to(device).squeeze()

                logits = model(acc, gyr, baro)
                loss = ce_loss(logits, label)
                val_losses.append(loss.item())

                pred = logits.argmax(dim=1)
                val_correct += (pred == label).sum().item()
                val_total += label.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        elapsed = time.time() - start

        # Per-class accuracy
        from sklearn.metrics import classification_report

        report = classification_report(
            all_labels,
            all_preds,
            target_names=["grip", "slip", "stationary"],
            output_dict=True,
            zero_division=0,
        )

        slip_f1 = report.get("1", {}).get("f1-score", 0.0)
        stat_f1 = report.get("2", {}).get("f1-score", 0.0)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
            f"slip_f1={slip_f1:.3f} stat_f1={stat_f1:.3f} "
            f"time={elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                },
                out_path / "best.pt",
            )
            logger.info(f"Saved best model to {out_path / 'best.pt'}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"Training complete. Best val acc: {best_val_acc:.4f}")
    return out_path / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="Train slip classifier")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--out-dir", default="models/slip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-seqs", type=int, default=20)
    parser.add_argument("--share-trunk", action="store_true", default=True)
    args = parser.parse_args()

    train_slip(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        out_dir=args.out_dir,
        seed=args.seed,
        max_train_seqs=args.max_train_seqs,
        share_trunk=args.share_trunk,
    )


if __name__ == "__main__":
    main()
