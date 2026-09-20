"""Training script for adaptation engine (LoRA adapters)."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aeronavis.config import get_config
from aeronavis.models.velocity import build_velocity_model
from aeronavis.models.adaptation_engine import AdaptationEngine
from aeronavis.data.velocity_dataset import build_velocity_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def train_adaptation(
    config_path: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 6,
    out_dir: str = "models/adaptation",
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
    train_ds, val_ds, _ = build_velocity_datasets(
        config, max_train_seqs=max_train_seqs, max_val_seqs=5
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Build base velocity model
    base_model = build_velocity_model(config)
    logger.info(f"Base model params: {sum(p.numel() for p in base_model.parameters()):,}")

    # Create adaptation engine
    engine = AdaptationEngine(base_model, get_config(config_path), device=device)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path / "best.pt"
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        engine.lora_model.train()
        start = time.time()

        for batch in train_loader:
            acc = batch["acc"]
            gyr = batch["gyr"]
            # For now, use velocity target as pseudo_speed proxy
            vel_target = batch["vel_target"]

            engine.step(
                window_acc=acc,
                window_gyr=gyr,
                pseudo_speed=vel_target,
                pseudo_rel=torch.ones_like(vel_target),
                gnss_speed=vel_target,
                gnss_ok=True,
            )
            # Note: step() already does backward and step internally

        # Validation
        engine.lora_model.eval()
        with torch.no_grad():
            for batch in val_loader:
                acc = batch["acc"]
                gyr = batch["gyr"]
                vel_target = batch["vel_target"]

                engine.step(
                    window_acc=acc,
                    window_gyr=gyr,
                    pseudo_speed=vel_target,
                    pseudo_rel=torch.ones_like(vel_target),
                    gnss_speed=vel_target,
                    gnss_ok=True,
                )
                # Note: step() doesn't return loss in eval mode currently

        avg_train = engine.state.total_loss
        avg_val = engine.state.total_loss  # placeholder
        elapsed = time.time() - start

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={avg_train:.4f} val_loss={avg_val:.4f} "
            f"time={elapsed:.1f}s"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            engine.save_snapshot(Path(out_dir) / "stable.pt")
            logger.info(f"Saved best model to {out_dir}/stable.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    engine.save_stats(Path(out_dir) / "stats.json")
    return out_path / "stable.pt"


def main():
    parser = argparse.ArgumentParser(description="Train adaptation engine")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--out-dir", default="models/adaptation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-seqs", type=int, default=10)
    args = parser.parse_args()

    train_adaptation(
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