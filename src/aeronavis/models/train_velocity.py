"""Training harness for velocity model.

Usage:
    python -m aeronavis.models.train_velocity --config config.yaml --epochs 60
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from aeronavis.config import get_config
from aeronavis.models.velocity import VelocityModel, build_velocity_model
from aeronavis.data.velocity_dataset import velocity_collate, build_velocity_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class VelocityTrainer:
    def __init__(
        self,
        model: VelocityModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config["lr"],
            total_steps=config["epochs"] * len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
        )
        self.criterion_mse = nn.MSELoss()
        self.epoch = 0
        self.best_val_ate = float("inf")
        self.patience_counter = 0

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        total_vel_loss = 0.0
        total_disp_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            acc = batch["acc"].to(self.device)  # (B, W, 3)
            gyr = batch["gyr"].to(self.device)
            att0 = batch["att0"].to(self.device)
            vel_target = batch["vel_target"].to(self.device)
            disp_target = batch["disp_target"].to(self.device)

            self.optimizer.zero_grad()
            vel_pred = self.model(acc, gyr, att0).squeeze(-1)  # (B,)

            # Velocity MSE
            vel_loss = self.criterion_mse(vel_pred, vel_target)

            # Displacement consistency: integrate velocity over window
            # Simple trapezoidal: disp = mean(vel) * window_duration
            window_duration = self.model.config.window / 100.0  # seconds
            disp_pred = vel_pred * window_duration
            disp_loss = self.criterion_mse(disp_pred, disp_target.to(self.device))

            loss = vel_loss + 0.5 * disp_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_vel_loss += vel_loss.item()
            total_disp_loss += disp_loss.item()
            n_batches += 1

        return {
            "loss": total_loss / n_batches,
            "vel_loss": total_vel_loss / n_batches,
            "disp_loss": total_disp_loss / n_batches,
        }

    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_vel_loss = 0.0
        total_disp_loss = 0.0
        n_batches = 0

        all_vel_pred = []
        all_vel_target = []
        all_disp_pred = []
        all_disp_target = []

        with torch.no_grad():
            for batch in self.val_loader:
                acc = batch["acc"].to(self.device)
                gyr = batch["gyr"].to(self.device)
                att0 = batch["att0"].to(self.device)
                vel_target = batch["vel_target"].to(self.device)
                disp_target = batch["disp_target"].to(self.device)

                vel_pred = self.model(acc, gyr, att0).squeeze(-1)

                vel_loss = self.criterion_mse(vel_pred, vel_target)
                window_duration = self.model.config.window / 100.0
                disp_pred = vel_pred * window_duration
                disp_loss = self.criterion_mse(disp_pred, disp_target)

                loss = vel_loss + 0.5 * disp_loss

                total_loss += loss.item()
                total_vel_loss += vel_loss.item()
                total_disp_loss += disp_loss.item()
                n_batches += 1

                all_vel_pred.append(vel_pred.cpu())
                all_vel_target.append(vel_target.cpu())
                all_disp_pred.append(disp_pred.cpu())
                all_disp_target.append(disp_target.cpu())

        # Compute ATE proxy: RMSE of velocity
        vel_pred_cat = torch.cat(all_vel_pred)
        vel_target_cat = torch.cat(all_vel_target)
        vel_rmse = torch.sqrt(torch.mean((vel_pred_cat - vel_target_cat) ** 2)).item()

        return {
            "loss": total_loss / n_batches,
            "vel_loss": total_vel_loss / n_batches,
            "disp_loss": total_disp_loss / n_batches,
            "vel_rmse": vel_rmse,
        }

    def save_checkpoint(self, path: Path, metric: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_ate": self.best_val_ate,
                "config": self.config,
                "model_config": self.model.config.__dict__,
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path} (metric: {metric:.4f})")

    def load_checkpoint(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.epoch = ckpt["epoch"]
        self.best_val_ate = ckpt["best_val_ate"]
        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")


def train_velocity(
    config_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    out_dir: str,
    seed: int,
):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    config = get_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build datasets
    train_ds, val_ds, test_ds = build_velocity_datasets(config)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=velocity_collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=velocity_collate, num_workers=0
    )

    # Build model
    model = build_velocity_model(config)

    trainer_config = {
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
    }
    trainer = VelocityTrainer(model, train_loader, val_loader, trainer_config, device)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_path = out_path / "best.pt"

    for epoch in range(epochs):
        trainer.epoch = epoch
        start = time.time()

        train_metrics = trainer.train_epoch()
        val_metrics = trainer.validate()

        elapsed = time.time() - start
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"vel_rmse={val_metrics['vel_rmse']:.4f} "
            f"time={elapsed:.1f}s"
        )

        # Early stopping on val vel_rmse (proxy for ATE)
        if val_metrics["vel_rmse"] < trainer.best_val_ate:
            trainer.best_val_ate = val_metrics["vel_rmse"]
            trainer.patience_counter = 0
            trainer.save_checkpoint(best_path, trainer.best_val_ate)
        else:
            trainer.patience_counter += 1
            if trainer.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"Training complete. Best val vel_rmse: {trainer.best_val_ate:.4f}")
    return best_path, trainer.best_val_ate


def main():
    parser = argparse.ArgumentParser(description="Train velocity model")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=60, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--out-dir", default="models/velocity", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    best_path, best_metric = train_velocity(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    logger.info(f"Best model saved to {best_path} with metric {best_metric:.4f}")


if __name__ == "__main__":
    main()
