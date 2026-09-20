"""Training script for visual odometry (UL-VIO style)."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from aeronavis.config import get_config
from aeronavis.data.vo_dataset import build_vo_dataloader
from aeronavis.models.visual_odo import build_visual_odo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def rotation_loss(R_pred: torch.Tensor, R_target: torch.Tensor) -> torch.Tensor:
    """
    Geodesic loss on SO(3): ||log(R_pred^T @ R_target)||^2
    Using 6-DoF representation: first 3 are translation, last 3 are rotation vector.
    """
    # Extract rotation vectors
    r_pred = R_pred[:, 3:]  # (B, 3)
    r_target = R_target[:, 3:]

    # Convert to rotation matrices
    R_pred_mat = rodrigues_batch(r_pred)
    R_target_mat = rodrigues_batch(r_target)

    # Relative rotation
    R_rel = torch.bmm(R_pred_mat.transpose(1, 2), R_target_mat)

    # Log map: angle = acos((trace(R) - 1) / 2)
    trace = R_rel.diagonal(offset=0, dim1=1, dim2=2).sum(dim=1)
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)

    return (angle**2).mean()


def rodrigues_batch(r_vec: torch.Tensor) -> torch.Tensor:
    """Batch Rodrigues: (B, 3) -> (B, 3, 3) rotation matrices."""
    theta = torch.norm(r_vec, dim=1, keepdim=True)  # (B, 1)
    r_hat = r_vec / (theta + 1e-8)  # (B, 3)

    # Skew-symmetric matrix
    K = torch.zeros(r_vec.shape[0], 3, 3, device=r_vec.device, dtype=r_vec.dtype)
    K[:, 0, 1] = -r_hat[:, 2]
    K[:, 0, 2] = r_hat[:, 1]
    K[:, 1, 0] = r_hat[:, 2]
    K[:, 1, 2] = -r_hat[:, 0]
    K[:, 2, 0] = -r_hat[:, 1]
    K[:, 2, 1] = r_hat[:, 0]

    eye_matrix = torch.eye(3, device=r_vec.device, dtype=r_vec.dtype).unsqueeze(0)
    R = (
        eye_matrix
        + torch.sin(theta).unsqueeze(-1) * K
        + (1 - torch.cos(theta)).unsqueeze(-1) * torch.bmm(K, K)
    )
    return R


def train_visual_odo(
    config_path: str,
    image_dir: str,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 8,
    out_dir: str = "models/visual_odo",
    seed: int = 42,
    num_samples: int = 20000,
):
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    config = get_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dataloader (synthetic)
    train_loader = build_vo_dataloader(
        image_dir=image_dir,
        batch_size=batch_size,
        num_workers=0,
        num_samples=int(num_samples * 0.8),
    )
    val_loader = build_vo_dataloader(
        image_dir=image_dir,
        batch_size=batch_size,
        num_workers=0,
        num_samples=int(num_samples * 0.2),
    )

    # Model
    model = build_visual_odo(config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_loader), pct_start=0.1
    )
    mse_loss = nn.MSELoss()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_path = out_path / "visual_odo.pt"
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        start = time.time()

        for batch in train_loader:
            pairs, targets = batch
            pairs = pairs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(pairs)

            # Loss: translation MSE + 0.3 * rotation geodesic
            trans_loss = mse_loss(preds[:, :3], targets[:, :3])
            rot_loss = rotation_loss(preds, targets)
            loss = trans_loss + 0.3 * rot_loss

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
                pairs, targets = batch
                pairs = pairs.to(device)
                targets = targets.to(device)

                preds = model(pairs)
                trans_loss = mse_loss(preds[:, :3], targets[:, :3])
                rot_loss = rotation_loss(preds, targets)
                loss = trans_loss + 0.3 * rot_loss
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        elapsed = time.time() - start

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={avg_train:.4f} val_loss={avg_val:.4f} "
            f"time={elapsed:.1f}s"
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
                Path(out_dir) / "visual_odo.pt",
            )
            logger.info(f"Saved best model to {out_dir}/visual_odo.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train visual odometry")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--image-dir", required=True, help="Directory with source images")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--out-dir", default="models/visual_odo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=20000)
    args = parser.parse_args()

    train_visual_odo(
        config_path=args.config,
        image_dir=args.image_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        out_dir=args.out_dir,
        seed=args.seed,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
