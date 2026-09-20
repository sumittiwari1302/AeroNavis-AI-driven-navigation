"""Training script for texture gate (3-class CNN)."""

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from aeronavis.config import get_config
from aeronavis.models.texture_gate import build_texture_gate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class TextureDataset(Dataset):
    """Texture gate dataset: 96x96 grayscale -> 3-class label."""

    def __init__(
        self,
        image_dir: str,
        num_samples: int = 50000,
        img_size: int = 96,
        # Class balance
        class_ratios: tuple = (0.4, 0.3, 0.3),  # rich, medium, poor
    ):
        self.image_dir = Path(image_dir)
        self.num_samples = num_samples
        self.img_size = img_size
        self.class_ratios = class_ratios

        # Find images
        self.image_files = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        if not self.image_files:
            raise ValueError(f"No images in {image_dir}")

        # Precompute samples
        self.samples = []
        self._build_samples()

    def _build_samples(self):
        n_rich = int(self.num_samples * self.class_ratios[0])
        n_medium = int(self.num_samples * self.class_ratios[1])
        n_poor = self.num_samples - n_rich - n_medium

        # Rich: natural images (sharp, textured)
        for _ in range(n_rich):
            img_path = random.choice(self.image_files)
            self.samples.append((img_path, 0))

        # Medium: slight blur / lower contrast
        for _ in range(n_medium):
            img_path = random.choice(self.image_files)
            self.samples.append((img_path, 1))

        # Poor: dark + heavy blur / uniform
        for _ in range(n_poor):
            img_path = random.choice(self.image_files)
            self.samples.append((img_path, 2))

        random.shuffle(self.samples)
        logger.info(f"TextureDataset: {len(self.samples)} samples built")

    def _load_and_augment(self, img_path: Path, label: int) -> torch.Tensor:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.random.randint(0, 255, (96, 96), dtype=np.uint8)
        img = cv2.resize(img, (96, 96))

        if label == 1:  # medium: slight blur
            img = cv2.GaussianBlur(img, (3, 3), 0.5)
        elif label == 2:  # poor: dark + heavy blur
            img = cv2.convertScaleAbs(img, alpha=0.3, beta=0)  # darken
            img = cv2.GaussianBlur(img, (7, 7), 2.0)

        # Normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0)  # (1, 96, 96)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        img = self._load_and_augment(img_path, label)
        return img, torch.tensor(label, dtype=torch.long)


def train_texture_gate(
    config_path: str,
    image_dir: str,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
    out_dir: str = "models/texture_gate",
    seed: int = 42,
    num_samples: int = 50000,
):
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    config = get_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dataset
    train_ds = TextureDataset(image_dir, num_samples=int(num_samples * 0.8))
    val_ds = TextureDataset(image_dir, num_samples=int(num_samples * 0.2))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = build_texture_gate(config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_loader), pct_start=0.1
    )
    criterion = nn.CrossEntropyLoss()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_path = out_path / "texture_gate.pt"
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        start = time.time()

        for batch in train_loader:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            pred = logits.argmax(dim=1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)

        # Validation
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)

                logits = model(imgs)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())

                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        elapsed = time.time() - start

        # Per-class
        from sklearn.metrics import classification_report

        report = classification_report(
            all_labels,
            all_preds,
            target_names=["rich", "medium", "poor"],
            output_dict=True,
            zero_division=0,
        )

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
            f"rich_f1={report['0']['f1-score']:.3f} "
            f"med_f1={report['1']['f1-score']:.3f} "
            f"poor_f1={report['2']['f1-score']:.3f} "
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
                Path(out_dir) / "texture_gate.pt",
            )
            logger.info(f"Saved best model to {out_dir}/texture_gate.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"Training complete. Best val acc: {best_val_acc:.4f}")
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train texture gate")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--image-dir", required=True, help="Directory with training images")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--out-dir", default="models/texture_gate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=50000)
    args = parser.parse_args()

    train_texture_gate(
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
