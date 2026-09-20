"""Synthetic VO data loader: generates homography-warped frame pairs with known 6-DoF.

Creates (frame_t-1, frame_t) pairs from a single image by applying random
homographies corresponding to small camera motions (2-5° rot, 0.2-3m trans).
"""

import logging
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class SyntheticVODataset(Dataset):
    """Synthetic VO dataset from single images via homography warps."""

    def __init__(
        self,
        image_dir: str,
        num_samples: int = 10000,
        img_size: tuple = (224, 224),
        # Motion ranges
        rot_deg_range: tuple = (2.0, 5.0),  # degrees
        trans_m_range: tuple = (0.2, 3.0),  # meters
        # Noise
        photometric_noise: float = 0.02,
        gaussian_blur_prob: float = 0.1,
    ):
        self.image_dir = Path(image_dir)
        self.num_samples = num_samples
        self.img_size = img_size
        self.rot_deg_range = rot_deg_range
        self.trans_m_range = trans_m_range
        self.photometric_noise = photometric_noise
        self.gaussian_blur_prob = gaussian_blur_prob

        # Find image files
        self.image_files = (
            list(self.image_dir.glob("*.jpg"))
            + list(self.image_dir.glob("*.png"))
            + list(self.image_dir.glob("*.jpeg"))
        )
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")

        # Camera intrinsics (normalized, for 224x224)
        self.fx = self.fy = 300.0
        self.cx = self.cy = 112.0
        self.K = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32
        )

        logger.info(
            f"SyntheticVODataset: {len(self.image_files)} source images, " f"{num_samples} samples"
        )

    def __len__(self) -> int:
        return self.num_samples

    def _random_se3(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random SE(3) transformation."""
        # Random rotation (axis-angle)
        rot_deg = random.uniform(*self.rot_deg_range)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        theta = np.deg2rad(rot_deg)
        R = cv2.Rodrigues(axis * theta)[0]

        # Random translation
        trans_m = random.uniform(*self.trans_m_range)
        t = np.random.randn(3)
        t = t / np.linalg.norm(t) * trans_m

        return R.astype(np.float32), t.astype(np.float32)

    def _se3_to_homography(
        self, R: np.ndarray, t: np.ndarray, plane_normal: np.ndarray = None, plane_dist: float = 1.0
    ) -> np.ndarray:
        """Convert SE(3) to homography assuming planar scene at distance d."""
        if plane_normal is None:
            plane_normal = np.array([0, 0, 1], dtype=np.float32)
        # H = K * (R - t * n^T / d) * K^-1
        K_inv = np.linalg.inv(self.K)
        H = self.K @ (R - np.outer(t, plane_normal) / plane_dist) @ K_inv
        return H

    def _apply_homography(self, img: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Warp image by homography."""
        h, w = self.img_size
        warped = cv2.warpPerspective(
            img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        return warped

    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        """Add photometric noise and optional blur."""
        # Gaussian noise
        noise = np.random.randn(*img.shape) * self.photometric_noise * 255
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Random Gaussian blur
        if random.random() < self.gaussian_blur_prob:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        return img

    def _se3_to_6dof(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Convert SE(3) to 6-DoF vector (tx,ty,tz, rx,ry,rz)."""
        # Translation
        trans = t

        # Rotation vector (axis-angle)
        rx, ry, rz = cv2.Rodrigues(R)[0].flatten()

        # Normalize translation by median scale (approx 1m)
        # This matches the normalized target convention
        scale = 1.0  # will be normalized by dataset stats
        trans_norm = trans / scale

        return np.array([trans_norm[0], trans_norm[1], trans_norm[2], rx, ry, rz], dtype=np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pick random source image
        img_path = random.choice(self.image_files)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback: synthetic texture
            img = np.random.randint(0, 255, self.img_size, dtype=np.uint8)

        img = cv2.resize(img, self.img_size)

        # Generate random motion
        R, t = self._random_se3()
        H = self._se3_to_homography(R, t)

        # Warp to get frame_t
        img_t = self._apply_homography(img, H)

        # Add noise to both frames
        img_tm1 = self._add_noise(img)
        img_t = self._add_noise(img_t)

        # Stack as 2-channel: (t-1, t)
        pair = np.stack([img_tm1, img_t], axis=0).astype(np.float32) / 255.0

        # Ground truth 6-DoF
        target = self._se3_to_6dof(R, t)

        return (
            torch.from_numpy(pair),  # (2, 224, 224)
            torch.from_numpy(target),  # (6,)
        )


class RealVODataset(Dataset):
    """Placeholder for real VO dataset (EuRoC/KITTI)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Real VO dataset not implemented. Use SyntheticVODataset.")


def build_vo_dataloader(
    image_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    num_samples: int = 10000,
    **kwargs,
) -> DataLoader:
    """Build VO training dataloader (synthetic by default)."""
    dataset = SyntheticVODataset(
        image_dir=image_dir,
        num_samples=num_samples,
        **kwargs,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )


if __name__ == "__main__":
    # Quick test with synthetic texture
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy image
        dummy = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        cv2.imwrite(os.path.join(tmpdir, "test.jpg"), dummy)

        dataset = SyntheticVODataset(tmpdir, num_samples=10)
        pair, target = dataset[0]
        print(f"Pair shape: {pair.shape}")  # (2, 224, 224)
        print(f"Target shape: {target.shape}")  # (6,)
        print(f"Target: {target}")

        loader = build_vo_dataloader(tmpdir, batch_size=4, num_samples=10)
        for batch in loader:
            print(f"Batch pair: {batch[0].shape}, target: {batch[1].shape}")
            break
