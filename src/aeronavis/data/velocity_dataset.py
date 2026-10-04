"""Velocity dataset: windowed IMU samples with ground-truth velocity labels.

Builds samples from NavSequence training splits (vehicle sources only).
Each sample is a 200-sample window @ 100 Hz with ground-truth forward velocity
at window center and integrated displacement over the window.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from aeronavis.config import Config
from aeronavis.data.preprocess import NavSequence

logger = logging.getLogger(__name__)


@dataclass
class VelocitySample:
    acc: torch.Tensor  # (W, 3)
    gyr: torch.Tensor  # (W, 3)
    att0: torch.Tensor  # (3, 3) rotation matrix device->canonical
    vel_target: torch.Tensor  # scalar forward velocity (m/s)
    disp_target: torch.Tensor  # scalar displacement over window (m)
    mask: torch.Tensor  # (W,) valid mask


def estimate_gravity_direction(acc: np.ndarray, window: int = 50) -> np.ndarray:
    """Estimate gravity direction from stationary start of sequence."""
    # Use first `window` samples where norm is closest to 9.81
    norms = np.linalg.norm(acc[:window], axis=1)
    stationary_idx = np.argmin(np.abs(norms - 9.80665))
    gravity_vec = acc[stationary_idx]
    if np.linalg.norm(gravity_vec) < 1e-6:
        return np.array([0.0, 0.0, -1.0])  # fallback
    return gravity_vec / np.linalg.norm(gravity_vec)


def rotation_matrix_from_gravity(gravity_device: np.ndarray) -> np.ndarray:
    """Build rotation matrix from device frame to gravity-aligned frame.

    Canonical frame: z-axis aligned with gravity (pointing down),
    x-axis = forward (projected heading), y-axis = right.
    """
    g = gravity_device / (np.linalg.norm(gravity_device) + 1e-8)
    # z_canonical = -g (gravity points down in canonical)
    z_canon = -g
    # Choose arbitrary x in horizontal plane, orthogonal to z
    if abs(z_canon[0]) < 0.9:
        x_ref = np.array([1.0, 0.0, 0.0])
    else:
        x_ref = np.array([0.0, 1.0, 0.0])
    x_canon = x_ref - np.dot(x_ref, z_canon) * z_canon
    x_canon = x_canon / (np.linalg.norm(x_canon) + 1e-8)
    y_canon = np.cross(z_canon, x_canon)
    # R_canonical_from_device: maps device -> canonical
    # Columns are canonical axes expressed in device frame
    R = np.column_stack([x_canon, y_canon, z_canon])
    return R.astype(np.float32)


def compute_att0_from_sequence(seq: NavSequence) -> np.ndarray:
    """Compute initial attitude matrix for a sequence."""
    # Use first 2s of accel to estimate gravity direction
    imu = seq.imu
    if len(imu) < 50:
        gravity_dir = np.array([0.0, 0.0, -1.0])
    else:
        acc_vals = imu[["acc_x", "acc_y", "acc_z"]].values[:200]
        gravity_dir = estimate_gravity_direction(acc_vals)
    return rotation_matrix_from_gravity(gravity_dir)


def forward_velocity_from_truth(
    truth: np.ndarray, center_idx: int, window: int, dt: float
) -> tuple[float, float]:
    """Mean ground speed and along-path displacement over a window.

    truth: (N, 4) columns [x, y, z, heading] at 10 Hz
    Returns: (speed_mps, disp_m)

    The label is the *magnitude* of the trajectory step, not the projection onto
    ``heading``. Projecting onto the stored heading silently corrupts the label
    because each source's heading lives in a different frame than its position:
    measured retention of forward/speed was 0.19 (io_vnbd), 0.67 (ronin) and
    0.97 (idol), with 36% of RoNIN windows landing negative because a carried
    Tango's yaw does not track the direction the person walks.

    Ground speed is frame-independent, non-negative, and is what a GNSS supplies,
    so it is the quantity an IMU-only estimator can actually be regressed onto.
    Heading stays a separate regressed/filtered output.
    """
    half = window // 2
    start = max(0, center_idx - half)
    end = min(len(truth), center_idx + half + 1)

    if end - start < 2:
        return 0.0, 0.0

    pos = truth[start:end, :2]  # (x, y)

    # Per-step displacements; speed is the magnitude, so it stays >= 0.
    dpos = np.diff(pos, axis=0)
    step_disp = np.hypot(dpos[:, 0], dpos[:, 1])
    total_disp = float(np.sum(step_disp))

    n_steps = end - start - 1
    mean_vel = total_disp / (n_steps * dt) if n_steps > 0 else 0.0

    return float(mean_vel), total_disp


class VelocityDataset(Dataset):
    """Dataset of IMU windows with velocity labels."""

    def __init__(
        self,
        sequences: list[NavSequence],
        config: Config,
        is_train: bool = True,
    ):
        self.config = config
        self.window = config.model.window
        self.stride = config.model.stride
        self.dt = 1.0 / config.data.imu_hz
        self.is_train = is_train

        self.samples: list[VelocitySample] = []
        self._build_samples(sequences)

        logger.info(
            f"VelocityDataset ({'train' if is_train else 'val/test'}): "
            f"{len(self.samples)} samples from {len(sequences)} sequences"
        )

    def _build_samples(self, sequences: list[NavSequence]):
        for seq in sequences:
            # Skip sequences without truth or wheel/gnss for velocity
            if seq.truth is None or len(seq.truth) < self.window // 10:
                continue

            # Get IMU data (already resampled to 100 Hz)
            imu = seq.imu
            if len(imu) < self.window:
                continue

            acc = imu[["acc_x", "acc_y", "acc_z"]].values.astype(np.float32)
            gyr = imu[["gyr_x", "gyr_y", "gyr_z"]].values.astype(np.float32)

            # Compute att0 once per sequence
            att0 = compute_att0_from_sequence(seq)

            # Truth at 10 Hz
            truth_df = seq.truth
            truth = truth_df[["x_m", "y_m", "z_m", "heading"]].values.astype(np.float32)
            truth_dt = 1.0 / self.config.data.truth_hz

            # Slide window with stride
            max_windows = 2000  # Cap windows per sequence for memory/speed
            n_possible = (len(acc) - self.window) // self.stride + 1
            if n_possible > max_windows:
                # Sample uniformly
                indices = np.linspace(0, n_possible - 1, max_windows, dtype=int)
            else:
                indices = np.arange(n_possible)

            for idx in indices:
                i = idx * self.stride
                # Center time of window
                t_center = (i + self.window // 2) * self.dt
                # Find closest truth index
                truth_times = np.arange(len(truth)) * truth_dt
                center_idx = int(np.argmin(np.abs(truth_times - t_center)))

                vel_target, disp_target = forward_velocity_from_truth(
                    truth, center_idx, self.window // 10, truth_dt
                )

                # Build sample
                acc_win = acc[i : i + self.window]
                gyr_win = gyr[i : i + self.window]

                # Skip if NaN
                if np.any(np.isnan(acc_win)) or np.any(np.isnan(gyr_win)):
                    continue

                sample = VelocitySample(
                    acc=torch.from_numpy(acc_win),
                    gyr=torch.from_numpy(gyr_win),
                    att0=torch.from_numpy(att0),
                    vel_target=torch.tensor(vel_target, dtype=torch.float32),
                    disp_target=torch.tensor(disp_target, dtype=torch.float32),
                    mask=torch.ones(self.window, dtype=torch.bool),
                )
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "acc": s.acc,  # (W, 3)
            "gyr": s.gyr,  # (W, 3)
            "att0": s.att0,  # (3, 3)
            "vel_target": s.vel_target,  # ()
            "disp_target": s.disp_target,  # ()
            "mask": s.mask,  # (W,)
        }


def velocity_collate(batch: list[dict]) -> dict:
    """Collate function for VelocityDataset."""
    return {
        "acc": torch.stack([b["acc"] for b in batch]),  # (B, W, 3)
        "gyr": torch.stack([b["gyr"] for b in batch]),  # (B, W, 3)
        "att0": torch.stack([b["att0"] for b in batch]),  # (B, 3, 3)
        "vel_target": torch.stack([b["vel_target"] for b in batch]),  # (B,)
        "disp_target": torch.stack([b["disp_target"] for b in batch]),  # (B,)
        "mask": torch.stack([b["mask"] for b in batch]),  # (B, W)
    }


TRAINABLE_SOURCES: tuple[str, ...] = ("io_vnbd", "ronin", "idol", "self_collected")


def build_velocity_datasets(
    config: Config,
    max_train_seqs: int = 20,
    max_val_seqs: int = 5,
    sources: tuple[str, ...] = TRAINABLE_SOURCES,
) -> tuple[VelocityDataset, VelocityDataset, VelocityDataset]:
    """Build train/val/test datasets by unioning every ingested source.

    Splits are computed per source so a single drive never straddles a boundary;
    the per-source buckets are then unioned. `max_*_seqs` applies per source, so
    a 121-sequence corpus cannot starve a 43-sequence one.
    """
    from aeronavis.data.preprocess import NavSequence
    from aeronavis.data.splits import make_splits

    processed_root = Path(config.paths.processed)

    train_seqs: list[NavSequence] = []
    val_seqs: list[NavSequence] = []
    test_seqs: list[NavSequence] = []
    found: list[str] = []

    for source in sources:
        cache_root = processed_root / source
        if not cache_root.exists():
            logger.info(f"{source}: no processed cache at {cache_root}, skipped")
            continue
        all_seqs = NavSequence.load_all(cache_root)
        if not all_seqs:
            logger.info(f"{source}: processed cache is empty, skipped")
            continue

        found.append(source)
        logger.info(f"Loaded {len(all_seqs)} sequences from {cache_root}")

        buckets = make_splits(all_seqs, config.splits)
        train_seqs.extend(buckets["train"][:max_train_seqs])
        val_seqs.extend(buckets["val"][:max_val_seqs])
        test_seqs.extend(buckets["test"][:max_val_seqs])

    if not found:
        raise RuntimeError(
            f"No processed caches found under {processed_root} for {sources}. Run Part 1 first."
        )

    logger.info(f"Training sources: {', '.join(found)}")
    logger.info(
        f"Using {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test sequences"
    )

    train_ds = VelocityDataset(train_seqs, config, is_train=True)
    val_ds = VelocityDataset(val_seqs, config, is_train=False)
    test_ds = VelocityDataset(test_seqs, config, is_train=False)

    return train_ds, val_ds, test_ds
