"""Generate slip/grip/stationary labels from IO-VNBD wheel speed data.

Labels:
- 0: grip (wheel and IMU agree)
- 1: slip_brake (wheel and IMU disagree significantly)
- 2: stationary (near-zero motion)

Saves CSVs under data/processed/labels/slip/
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from aeronavis.config import get_config
from aeronavis.data.preprocess import NavSequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_imu_speed(acc: np.ndarray, dt: float, gravity: float = 9.80665) -> np.ndarray:
    """Estimate speed from gravity-free accel projection."""
    # Remove gravity component (assume z-axis aligned)
    acc_horiz = np.sqrt(acc[:, 0] ** 2 + acc[:, 1] ** 2)
    # Simple integration with high-pass to remove drift
    speed = np.cumsum(acc_horiz) * dt
    # High-pass filter to remove integration drift
    from scipy.signal import butter, filtfilt

    b, a = butter(2, 0.01 / (1.0 / dt / 2), btype="highpass")
    speed = filtfilt(b, a, speed)
    return np.maximum(speed, 0.0)


def label_slip(
    seq: NavSequence,
    config,
    slip_thresh: float = 0.25,
    slip_duration_s: float = 0.4,
    stationary_acc_thresh: float = 0.5,
    stationary_gyro_thresh: float = 0.05,
    stationary_wheel_thresh: float = 0.1,
) -> np.ndarray:
    """
    Generate slip labels for a sequence.

    Returns: array of labels (0=grip, 1=slip_brake, 2=stationary) per 100Hz window center.
    """
    dt = 1.0 / config.data.imu_hz

    # Get IMU data
    imu = seq.imu[["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]].values
    wheel = seq.wheel

    if wheel is None or len(wheel) == 0:
        logger.warning(f"{seq.seq_id}: no wheel data, cannot label")
        return np.full(len(seq.imu) // 10, 0, dtype=int)  # default to grip

    # Wheel speed at window centers
    wheel_speed = wheel["wheel_speed_mps"].values
    wheel_ts = wheel["ts"].values

    # Compute IMU-derived speed
    imu_speed = compute_imu_speed(imu, dt)

    # Resample both to window centers (every 10 samples = 10Hz)
    window_stride = config.model.stride
    window_len = config.model.window

    # For each window center, get wheel speed and IMU speed
    n_windows = (len(imu) - window_len) // window_stride + 1
    labels = np.zeros(n_windows, dtype=int)

    for w in range(n_windows):
        center_idx = w * window_stride + config.model.window // 2
        t_center = center_idx * dt

        # Get wheel speed at this time (interpolate)
        if len(wheel_ts) > 1:
            v_wheel = np.interp(t_center, wheel_ts, wheel_speed)
        else:
            v_wheel = wheel_speed[0]

        # Get IMU speed at this time
        if center_idx < len(imu_speed):
            v_imu = imu_speed[center_idx]
        else:
            v_imu = 0.0

        # Stationary check
        window_imu = imu[center_idx - 5 : center_idx + 5]
        acc_norm = np.linalg.norm(window_imu[:, :3], axis=1).mean()
        gyro_norm = np.linalg.norm(window_imu[:, 3:], axis=1).mean()
        G = 9.80665

        if (
            acc_norm < G + stationary_acc_thresh
            and gyro_norm < stationary_gyro_thresh
            and v_wheel < stationary_wheel_thresh
        ):
            labels[w] = 2  # stationary
            continue

        # Slip check
        if v_wheel > 0.5:  # only check slip when moving
            rel_diff = abs(v_wheel - v_imu) / max(v_wheel, 0.5)
            if rel_diff > slip_thresh:
                # Check duration
                labels[w] = 1  # slip/brake
            else:
                labels[w] = 0  # grip
        else:
            labels[w] = 0  # grip at low speed

    return labels


def generate_all_labels(config, out_dir: Path):
    """Generate labels for all sequences in processed cache."""
    from aeronavis.data.preprocess import NavSequence

    cache_root = Path(config.paths.processed) / "io_vnbd"
    if not cache_root.exists():
        raise RuntimeError(f"Processed cache not found: {cache_root}")

    sequences = NavSequence.load_all(cache_root)
    logger.info(f"Loaded {len(sequences)} sequences for labeling")

    out_dir.mkdir(parents=True, exist_ok=True)

    for seq in sequences:
        labels = label_slip(seq, config)
        # Save per-window labels
        out_file = out_dir / f"{seq.seq_id}_slip.csv"
        df = pd.DataFrame(
            {
                "window_idx": np.arange(len(labels)),
                "label": labels,
            }
        )
        df.to_csv(out_file, index=False)
        logger.info(
            f"  {seq.seq_id}: {len(labels)} windows, "
            f"grip={np.sum(labels==0)}, slip={np.sum(labels==1)}, stationary={np.sum(labels==2)}"
        )

    # Also create a combined manifest
    manifest = []
    for seq in sequences:
        labels = label_slip(seq, config)
        manifest.append(
            {
                "seq_id": seq.seq_id,
                "n_windows": len(labels),
                "grip_pct": np.mean(labels == 0),
                "slip_pct": np.mean(labels == 1),
                "stationary_pct": np.mean(labels == 2),
            }
        )
    pd.DataFrame(manifest).to_csv(out_dir / "label_manifest.csv", index=False)
    logger.info(f"Label manifest saved to {out_dir / 'label_manifest.csv'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate slip labels from IO-VNBD")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="data/processed/labels/slip")
    args = parser.parse_args()

    config = get_config(args.config)
    generate_all_labels(config, Path(args.out))
    logger.info("Label generation complete.")


if __name__ == "__main__":
    main()
