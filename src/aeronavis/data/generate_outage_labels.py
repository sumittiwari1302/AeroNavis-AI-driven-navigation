"""Generate outage labels from GNSS data for forecaster training.

Creates per-window outage labels (0=no outage, 1=outage) from GNSS fix gaps.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from aeronavis.config import get_config
from aeronavis.data.preprocess import NavSequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def derive_outage_labels(
    gnss_ts: np.ndarray,
    window_ts: np.ndarray,
    min_gap_s: float = 2.0,
) -> np.ndarray:
    """
    Derive outage labels per velocity window from GNSS timestamps.
    
    Args:
        gnss_ts: GNSS fix timestamps
        window_ts: Velocity window timestamps (center of each window)
        min_gap_s: Minimum gap to consider as outage
        
    Returns:
        Binary labels per window (1=outage, 0=GNSS available)
    """
    if len(gnss_ts) < 2:
        return np.ones(len(window_ts), dtype=int)
    
    # Sort GNSS timestamps
    gnss_ts = np.sort(gnss_ts)
    
    # Find outage intervals
    gaps = np.diff(gnss_ts) > min_gap_s
    outage_starts = gnss_ts[:-1][gaps]
    outage_ends = gnss_ts[1:][gaps]
    
    # Label each window
    labels = np.zeros(len(window_ts), dtype=int)
    for i, t in enumerate(window_ts):
        # Check if window center falls in any outage interval
        for start, end in zip(outage_starts, outage_ends):
            if start <= t < end:
                labels[i] = 1
                break
    
    return labels


def generate_outage_labels_for_sequence(
    sequence: NavSequence,
    config,
    labels_dir: Path,
) -> int:
    """Generate outage labels for a single sequence."""
    if sequence.gnss is None or len(sequence.gnss) == 0:
        logger.warning(f"{sequence.seq_id}: No GNSS data, skipping")
        return 0
    
    if sequence.imu is None or len(sequence.imu) == 0:
        logger.warning(f"{sequence.seq_id}: No IMU data, skipping")
        return 0
    
    # Get window timestamps from velocity dataset logic
    imu_ts = sequence.imu["ts"].values
    window = config.model.window
    stride = config.model.stride
    dt = 1.0 / config.data.imu_hz
    
    # Window centers
    n_windows = (len(imu_ts) - window) // stride + 1
    if n_windows <= 0:
        logger.warning(f"{sequence.seq_id}: Not enough IMU data for windows")
        return 0
    
    window_centers = []
    for w_idx in range(n_windows):
        start_idx = w_idx * stride
        center_idx = start_idx + window // 2
        if center_idx < len(imu_ts):
            window_centers.append(imu_ts[center_idx])
    
    window_centers = np.array(window_centers)
    
    # Get GNSS timestamps
    gnss_ts = sequence.gnss["ts"].values
    
    # Derive labels
    labels = derive_outage_labels(gnss_ts, window_centers, config.data.outage_min_gap_s)
    
    # Save labels
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_path = labels_dir / f"{sequence.seq_id}_outage.csv"
    
    labels_df = pd.DataFrame({
        "window_idx": np.arange(len(labels)),
        "window_center_ts": window_centers,
        "label": labels,
    })
    labels_df.to_csv(labels_path, index=False)
    
    outage_pct = labels.mean() * 100
    logger.info(f"{sequence.seq_id}: {len(labels)} windows, {outage_pct:.1f}% outage")
    
    return len(labels)


def generate_all_outage_labels(
    source: str = "io_vnbd",
    labels_dir: str = "data/processed/labels/outage",
) -> int:
    """Generate outage labels for all sequences in a source."""
    config = get_config()
    labels_path = Path(labels_dir)
    labels_path.mkdir(parents=True, exist_ok=True)
    
    cache_root = Path(config.paths.processed) / source
    sequences = NavSequence.load_all(cache_root)
    
    total_windows = 0
    for seq in sequences:
        n = generate_outage_labels_for_sequence(seq, config, labels_path)
        total_windows += n
    
    logger.info(f"Generated outage labels for {len(sequences)} sequences, {total_windows} total windows")
    return total_windows


def generate_slip_labels_for_sequence(
    sequence: NavSequence,
    config,
    labels_dir: Path,
) -> int:
    """Generate slip labels from wheel speed vs IMU-derived speed.
    
    Label: 0=grip, 1=slip, 2=stationary
    """
    if sequence.wheel is None or len(sequence.wheel) == 0:
        return 0
    
    # This is a simplified version - real implementation would use
    # IMU-derived speed vs wheel speed to detect slip
    # For now, create placeholder labels
    
    imu_ts = sequence.imu["ts"].values
    window = config.model.window
    stride = config.model.stride
    
    n_windows = (len(imu_ts) - window) // stride + 1
    if n_windows <= 0:
        return 0
    
    # Placeholder: random labels for now
    # Real implementation would compute speed difference
    np.random.seed(42)
    labels = np.random.choice([0, 1, 2], size=n_windows, p=[0.7, 0.2, 0.1])
    
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_path = labels_dir / f"{sequence.seq_id}_slip.csv"
    
    labels_df = pd.DataFrame({
        "window_idx": np.arange(n_windows),
        "label": labels,
    })
    labels_df.to_csv(labels_path, index=False)
    
    return n_windows


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate labels for forecaster/slip training")
    parser.add_argument("--source", default="io_vnbd", help="Data source")
    parser.add_argument("--outage-dir", default="data/processed/labels/outage", help="Outage labels output dir")
    parser.add_argument("--slip-dir", default="data/processed/labels/slip", help="Slip labels output dir")
    parser.add_argument("--only-outage", action="store_true", help="Only generate outage labels")
    parser.add_argument("--only-slip", action="store_true", help="Only generate slip labels")
    
    args = parser.parse_args()
    
    config = get_config()
    
    if not args.only_slip:
        logger.info("Generating outage labels...")
        generate_all_outage_labels(args.source, args.outage_dir)
    
    if not args.only_outage:
        logger.info("Generating slip labels...")
        # Slip labels are generated by generate_slip_labels.py
        logger.info("Slip labels generated by generate_slip_labels.py")


if __name__ == "__main__":
    main()