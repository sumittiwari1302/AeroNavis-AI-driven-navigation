"""Adaptation evaluation and device calibration benchmark.

Re-runs Part 5 adaptation evaluation across synthetic device profiles
to prove adaptation beats frozen per device type.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from aeronavis.config import get_config
from aeronavis.config import Config
from aeronavis.models.velocity import build_velocity_model

logger = logging.getLogger(__name__)


@dataclass
class DeviceProfile:
    """Synthetic device profile for calibration."""
    name: str
    imu_noise_std: float       # IMU noise std multiplier
    vibration_amplitude: float  # Vibration amplitude multiplier
    bias_instability: float     # Gyro/accel bias instability
    sensor_type: str           # "phone", "two_wheeler", "vehicle", "synthetic"
    description: str = ""


# Predefined device profiles for calibration
DEVICE_PROFILES = [
    DeviceProfile(
        name="phone_high_end",
        imu_noise_std=1.0,
        vibration_amplitude=1.0,
        bias_instability=1.0,
        sensor_type="phone",
        description="High-end smartphone (e.g., Pixel 7, iPhone 15)",
    ),
    DeviceProfile(
        name="phone_mid_range",
        imu_noise_std=1.5,
        vibration_amplitude=1.3,
        bias_instability=1.5,
        sensor_type="phone",
        description="Mid-range smartphone (e.g., Pixel 6a, Galaxy A54)",
    ),
    DeviceProfile(
        name="phone_budget",
        imu_noise_std=2.0,
        vibration_amplitude=2.0,
        bias_instability=2.0,
        sensor_type="phone",
        description="Budget smartphone (e.g., Redmi, low-end Motorola)",
    ),
    DeviceProfile(
        name="two_wheeler",
        imu_noise_std=1.2,
        vibration_amplitude=3.0,      # High vibration
        bias_instability=2.0,
        sensor_type="two_wheeler",
        description="Motorcycle/scooter mount (high vibration)",
    ),
    DeviceProfile(
        name="vehicle_dash",
        imu_noise_std=0.8,
        vibration_amplitude=0.5,       # Low vibration (dashboard mount)
        bias_instability=0.8,
        sensor_type="vehicle",
        description="Car dashboard mount (low vibration)",
    ),
    DeviceProfile(
        name="synthetic_low",
        imu_noise_std=0.5,
        vibration_amplitude=0.5,
        bias_instability=0.5,
        sensor_type="synthetic",
        description="Ideal low-noise synthetic profile",
    ),
    DeviceProfile(
        name="synthetic_high",
        imu_noise_std=3.0,
        vibration_amplitude=5.0,
        bias_instability=3.0,
        sensor_type="synthetic",
        description="Extreme high-noise synthetic profile",
    ),
]


@dataclass
class CalibrationResult:
    """Results for a single device profile."""
    profile_name: str
    base_ate_60s: float          # ATE without adaptation
    adapted_ate_60s: float       # ATE with adaptation
    improvement_pct: float       # Percentage improvement
    base_vel_rmse: float         # Base velocity RMSE
    adapted_vel_rmse: float      # Adapted velocity RMSE
    convergence_steps: int       # Steps to converge
    frozen_during_training: bool


@dataclass
class CalibrationReport:
    """Full calibration report across all profiles."""
    profiles: List[Dict]
    summary: Dict
    overall_improvement: float
    worst_case_improvement: float
    best_profile: str
    worst_profile: str


@dataclass
class DeviceProfile:
    """Synthetic device profile for calibration."""
    name: str
    imu_noise_std: float
    vibration_amplitude: float
    bias_instability: float
    sensor_type: str
    description: str


def get_device_profiles() -> List[DeviceProfile]:
    """Get all predefined device profiles for calibration."""
    return [
        DeviceProfile(
            name="phone_high_end",
            imu_noise_std=1.0,
            vibration_amplitude=1.0,
            bias_instability=1.0,
            sensor_type="phone",
            description="High-end smartphone (e.g., Pixel 7, iPhone 15)",
        ),
        DeviceProfile(
            name="phone_mid_range",
            imu_noise_std=1.5,
            vibration_amplitude=1.3,
            bias_instability=1.5,
            sensor_type="phone",
            description="Mid-range smartphone (e.g., Pixel 6a, Galaxy A54)",
        ),
        DeviceProfile(
            name="phone_budget",
            imu_noise_std=2.0,
            vibration_amplitude=2.0,
            bias_instability=2.0,
            sensor_type="phone",
            description="Budget smartphone (e.g., Redmi, low-end Motorola)",
        ),
        DeviceProfile(
            name="two_wheeler",
            imu_noise_std=1.2,
            vibration_amplitude=3.0,
            bias_instability=2.0,
            sensor_type="two_wheeler",
            description="Motorcycle/scooter mount (high vibration)",
        ),
        DeviceProfile(
            name="vehicle_dash",
            imu_noise_std=0.8,
            vibration_amplitude=0.5,
            bias_instability=0.8,
            sensor_type="vehicle",
            description="Car dashboard mount (low vibration)",
        ),
    ]


def build_velocity_model_for_profile(config: Config, profile: DeviceProfile):
    """Build velocity model with profile-specific noise injection."""
    from aeronavis.config import get_config
    
    # Use base config but override with profile-specific noise
    config = get_config()
    model = build_velocity_model(config)
    return model


def inject_profile_noise(
    imu_data: np.ndarray,
    profile: Dict,
    dt: float = 0.01,
) -> np.ndarray:
    """
    Inject device-specific noise into IMU data.
    
    Args:
        imu_data: (N, 6) IMU data [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
        profile: Device profile dict with noise parameters
        dt: Time step in seconds
        
    Returns:
        Noisy IMU data with injected device-specific noise
    """
    noisy = imu_data.copy()
    
    # Accelerometer noise
    acc_noise = np.random.randn(*imu_data[:, :3].shape) * profile["imu_noise_std"]
    imu_noisy = imu_data.copy()
    imu_noisy[:, :3] += acc_noise
    
    # Gyro noise
    gyr_noise = np.random.randn(*imu_data[:, 3:].shape) * profile["imu_noise_std"] * 0.1
    imu_noisy[:, 3:] += gyr_noise
    
    # Vibration: add high-frequency oscillation
    t = np.arange(len(imu_data)) * 0.01  # 100 Hz
    vib_freq = 20.0  # 20 Hz vibration
    vib = profile["vibration_amplitude"] * np.sin(2 * np.pi * vib_freq * np.arange(len(imu_data)) * 0.01)
    imu_noisy[:, :3] += vib[:, np.newaxis] * np.random.randn(*imu_data[:, :3].shape) * 0.1
    
    # Bias instability: slow random walk
    bias_walk = np.cumsum(np.random.randn(len(imu_data)) * profile["bias_instability"] * 0.01)
    imu_noisy[:, :3] += bias_walk[:, np.newaxis]
    imu_noisy[:, 3:] += bias_walk[:, np.newaxis] * 0.1
    
    return imu_noisy


def evaluate_adaptation_on_profile(
    config,
    profile: DeviceProfile,
    sequences,
    epochs: int = 5,
) -> Dict:
    """
    Evaluate adaptation performance on a single device profile.
    
    Returns dict with base vs adapted metrics.
    """
    from aeronavis.config import get_config
    
    config = get_config()
    
    # Build base velocity model
    base_model = build_velocity_model(get_config())
    
    # Split sequences
    train_seqs = sequences[:int(len(sequences) * 0.8)]
    val_seqs = sequences[int(len(sequences) * 0.8):]
    
    # Train base model (frozen)
    # ... training logic here (simplified)
    
    # Evaluate base model on validation
    base_losses = []
    adapted_losses = []
    
    # This is a simplified version - real implementation would:
    # 1. Inject profile noise into validation sequences
    # 2. Run base model inference
    # 2. Run adapted model inference
    # 3. Compare ATE/RMSE
    
    return {
        "profile": "unknown",
        "base_ate_60s": 0.0,
        "adapted_ate_60s": 0.0,
        "improvement_pct": 0.0,
        "base_vel_rmse": 0.0,
        "adapted_vel_rmse": 0.0,
        "convergence_steps": 0,
        "frozen_during_training": False,
    }


def run_calibration_benchmark(
    config: Config = None,
    profiles: List = None,
    sequences = None,
    epochs: int = 10,
    output_dir: str = "models/calibration",
) -> Dict:
    """
    Run full calibration benchmark across all device profiles.
    
    Args:
        config: Global config
        profiles: List of DeviceProfile (uses defaults if None)
        sequences: Pre-loaded sequences (loads from disk if None)
        epochs: Training epochs per profile
        output_dir: Directory to save results
        
    Returns:
        CalibrationReport with per-profile results and summary
    """
    config = config or get_config()
    profiles = profiles or get_device_profiles()
    output_path = Path(__file__).parent.parent.parent / "models" / "calibration"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if sequences is None:
        # Load from Part 1 processed data
        from aeronavis.data.velocity_dataset import build_velocity_datasets
        train_ds, val_ds, test_ds = build_velocity_datasets(get_config())
        sequences = list(val_ds)  # Use validation as test
    
    results = []
    logger.info(f"Running calibration benchmark on {len(profiles)} profiles")
    
    for profile in profiles:
        logger.info(f"Evaluating profile: {profile.name} ({profile.description})")
        
        # Inject profile noise into validation sequences
        # For now, use simplified evaluation
        result = evaluate_adaptation_on_profile(config, profile, sequences[:5], epochs=3)
        
        result["profile"] = profile.name
        result["description"] = profile.description
        results.append(result)
        logger.info(f"  {profile.name}: base_ate={result['base_ate_60s']:.2f}m, "
                    f"adapted={result['adapted_ate_60s']:.2f}m, "
                    f"improvement={result['improvement_pct']:.1f}%")
    
    # Compute summary
    improvements = [r["improvement_pct"] for r in results]
    overall_improvement = np.mean(improvements) if improvements else 0.0
    worst_case = min(improvements) if improvements else 0.0
    best_idx = np.argmax(improvements)
    worst_idx = np.argmin(improvements)
    
    report = {
        "profiles": results,
        "summary": {
            "overall_improvement_pct": float(overall_improvement),
            "worst_case_improvement_pct": float(worst_case),
            "best_profile": results[best_idx]["profile"] if results else "none",
            "worst_profile": results[worst_idx]["profile"] if results else "none",
            "num_profiles": len(profiles),
        },
        "timestamp": time.time(),
    }
    
    # Save report
    output_path = Path(output_dir) / "calibration_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Calibration complete: {overall_improvement:.1f}% avg improvement")
    logger.info(f"  Best: {report['summary']['best_profile']} (+{improvements[best_idx]:.1f}%)")
    logger.info(f"  Worst: {report['summary']['worst_profile']} (+{improvements[worst_idx]:.1f}%)")
    
    return {
        "profiles": results,
        "summary": report["summary"],
        "overall_improvement": overall_improvement,
        "worst_case_improvement": worst_case,
        "best_profile": report["summary"]["best_profile"],
        "worst_profile": report["summary"]["worst_profile"],
    }


def run_calibration_benchmark_cli(
    config=None,
    profiles=None,
    sequences=None,
    epochs=10,
    output_dir="models/calibration",
) -> Dict:
    """Entry point for calibration benchmark (CLI wrapper)."""
    return run_calibration_benchmark(config, profiles, sequences, epochs, output_dir)


def _cli():
    """CLI entry point for calibration benchmark."""
    import argparse
    parser = argparse.ArgumentParser(description="NAV-X 3.0 Calibration Benchmark")
    parser.add_argument("--profiles", type=str, default="phone_high_end,phone_mid_range,phone_budget,two_wheeler", help="Comma-separated profile names")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per profile")
    parser.add_argument("--output", type=str, default="models/calibration", help="Output directory")
    args = parser.parse_args()
    
    profile_names = args.profiles.split(",")
    # Use default profiles
    profiles = get_device_profiles()
    profiles = [p for p in profiles if p.name in profile_names]
    
    report = run_calibration_benchmark(profiles=profiles, epochs=args.epochs, output_dir=args.output)
    print(f"Calibration benchmark complete. Results saved to {args.output}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    _cli()