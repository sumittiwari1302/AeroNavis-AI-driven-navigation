"""NAV-X evaluation package."""

from navx.eval.metrics import (
    compute_ate,
    compute_rte,
    compute_drift_pct_km,
    heading_error,
    umeyama_alignment,
    align_trajectory,
    latency_ms,
    energy_mah,
    compute_all_metrics,
    save_metrics,
    ATEResult,
    RTEResult,
    DriftResult,
    HeadingErrorResult,
    LatencyResult,
    EnergyResult,
    MetricsResult,
)

from navx.eval.forced_blackout import (
    ForcedBlackoutEvaluator,
    BlackoutSchedule,
    BlackoutResult,
    BaselineResult,
    BlackoutComparison,
    run_blackout_protocol,
)

from navx.eval.adapt_eval import (
    DeviceProfile,
    CalibrationResult,
    CalibrationReport,
    run_calibration_benchmark,
)

__all__ = [
    # Metrics
    "compute_ate",
    "compute_rte",
    "compute_drift_pct_km",
    "heading_error",
    "umeyama_alignment",
    "align_trajectory",
    "latency_ms",
    "energy_mah",
    "compute_all_metrics",
    "save_metrics",
    "ATEResult",
    "RTEResult",
    "DriftResult",
    "HeadingErrorResult",
    "LatencyResult",
    "EnergyResult",
    "MetricsResult",
    # Forced blackout
    "ForcedBlackoutEvaluator",
    "BlackoutSchedule",
    "BlackoutResult",
    "BaselineResult",
    "BlackoutComparison",
    "run_blackout_protocol",
    # Adaptation eval
    "DeviceProfile",
    "CalibrationResult",
    "CalibrationReport",
    "run_calibration_benchmark",
]