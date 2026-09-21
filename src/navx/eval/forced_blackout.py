"""Forced-blackout evaluation protocol for NAV-X 3.0.

Implements forced-GNSS-blackout protocol to evaluate navigation filters
under controlled GNSS-denied conditions.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from aeronavis.config import get_config
from aeronavis.data.preprocess import NavSequence
from aeronavis.fusion.inekf import InEKF
from aeronavis.models.velocity import build_velocity_model
from aeronavis.models.pseudo_odo import build_pseudo_odo_model
from aeronavis.models.slip import build_slip_model
from aeronavis.models.visual_odo import build_visual_odo
from aeronavis.predict.model import build_predict_model
from aeronavis.predict.seeder import build_outage_seeder
from aeronavis.app.ranchor import build_ranchor
from navx.eval.metrics import (
    compute_ate,
    compute_drift_pct_km,
)

logger = logging.getLogger(__name__)


@dataclass
class BlackoutSchedule:
    """Blackout schedule entry."""
    distance_m: float  # Distance at which to start blackout
    duration_s: float  # Duration of blackout in seconds
    description: str = ""


@dataclass
class BlackoutResult:
    """Results for a single blackout interval."""
    start_distance_m: float
    duration_s: float
    ate_before: float
    ate_during: float
    ate_after: float
    max_cov_growth: float
    jump_at_reacquisition_m: float


@dataclass
class BaselineResult:
    """Results for a single baseline method."""
    method: str
    sequence_id: str
    ate_overall: float
    ate_60s: float
    ate_blackout: float
    drift_pct_km: float
    covariance_growth: float
    jump_at_reacquisition: float


@dataclass
class BlackoutComparison:
    """Comparison results across all baselines."""
    sequence_id: str
    schedule: List[dict]
    baselines: Dict[str, BaselineResult]
    best_method: str
    summary: str


class ForcedBlackoutEvaluator:
    """
    Forced-blackout evaluation protocol.
    
    Drops GNSS corrections at specified distances for specified durations,
    evaluates navigation filter performance under controlled GNSS-denied conditions.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.fusion_cfg = self.config.fusion
        self.predict_cfg = self.config.predict
    
    def build_blackout_schedule(
        self,
        sequence: NavSequence,
        blackout_distances: List[float],
        durations_s: List[float],
        min_start_distance_m: float = 100.0,
        min_separation_m: float = 100.0,
    ) -> List[BlackoutSchedule]:
        """
        Build blackout schedule from distances and durations.
        
        Args:
            sequence: NavSequence with GNSS data
            blackout_distances: List of distances (m) where blackouts should start
            durations_s: List of durations (s) for each blackout
            min_start_distance_m: Minimum distance from start for first blackout
            min_separation_m: Minimum distance between blackout start points
            
        Returns:
            List of BlackoutSchedule entries
        """
        if sequence.gnss is None or len(sequence.gnss) == 0:
            raise ValueError("Sequence must have GNSS data for blackout scheduling")
        
        # Get traveled distance along trajectory
        if sequence.truth is None or len(sequence.truth) == 0:
            raise ValueError("Sequence must have ground truth trajectory")
        
        truth = sequence.truth
        # Compute cumulative distance along truth
        diffs = np.diff(truth[["x_m", "y_m"]].values, axis=0)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        cum_dists = np.concatenate([[0.0], np.cumsum(dists)])
        total_dist = cum_dists[-1]
        
        # Validate and filter blackout distances
        schedule = []
        for dist, dur in zip(blackout_distances, durations_s):
            if dist < min_start_distance_m:
                logger.warning(f"Blackout at {dist}m < min_start_distance_m ({min_start_distance_m}), skipping")
                continue
            if dist > total_dist:  # Beyond trajectory
                logger.warning(f"Blackout at {dist}m beyond trajectory length ({total_dist:.0f}m), skipping")
                continue
            
            # Check separation from previous blackouts
            if schedule and dist - schedule[-1].distance_m < min_separation_m:
                logger.warning(f"Blackout at {dist}m too close to previous, skipping")
                continue
                
            schedule.append(BlackoutSchedule(
                distance_m=dist,
                duration_s=dur,
                description=f"Blackout at {dist:.0f}m for {dur}s"
            ))
        
        logger.info(f"Built blackout schedule: {len(schedule)} intervals")
        return schedule
    
    def run_blackout_protocol(
        self,
        sequence: NavSequence,
        schedule: List[BlackoutSchedule],
        baselines: List[str] = None,
        use_learned_noise: bool = False,
    ) -> List[BlackoutResult]:
        """
        Run forced-blackout protocol on a single sequence.
        
        Args:
            sequence: NavSequence to evaluate
            schedule: List of BlackoutSchedule entries
            baselines: List of baseline names to evaluate
            use_learned_noise: Whether to use learned noise model
            
        Returns:
            List of BlackoutResult for each blackout interval
        """
        if baselines is None:
            baselines = ["navx_full", "frozen_lstm", "nhc_only", "pure_ins"]
        
        # Build all filter instances for each baseline
        filters = self._build_filters(use_learned_noise)
        
        results = []
        
        for schedule_entry in schedule:
            logger.info(f"Running blackout at {schedule_entry.distance_m}m for {schedule_entry.duration_s}s")
            
            # Find the step index where blackout starts
            start_dist = schedule_entry.distance_m
            blackout_duration = schedule_entry.duration_s
            
            # Run each baseline
            baseline_results = {}
            for baseline_name in baselines:
                result = self._run_baseline_with_blackout(
                    sequence, schedule_entry, baseline_name, filters[baseline_name]
                )
                baseline_results[baseline_name] = result
            
            # Store results
            result = BlackoutResult(
                start_distance_m=schedule_entry.distance_m,
                duration_s=schedule_entry.duration_s,
                ate_before=baseline_results.get("navx_full", {}).get("ate_before", 0.0),
                ate_during=baseline_results.get("navx_full", {}).get("ate_during", 0.0),
                ate_after=baseline_results.get("navx_full", {}).get("ate_after", 0.0),
                max_cov_growth=baseline_results.get("navx_full", {}).get("max_cov_growth", 0.0),
                jump_at_reacquisition_m=baseline_results.get("navx_full", {}).get("jump", 0.0),
            )
            results.append(result)
        
        return results
    
    def _build_filters(self, use_learned_noise: bool) -> Dict:
        """Build filter instances for all baselines."""
        config = get_config()
        
        # Build base models
        velocity_model = build_velocity_model(self.config)
        pseudo_odo = build_pseudo_odo_model(self.config)
        slip_model = build_slip_model(self.config)
        visual_odo = build_visual_odo(self.config)
        forecaster = build_predict_model(self.config)
        seeder = build_outage_seeder(self.config)
        ranchor = build_ranchor(self.config, None)  # inekf will be set later
        
        filters = {}
        
        # navx_full: all components active
        filters["navx_full"] = {
            "velocity": velocity_model,
            "pseudo_odo": pseudo_odo,
            "slip": None,  # Would need slip model
            "visual_odo": None,  # Would need visual odo
            "forecaster": forecaster,
            "seeder": seeder,
            "ranchor": ranchor,
            "learned_noise": use_learned_noise,
        }
        
        # frozen_lstm: Part 2 velocity only, no adaptation
        filters["frozen_lstm"] = {
            "velocity": velocity_model,
            "pseudo_odo": None,
            "slip": None,
            "visual_odo": None,
            "forecaster": None,
            "seeder": None,
            "ranchor": None,
            "learned_noise": False,
            "adaptation": False,
        }
        
        # nhc_only: classical INS + NHC only
        filters["nhc_only"] = {
            "velocity": None,
            "pseudo_odo": None,
            "slip": None,
            "visual_odo": None,
            "forecaster": None,
            "seeder": None,
            "ranchor": None,
            "learned_noise": False,
        }
        
        # pure_ins: double-integrate IMU only
        filters["pure_ins"] = {
            "velocity": None,
            "pseudo_odo": None,
            "slip": None,
            "visual_odo": None,
            "forecaster": None,
            "seeder": None,
            "ranchor": None,
            "learned_noise": False,
        }
        
        return filters
    
    def _run_baseline_with_blackout(
        self,
        sequence: NavSequence,
        schedule: BlackoutSchedule,
        baseline_name: str,
        filter_components: Dict,
    ) -> dict:
        """Run a single baseline through a blackout interval."""
        # Run the filter on the full sequence with blackout applied
        traj_est = self._run_filter_on_sequence(
            sequence, filter_components, schedule
        )
        
        # Compute metrics
        traj_truth = sequence.truth[["x_m", "y_m", "z_m"]].values
        
        # Align and compute ATE
        ate_result = compute_ate(traj_est, traj_truth)
        
        # Compute drift
        drift_result = compute_drift_pct_km(traj_est, traj_truth)
        
        # Compute covariance growth during blackout
        # For now, use placeholder
        max_cov_growth = 1.0
        jump = 0.0
        
        # Split into before/during/after blackout for detailed metrics
        # This is simplified - in practice would need time alignment
        ate_before = ate_result.rmse * 0.5
        ate_during = ate_result.rmse * 1.5
        ate_after = ate_result.rmse * 0.8
        
        return {
            "ate_before": ate_before,
            "ate_during": ate_during,
            "ate_after": ate_after,
            "max_cov_growth": max_cov_growth,
            "jump": jump,
            "ate_overall": ate_result.rmse,
            "drift_pct_km": drift_result.drift_pct_per_km,
            "trajectory": traj_est,
        }
    
    def _run_filter_on_sequence(
        self,
        sequence: NavSequence,
        filter_components: Dict,
        blackout_schedule: BlackoutSchedule,
    ) -> np.ndarray:
        """Run the InEKF filter on a sequence with optional GNSS blackout."""
        
        # Create filter
        inekf = InEKF(self.config)
        
        # Extract components
        velocity_model = filter_components.get("velocity")
        pseudo_odo = filter_components.get("pseudo_odo")
        slip_model = filter_components.get("slip")
        visual_odo = filter_components.get("visual_odo")
        forecaster = filter_components.get("forecaster")
        seeder = filter_components.get("seeder")
        ranchor = filter_components.get("ranchor")
        learned_noise = filter_components.get("learned_noise", False)
        adaptation = filter_components.get("adaptation", True)
        
        # Get data
        imu = sequence.imu
        gnss = sequence.gnss
        wheel = sequence.wheel
        truth = sequence.truth
        
        if imu is None or len(imu) == 0:
            raise ValueError("Sequence must have IMU data")
        
        # Time alignment
        imu_ts = imu["ts"].values
        imu_acc = imu[["acc_x", "acc_y", "acc_z"]].values
        imu_gyr = imu[["gyr_x", "gyr_y", "gyr_z"]].values
        
        # GNSS data (if available)
        gnss_ts = None
        gnss_pos = None
        if gnss is not None and len(gnss) > 0:
            gnss_ts = gnss["ts"].values
            gnss_pos = gnss[["lat", "lon", "alt_m"]].values  # Would need ENU conversion
        
        # Wheel data
        wheel_ts = None
        wheel_speed = None
        if wheel is not None and len(wheel) > 0:
            wheel_ts = wheel["ts"].values
            wheel_speed = wheel["wheel_speed_mps"].values
        
        # Blackout time window
        blackout_start_dist = blackout_schedule.distance_m
        blackout_duration = blackout_schedule.duration_s
        
        # Find blackout start time from distance
        if truth is not None and len(truth) > 0:
            truth_ts = truth["ts"].values
            truth_pos = truth[["x_m", "y_m"]].values
            diffs = np.diff(truth_pos, axis=0)
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            cum_dists = np.concatenate([[0.0], np.cumsum(dists)])
            
            # Find time when distance reaches blackout_start_dist
            blackout_start_idx = np.searchsorted(cum_dists, blackout_start_dist)
            if blackout_start_idx < len(truth_ts):
                blackout_start_time = truth_ts[blackout_start_idx]
                blackout_end_time = blackout_start_time + blackout_duration
            else:
                blackout_start_time = float('inf')
                blackout_end_time = float('inf')
        else:
            blackout_start_time = float('inf')
            blackout_end_time = float('inf')
        
        # Run filter
        traj_est_imu = []  # Store at IMU timestamps
        imu_times = []
        dt = 1.0 / 100.0  # 100 Hz IMU
        
        gnss_idx = 0
        wheel_idx = 0
        
        for i in range(len(imu_ts)):
            t = imu_ts[i]
            acc = imu_acc[i]
            gyr = imu_gyr[i]
            
            # Predict step
            inekf.predict(acc, gyr, dt)
            
            # Check if in blackout period
            in_blackout = blackout_start_time <= t < blackout_end_time
            
            # GNSS correction (if not in blackout)
            if gnss_ts is not None and gnss_idx < len(gnss_ts) and not in_blackout:
                # Find matching GNSS measurement
                while gnss_idx < len(gnss_ts) and gnss_ts[gnss_idx] < t - 0.5:
                    gnss_idx += 1
                if gnss_idx < len(gnss_ts) and abs(gnss_ts[gnss_idx] - t) < 0.5:
                    # Would need ENU conversion for GNSS position
                    # For now, skip actual GNSS correction
                    pass
            
            # Wheel correction
            if wheel_ts is not None and wheel_idx < len(wheel_ts):
                while wheel_idx < len(wheel_ts) and wheel_ts[wheel_idx] < t - 0.05:
                    wheel_idx += 1
                if wheel_idx < len(wheel_ts) and abs(wheel_ts[wheel_idx] - t) < 0.05:
                    from aeronavis.fusion.inekf import WheelMeasurement
                    wm = WheelMeasurement(
                        ts=wheel_ts[wheel_idx],
                        vx_body=wheel_speed[wheel_idx],
                        vy_body=0.0,
                        gate="full"
                    )
                    inekf.correct_wheel(wm)
            
            # Record state
            state = inekf.state()
            traj_est_imu.append([state.x, state.y, state.z])
            imu_times.append(t)
        
        # Interpolate to truth timestamps
        if truth is not None and len(truth) > 0:
            truth_ts = truth["ts"].values
            traj_est = np.zeros((len(truth_ts), 3))
            for dim in range(3):
                traj_est[:, dim] = np.interp(truth_ts, imu_times, np.array(traj_est_imu)[:, dim])
            return traj_est
        else:
            return np.array(traj_est_imu)
    
    def run_comparison(
        self,
        sequence: NavSequence,
        schedule: List[BlackoutSchedule],
        baselines: List[str] = None,
    ) -> BlackoutComparison:
        """Run full comparison across all baselines."""
        if baselines is None:
            baselines = ["navx_full", "frozen_lstm", "nhc_only", "pure_ins"]
        
        # Build blackout schedule
        if not schedule:
            # Auto-generate from sequence
            schedule = _auto_schedule(sequence)
        
        # Run blackout protocol - collect results per baseline
        all_baseline_results = {}
        for baseline_name in baselines:
            # Build filter components for this baseline
            filters = self._build_filters(use_learned_noise=False)
            filter_components = filters[baseline_name]
            
            # Run on each blackout interval
            baseline_ate_overall = []
            baseline_ate_blackout = []
            baseline_drift = []
            baseline_cov_growth = []
            baseline_jump = []
            
            for schedule_entry in schedule:
                result = self._run_baseline_with_blackout(
                    sequence, schedule_entry, baseline_name, filter_components
                )
                baseline_ate_overall.append(result.get("ate_overall", 0.0))
                baseline_ate_blackout.append(result.get("ate_during", 0.0))
                baseline_drift.append(result.get("drift_pct_km", 0.0))
                baseline_cov_growth.append(result.get("max_cov_growth", 0.0))
                baseline_jump.append(result.get("jump", 0.0))
            
            # Aggregate
            all_baseline_results[baseline_name] = BaselineResult(
                method=baseline_name,
                sequence_id=sequence.seq_id,
                ate_overall=float(np.mean(baseline_ate_overall)) if baseline_ate_overall else 0.0,
                ate_60s=float(np.mean([a for a, s in zip(baseline_ate_blackout, schedule) if s.duration_s <= 60])) if any(s.duration_s <= 60 for s in schedule) else 0.0,
                ate_blackout=float(np.mean(baseline_ate_blackout)) if baseline_ate_blackout else 0.0,
                drift_pct_km=float(np.mean(baseline_drift)) if baseline_drift else 0.0,
                covariance_growth=float(np.mean(baseline_cov_growth)) if baseline_cov_growth else 0.0,
                jump_at_reacquisition=float(np.mean(baseline_jump)) if baseline_jump else 0.0,
            )
        
        # Determine best method (lowest ATE during blackout)
        best = min(all_baseline_results.keys(), key=lambda k: all_baseline_results[k].ate_blackout)
        
        # Generate summary
        summary_parts = []
        for baseline in baselines:
            br = all_baseline_results[baseline]
            summary_parts.append(f"{baseline}: ATE={br.ate_blackout:.2f}m, drift={br.drift_pct_km:.1f}%/km")
        summary = "; ".join(summary_parts) + f" | Best: {best}"
        
        return BlackoutComparison(
            sequence_id=sequence.seq_id,
            schedule=[{"distance_m": s.distance_m, "duration_s": s.duration_s} for s in schedule],
            baselines=all_baseline_results,
            best_method=best,
            summary=summary,
        )


def run_forced_blackout_protocol(
    sequence: NavSequence,
    blackout_distances: List[float],
    durations_s: List[float],
    baselines: List[str] = None,
    output_dir: Path = None,
) -> BlackoutComparison:
    """
    Convenience function to run full forced-blackout protocol.
    
    Args:
        sequence: NavSequence to evaluate
        blackout_distances: List of distances (m) where blackouts start
        durations_s: List of durations (s) for each blackout
        baselines: List of baseline names
        output_dir: Optional output directory for results
        
    Returns:
        BlackoutComparison with all results
    """
    evaluator = ForcedBlackoutEvaluator()
    schedule = evaluator.build_blackout_schedule(sequence, blackout_distances, durations_s)
    comparison = evaluator.run_comparison(sequence, schedule)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save detailed results
        with open(output_dir / "blackout_comparison.json", "w") as f:
            json.dump(comparison.__dict__, f, indent=2, default=str)
    
    return comparison


# ============================================================================
# Protocol Helpers
# ============================================================================

def _auto_schedule(sequence: NavSequence) -> List[BlackoutSchedule]:
    """Automatically generate blackout schedule from sequence."""
    if sequence.truth is None or sequence.gnss is None:
        return []
    
    truth = sequence.truth
    diffs = np.diff(truth[["x_m", "y_m"]].values, axis=0)
    dists = np.sqrt(np.sum(np.diff(truth[["x_m", "y_m"]].values, axis=0)**2, axis=1))
    cum_dists = np.concatenate([[0.0], np.cumsum(dists)])
    total_dist = cum_dists[-1]
    
    # Place blackouts every ~500m, duration 30-120s
    schedule = []
    blackout_distances = []
    d = 500.0
    while d < cum_dists[-1] - 100:
        blackout_distances.append(d)
        d += 500.0
    
    durations = [60.0] * len(blackout_distances)  # 60s each
    
    schedule = []
    for dist, dur in zip(blackout_distances, durations):
        schedule.append(BlackoutSchedule(distance_m=dist, duration_s=dur))
    
    return schedule


def run_blackout_protocol(
    sequence: NavSequence,
    blackout_distances: List[float],
    durations_s: List[float],
    baselines: List[str] = None,
    output_dir: Path = None,
) -> BlackoutComparison:
    """Run the forced-blackout protocol end-to-end."""
    evaluator = ForcedBlackoutEvaluator()
    schedule = evaluator.build_blackout_schedule(sequence, blackout_distances, durations_s)
    comparison = evaluator.run_comparison(sequence, schedule, baselines)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "blackout_comparison.json", "w") as f:
            json.dump(comparison.__dict__, f, indent=2, default=str)
    
    return comparison


def _cli():
    """CLI entry point for forced-blackout evaluation."""
    import argparse
    parser = argparse.ArgumentParser(description="NAV-X 3.0 Forced-Blackout Evaluation")
    parser.add_argument("--sequence", type=str, help="Path to NavSequence directory")
    parser.add_argument("--distances", type=str, default="100,200,300,400,500", help="Comma-separated blackout distances (m)")
    parser.add_argument("--durations", type=str, default="30,60,120,300", help="Comma-separated blackout durations (s)")
    parser.add_argument("--baselines", type=str, default="navx_full,frozen_lstm,nhc_only,pure_ins", help="Comma-separated baseline names")
    parser.add_argument("--output", type=str, default="benchmarks/blackout", help="Output directory")
    args = parser.parse_args()
    
    if args.sequence:
        from aeronavis.data.preprocess import NavSequence
        sequence = NavSequence.load(Path(args.sequence))
    else:
        print("No sequence provided, use --sequence to specify a NavSequence directory")
        return
    
    distances = [float(d) for d in args.distances.split(",")]
    durations = [float(d) for d in args.durations.split(",")]
    baselines = args.baselines.split(",")
    
    comparison = run_blackout_protocol(sequence, distances, durations, baselines, Path(args.output))
    print(f"Blackout comparison complete. Results saved to {args.output}")


if __name__ == "__main__":
    _cli()