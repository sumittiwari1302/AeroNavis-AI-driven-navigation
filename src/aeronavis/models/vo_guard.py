"""VO integrity guard: decides when visual odometry measurement is trustworthy.

Contract for Part 7 (fusion):
    decide(gate, motion_residual, speed_mps) -> bool
"""

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VOGuardConfig:
    """Thresholds for VO integrity guard."""

    # Texture gate thresholds (from texture_gate.py: 0=rich, 1=medium, 2=poor)
    gate_poor: int = 2
    # Motion residual threshold (reprojection error in pixels after 2-point RANSAC)
    motion_residual_thresh: float = 2.0
    # Minimum speed for reliable scale estimation
    min_speed_mps: float = 0.5
    # Maximum consecutive poor frames before forcing re-init
    max_poor_streak: int = 10


def make_vo_guard(config: VOGuardConfig) -> callable:
    """Factory returning the decide function."""

    poor_streak = 0

    def decide(
        gate: int,
        motion_residual: float,
        speed_mps: float,
    ) -> bool:
        """
        Decide if VO measurement can be used.

        Args:
            gate: texture gate class (0=rich, 1=medium, 2=poor)
            motion_residual: reprojection error after 2-point RANSAC (pixels)
            speed_mps: current forward speed (m/s)

        Returns:
            True if VO measurement may be used by filter
        """
        nonlocal poor_streak

        # Rule 1: Poor texture -> reject
        if gate >= config.gate_poor:
            poor_streak += 1
            logger.debug(f"VO guard: gate={gate} (poor) -> reject, streak={poor_streak}")
            return False

        # Rule 2: High motion residual -> reject
        if motion_residual > config.motion_residual_thresh:
            logger.debug(
                f"VO guard: motion_residual={motion_residual:.2f} > "
                f"{config.motion_residual_thresh} -> reject"
            )
            return False

        # Rule 3: Too slow for scale estimation -> reject
        if speed_mps < config.min_speed_mps:
            logger.debug(f"VO guard: speed={speed_mps:.2f} < " f"{config.min_speed_mps} -> reject")
            return False

        # Accept
        poor_streak = 0
        logger.debug(
            f"VO guard: gate={gate}, residual={motion_residual:.2f}, "
            f"speed={speed_mps:.2f} -> accept"
        )
        return True

    def reset():
        nonlocal poor_streak
        poor_streak = 0

    return decide, reset


def build_vo_guard(config) -> tuple[callable, callable]:
    """Build VO guard from global config."""
    guard_config = VOGuardConfig(
        motion_residual_thresh=2.0,
        min_speed_mps=0.5,
    )
    return make_vo_guard(guard_config)


if __name__ == "__main__":
    # Quick test
    config = VOGuardConfig()
    decide, reset = make_vo_guard(config)

    # Test cases
    assert decide(0, 1.0, 5.0) is True  # rich, low residual, fast -> accept
    assert decide(1, 1.0, 5.0) is True  # medium, low residual, fast -> accept
    assert not decide(2, 1.0, 5.0)  # poor -> reject
    assert not decide(0, 3.0, 5.0)  # high residual -> reject
    assert not decide(0, 1.0, 0.3)  # too slow -> reject
    print("VO guard tests passed!")
