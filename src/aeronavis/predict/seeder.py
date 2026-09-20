"""Outage seeder: manages filter trust transitions based on outage predictions."""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np

from aeronavis.config import Config

logger = logging.getLogger(__name__)


@dataclass
class SeederConfig:
    """Configuration for the outage seeder."""

    threshold: float = 0.85
    horizons: List[int] = None
    handover_window_s: float = 5.0
    covariance_ramp_factor: float = 2.0
    max_covariance_factor: float = 8.0
    reseed_jump_max_m: float = 0.3

    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [5, 10, 15]


@dataclass
class NavEvent:
    """Navigation event emitted by the seeder."""

    event_type: str  # "PREDICT_UPCOMING", "ENTER_OUTAGE", "EXIT_OUTAGE", "HANDOVER_COMPLETE"
    timestamp: float
    horizon: Optional[int] = None
    probability: float = 0.0
    covariance_factor: float = 1.0
    anchor_pose: Optional[Dict[str, float]] = None


class NavState:
    """Current navigation state passed to seeder."""

    def __init__(
        self,
        timestamp: float,
        position: np.ndarray,  # (3,) ENU
        velocity: np.ndarray,  # (3,) ENU
        orientation: np.ndarray,  # (3,) quaternion or euler
        gnss_ok: bool = True,
        gnss_covariance: Optional[np.ndarray] = None,
    ):
        self.timestamp = timestamp
        self.position = position
        self.velocity = velocity
        self.orientation = orientation
        self.gnss_ok = gnss_ok
        self.gnss_covariance = gnss_covariance


class OutageSeeder:
    """Manages filter trust transitions based on outage predictions."""

    def __init__(
        self,
        config: Config,
        threshold: float = 0.85,
        horizons: Optional[List[int]] = None,
    ):
        self.config = config
        self.seeder_cfg = self._build_seeder_config(config)
        self.threshold = threshold

        # State
        self._outage_active = False
        self._outage_start_time = None
        self._prediction_history = []  # List of (timestamp, probs)
        self._last_good_pose = None
        self._covariance_factor = 1.0
        self._handover_active = False
        self._handover_start = None

    def _build_seeder_config(self, config: Config) -> "SeederConfig":
        # Extract seeder params from config
        return SeederConfig(
            threshold=config.predict.get("threshold", 0.85)
            if hasattr(config.predict, "get")
            else 0.85,
            horizons=[5, 10, 15],
            handover_window_s=config.fusion.get("reseed_jump_max_m", 5.0)
            if hasattr(config.fusion, "get")
            else 5.0,
            covariance_ramp_factor=2.0,
            max_covariance_factor=8.0,
            reseed_jump_max_m=config.fusion.get("reseed_jump_max_m", 0.3)
            if hasattr(config.fusion, "get")
            else 0.3,
        )

    def update(
        self,
        probs: Dict[int, float],  # {horizon: probability}
        now_nav: NavState,
    ) -> List[NavEvent]:
        """
        Update seeder with latest predictions and navigation state.

        Returns:
            List of NavEvent to be consumed by the fusion filter.
        """
        events = []
        now = now_nav.timestamp

        # Store prediction
        self._prediction_history.append((now_nav.timestamp, probs))
        # Keep last 60 seconds
        self._prediction_history = [(t, p) for t, p in self._prediction_history if t >= now - 60]

        # Check if any horizon exceeds threshold
        max_prob = max(probs.values())
        any_upcoming = any(p >= self.threshold for p in probs.values())

        if any_upcoming and not self._outage_active:
            # PREDICT_UPCOMING: start covariance inflation
            events.append(
                NavEvent(
                    event_type="PREDICT_UPCOMING",
                    timestamp=now_nav.timestamp,
                    probability=max_prob,
                    covariance_factor=self._covariance_factor,
                )
            )
            self._outage_active = True
            self._outage_start_time = now
            self._last_good_pose = {
                "position": now_nav.position.copy(),
                "orientation": now_nav.orientation.copy(),
                "timestamp": now,
            }
            logger.info(f"PREDICT_UPCOMING at {now:.1f}s, max_prob={max_prob:.2f}")

        elif self._outage_active and not now_nav.gnss_ok:
            # ENTER_OUTAGE: confirm outage entered
            events.append(
                NavEvent(
                    event_type="ENTER_OUTAGE",
                    timestamp=now_nav.timestamp,
                )
            )
            logger.info(f"ENTER_OUTAGE at {now:.1f}s")

        elif self._outage_active and now_nav.gnss_ok:
            # EXIT_OUTAGE: GNSS reacquired
            # Check if prediction has settled (probabilities dropped)
            recent_probs = [p for t, p in self._prediction_history if t >= now - 3]
            if recent_probs:
                avg_max = np.mean([max(p.values()) for _, p in recent_probs])
                if avg_max < self.threshold * 0.5:
                    events.append(
                        NavEvent(
                            event_type="EXIT_OUTAGE",
                            timestamp=now_nav.timestamp,
                            anchor_pose=self._last_good_pose,
                        )
                    )
                    self._outage_active = False
                    self._outage_start_time = None
                    self._handover_active = True
                    self._handover_start = now
                    logger.info(f"EXIT_OUTAGE at {now:.1f}s, starting handover")

        # Handover logic: smooth covariance inflation/deflation
        if self._outage_active and not self._handover_active:
            # In outage, ramp up covariance
            self._covariance_factor = min(
                self._covariance_factor * 1.1,
                self.config.max_covariance_factor
                if hasattr(self, "config") and hasattr(self.config, "max_covariance_factor")
                else 8.0,
            )
        elif self._handover_active:
            # Handover: blend toward re-acquired GNSS
            elapsed = now - self._handover_start
            window = self._get_handover_window()
            if elapsed >= window:
                events.append(
                    NavEvent(
                        event_type="HANDOVER_COMPLETE",
                        timestamp=now_nav.timestamp,
                    )
                )
                self._handover_active = False
                self._covariance_factor = 1.0
                logger.info(f"HANDOVER_COMPLETE at {now:.1f}s")
            else:
                # Blend factor
                alpha = elapsed / self._get_handover_window()
                self._covariance_factor = 1.0 + (self._covariance_factor - 1.0) * (1 - alpha)

        return events

    def _get_handover_window(self) -> float:
        return 5.0  # config.fusion.reseed_jump_max_m / speed... simplified

    def get_covariance_factor(self) -> float:
        """Current GNSS covariance inflation factor."""
        return self._covariance_factor

    def get_last_good_pose(self) -> Optional[Dict]:
        return self._last_good_pose

    def is_outage_active(self) -> bool:
        return self._outage_active

    def is_handover_active(self) -> bool:
        return self._handover_active

    def get_prediction_history(self) -> List[tuple]:
        return self._prediction_history

    def reset(self):
        """Reset seeder state."""
        self._outage_active = False
        self._outage_start_time = None
        self._prediction_history = []
        self._last_good_pose = None
        self._covariance_factor = 1.0
        self._handover_active = False
        self._handover_start = None


def build_outage_seeder(config: Config) -> "OutageSeeder":
    """Factory to build seeder from global config."""
    return OutageSeeder(config)
