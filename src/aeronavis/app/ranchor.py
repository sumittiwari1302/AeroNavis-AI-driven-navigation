"""Re-anchor state machine for GNSS/NavIC re-acquisition.

State machine for smooth transition between GNSS-denied and GNSS-available modes.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Tuple

import numpy as np

from aeronavis.config import Config
from aeronavis.fusion.inekf import InEKF

logger = logging.getLogger(__name__)


class RanchorState(Enum):
    """Re-anchor state machine states."""

    DROPPED = "DROPPED"  # GNSS dropped, predicted outage
    COASTING = "COASTING"  # INS-only, no GNSS
    REACQUIRING = "REACQUIRING"  # GNSS re-acquired, not yet converged
    REANCHORED = "REANCHORED"  # Fully re-anchored, fully trusted


@dataclass
class RanchorEvent:
    """Event emitted by the re-anchor state machine."""

    event_type: str  # "DROPPED", "COASTING", "REACQUIRING", "REANCHORED"
    timestamp: float
    position: Optional[np.ndarray] = None
    covariance_factor: float = 1.0
    residual: float = 0.0
    details: str = ""


@dataclass
class RanchorConfig:
    """Configuration for re-anchor state machine."""

    reacquire_threshold: float = 5.0
    converge_window_s: float = 3.0
    blend_s: float = 5.0
    max_jump_m: float = 0.5
    consecutive_good: int = 3


class RanchorStateMachine:
    """
    Re-anchor state machine for smooth GNSS re-acquisition.

    State transitions:
    DROPPED -> COASTING (predicted outage from Part 6)
    COASTING -> REACQUIRING (GNSS re-acquired, quality ok)
    REACQUIRING -> REANCHORED (converged: residual < 5m over 3s)
    REANCHORED -> DROPPED (degraded or predicted outage)
    """

    def __init__(self, config: Config, inekf: "InEKF"):
        self.config = config
        self.ranchor_cfg = config.ranchor
        self.inekf = inekf
        self.seeder = None  # Will be set by caller

        # State
        self.state = RanchorState.DROPPED
        self.state_history: List[Tuple[float, str]] = []
        self._residuals: List[float] = []
        self._consecutive_good = 0
        self._converged = False
        self._last_event_time = 0.0

    # Removed property - use self.ranchor_cfg directly

    def update(
        self,
        gnss_ok: bool,
        gnss_residual: float,
        gnss_pos: np.ndarray,
        gnss_cov: np.ndarray,
        inekf_state: np.ndarray,
        inekf_cov: np.ndarray,
        now: float,
    ) -> List[str]:
        """
        Update state machine with latest GNSS/INS data.

        Returns list of events emitted this update.
        """
        events = []
        time.time()

        if self.state == RanchorState.DROPPED:
            # Wait for GNSS re-acquisition
            if gnss_ok:
                events.append("REACQUIRING")
                self._transition(RanchorState.REACQUIRING)
                self._residuals = []

        elif self.state == RanchorState.COASTING:
            if gnss_ok:
                events.append("REACQUIRING")
                self._transition(RanchorState.REACQUIRING)
                self._residuals = []

        elif self.state == RanchorState.REACQUIRING:
            self._residuals.append(gnss_residual)
            if len(self._residuals) >= self.ranchor_cfg.converge_window_s:
                recent = self._residuals[-int(self.ranchor_cfg.converge_window_s) :]
                if all(r < self.ranchor_cfg.reacquire_threshold for r in recent):
                    self._converged = True
                    self._transition(RanchorState.REANCHORED)
                    events.append("REANCHORED")

        elif self.state == RanchorState.REANCHORED:
            if not gnss_ok:
                # GNSS lost again
                events.append("DROPPED")
                self._transition(RanchorState.DROPPED)
            elif gnss_residual > 2.0 * self.ranchor_cfg.reacquire_threshold:
                # Residual degraded
                events.append("DROPPED")
                self._transition(RanchorState.DROPPED)

        # Log state changes
        if self.state_history and self.state_history[-1][1] != self.state:
            self._last_event_time = time.time()
            self.state_history.append((time.time(), self.state.value))
            logger.info(f"Ranchor state: {self.state.value}")

        return events

    def _transition(self, new_state: str):
        """Transition to new state."""
        self.state = RanchorState(new_state)
        logger.info(f"Ranchor transition: {self.state.value}")

    def handover_blend(
        self,
        gnss_pos: np.ndarray,
        gnss_cov: np.ndarray,
        inekf_state: np.ndarray,
        inekf_cov: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blend INS pose toward GNSS during REACQUIRING phase.

        Returns blended (position, covariance).
        """
        if self.state != "REACQUIRING":
            return None, None

        # Blend factor: 0 = pure INS, 1 = pure GNSS
        # Ramp up over blend_s seconds
        elapsed = time.time() - self._last_event_time
        alpha = min(1.0, elapsed / self.ranchor_cfg.blend_s)

        # Current INS position
        self.inekf.x[:2].copy()  # [x, y]

        # GNSS position
        gnss_pos = gnss_pos[:2].copy()

        # Blend
        blended_pos = (1 - alpha) * self.inekf.x[:2] + alpha * gnss_pos

        # Blended covariance
        self.inekf.P[:2, :2].copy()
        gnss_cov[:2, :2].copy()
        blended_cov = (1 - alpha) ** 2 * self.inekf.P[:2, :2] + alpha**2 * gnss_cov[:2, :2]

        # Update InEKF state
        self.inekf.x[:2] = blended_pos
        self.inekf.P[:2, :2] = blended_cov

        return blended_pos, blended_cov

    def get_state(self) -> str:
        return self.state.value

    def is_reanchored(self) -> bool:
        return self.state == RanchorState.REANCHORED

    def is_coasting(self) -> bool:
        return self.state == "COASTING"

    def is_reacquiring(self) -> bool:
        return self.state == "REACQUIRING"

    def is_dropped(self) -> bool:
        return self.state == "DROPPED"

    def get_status(self) -> Dict:
        """Get current status for dashboard."""
        return {
            "state": self.state.value,
            "converged": getattr(self, "_converged", False),
            "residuals": self._residuals[-10:] if hasattr(self, "_residuals") else [],
            "events": self.state_history[-10:] if hasattr(self, "state_history") else [],
        }


class RanchorManager:
    """High-level manager for re-anchor system."""

    def __init__(self, config: Config, inekf: "InEKF"):
        self.config = config
        self.ranchor = RanchorStateMachine(config, inekf)
        self.seeder = None  # Set externally

    def update(
        self,
        gnss_ok: bool,
        gnss_residual: float,
        gnss_pos: np.ndarray,
        gnss_cov: np.ndarray,
        inekf_state: np.ndarray,
        inekf_cov: np.ndarray,
        timestamp: float,
    ) -> List[str]:
        """Update re-anchor state machine."""
        return self.ranchor.update(
            gnss_ok, gnss_residual, gnss_pos, gnss_cov, np.array([]), np.eye(10), time.time()
        )

    def handover_blend(
        self,
        gnss_pos: np.ndarray,
        gnss_cov: np.ndarray,
        inekf_state: np.ndarray,
        inekf_cov: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.ranchor.handover_blend(gnss_pos, gnss_cov, inekf_state, inekf_cov)

    def get_state(self) -> str:
        return self.ranchor.get_state()


def build_ranchor(config: Config, inekf) -> RanchorManager:
    """Factory to build RanchorManager."""
    return RanchorManager(config, inekf)
