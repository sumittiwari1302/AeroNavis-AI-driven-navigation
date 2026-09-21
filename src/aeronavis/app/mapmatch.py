"""Map-matching using Hidden-Markov Viterbi algorithm.

Takes the fused pose from InEKF + covariance and snaps to road segments
using Hidden-Markov Viterbi over candidate road segments.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np

from aeronavis.app.maps import MapStore, MapSegment
from aeronavis.config import get_config, Config

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of map-matching for a single pose."""

    pose_snapped: bool
    x: float
    y: float
    yaw: float
    road_segment_id: Optional[int]
    conf: float


@dataclass
class CandidateState:
    """Candidate state in Viterbi algorithm."""

    segment_id: int
    x: float
    y: float
    yaw: float
    emission_prob: float


class MapMatcher:
    """Hidden-Markov Viterbi map-matching for pose-to-road snapping."""

    def __init__(self, config: Config, map_store: "MapStore"):
        self.config = config
        self.map_store = map_store
        self.mm_cfg = config.mapmatch

        # Parameters from config
        self.candidate_radius = self.mapmatch_cfg.candidate_radius
        self.viterbi_window = self.mapmatch_cfg.viterbi_window
        self.emission_gps_weight = self.mapmatch_cfg.emission_gps_weight
        self.emission_heading_weight = self.mapmatch_cfg.emission_heading_weight
        self.transition_speed_weight = self.mapmatch_cfg.transition_speed_weight
        self.gnss_health_threshold = self.mapmatch_cfg.gnss_health_threshold

        # Internal state
        self.viterbi_states: List[List[Dict]] = []  # Viterbi trellis
        self.last_snapped_segment: Optional[int] = None

    @property
    def mapmatch_cfg(self):
        return self.config.mapmatch

    def find_candidates(self, x: float, y: float, yaw: float, radius: float = None) -> List[tuple]:
        """
        Find candidate road segments within radius.
        Returns list of (segment_id, distance, bearing_diff).
        """
        if not self.map_store._rtree_idx:
            self.map_store.build_rtree()

        radius = self.candidate_radius
        segment_ids = self.map_store.query_segments_nearby(x, y, radius)
        candidates = []

        for seg_id in segment_ids:
            seg = self.map_store.get_segment(seg_id)
            if not seg:
                continue

            # Get segment midpoint
            n1 = self.map_store.get_node(seg.n1)
            n2 = self.map_store.get_node(seg.n2)
            if not n1 or not n2:
                continue

            mid_lat = (n1.lat + n2.lat) / 2
            mid_lon = (n1.lon + n2.lon) / 2

            # Distance from pose to segment midpoint
            dist = haversine_m(x, y, mid_lat, mid_lon)

            # Bearing difference
            bearing_diff = abs(yaw - seg.bearing_deg)
            bearing_diff = min(bearing_diff, 360 - bearing_diff)

            candidates.append((seg.id, dist, bearing_diff))

        # Sort by distance
        candidates.sort(key=lambda x: x[1])
        return candidates[:20]  # Limit candidates

    def emission_probability(
        self,
        gps_x: float,
        gps_y: float,
        gps_yaw: float,
        seg: "MapSegment",
        gnss_health: float,
        gps_cov: np.ndarray = None,
    ) -> float:
        """
        Compute emission probability for a segment given GNSS observation.
        Higher probability = better match.
        """
        # Get segment midpoint
        n1 = self.map_store.get_node(seg.n1)
        n2 = self.map_store.get_node(seg.n2)
        if not n1 or not n2:
            return 0.0

        mid_lat = (n1.lat + n2.lat) / 2
        mid_lon = (n1.lon + n2.lon) / 2

        # Distance error (meters)
        dist = haversine_m(gps_x, gps_y, mid_lat, mid_lon)

        # Heading error (degrees)
        heading_diff = abs(gps_yaw - seg.bearing_deg)
        heading_diff = min(heading_diff, 360 - heading_diff)

        # GPS emission: Gaussian in distance and heading
        # Standard deviations from config
        pos_std = self.config.fusion.gnss_pos_std
        yaw_std = np.deg2rad(10.0)  # 10 deg std

        # Distance likelihood
        dist_prob = np.exp(-0.5 * (dist / pos_std) ** 2)

        # Heading likelihood
        yaw_diff_rad = np.deg2rad(
            min(abs(gps_yaw - seg.bearing_deg), 360 - abs(gps_yaw - seg.bearing_deg))
        )
        yaw_std = np.deg2rad(10.0)
        heading_prob = np.exp(-0.5 * (yaw_diff_rad / yaw_std) ** 2)

        # Combine with weights
        emission = (
            self.emission_gps_weight * dist_prob + self.emission_heading_weight * heading_prob
        ) / (self.emission_gps_weight + self.emission_heading_weight)

        # If GNSS health is low, reduce GPS influence
        if gnss_health < 0.5:
            emission = emission * gnss_health + 0.1 * (1 - gnss_health)

        return max(emission, 1e-6)

    def transition_probability(
        self, seg_from: "MapSegment", seg_to: "MapSegment", speed: float, dt: float
    ) -> float:
        """
        Transition probability between two segments based on
        physical connectivity and speed consistency.
        """
        # Check connectivity
        if (
            seg_from.n2 != seg_to.n1
            and seg_from.n1 != seg_to.n2
            and seg_from.n1 != seg_to.n1
            and seg_from.n2 != seg_to.n2
        ):
            return 1e-6  # Not connected

        # Speed consistency: expected distance = speed * dt
        n1 = self.map_store.get_node(seg_from.n2)
        n2 = self.map_store.get_node(seg_to.n1)
        if not n1 or not n2:
            return 0.001

        expected_dist = seg_from.length_m  # approximate
        expected_speed = expected_dist / 1.0  # per second
        speed_diff = abs(speed - expected_speed)

        # Gaussian on speed difference
        speed_prob = np.exp(-0.5 * (speed_diff / 5.0) ** 2)

        return max(speed_prob, 1e-6)

    def viterbi_step(
        self,
        prev_states: List[Dict],
        candidates: List[tuple],
        x: float,
        y: float,
        yaw: float,
        speed: float,
        gnss_health: float,
        dt: float,
        gps_cov: np.ndarray = None,
    ) -> List[Dict]:
        """
        One Viterbi step: compute best path to each candidate.
        """
        new_states = []

        for seg_id, dist, bearing_diff in candidates:
            seg = self.map_store.get_segment(seg_id)
            if not seg:
                continue

            # Emission probability
            emission = self.emission_probability(
                x, y, yaw, self.map_store.get_segment(seg_id), 1.0, None
            )

            # Find best predecessor
            best_score = -np.inf
            best_prev_id = None

            for prev in prev_states:
                prev_seg = self.map_store.get_segment(prev["segment_id"])
                if not prev_seg:
                    continue

                trans_prob = self.transition_probability(
                    self.map_store.get_segment(prev["segment_id"]),
                    self.map_store.get_segment(seg_id),
                    prev.get("speed", 10.0),
                    1.0,
                )

                score = (
                    prev["log_prob"] + np.log(max(emission, 1e-6)) + np.log(max(trans_prob, 1e-6))
                )

                if score > best_score:
                    best_score = score
                    best_prev_id = prev["segment_id"]

            new_state = {
                "segment_id": seg_id,
                "x": x,
                "y": y,
                "yaw": yaw,
                "speed": 10.0,  # placeholder
                "log_prob": best_score,
                "prev_id": best_prev_id,
            }
            new_states.append(new_state)

        # Normalize probabilities (log-space)
        if new_states:
            max_log = max(s["log_prob"] for s in new_states)
            for s in new_states:
                s["log_prob"] -= max_log

        return new_states

    def match(
        self,
        x: float,
        y: float,
        yaw: float,
        speed: float,
        gnss_health: float,
        dt: float,
        gnss_pos: Optional[np.ndarray] = None,
        gnss_cov: Optional[np.ndarray] = None,
    ) -> MatchResult:
        """
        Main entry point: match a single pose to the road network.
        Uses Viterbi over sliding window.
        """
        # Check GNSS health for snapping decision
        gnss_healthy = gnss_health >= self.gnss_health_threshold

        # Find candidates
        candidates = self.find_candidates(x, y, yaw)

        if not candidates:
            return MatchResult(
                pose_snapped=False, x=x, y=y, yaw=yaw, road_segment_id=None, conf=0.0
            )

        # Initialize or extend Viterbi
        if not self.viterbi_states:
            # First observation
            for seg_id, dist, bearing_diff in candidates:
                self.map_store.get_segment(seg_id)
                emission = self.emission_probability(
                    x, y, yaw, self.map_store.get_segment(seg_id), 1.0
                )
                self.viterbi_states.append(
                    [
                        {
                            "segment_id": seg_id,
                            "log_prob": np.log(max(emission, 1e-6)),
                            "prev_id": None,
                        }
                    ]
                )
        else:
            # Viterbi step
            new_states = self.viterbi_step(
                self.viterbi_states[-1],
                [(seg_id, dist, bd) for seg_id, dist, bd in candidates],
                x,
                y,
                yaw,
                10.0,
                1.0,
                1.0,
            )
            self.viterbi_states.append(new_states)

            # Limit window
            if len(self.viterbi_states) > self.viterbi_window:
                self.viterbi_states.pop(0)

        # Find best current state
        if not self.viterbi_states:
            return MatchResult(
                pose_snapped=False, x=x, y=y, yaw=yaw, road_segment_id=None, conf=0.0
            )

        best_state = max(self.viterbi_states[-1], key=lambda s: s["log_prob"])
        best_seg = self.map_store.get_segment(best_state["segment_id"])

        # Get snapped position (segment midpoint)
        n1 = self.map_store.get_node(best_seg.n1)
        n2 = self.map_store.get_node(best_seg.n2)
        if n1 and n2:
            snapped_x = (n1.lat + n2.lat) / 2
            snapped_y = (n1.lon + n2.lon) / 2
            snapped_yaw = best_seg.bearing_deg
        else:
            snapped_x, snapped_y, snapped_yaw = 0, 0, 0

        # Confidence from log-prob
        max_log = max(s["log_prob"] for s in self.viterbi_states[-1])
        conf = np.exp(best_state["log_prob"] - max_log) if max_log > -np.inf else 0.0

        # Snapping decision
        gnss_healthy = True  # placeholder
        snapped = gnss_healthy and self.gnss_health_threshold < 1.0

        return MatchResult(
            pose_snapped=snapped,
            x=snapped_x if snapped else x,
            y=snapped_y if snapped else y,
            yaw=snapped_yaw if snapped else yaw,
            road_segment_id=best_state["segment_id"],
            conf=float(conf),
        )

    def reset(self):
        """Reset Viterbi state."""
        self.viterbi_states = []
        self.last_snapped_segment = None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = np.deg2rad(lat2 - lat1)
    np.deg2rad(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.deg2rad(lat1))
        * np.cos(np.deg2rad(lat2))
        * np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371000.0 * c


def build_map_matcher(config: Config = None, map_store=None):
    """Factory function to create MapMatcher."""
    config = config or get_config()
    return MapMatcher(config, map_store)
