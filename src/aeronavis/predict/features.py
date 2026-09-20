"""Feature extractor for GNSS outage prediction.

Per-second feature vector from GNSS + IMU + barometer + optional map context.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from aeronavis.config import Config
from aeronavis.data.preprocess import NavSequence

logger = logging.getLogger(__name__)


@dataclass
class PredictFeatureConfig:
    """Configuration for feature extraction."""

    max_satellites: int = 32
    sequence_length: int = 30  # 30 seconds
    feature_dim: int = 104  # 32*3 + 8


class GNSSFeatureExtractor:
    """Extract per-second features for GNSS outage prediction."""

    def __init__(self, config: PredictFeatureConfig):
        self.config = config
        self.max_sats = config.max_satellites
        self.seq_len = config.sequence_length

    def extract_sequence(
        self,
        seq: NavSequence,
        t_start: float,
        t_end: float,
    ) -> np.ndarray:
        """
        Extract feature sequence from t_start to t_end (seconds).

        Returns:
            np.ndarray of shape (T, F) where T = sequence_length, F = feature_dim
        """
        # For now, return synthetic features since we don't have full GNSS data structure
        # In production, this would extract from actual GNSS data in seq
        T = self.config.sequence_length
        F = self.config.feature_dim
        return np.random.randn(T, F).astype(np.float32)

    def extract_satellite_features(self, gnss_df: pd.DataFrame, t: float) -> np.ndarray:
        """
        Extract per-satellite features at time t.

        Returns:
            (max_sats, 3) array: [cno_db_hz, elev_deg, azim_deg] per satellite
        """
        # Pad to max_satellites
        sats = gnss_df[gnss_df["ts"] == t]
        n = len(sats)
        feats = np.full((self.max_sats, 3), -1.0, dtype=np.float32)

        if n > 0:
            idx = min(n, self.max_sats)
            sats = sats.iloc[:idx]
            feats[:idx, 0] = sats.get("cno_db_hz", sats.get("snr", 0)).values
            feats[:idx, 1] = sats.get("elev_deg", sats.get("elevation", 0)).values
            feats[:idx, 2] = sats.get("azim_deg", sats.get("azimuth", 0)).values

        return feats

    def extract_global_features(self, seq: NavSequence, t: float) -> np.ndarray:
        """
        Extract global features at time t.

        Returns:
            8-dim array: [n_sats, pdop, hdop, gnss_speed, gnss_heading,
                          imu_speed, baro_trend, fix_quality]
        """
        feats = np.zeros(8, dtype=np.float32)

        # GNSS features
        if seq.gnss is not None and len(seq.gnss) > 0:
            gnss_t = seq.gnss[seq.gnss["ts"] <= t]
            if len(gnss_t) > 0:
                latest = gnss_t.iloc[-1]
                feats[0] = latest.get("n_sats", 0)  # n_sats
                feats[1] = latest.get("pdop", 0)  # pdop
                feats[2] = latest.get("hdop", 0)  # hdop
                feats[3] = latest.get("speed_mps", 0)  # gnss_speed
                feats[4] = latest.get("heading_deg", 0)  # gnss_heading
                feats[7] = latest.get("fix_quality", 0)  # fix_quality

        # IMU speed estimate
        if seq.imu is not None and len(seq.imu) > 0:
            imu_t = seq.imu[seq.imu["ts"] <= t]
            if len(imu_t) > 1:
                # Estimate speed from IMU (simplified)
                acc = imu_t[["acc_x", "acc_y", "acc_z"]].values
                acc_mag = np.linalg.norm(acc, axis=1)
                feats[5] = float(np.mean(acc_mag[-10:]) * 0.01)  # rough estimate

        # Barometer trend
        if seq.baro is not None and len(seq.baro) > 1:
            baro_t = seq.baro[seq.baro["ts"] <= t]
            if len(baro_t) >= 2:
                dt = baro_t["ts"].diff().iloc[-1]
                dp = baro_t["pressure_hpa"].diff().iloc[-1]
                if dt > 0:
                    feats[6] = float(dp / dt)  # baro_trend (hPa/s)

        return feats

    def extract_map_context(self, seq: NavSequence, t: float) -> np.ndarray:
        """
        Extract optional map context features.

        Returns:
            3-dim array: [tunnel_flag, valley_flag, building_density]
        """
        feats = np.zeros(3, dtype=np.float32)

        if seq.metadata and "map_context" in seq.metadata:
            ctx = seq.metadata["map_context"]
            feats[0] = float(ctx.get("tunnel_flag", 0))
            feats[1] = float(ctx.get("valley_flag", 0))
            feats[2] = float(ctx.get("building_density", 0.0))

        return feats

    def build_feature_vector(self, seq: NavSequence, t: float) -> np.ndarray:
        """
        Build complete feature vector at time t.

        Returns:
            (feature_dim,) array: concat(satellite_feats, global_feats, map_feats)
        """
        sat_feats = self.extract_satellite_features(
            seq.gnss if seq.gnss is not None else pd.DataFrame(), t
        )
        global_feats = self.extract_global_features(seq, t)
        map_feats = self.extract_map_context(seq, t)

        return np.concatenate([sat_feats.flatten(), global_feats, map_feats])

    def extract_sequence_features(
        self,
        seq: NavSequence,
        t_start: float,
        t_end: float,
    ) -> np.ndarray:
        """
        Extract feature sequence from t_start to t_end.

        Returns:
            (T, F) array where T = sequence_length, F = feature_dim
        """
        T = self.config.sequence_length
        feats = np.zeros((T, self.config.feature_dim), dtype=np.float32)

        for i, t in enumerate(np.linspace(t_start, t_end, T)):
            if i >= T:
                break
            feats[i] = self.build_feature_vector(seq, t)

        return feats


def extract_sequence(seq: NavSequence, t_start: float, t_end: float) -> np.ndarray:
    """Convenience function to extract features from a sequence."""
    config = PredictFeatureConfig()
    extractor = GNSSFeatureExtractor(config)
    return extractor.extract_sequence_features(seq, t_start, t_end)


def build_predict_feature_extractor(config: Config) -> GNSSFeatureExtractor:
    """Factory to build feature extractor from global config."""
    feature_config = PredictFeatureConfig(
        max_satellites=32,
        sequence_length=30,
        feature_dim=104,
    )
    return GNSSFeatureExtractor(feature_config)
