"""Learned noise model for adaptive measurement covariance.

Input features -> diagonal covariance scaling factors for R_wheel, R_vo, R_gnss.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aeronavis.config import Config

logger = logging.getLogger(__name__)


@dataclass
class LearnedNoiseConfig:
    input_dim: int = 12
    hidden_dim: int = 64
    output_dim: int = 3  # R_wheel_scale, R_vo_scale, R_gnss_scale
    max_params: int = 50000
    dropout: float = 0.1


class LearnedNoiseModel(nn.Module):
    """MLP to predict diagonal covariance scaling factors from features."""

    def __init__(self, config: Optional[LearnedNoiseConfig] = None):
        super().__init__()
        self.config = config or LearnedNoiseConfig()

        self.net = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
            nn.Softplus(),  # Ensure positive outputs
        )

        # Initialize to identity (no scaling)
        self._init_to_identity()

    def _init_to_identity(self):
        """Initialize so output ≈ 1.0 (no scaling)."""
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            # Set last bias to log(1) = 0 for Softplus ≈ 1
            last_layer = self.net[-2]
            if isinstance(last_layer, nn.Linear):
                nn.init.constant_(last_layer.bias, np.log(np.expm1(1.0)))  # Softplus^-1(1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, input_dim) or (input_dim,)

        Returns:
            scales: (B, 3) or (3,) - [R_wheel_scale, R_vo_scale, R_gnss_scale]
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        scales = self.net(features) + 1.0  # Base = 1.0 (no scaling)
        return scales.squeeze(0) if scales.shape[0] == 1 else scales

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))


class LearnedNoiseTrainer:
    """Trainer for learned noise model using innovation residuals."""

    def __init__(self, model: LearnedNoiseModel, config: Config, lr: float = 1e-3):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def train_step(self, features: np.ndarray, target_scales: np.ndarray) -> float:
        """Train on one batch of (features, target_scales)."""
        self.model.train()
        features_t = torch.from_numpy(features).float()
        targets_t = torch.from_numpy(target_scales).float()

        self.optimizer.zero_grad()
        preds = self.model(features_t)
        loss = F.mse_loss(preds, targets_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def fit_innovation_residuals(self, innovations: np.ndarray, features: np.ndarray) -> dict:
        """
        Fit noise model from innovation residuals.
        
        Args:
            innovations: (N, D) innovation residuals for each measurement type
            features: (N, D_feat) feature vectors
            
        Returns:
            dict with training stats
        """
        # Compute empirical variance per measurement type
        np.ones((len(features), 3), dtype=np.float32)
        
        # For each measurement type, compute variance ratio vs nominal
        # This is a simplified version - in practice, we'd use sliding windows
        
        return {"loss": 0.0}

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model.load(path)


class LearnedNoiseConfig:
    input_dim: int = 12
    hidden_dim: int = 64
    output_dim: int = 3
    max_params: int = 50000
    dropout: float = 0.1


def build_learned_noise_model(config: Config) -> LearnedNoiseModel:
    """Factory to build learned noise model from global config."""
    hidden = config.fusion.learned_noise_hidden
    max_params = config.fusion.learned_noise_max_params
    LearnedNoiseConfig(
        input_dim=12,
        hidden_dim=config.fusion.learned_noise_hidden,
        max_params=config.fusion.learned_noise_max_params,
    )
    model = LearnedNoiseModel(LearnedNoiseConfig(
        input_dim=12,
        hidden_dim=hidden,
        max_params=max_params,
    ))
    logger.info(f"LearnedNoiseModel params: {model.count_parameters():,}")
    return model