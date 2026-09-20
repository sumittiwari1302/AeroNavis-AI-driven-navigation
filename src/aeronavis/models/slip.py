"""Slip / Grip / Stationary classifier.

Shares CNN-GRU trunk with pseudo_odo (optional) + small classification head.
Classes: 0=grip, 1=slip_brake, 2=stationary.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from aeronavis.config import Config
from aeronavis.models.pseudo_odo import (
    CNNEncoder,
    GRUEncoder,
    PseudoOdoConfig,
    get_pseudo_odo_config,
    SpectrogramExtractor,
    PreNet,
)

logger = logging.getLogger(__name__)


@dataclass
class SlipConfig:
    window: int = 200
    hidden: int = 64
    max_params: int = 150_000
    num_classes: int = 3
    dropout: float = 0.1
    share_trunk: bool = True


def get_slip_config(config: Config) -> SlipConfig:
    return SlipConfig(
        window=config.model.window,
        hidden=config.model.slip_hidden,
        max_params=config.model.slip_max_params,
    )


class SlipClassifier(nn.Module):
    """Slip/grip/stationary classifier with optional shared trunk."""

    def __init__(
        self,
        config: Optional[SlipConfig] = None,
        pseudo_config: Optional[PseudoOdoConfig] = None,
        shared_trunk: Optional[nn.ModuleDict] = None,
    ):
        super().__init__()
        self.config = config or SlipConfig()
        self.pseudo_config = pseudo_config or PseudoOdoConfig()
        self.share_trunk = self.config.share_trunk

        if shared_trunk is not None:
            self.cnn = shared_trunk["cnn"]
            self.gru = shared_trunk["gru"]
            self.spec_extractor = shared_trunk["spec_extractor"]
            self.prenet = shared_trunk["prenet"]
            # Freeze or not depending on training strategy
        else:
            # Build own trunk
            self.spec_extractor = SpectrogramExtractor(
                n_bins=self.pseudo_config.spec_bins,
                fs=100.0,
                fmax=5.0,
            )
            self.prenet = PreNet(
                spec_bins=self.pseudo_config.spec_bins,
                out_dim=self.pseudo_config.prenet_out,
            )
            self.cnn = CNNEncoder(
                in_channels=6,
                channels=self.pseudo_config.cnn_channels,
                kernel=self.pseudo_config.cnn_kernel,
                dropout=self.pseudo_config.dropout,
            )
            self.gru = GRUEncoder(
                input_dim=self.pseudo_config.cnn_channels[-1],
                hidden=self.pseudo_config.gru_hidden,
                layers=self.pseudo_config.gru_layers,
                dropout=self.pseudo_config.dropout,
            )

        # Classification head
        combined_dim = self.pseudo_config.gru_hidden + self.pseudo_config.prenet_out
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, self.config.hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden, self.config.num_classes),
        )

    def forward(
        self,
        acc: torch.Tensor,
        gyr: torch.Tensor,
        baro_delta: torch.Tensor,
    ) -> torch.Tensor:
        """Returns logits for 3 classes: (B, 3)."""
        B, W, _ = acc.shape

        # Spectrogram from vertical accel
        spec = self.spec_extractor(acc[:, :, 2])

        # Pre-net (needs baro_delta - use dummy if not provided for standalone)
        if baro_delta is None:
            baro_delta = torch.zeros(B, 1, device=acc.device, dtype=acc.dtype)
        prenet_feat = self.prenet(baro_delta, spec)  # (B, 32)

        # CNN-GRU encoder
        imu = torch.cat([acc, gyr], dim=-1).transpose(1, 2)  # (B, 6, W)
        cnn_feat = self.cnn(imu)
        gru_feat = self.gru(cnn_feat)  # (B, hidden)

        # Combine and classify
        combined = torch.cat([gru_feat, prenet_feat], dim=-1)
        logits = self.classifier(combined)  # (B, 3)
        return logits

    def get_trunk_modules(self) -> dict:
        """Return trunk modules for sharing."""
        return {
            "cnn": self.cnn,
            "gru": self.gru,
            "spec_extractor": self.spec_extractor,
            "prenet": self.prenet,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_summary(self) -> str:
        total = self.count_parameters()
        lines = [
            "SlipClassifier Summary",
            f"  Total params: {total:,}",
            f"  Max allowed: {self.config.max_params:,}",
            f"  Within budget: {total <= self.config.max_params}",
            f"  Shared trunk: {self.share_trunk}",
            f"  Classes: {self.config.num_classes}",
        ]
        return "\n".join(lines)


def build_slip_model(
    config: Config,
    shared_trunk: Optional[dict] = None,
) -> SlipClassifier:
    slip_config = get_slip_config(config)
    pseudo_config = get_pseudo_odo_config(config)
    model = SlipClassifier(
        config=slip_config,
        pseudo_config=pseudo_config,
        shared_trunk=shared_trunk,
    )
    logger.info(model.model_summary())
    assert model.count_parameters() <= slip_config.max_params, (
        f"SlipClassifier has {model.count_parameters():,} params, "
        f"exceeds budget {slip_config.max_params:,}"
    )
    return model
