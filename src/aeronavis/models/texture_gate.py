"""Texture gate: 3-class CNN classifier for visual frame quality.

Classes: 0=rich, 1=medium, 2=poor (dark/textureless).
Input: 96x96 grayscale uint8 frame.
Output: 3-class softmax.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from aeronavis.config import Config

logger = logging.getLogger(__name__)


@dataclass
class TextureGateConfig:
    input_size: int = 96
    num_classes: int = 3
    max_params: int = 100_000
    channels: tuple = (16, 32, 64)
    dropout: float = 0.1


def get_texture_gate_config(config: Config) -> TextureGateConfig:
    return TextureGateConfig()


class TextureGate(nn.Module):
    """3-layer CNN texture quality classifier."""

    def __init__(self, config: Optional[TextureGateConfig] = None):
        super().__init__()
        self.config = config or TextureGateConfig()

        # 3 conv layers with decreasing spatial resolution
        layers = []
        in_ch = 1
        for i, c_out in enumerate(self.config.channels):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, c_out, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(self.config.dropout),
                )
            )
            in_ch = c_out
        self.features = nn.Sequential(*layers)

        # After 3 stride-2 layers: 96 -> 48 -> 24 -> 12
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.config.channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(64, self.config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 96, 96) grayscale -> logits (B, 3)"""
        x = self.features(x)  # (B, 128, 12, 12)
        x = self.pool(x)  # (B, 128, 1, 1)
        x = self.classifier(x)  # (B, 3)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_summary(self) -> str:
        total = self.count_parameters()
        lines = [
            "TextureGate Summary",
            f"  Total params: {total:,}",
            f"  Max allowed: {self.config.max_params:,}",
            f"  Within budget: {total <= self.config.max_params}",
            f"  Channels: {self.config.channels}",
            f"  Input: 1x{self.config.input_size}x{self.config.input_size}",
        ]
        return "\n".join(lines)


def build_texture_gate(config: Config) -> TextureGate:
    gate_config = get_texture_gate_config(config)
    model = TextureGate(gate_config)
    logger.info(model.model_summary())
    assert model.count_parameters() <= gate_config.max_params, (
        f"TextureGate has {model.count_parameters():,} params, "
        f"exceeds budget {gate_config.max_params:,}"
    )
    return model
