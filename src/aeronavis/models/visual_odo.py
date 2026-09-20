"""Visual odometry net (UL-VIO style, < 1M params).

Input: 2-channel (t-1, t) grayscale pair at 224x224.
Output: 6-DoF delta pose (tx, ty, tz, rx, ry, rz) normalized.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from aeronavis.config import Config

logger = logging.getLogger(__name__)


@dataclass
class VisualOdoConfig:
    input_size: int = 224
    in_channels: int = 2  # stacked (t-1, t) grayscale
    max_params: int = 1_000_000
    stem_channels: int = 32
    block_channels: tuple = (32, 32, 64, 64)
    fc_hidden: int = 64
    dropout: float = 0.1


def get_visual_odo_config(config: Config) -> VisualOdoConfig:
    return VisualOdoConfig()


class ResidualBlock(nn.Module):
    """Residual block with 2 conv3x3."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class VisualOdoNet(nn.Module):
    """UL-VIO style visual odometry network."""

    def __init__(self, config: Optional[VisualOdoConfig] = None):
        super().__init__()
        self.config = config or VisualOdoConfig()

        # Stem: conv 7x7 s=2
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.config.in_channels,
                self.config.stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(self.config.stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 4 residual blocks
        blocks = []
        in_ch = self.config.stem_channels
        for i, c_out in enumerate(self.config.block_channels):
            stride = 2 if i in (1, 3) else 1  # stride 2 at block 2 and 4
            blocks.append(ResidualBlock(in_ch, c_out, stride=stride, dropout=self.config.dropout))
            blocks.append(ResidualBlock(c_out, c_out, stride=1, dropout=self.config.dropout))
            in_ch = c_out
        self.blocks = nn.Sequential(*blocks)

        # Global pooling + FC
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.config.block_channels[-1], self.config.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.fc_hidden, 6),  # tx,ty,tz, rx,ry,rz
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, 224, 224) -> 6-DoF delta pose"""
        x = self.stem(x)  # (B, 64, 56, 56)
        x = self.blocks(x)  # (B, 128, 14, 14)
        x = self.pool(x)  # (B, 128, 1, 1)
        x = self.fc(x)  # (B, 6)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_summary(self) -> str:
        total = self.count_parameters()
        lines = [
            "VisualOdoNet Summary",
            f"  Total params: {total:,}",
            f"  Max allowed: {self.config.max_params:,}",
            f"  Within budget: {total <= self.config.max_params}",
            f"  Stem: {self.config.stem_channels}",
            f"  Blocks: {self.config.block_channels}",
            f"  FC hidden: {self.config.fc_hidden}",
        ]
        return "\n".join(lines)


def build_visual_odo(config: Config) -> VisualOdoNet:
    model_config = get_visual_odo_config(config)
    model = VisualOdoNet(model_config)
    logger.info(model.model_summary())
    assert model.count_parameters() <= model_config.max_params, (
        f"VisualOdoNet has {model.count_parameters():,} params, "
        f"exceeds budget {model_config.max_params:,}"
    )
    return model
