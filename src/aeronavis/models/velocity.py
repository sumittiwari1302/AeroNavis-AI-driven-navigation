"""Learned velocity model: TCN + LSTM fusion with gravity-aligned canonical frame.

Architecture (RoNIN/EqNIO school):
- Input: raw IMU window (acc, gyr) + initial attitude matrix
- Canonicalization: rotate acc into gravity-aligned frame via att0^T
- Encoder A: TCN (5 residual blocks, dilations 1,2,4,8,16)
- Encoder B: LSTM (2-layer, hidden=128)
- Fusion: concat features -> FC(256->128) -> FC(128->1) + residual 0.2
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aeronavis.config import Config

logger = logging.getLogger(__name__)


@dataclass
class VelocityModelConfig:
    window: int = 200
    stride: int = 10
    hidden: int = 64
    max_params: int = 500_000
    tcn_channels: tuple = (32, 40, 64, 64, 64)
    tcn_kernel: int = 5
    tcn_dilations: tuple = (1, 2, 4, 8, 16)
    lstm_layers: int = 2
    lstm_hidden: int = 64
    fusion_hidden: int = 128
    dropout: float = 0.1
    residual_vel: float = 0.2


def get_velocity_config(config: Config) -> VelocityModelConfig:
    return VelocityModelConfig(
        window=config.model.window,
        stride=config.model.stride,
        hidden=config.model.velocity_hidden,
        max_params=config.model.max_params,
    )


class TCNResidualBlock(nn.Module):
    """Causal TCN residual block with GroupNorm + GeLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = out[:, :, : -self.conv1.padding[0]] if self.conv1.padding[0] > 0 else out
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, : -self.conv2.padding[0]] if self.conv2.padding[0] > 0 else out
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = out + residual
        out = F.gelu(out)
        return out


class TCNEncoder(nn.Module):
    """TCN encoder stack with exponentially increasing dilations."""

    def __init__(self, config: VelocityModelConfig):
        super().__init__()
        channels = [6] + list(config.tcn_channels)
        self.blocks = nn.ModuleList()
        for i, (c_in, c_out) in enumerate(zip(channels[:-1], channels[1:])):
            self.blocks.append(
                TCNResidualBlock(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=config.tcn_kernel,
                    dilation=config.tcn_dilations[i],
                    dropout=config.dropout,
                )
            )
        self.out_proj = nn.Linear(config.tcn_channels[-1], config.hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, W) -> (B, C, W)
        for block in self.blocks:
            x = block(x)
        # Take last timestep
        feat = x[:, :, -1]  # (B, C)
        feat = self.out_proj(feat)  # (B, hidden)
        return feat


class LSTMEncoder(nn.Module):
    """2-layer LSTM encoder, returns final hidden state."""

    def __init__(self, config: VelocityModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=6,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
        )
        self.out_proj = nn.Linear(config.lstm_hidden, config.hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, W) -> (B, W, 6)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, B, hidden) -> take last layer
        feat = h_n[-1]  # (B, hidden)
        feat = self.out_proj(feat)
        return feat


class VelocityHead(nn.Module):
    """Fusion head: concat TCN + LSTM features -> velocity scalar."""

    def __init__(self, config: VelocityModelConfig):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden * 2, config.fusion_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden, config.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, 1),
        )
        self.register_buffer("residual_vel", torch.tensor(config.residual_vel))

    def forward(self, tcn_feat: torch.Tensor, lstm_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([tcn_feat, lstm_feat], dim=-1)
        vel = self.fusion(fused) + self.residual_vel
        return vel


class VelocityModel(nn.Module):
    """Full velocity model with canonical frame rotation."""

    def __init__(self, config: Optional[VelocityModelConfig] = None):
        super().__init__()
        self.config = config or VelocityModelConfig()
        self.tcn = TCNEncoder(self.config)
        self.lstm = LSTMEncoder(self.config)
        self.head = VelocityHead(self.config)

    def canonicalize(self, acc: torch.Tensor, att0: torch.Tensor) -> torch.Tensor:
        """Rotate acceleration from device frame to gravity-aligned frame.

        acc: (B, 3, W) or (B, W, 3)
        att0: (B, 3, 3) rotation matrix from device to canonical frame
        Returns: (B, 3, W) in canonical frame
        """
        if acc.dim() == 3 and acc.shape[1] == 3:
            acc = acc.transpose(1, 2)  # (B, W, 3)
        # acc: (B, W, 3), att0: (B, 3, 3)
        # att0^T @ acc^T -> (B, 3, W) -> transpose -> (B, W, 3)
        acc_canonical = torch.bmm(att0.transpose(1, 2), acc.transpose(1, 2)).transpose(1, 2)
        return acc_canonical.transpose(1, 2)  # (B, 3, W)

    def forward(
        self,
        acc: torch.Tensor,
        gyr: torch.Tensor,
        att0: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning velocity scalar.

        Args:
            acc: (B, W, 3) acceleration in device frame
            gyr: (B, W, 3) angular velocity in device frame
            att0: (B, 3, 3) initial attitude matrix (device -> canonical)
        Returns:
            vel: (B, 1) forward velocity in m/s
        """
        # Canonicalize acceleration
        acc_canon = self.canonicalize(acc, att0)  # (B, 3, W)

        # Concatenate canonical acc + gyro (not canonicalized)
        x = torch.cat([acc_canon, gyr.transpose(1, 2)], dim=1)  # (B, 6, W)

        # Encode
        tcn_feat = self.tcn(x)
        lstm_feat = self.lstm(x)

        # Fuse and predict
        vel = self.head(tcn_feat, lstm_feat)
        return vel

    def forward_train(
        self,
        acc: torch.Tensor,
        gyr: torch.Tensor,
        att0: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for training, returning velocity and intermediate features."""
        acc_canon = self.canonicalize(acc, att0)
        x = torch.cat([acc_canon, gyr.transpose(1, 2)], dim=1)
        tcn_feat = self.tcn(x)
        lstm_feat = self.lstm(x)
        vel = self.head(tcn_feat, lstm_feat)
        return vel, (tcn_feat, lstm_feat)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_summary(self) -> str:
        total = self.count_parameters()
        lines = [
            "VelocityModel Summary",
            f"  Total params: {total:,}",
            f"  Max allowed: {self.config.max_params:,}",
            f"  Within budget: {total <= self.config.max_params}",
            f"  TCN channels: {self.config.tcn_channels}",
            f"  LSTM hidden: {self.config.lstm_hidden}",
            f"  Fusion hidden: {self.config.fusion_hidden}",
        ]
        return "\n".join(lines)


def build_velocity_model(config: Config) -> VelocityModel:
    """Factory function to build model from global config."""
    model_config = get_velocity_config(config)
    model = VelocityModel(model_config)
    logger.info(model.model_summary())
    assert (
        model.count_parameters() <= model_config.max_params
    ), f"Model has {model.count_parameters():,} params, exceeds budget {model_config.max_params:,}"
    return model
