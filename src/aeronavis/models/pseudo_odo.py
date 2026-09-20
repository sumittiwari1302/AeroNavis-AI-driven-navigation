"""Pseudo wheel odometry (DeepOdo-style) with slip/loss-of-grip reliability head.

Recreates wheel speed from phone IMU + barometer + vertical accel spectrogram.
Architecture: Pre-net (baro + spec) → CNN-GRU encoder → wheel_speed + reliability heads.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from aeronavis.config import Config

logger = logging.getLogger(__name__)


@dataclass
class PseudoOdoConfig:
    window: int = 200
    hidden: int = 96
    max_params: int = 300_000
    cnn_channels: tuple = (48, 64, 96)
    cnn_kernel: int = 5
    gru_layers: int = 2
    gru_hidden: int = 96
    prenet_hidden: int = 32
    spec_bins: int = 8
    prenet_out: int = 32
    head_hidden: int = 128
    dropout: float = 0.1


def get_pseudo_odo_config(config: Config) -> PseudoOdoConfig:
    return PseudoOdoConfig(
        window=config.model.window,
        hidden=config.model.pseudo_odo_hidden,
        max_params=config.model.pseudo_odo_max_params,
    )


class SpectrogramExtractor(nn.Module):
    """Extract 8-bin spectrogram from vertical accel (0-5 Hz)."""

    def __init__(self, n_bins: int = 8, fs: float = 100.0, fmax: float = 5.0):
        super().__init__()
        self.n_bins = n_bins
        self.fs = fs
        self.fmax = fmax
        # Pre-compute FFT frequencies
        self.register_buffer("freqs", torch.fft.rfftfreq(200, 1.0 / fs))
        self.register_buffer("bin_edges", torch.linspace(0, fmax, n_bins + 1))

    def forward(self, acc_z: torch.Tensor) -> torch.Tensor:
        """acc_z: (B, W) -> spec: (B, n_bins)"""
        # FFT on each window
        fft = torch.fft.rfft(acc_z, dim=-1)  # (B, n_freq)
        mag = fft.abs()
        # Bin into 0-5 Hz
        spec = torch.zeros(acc_z.shape[0], self.n_bins, device=acc_z.device)
        for i in range(self.n_bins):
            lo = self.bin_edges[i]
            hi = self.bin_edges[i + 1]
            mask = (self.freqs >= lo) & (self.freqs < hi)
            if mask.any():
                spec[:, i] = mag[:, mask].mean(dim=-1)
        return spec  # (B, n_bins)


class PreNet(nn.Module):
    """Pre-net: baro_delta (1) + spec (8) -> 32."""

    def __init__(self, spec_bins: int = 8, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + spec_bins, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_dim),
            nn.GELU(),
        )

    def forward(self, baro_delta: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        # baro_delta: (B, 1), spec: (B, n_bins)
        x = torch.cat([baro_delta, spec], dim=-1)  # (B, 1 + n_bins)
        return self.net(x)  # (B, out_dim)


class CNNEncoder(nn.Module):
    """3 conv1d blocks over concat(acc, gyr)."""

    def __init__(
        self,
        in_channels: int = 6,
        channels: tuple = (48, 64, 96),
        kernel: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        c_in = in_channels
        for c_out in channels:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2, groups=1),
                    nn.GroupNorm(min(8, c_out), c_out),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            c_in = c_out
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, W)
        for block in self.blocks:
            x = block(x)
        return x  # (B, C, W)


class GRUEncoder(nn.Module):
    """2-layer GRU over CNN features."""

    def __init__(self, input_dim: int, hidden: int = 96, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
            bidirectional=False,
        )
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, W) -> (B, W, C)
        x = x.transpose(1, 2)
        _, h_n = self.gru(x)
        # h_n: (num_layers, B, hidden) -> take last layer
        return h_n[-1]  # (B, hidden)


class PseudoOdoModel(nn.Module):
    """Full pseudo wheel odometry model with reliability head."""

    def __init__(self, config: Optional[PseudoOdoConfig] = None):
        super().__init__()
        self.config = config or PseudoOdoConfig()

        # Spectrogram extractor (fixed, no params)
        self.spec_extractor = SpectrogramExtractor(
            n_bins=self.config.spec_bins,
            fs=100.0,
            fmax=5.0,
        )

        # Pre-net: baro + spec -> 32
        self.prenet = PreNet(
            spec_bins=self.config.spec_bins,
            out_dim=self.config.prenet_out,
        )

        # CNN-GRU encoder
        self.cnn = CNNEncoder(
            in_channels=6,
            channels=self.config.cnn_channels,
            kernel=self.config.cnn_kernel,
            dropout=self.config.dropout,
        )
        self.gru = GRUEncoder(
            input_dim=self.config.cnn_channels[-1],
            hidden=self.config.gru_hidden,
            layers=self.config.gru_layers,
            dropout=self.config.dropout,
        )

        # Heads
        combined_dim = self.config.gru_hidden + self.config.prenet_out
        self.wheel_head = nn.Sequential(
            nn.Linear(combined_dim, self.config.head_hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.head_hidden, 1),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(combined_dim, self.config.head_hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.head_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        acc: torch.Tensor,  # (B, W, 3)
        gyr: torch.Tensor,  # (B, W, 3)
        baro_delta: torch.Tensor,  # (B, 1) pressure change over window
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (wheel_speed, reliability)."""
        B, W, _ = acc.shape

        # Spectrogram from vertical accel
        spec = self.spec_extractor(acc[:, :, 2])  # (B, 8)

        # Pre-net
        prenet_feat = self.prenet(baro_delta, spec)  # (B, 32)

        # CNN-GRU on IMU
        imu = torch.cat([acc, gyr], dim=-1).transpose(1, 2)  # (B, 6, W)
        cnn_feat = self.cnn(imu)  # (B, C, W)
        gru_feat = self.gru(cnn_feat)  # (B, 96)

        # Combine
        combined = torch.cat([gru_feat, prenet_feat], dim=-1)  # (B, 96+32)

        # Heads
        wheel_speed = self.wheel_head(combined).squeeze(-1)  # (B,)
        reliability = self.reliability_head(combined).squeeze(-1)  # (B,)

        return wheel_speed, reliability

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_summary(self) -> str:
        total = self.count_parameters()
        lines = [
            "PseudoOdoModel Summary",
            f"  Total params: {total:,}",
            f"  Max allowed: {self.config.max_params:,}",
            f"  Within budget: {total <= self.config.max_params}",
            f"  CNN channels: {self.config.cnn_channels}",
            f"  GRU hidden: {self.config.gru_hidden}",
            f"  Pre-net out: {self.config.prenet_out}",
        ]
        return "\n".join(lines)


def build_pseudo_odo_model(config: Config) -> PseudoOdoModel:
    model_config = get_pseudo_odo_config(config)
    model = PseudoOdoModel(model_config)
    logger.info(model.model_summary())
    assert model.count_parameters() <= model_config.max_params, (
        f"PseudoOdoModel has {model.count_parameters():,} params, "
        f"exceeds budget {model_config.max_params:,}"
    )
    return model
