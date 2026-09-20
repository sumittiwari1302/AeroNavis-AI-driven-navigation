"""Transformer sequence model for GNSS outage prediction.

Input: (B, 30, F) where F ≈ 104 (32*3 + 8)
Output: per-horizon degradation probability for h ∈ {5, 10, 15}
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn

from aeronavis.config import Config

logger = logging.getLogger(__name__)


@dataclass
class PredictModelConfig:
    """Configuration for the prediction model."""

    input_dim: int = 104
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ff_dim: int = 512
    dropout: float = 0.1
    horizons: List[int] = None
    max_len: int = 30
    max_params: int = 2_000_000

    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [5, 10, 15]


def get_predict_model_config(config: Config) -> PredictModelConfig:
    return PredictModelConfig(
        d_model=config.model.get("predict_d_model", 128) if hasattr(config.model, "get") else 128,
    )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class PredictTransformer(nn.Module):
    """Transformer encoder for GNSS outage prediction."""

    def __init__(self, config: PredictModelConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_len)

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    ff_dim=config.ff_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # Output heads per horizon
        self.heads = nn.ModuleDict(
            {
                str(h): nn.Sequential(
                    nn.Linear(config.d_model, 64),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                )
                for h in config.horizons
            }
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, F)
        padding_mask: Optional[torch.Tensor] = None,  # (B, T) - True for padding
    ) -> Dict[int, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, T, F) input features
            padding_mask: (B, T) boolean mask, True for padded positions

        Returns:
            Dict mapping horizon -> probability (B,)
        """
        B, T, F = x.shape

        # Project to d_model
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.dropout(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoder
        # Convert padding mask to attention mask (True = attend)
        if padding_mask is not None:
            pass

        for layer in self.layers:
            x = layer(x, mask=padding_mask)

        # Pool over time (mean pooling over valid positions)
        if padding_mask is not None:
            valid_mask = (~padding_mask).float().unsqueeze(-1)  # (B, T, 1)
            x = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)  # (B, d_model)

        # Per-horizon heads
        outputs = {}
        for h, head in self.heads.items():
            outputs[int(h)] = head(x).squeeze(-1)  # (B,)

        return outputs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_summary(self) -> str:
        total = self.count_parameters()
        lines = [
            "PredictTransformer Summary",
            f"  Total params: {total:,}",
            f"  Max allowed: {self.config.max_params:,}",
            f"  Within budget: {total <= self.config.max_params}",
            f"  d_model: {self.config.d_model}",
            f"  n_layers: {self.config.n_layers}",
            f"  n_heads: {self.config.n_heads}",
            f"  horizons: {self.config.horizons}",
        ]
        return "\n".join(lines)


def build_predict_model(config: Config) -> "PredictTransformer":
    """Factory to build prediction model from global config."""
    model_config = PredictModelConfig()
    model = PredictTransformer(model_config)
    logger.info(model.model_summary())
    assert model.count_parameters() <= model_config.max_params, (
        f"PredictTransformer has {model.count_parameters():,} params, "
        f"exceeds budget {model_config.max_params:,}"
    )
    return model
