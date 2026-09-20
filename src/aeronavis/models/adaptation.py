"""LoRA-style low-rank adapters for test-time adaptation.

Wraps the frozen velocity model with low-rank adapters (LoRA) for online
self-supervised adaptation. Designed for on-device CPU inference with
≤ 200 kB extra weights and ≤ 15 ms added latency.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn

from aeronavis.models.velocity import VelocityModel
from aeronavis.config import Config

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    max_adapter_bytes: int = 200_000  # 200 kB


class LoRALinear(nn.Module):
    """LoRA wrapper for a frozen nn.Linear layer."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
    ):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base.in_features
        out_features = base.out_features

        # LoRA matrices: A (in x r), B (r x out)
        # Init: A ~ kaiming_uniform, B = 0
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base(x) + scaling * (x @ A @ B)
        base_out = self.base(x)
        lora_out = (x @ self.A) @ self.B * self.scaling
        return base_out + lora_out

    def count_parameters(self) -> int:
        return self.A.numel() + self.B.numel()

    def state_dict_lora(self) -> dict:
        return {"A": self.A.detach().cpu(), "B": self.B.detach().cpu()}

    def load_lora(self, state: dict):
        self.A.data.copy_(state["A"].to(self.A.device))
        self.B.data.copy_(state["B"].to(self.B.device))


class LoRAVelocityModel(nn.Module):
    """Velocity model with LoRA adapters on target layers."""

    def __init__(
        self,
        base_model: VelocityModel,
        config: Optional[LoRAConfig] = None,
        target_layers: Optional[List[str]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.lora_config = config or LoRAConfig()
        self.lora_layers = nn.ModuleDict()

        # Default target layers: fusion FC + two heads
        if target_layers is None:
            target_layers = [
                "fusion",
                "head.0",  # FC(256->128)
                "head.3",  # FC(128->1)
            ]

        self._wrap_layers(target_layers)
        self._freeze_base()
        self._validate_budget()

    def _wrap_layers(self, target_layers: List[str]):
        """Replace target linear layers with LoRALinear."""
        for name in target_layers:
            module = self._get_module(name)
            if module is None:
                logger.warning(f"LoRA target '{name}' not found in base model")
                continue
            if not isinstance(module, nn.Linear):
                logger.warning(f"LoRA target '{name}' is not nn.Linear, skipping")
                continue

            lora = LoRALinear(
                module,
                rank=self.lora_config.rank,
                alpha=self.lora_config.alpha,
            )
            # ModuleDict keys can't contain dots, replace with underscore
            key = name.replace(".", "_")
            self.lora_layers[key] = lora
            self._set_module(name, lora)
            logger.info(
                f"Wrapped LoRA around '{name}' "
                f"({module.in_features}->{module.out_features}, r={self.lora_config.rank})"
            )

    def _get_module(self, name: str) -> Optional[nn.Module]:
        """Get module by dotted path from base_model."""
        parts = name.split(".")
        module = self.base_model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _set_module(self, name: str, new_module: nn.Module):
        """Replace module by dotted path in base_model."""
        parts = name.split(".")
        parent = self.base_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def _freeze_base(self):
        """Freeze all base model parameters except LoRA."""
        for p in self.base_model.parameters():
            p.requires_grad = False
        # Ensure LoRA params are trainable
        for lora in self.lora_layers.values():
            for p in lora.parameters():
                p.requires_grad = True

    def _validate_budget(self):
        """Check adapter memory budget."""
        total_params = sum(layer.count_parameters() for layer in self.lora_layers.values())
        bytes_used = total_params * 4  # fp32
        logger.info(f"LoRA total params: {total_params:,} ({bytes_used:,} bytes)")
        assert bytes_used <= self.lora_config.max_adapter_bytes, (
            f"LoRA adapters exceed budget: {bytes_used} > "
            f"{self.lora_config.max_adapter_bytes} bytes"
        )

    def forward(self, acc, gyr, att0):
        """Forward pass with LoRA adapters active."""
        return self.base_model(acc, gyr, att0)

    def forward_train(self, acc, gyr, att0):
        """Forward pass returning velocity and intermediate features."""
        return self.base_model.forward_train(acc, gyr, att0)

    def lora_parameters(self):
        """Return LoRA parameters for optimizer."""
        params = []
        for lora in self.lora_layers.values():
            params.extend(lora.parameters())
        return params

    def lora_state_dict(self) -> dict:
        return {name: lora.state_dict_lora() for name, lora in self.lora_layers.items()}

    def load_lora(self, state: dict):
        for name, lora in self.lora_layers.items():
            if name in state:
                lora.load_lora(state[name])

    def save_lora(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.lora_state_dict(), path)
        logger.info(f"Saved LoRA adapters to {path}")

    def load_lora_file(self, path: Path):
        state = torch.load(path, map_location="cpu")
        self.load_lora(state)
        logger.info(f"Loaded LoRA adapters from {path}")

    def count_parameters(self) -> int:
        return sum(layer.count_parameters() for layer in self.lora_layers.values())


def wrap_model_with_lora(
    base_model: VelocityModel,
    config: Config,
    target_layers: Optional[List[str]] = None,
) -> LoRAVelocityModel:
    """Factory to wrap a velocity model with LoRA adapters."""
    if target_layers is None:
        target_layers = [
            "tcn.out_proj",
            "lstm.out_proj",
            "head.fusion.0",
            "head.fusion.3",
            "head.fusion.6",
        ]
    lora_config = LoRAConfig(
        rank=config.model.lora_rank,
        alpha=config.model.lora_alpha,
        max_adapter_bytes=config.model.lora_max_bytes,
    )
    return LoRAVelocityModel(base_model, lora_config, target_layers)


# --- Compatibility with Config class ---
# The Config class uses dataclass, so we need to add LoRA params to ModelConfig
# This is handled in config.py by adding lora_rank, lora_alpha, lora_max_bytes