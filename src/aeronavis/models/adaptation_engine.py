"""Adaptation engine for online test-time adaptation of the velocity model.

Implements the online self-supervised update rule with LoRA adapters,
including guards against destructive over-fitting.
"""

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from aeronavis.models.velocity import VelocityModel
from aeronavis.models.adaptation import LoRAVelocityModel
from aeronavis.config import Config

logger = logging.getLogger(__name__)


@dataclass
class AdaptationState:
    """Current health state of the adaptation engine."""

    active: bool = True
    step_count: int = 0
    total_loss: float = 0.0
    l_vel_loss: float = 0.0
    l_smooth_loss: float = 0.0
    l_gnss_loss: float = 0.0
    params_delta_norm: float = 0.0
    gnss_residual_count: int = 0
    last_gnss_residual: float = 0.0
    frozen: bool = False
    loss_history: list = field(default_factory=list)


class AdaptationEngine:
    """Online test-time adaptation engine for the velocity model."""

    def __init__(
        self,
        velocity_model: VelocityModel,
        config: Config,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.adapt_cfg = config.adaptation
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Wrap velocity model with LoRA
        self.lora_model = LoRAVelocityModel(
            base_model=velocity_model,
            config=None,  # Will use internal defaults
            target_layers=[
                "fusion",
                "head.0",  # FC(256->128)
                "head.3",  # FC(128->1)
            ],
        )
        self.lora_model.to(self.device)

        # Optimizer (only LoRA params)
        self.optimizer = optim.SGD(
            self.lora_model.lora_parameters(),
            lr=self.adapt_cfg.lr,
            momentum=self.adapt_cfg.momentum,
        )

        # State
        self.state = AdaptationState()
        self._loss_window = deque(
            maxlen=self.adapt_cfg.loss_window_s * 100 // 50
        )  # ~30s at 50ms steps
        self._base_loss = float("inf")
        self._stable_snapshot = None
        self._step_buffer = []

    def step(
        self,
        window_acc: torch.Tensor,  # (B, W, 3)
        window_gyr: torch.Tensor,  # (B, W, 3)
        pseudo_speed: Optional[torch.Tensor],  # (B,) or None
        pseudo_rel: Optional[torch.Tensor],  # (B,) or None
        gnss_speed: Optional[torch.Tensor],  # (B,) or None
        gnss_ok: bool,
    ) -> float:
        """Perform one adaptation step. Returns current total loss."""
        self.state.step_count += 1

        # Move to device
        window_acc = window_acc.to(self.device)
        window_gyr = window_gyr.to(self.device)

        # Forward pass
        self.lora_model.train()
        vel_pred = self.lora_model(window_acc, window_gyr, None)

        # Compute losses
        loss = torch.tensor(0.0, device=self.device)
        l_vel_loss = torch.tensor(0.0, device=self.device)
        l_smooth_loss = torch.tensor(0.0, device=self.device)
        l_gnss_loss = torch.tensor(0.0, device=self.device)

        # L_vel: velocity consistency with pseudo-odo
        if pseudo_speed is not None and pseudo_rel is not None:
            pseudo_speed = pseudo_speed.to(self.device)
            pseudo_rel = pseudo_rel.to(self.device)
            # Only use high-reliability pseudo-odo
            mask = pseudo_rel > 0.7
            if mask.any():
                l_vel = F.mse_loss(vel_pred[mask], pseudo_speed[mask])
                l_vel_loss = l_vel
                loss = loss + self.config.adaptation.l_vel_weight * l_vel

        # L_smooth: temporal smoothness prior
        if hasattr(self, "_prev_vel_pred") and self._prev_vel_pred is not None:
            l_smooth = F.mse_loss(vel_pred, self._prev_vel_pred)
            l_smooth_loss = l_smooth
            loss = loss + self.config.adaptation.l_smooth_weight * l_smooth
        self._prev_vel_pred = vel_pred.detach()

        # L_gnss: GNSS speed regression (strongest signal when available)
        if gnss_ok and gnss_speed is not None:
            gnss_speed = gnss_speed.to(self.device)
            l_gnss = F.mse_loss(vel_pred, gnss_speed)
            l_gnss_loss = l_gnss
            loss = loss + self.config.adaptation.l_gnss_weight * l_gnss

            # Track GNSS residual for kill switch
            with torch.no_grad():
                residual = (vel_pred - gnss_speed).abs().mean().item()
                self.state.last_gnss_residual = residual
                if residual > 1.5:  # arbitrary threshold for "worsening"
                    self.state.gnss_residual_count += 1
                else:
                    self.state.gnss_residual_count = 0

        # In pure blackout with no GNSS and low pseudo reliability, scale smooth loss
        if not gnss_ok and (
            pseudo_speed is None or (pseudo_rel is not None and pseudo_rel.max() < 0.5)
        ):
            loss = loss * 0.5

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.lora_model.lora_parameters(), self.config.adaptation.clip_norm
        )
        self.optimizer.step()

        # Update state
        self.state.total_loss = loss.item()
        self.state.l_vel_loss = (
            l_vel_loss.item() if isinstance(l_vel_loss, torch.Tensor) else l_vel_loss
        )
        self.state.l_smooth_loss = (
            l_smooth_loss.item() if isinstance(l_smooth_loss, torch.Tensor) else l_smooth_loss
        )
        self.state.l_gnss_loss = (
            l_gnss_loss.item() if isinstance(l_gnss_loss, torch.Tensor) else l_gnss_loss
        )
        self._loss_window.append(loss.item())

        # Guards
        self._check_guards()

        # Stats
        self._update_stats()

        return loss.item()

    def _check_guards(self):
        """Check adaptation guards: freeze if loss not improving, kill switch on GNSS residual."""
        # Freeze if loss window not improving vs base
        if len(self._loss_window) >= self.adapt_cfg.loss_window_s * 2:
            recent_mean = sum(list(self._loss_window)[-10:]) / 10
            if self._base_loss == float("inf"):
                self._base_loss = sum(self._loss_window) / len(self._loss_window)
            elif recent_mean > self._base_loss * (1 + self.adapt_cfg.freeze_threshold_pct / 100):
                if not self.state.frozen:
                    logger.warning("Adaptation frozen: loss not improving vs base")
                    self.freeze_and_snapshot()

        # Kill switch: 3 consecutive GNSS residual worsenings
        if self.state.gnss_residual_count >= self.config.adaptation.gnss_residual_patience:
            logger.warning("Kill switch triggered: GNSS residual worsened 3x")
            self.revert_to_snapshot()

    def _update_stats(self):
        """Update rolling stats."""
        if hasattr(self.lora_model, "lora_layers"):
            total_delta = 0.0
            for lora in self.lora_model.lora_layers.values():
                if hasattr(lora, "A") and hasattr(lora, "B"):
                    total_delta += lora.A.grad.norm().item() if lora.A.grad is not None else 0
                    total_delta += lora.B.grad.norm().item() if lora.B.grad is not None else 0
            self.state.params_delta_norm = total_delta

        self.state.loss_history.append(
            {
                "step": self.state.step_count,
                "total": self.state.total_loss,
                "l_vel": self.state.l_vel_loss,
                "l_smooth": self.state.l_smooth_loss,
                "l_gnss": self.state.l_gnss_loss,
            }
        )

    def infer(self, window_acc, window_gyr, att0) -> float:
        """Forward pass with adapters ACTIVE (inference mode)."""
        self.lora_model.eval()
        with torch.no_grad():
            window_acc = window_acc.to(self.device)
            window_gyr = window_gyr.to(self.device)
            if att0 is not None:
                att0 = att0.to(self.device)
            vel = self.lora_model(window_acc, window_gyr, att0)
            return vel.item() if vel.numel() == 1 else vel

    def state(self) -> dict:
        """Return health state for dashboard."""
        return {
            "active": self.state.active,
            "frozen": self.state.frozen,
            "step_count": self.state.step_count,
            "loss": {
                "total": self.state.total_loss,
                "l_vel": self.state.l_vel_loss,
                "l_smooth": self.state.l_smooth_loss,
                "l_gnss": self.state.l_gnss_loss,
            },
            "params_delta_norm": self.state.params_delta_norm,
            "gnss_residual_count": self.state.gnss_residual_count,
            "last_gnss_residual": self.state.last_gnss_residual,
            "loss_history_len": len(self.state.loss_history),
        }

    def freeze_and_snapshot(self):
        """Freeze adapters and save stable snapshot."""
        self.state.frozen = True
        for p in self.lora_model.lora_parameters():
            p.requires_grad = False
        self._stable_snapshot = self.lora_model.lora_state_dict()
        self.save_snapshot(Path("models/adaptation/stable.pt"))
        logger.info("Adapters frozen and snapshot saved")

    def revert_to_snapshot(self):
        """Revert adapters to stable snapshot."""
        if self._stable_snapshot is not None:
            self.lora_model.load_lora(self._stable_snapshot)
            for p in self.lora_model.lora_parameters():
                p.requires_grad = True
            self.state.frozen = False
            self.state.gnss_residual_count = 0
            logger.warning("Reverted adapters to stable snapshot")

    def save_snapshot(self, path: Path):
        """Save current adapter state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "lora_state": self.lora_model.lora_state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "state": self.state.__dict__,
            },
            path,
        )

    def load_snapshot(self, path: Path):
        """Load adapter state."""
        ckpt = torch.load(path, map_location="cpu")
        self.lora_model.load_lora(ckpt["lora_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        logger.info(f"Loaded adaptation snapshot from {path}")

    def save_stats(self, path: Path):
        """Save adaptation stats as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        stats = self.state()
        stats["loss_history"] = self.state.loss_history
        with open(path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Saved adaptation stats to {path}")
