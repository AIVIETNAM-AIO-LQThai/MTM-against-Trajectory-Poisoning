from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.data.batching import DTBatch
from src.methods.dt.losses import masked_action_mse
from src.methods.dt.model import DecisionTransformer


@dataclass
class DTTrainMetrics:
    loss: float
    grad_norm_pre_clip: float
    learning_rate: float


class DTTrainer:
    """
    Production trainer for vanilla Decision Transformer.

    Reference behavior:
      - action-prediction MSE only
      - padding excluded through attention mask
      - AdamW optimizer
      - gradient clipping at 0.25
      - LR scheduler stepped after every optimizer update
    """

    def __init__(
        self,
        model: DecisionTransformer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        device: torch.device,
        grad_clip_norm: float = 0.25,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_clip_norm = grad_clip_norm

    def train_step(
        self,
        batch: DTBatch,
    ) -> DTTrainMetrics:

        self.model.train()

        states = torch.as_tensor(
            batch.states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            batch.actions,
            dtype=torch.float32,
            device=self.device,
        )

        returns_to_go = torch.as_tensor(
            batch.rtg[:, :-1],
            dtype=torch.float32,
            device=self.device,
        )

        timesteps = torch.as_tensor(
            batch.timesteps,
            dtype=torch.long,
            device=self.device,
        )

        attention_mask = torch.as_tensor(
            batch.attention_mask,
            dtype=torch.long,
            device=self.device,
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        _, action_preds, _ = self.model(
            states,
            actions,
            returns_to_go,
            timesteps,
            attention_mask,
        )

        loss = masked_action_mse(
            action_preds,
            actions,
            attention_mask,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite DT loss detected: {loss.item()}"
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.grad_clip_norm,
        )

        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                "Non-finite gradient norm detected: "
                f"{grad_norm.item()}"
            )

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        learning_rate = float(
            self.optimizer.param_groups[0]["lr"]
        )

        return DTTrainMetrics(
            loss=float(loss.detach().cpu()),
            grad_norm_pre_clip=float(grad_norm.detach().cpu()),
            learning_rate=learning_rate,
        )