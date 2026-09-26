from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from src.methods.dt.losses import masked_action_mse
from src.methods.dt_mtm.diagnostics import shared_gradient_diagnostics
from src.methods.dt_mtm.losses import compose_dt_mtm_loss
from src.methods.dt_mtm.model import DTMTMModel
from src.methods.mtm.losses import reference_mtm_loss


@dataclass(frozen=True)
class DTMTMTrainBatch:
    states: object
    actions: object
    returns_to_go: object
    timesteps: object
    attention_mask: object
    mtm_trajectories: Mapping[str, object]
    mtm_masks: Mapping[str, object]


@dataclass(frozen=True)
class DTMTMTrainMetrics:
    total_loss: float
    dt_loss: float
    mtm_loss: float
    mtm_scaled_loss: float
    mtm_state_loss: float
    mtm_action_loss: float
    mtm_return_loss: float

    # Gradient diagnostics before any clipping.
    grad_norm_pre_clip: float
    dt_grad_norm_pre_clip: float
    auxiliary_grad_norm_pre_clip: float

    # Objective interaction on the parameters shared by both losses.
    shared_dt_grad_norm: float
    shared_mtm_grad_norm: float
    shared_mtm_scaled_grad_norm: float
    shared_grad_cosine: float

    learning_rate: float


def _parameter_grad_norm(parameters) -> torch.Tensor:
    """Compute an L2 norm over existing gradients without mutating them."""

    squares: list[torch.Tensor] = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None:
            squares.append(
                torch.sum(
                    gradient.detach().to(torch.float32) ** 2
                )
            )

    if not squares:
        return torch.zeros((), dtype=torch.float32)

    return torch.sqrt(torch.stack(squares).sum())


class DTMTMTrainer:
    """
    One-step trainer for the joint DT + MTM objective.

    The Group-1 Decision Transformer clipping contract is preserved on the
    DT parameter set itself. Auxiliary-only MTM/bridge parameters are clipped
    separately so a large MTM-only gradient cannot shrink the entire DT update
    merely by inflating a global clipping norm.
    """

    def __init__(
        self,
        model: DTMTMModel,
        optimizer: torch.optim.Optimizer,
        scheduler: object | None,
        device: torch.device,
        *,
        lambda_mtm: float,
        grad_clip_norm: float = 0.25,
        auxiliary_grad_clip_norm: float | None = None,
        diagnose_shared_gradients: bool = True,
    ) -> None:
        if lambda_mtm < 0.0:
            raise ValueError("lambda_mtm must be non-negative")
        if grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive")

        if auxiliary_grad_clip_norm is None:
            auxiliary_grad_clip_norm = grad_clip_norm
        if auxiliary_grad_clip_norm <= 0.0:
            raise ValueError("auxiliary_grad_clip_norm must be positive")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.lambda_mtm = float(lambda_mtm)
        self.grad_clip_norm = float(grad_clip_norm)
        self.auxiliary_grad_clip_norm = float(auxiliary_grad_clip_norm)
        self.diagnose_shared_gradients = bool(diagnose_shared_gradients)

    def _tensor(
        self,
        value: object,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _parameter_partitions(self) -> tuple[tuple[torch.nn.Parameter, ...], tuple[torch.nn.Parameter, ...]]:
        dt_parameters = tuple(
            parameter
            for parameter in self.model.dt.parameters()
            if parameter.requires_grad
        )
        dt_ids = {id(parameter) for parameter in dt_parameters}

        auxiliary_parameters = tuple(
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in dt_ids
        )

        return dt_parameters, auxiliary_parameters

    def train_step(self, batch: DTMTMTrainBatch) -> DTMTMTrainMetrics:
        self.model.train()

        states = self._tensor(batch.states, dtype=torch.float32)
        actions = self._tensor(batch.actions, dtype=torch.float32)
        returns_to_go = self._tensor(batch.returns_to_go, dtype=torch.float32)
        timesteps = self._tensor(batch.timesteps, dtype=torch.long)
        attention_mask = self._tensor(batch.attention_mask, dtype=torch.long)

        mtm_trajectories = {
            key: self._tensor(value, dtype=torch.float32)
            for key, value in batch.mtm_trajectories.items()
        }
        mtm_masks = {
            key: self._tensor(value, dtype=torch.float32)
            for key, value in batch.mtm_masks.items()
        }

        self.optimizer.zero_grad(set_to_none=True)

        _, action_preds, _ = self.model.forward_dt(
            states,
            actions,
            returns_to_go,
            timesteps,
            attention_mask,
        )
        dt_loss = masked_action_mse(
            action_preds,
            actions,
            attention_mask,
        )

        mtm_predictions = self.model.forward_mtm(
            mtm_trajectories,
            mtm_masks,
        )
        mtm_loss = reference_mtm_loss(
            mtm_trajectories,
            mtm_predictions,
            mtm_masks,
        )

        joint_loss = compose_dt_mtm_loss(
            dt_loss,
            mtm_loss,
            lambda_mtm=self.lambda_mtm,
        )

        if self.diagnose_shared_gradients:
            grad_diag = shared_gradient_diagnostics(
                self.model,
                dt_loss,
                joint_loss.mtm_weighted_loss,
                lambda_mtm=self.lambda_mtm,
            )
            shared_dt_norm = grad_diag.dt_norm
            shared_mtm_norm = grad_diag.mtm_norm_unscaled
            shared_mtm_scaled_norm = grad_diag.mtm_norm_scaled
            shared_cosine = grad_diag.cosine_similarity
        else:
            shared_dt_norm = float("nan")
            shared_mtm_norm = float("nan")
            shared_mtm_scaled_norm = float("nan")
            shared_cosine = float("nan")

        # At lambda=0, backpropagate the DT loss directly rather than
        # dt_loss + 0 * mtm_loss. This keeps MTM-only parameter.grad as None,
        # so AdamW cannot move the auxiliary branch via decoupled weight decay.
        backward_loss = (
            dt_loss
            if self.lambda_mtm == 0.0
            else joint_loss.total_loss
        )
        backward_loss.backward()

        dt_parameters, auxiliary_parameters = self._parameter_partitions()
        all_parameters = dt_parameters + auxiliary_parameters

        global_grad_norm = _parameter_grad_norm(all_parameters)
        dt_grad_norm = _parameter_grad_norm(dt_parameters)
        auxiliary_grad_norm = _parameter_grad_norm(auxiliary_parameters)

        for name, value in (
            ("global", global_grad_norm),
            ("dt", dt_grad_norm),
            ("auxiliary", auxiliary_grad_norm),
        ):
            if not torch.isfinite(value):
                raise FloatingPointError(
                    f"Non-finite {name} gradient norm detected: {value.item()}"
                )

        # Preserve Group-1 clipping semantics on the complete DT parameter set.
        torch.nn.utils.clip_grad_norm_(
            dt_parameters,
            max_norm=self.grad_clip_norm,
        )

        # Clip auxiliary-only parameters independently. This prevents the size
        # of the MTM branch from changing the clipping scale applied to DT.
        if auxiliary_parameters:
            torch.nn.utils.clip_grad_norm_(
                auxiliary_parameters,
                max_norm=self.auxiliary_grad_clip_norm,
            )

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        learning_rate = float(self.optimizer.param_groups[0]["lr"])

        full = mtm_loss.full_losses
        required = {"states", "actions", "returns"}
        if set(full) != required:
            raise RuntimeError(
                "Unexpected MTM loss modalities: "
                f"expected {sorted(required)}, got {sorted(full)}"
            )

        return DTMTMTrainMetrics(
            total_loss=float(joint_loss.total_loss.detach().cpu()),
            dt_loss=float(dt_loss.detach().cpu()),
            mtm_loss=float(joint_loss.mtm_weighted_loss.detach().cpu()),
            mtm_scaled_loss=float(joint_loss.mtm_scaled_loss.detach().cpu()),
            mtm_state_loss=float(full["states"].detach().cpu()),
            mtm_action_loss=float(full["actions"].detach().cpu()),
            mtm_return_loss=float(full["returns"].detach().cpu()),
            grad_norm_pre_clip=float(global_grad_norm.detach().cpu()),
            dt_grad_norm_pre_clip=float(dt_grad_norm.detach().cpu()),
            auxiliary_grad_norm_pre_clip=float(auxiliary_grad_norm.detach().cpu()),
            shared_dt_grad_norm=shared_dt_norm,
            shared_mtm_grad_norm=shared_mtm_norm,
            shared_mtm_scaled_grad_norm=shared_mtm_scaled_norm,
            shared_grad_cosine=shared_cosine,
            learning_rate=learning_rate,
        )
