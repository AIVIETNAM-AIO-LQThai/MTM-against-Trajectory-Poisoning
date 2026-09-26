from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from src.methods.mtm.losses import MTMLossOutput


@dataclass(frozen=True)
class DTMTMLossOutput:
    """Explicit decomposition of the joint DT + MTM objective."""

    total_loss: torch.Tensor
    dt_loss: torch.Tensor
    mtm_unweighted_loss: torch.Tensor
    mtm_weighted_loss: torch.Tensor
    mtm_scaled_loss: torch.Tensor
    lambda_mtm: float
    modality_weights: dict[str, float]
    mtm_full_losses: dict[str, torch.Tensor]


def compose_dt_mtm_loss(
    dt_loss: torch.Tensor,
    mtm_loss: MTMLossOutput,
    *,
    lambda_mtm: float,
    modality_weights: Mapping[str, float] | None = None,
) -> DTMTMLossOutput:
    """
    Compose

        L_total = L_DT + lambda_mtm * L_MTM.

    With ``modality_weights=None`` this preserves the validated Group-3
    reference objective exactly: the full reconstruction losses for states,
    actions, and returns are summed with weight 1.

    Optional modality weights exist only for explicitly predeclared later
    ablations; Group 4's primary experiment should leave them unset.
    """

    if lambda_mtm < 0:
        raise ValueError("lambda_mtm must be non-negative")

    if not torch.isfinite(dt_loss):
        raise FloatingPointError("Non-finite DT loss")

    if modality_weights is None:
        weights = {
            key: 1.0
            for key in mtm_loss.full_losses.keys()
        }
        weighted_mtm = mtm_loss.total_loss
    else:
        weights = {key: float(value) for key, value in modality_weights.items()}

        if set(weights.keys()) != set(mtm_loss.full_losses.keys()):
            raise ValueError(
                "modality_weights keys must exactly match MTM loss modalities"
            )

        if any(value < 0 for value in weights.values()):
            raise ValueError("modality weights must be non-negative")

        weighted_mtm = torch.stack(
            [
                mtm_loss.full_losses[key] * weights[key]
                for key in mtm_loss.full_losses.keys()
            ]
        ).sum()

    if not torch.isfinite(weighted_mtm):
        raise FloatingPointError("Non-finite MTM loss")

    scaled_mtm = weighted_mtm * float(lambda_mtm)
    total = dt_loss + scaled_mtm

    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite joint DT+MTM loss")

    return DTMTMLossOutput(
        total_loss=total,
        dt_loss=dt_loss,
        mtm_unweighted_loss=mtm_loss.total_loss,
        mtm_weighted_loss=weighted_mtm,
        mtm_scaled_loss=scaled_mtm,
        lambda_mtm=float(lambda_mtm),
        modality_weights=weights,
        mtm_full_losses=dict(mtm_loss.full_losses),
    )
