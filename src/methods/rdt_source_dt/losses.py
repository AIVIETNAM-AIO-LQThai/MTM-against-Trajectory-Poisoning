from __future__ import annotations

import torch
import torch.nn.functional as F


def source_masked_action_mse(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Match the public RDT DT loss reduction exactly.

    The elementwise MSE is masked, then mean() is taken across the
    complete B x K x action_dim tensor. Padded positions therefore
    remain in the denominator.
    """

    if (
        predicted_actions.shape
        != target_actions.shape
    ):
        raise ValueError(
            "prediction/target shape mismatch"
        )

    if mask.shape != (
        predicted_actions.shape[
            0
        ],
        predicted_actions.shape[
            1
        ],
    ):
        raise ValueError(
            "mask shape mismatch"
        )

    loss = F.mse_loss(
        predicted_actions,
        target_actions.detach(),
        reduction="none",
    )

    return (
        loss
        * mask.unsqueeze(
            -1
        )
    ).mean()
