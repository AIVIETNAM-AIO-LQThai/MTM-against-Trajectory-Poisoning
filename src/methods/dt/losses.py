from __future__ import annotations

import torch


def masked_action_mse(
    action_preds: torch.Tensor,
    action_targets: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Decision Transformer action-prediction loss.

    Only valid trajectory positions contribute to the MSE.
    Left-padded positions (attention_mask == 0) are ignored.

    Parameters
    ----------
    action_preds:
        Shape (B, K, action_dim).

    action_targets:
        Shape (B, K, action_dim).

    attention_mask:
        Shape (B, K).
        1 = valid transition
        0 = padding

    Returns
    -------
    torch.Tensor
        Scalar MSE loss.
    """

    if action_preds.shape != action_targets.shape:
        raise ValueError(
            "action_preds and action_targets must have "
            f"identical shapes, got "
            f"{action_preds.shape} and "
            f"{action_targets.shape}"
        )

    if action_preds.ndim != 3:
        raise ValueError(
            "Expected action tensors with shape "
            "(B, K, action_dim)."
        )

    batch_size, seq_length, action_dim = (
        action_preds.shape
    )

    if attention_mask.shape != (
        batch_size,
        seq_length,
    ):
        raise ValueError(
            "attention_mask must have shape "
            f"({batch_size}, {seq_length}), "
            f"got {attention_mask.shape}"
        )

    valid = (
        attention_mask.reshape(-1) > 0
    )

    if not torch.any(valid):
        raise ValueError(
            "Batch contains no valid action positions."
        )

    preds = action_preds.reshape(
        -1,
        action_dim,
    )[valid]

    targets = action_targets.reshape(
        -1,
        action_dim,
    )[valid]

    return torch.mean(
        (preds - targets) ** 2
    )