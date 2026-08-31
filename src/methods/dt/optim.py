from __future__ import annotations

import torch


def create_dt_optimizer_and_scheduler(
    model: torch.nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_steps: int = 10_000,
):
    """
    Reference Decision Transformer optimizer and warmup schedule.
    """

    if warmup_steps <= 0:
        raise ValueError(
            "warmup_steps must be positive."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            (step + 1) / warmup_steps,
            1.0,
        ),
    )

    return optimizer, scheduler