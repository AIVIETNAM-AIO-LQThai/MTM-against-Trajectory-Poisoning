from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def reference_mtm_lr_multiplier(
    step: int,
    *,
    warmup_steps: int,
    num_train_steps: int,
) -> float:
    """
    Reproduce the reference MTM schedule.

    1. Linear warmup:
           step / warmup_steps

    2. Cosine decay after warmup:
           0.5 * (
               1 + cos(
                   progress * pi
               )
           )

    where:

        progress =
            (step - warmup_steps)
            /
            (num_train_steps - warmup_steps)
    """

    if warmup_steps <= 0:
        raise ValueError(
            "warmup_steps must be positive"
        )

    if num_train_steps <= (
        warmup_steps
    ):
        raise ValueError(
            "num_train_steps must exceed "
            "warmup_steps"
        )

    if step < 0:
        raise ValueError(
            "step must be non-negative"
        )

    if step < warmup_steps:
        return (
            float(step)
            / float(warmup_steps)
        )

    adjusted_step = (
        step - warmup_steps
    )

    decay_steps = (
        num_train_steps
        - warmup_steps
    )

    progress = (
        float(adjusted_step)
        / float(decay_steps)
    )

    # Protect against accidental calls beyond
    # the intended training horizon.
    progress = min(
        max(
            progress,
            0.0,
        ),
        1.0,
    )

    return 0.5 * (
        1.0
        + math.cos(
            progress
            * math.pi
        )
    )


def create_reference_mtm_optimizer_and_scheduler(
    parameters: Iterable[
        torch.nn.Parameter
    ],
    *,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.005,
    warmup_steps: int = 40_000,
    num_train_steps: int = 140_010,
) -> tuple[
    AdamW,
    LambdaLR,
]:
    """
    Reference MTM optimizer + scheduler.

    AdamW uses PyTorch defaults for beta/epsilon because the
    reference implementation does not override them.
    """

    if learning_rate <= 0.0:
        raise ValueError(
            "learning_rate must be positive"
        )

    if weight_decay < 0.0:
        raise ValueError(
            "weight_decay must be non-negative"
        )

    optimizer = AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            reference_mtm_lr_multiplier(
                step,
                warmup_steps=(
                    warmup_steps
                ),
                num_train_steps=(
                    num_train_steps
                ),
            )
        ),
    )

    return (
        optimizer,
        scheduler,
    )