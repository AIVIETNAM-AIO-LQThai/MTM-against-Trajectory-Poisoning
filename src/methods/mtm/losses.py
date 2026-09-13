from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class MTMLossOutput:
    """
    Reference MTM reconstruction losses.

    total_loss:
        Training objective used by the official continuous MTM.

    full_losses:
        Full reconstruction MSE for each modality.

    masked_losses:
        Diagnostic reconstruction error at hidden positions.

    visible_losses:
        Diagnostic reconstruction error at conditioned/visible positions.

    Important:
        Official MTM optimizes full_losses, NOT masked_losses.
    """

    total_loss: torch.Tensor

    full_losses: dict[
        str,
        torch.Tensor,
    ]

    masked_losses: dict[
        str,
        torch.Tensor,
    ]

    visible_losses: dict[
        str,
        torch.Tensor,
    ]


def _expand_reference_mask(
    mask: torch.Tensor,
    *,
    trajectory_length: int,
    tokens_per_timestep: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert reference MTM mask to shape [T, P].

    Accepted:
        [T]
        [T, P]

    Convention:
        1 = visible
        0 = hidden
    """

    mask = torch.as_tensor(
        mask,
        device=device,
    )

    if mask.ndim == 1:
        if mask.shape[0] != (
            trajectory_length
        ):
            raise ValueError(
                "1D mask has wrong "
                "trajectory length"
            )

        mask = (
            mask[:, None]
            .repeat(
                1,
                tokens_per_timestep,
            )
        )

    elif mask.ndim == 2:
        expected_shape = (
            trajectory_length,
            tokens_per_timestep,
        )

        if tuple(
            mask.shape
        ) != expected_shape:
            raise ValueError(
                "2D mask has wrong shape"
            )

    else:
        raise ValueError(
            "mask must have shape "
            "[T] or [T,P]"
        )

    if not bool(
        (
            (mask == 0)
            | (mask == 1)
        ).all()
    ):
        raise ValueError(
            "mask values must be only "
            "0 or 1"
        )

    return mask.to(
        dtype=torch.float32
    )


def _diagnostic_loss(
    raw_loss: torch.Tensor,
    selector: torch.Tensor,
) -> torch.Tensor:
    """
    Reproduce the official MTM diagnostic reduction.

    raw_loss:
        [B,T,P,D]

    selector:
        [T,P]

    Important reference behavior:

    denominator = number of selected TOKENS

    It does NOT multiply the denominator by feature dimension D.

    Therefore masked/visible diagnostics are not numerically
    comparable to the ordinary full-element MSE when D > 1.
    """

    denominator = (
        selector.sum()
    )

    if denominator.item() == 0:
        # Official code reaches a division-by-zero here.
        #
        # We explicitly return NaN instead so the undefined
        # diagnostic is visible without hiding the condition.
        return torch.full(
            (),
            float("nan"),
            dtype=raw_loss.dtype,
            device=raw_loss.device,
        )

    selected = (
        raw_loss
        * selector[
            None,
            :,
            :,
            None,
        ]
    )

    per_batch = (
        selected.sum(
            dim=(1, 2, 3)
        )
        / denominator
    )

    return per_batch.mean()


def reference_mtm_loss(
    targets: Mapping[
        str,
        torch.Tensor,
    ],
    predictions: Mapping[
        str,
        torch.Tensor,
    ],
    masks: Mapping[
        str,
        torch.Tensor,
    ],
    *,
    loss_keys: Sequence[str]
    | None = None,
    reduce_use_sum: bool = False,
) -> MTMLossOutput:
    """
    Reproduce the continuous reference MTM reconstruction loss.

    The official D4RL configuration uses:

        norm = "none"

    therefore this function intentionally does NOT apply the
    optional l2/mae target normalization paths found elsewhere
    in the generic reference implementation.

    Continuous modalities use elementwise squared error.

    Full loss:

        raw_loss.mean(dim=(2,3)).mean()

    by default.

    Training objective:

        sum(full_loss[modality])

    Masked and visible losses are diagnostics only.
    """

    if set(
        targets.keys()
    ) != set(
        predictions.keys()
    ):
        raise ValueError(
            "target and prediction "
            "modalities differ"
        )

    if set(
        targets.keys()
    ) != set(
        masks.keys()
    ):
        raise ValueError(
            "target and mask "
            "modalities differ"
        )

    if len(targets) == 0:
        raise ValueError(
            "targets must not be empty"
        )

    full_losses: dict[
        str,
        torch.Tensor,
    ] = {}

    masked_losses: dict[
        str,
        torch.Tensor,
    ] = {}

    visible_losses: dict[
        str,
        torch.Tensor,
    ] = {}

    mse = nn.MSELoss(
        reduction="none"
    )

    for key in targets.keys():

        target = targets[
            key
        ].to(
            torch.float32
        )

        prediction = predictions[
            key
        ].to(
            torch.float32
        )

        if target.ndim != 4:
            raise ValueError(
                f"{key}: target must "
                "have shape [B,T,P,D]"
            )

        if prediction.shape != (
            target.shape
        ):
            raise ValueError(
                f"{key}: prediction shape "
                "does not match target"
            )

        (
            _,
            trajectory_length,
            tokens_per_timestep,
            _,
        ) = target.shape

        mask = (
            _expand_reference_mask(
                masks[key],
                trajectory_length=(
                    trajectory_length
                ),
                tokens_per_timestep=(
                    tokens_per_timestep
                ),
                device=(
                    target.device
                ),
            )
        )

        raw_loss = mse(
            prediction,
            target,
        )

        if reduce_use_sum:
            full_loss = (
                raw_loss.sum(
                    dim=(2, 3)
                )
                .mean()
            )

        else:
            full_loss = (
                raw_loss.mean(
                    dim=(2, 3)
                )
                .mean()
            )

        visible_loss = (
            _diagnostic_loss(
                raw_loss,
                mask,
            )
        )

        hidden_selector = (
            1.0 - mask
        )

        masked_loss = (
            _diagnostic_loss(
                raw_loss,
                hidden_selector,
            )
        )

        full_losses[
            key
        ] = full_loss

        masked_losses[
            key
        ] = masked_loss

        visible_losses[
            key
        ] = visible_loss

    if loss_keys is None:
        selected_keys = tuple(
            full_losses.keys()
        )

    else:
        selected_keys = tuple(
            loss_keys
        )

        unknown = set(
            selected_keys
        ) - set(
            full_losses.keys()
        )

        if unknown:
            raise ValueError(
                "unknown loss_keys: "
                f"{sorted(unknown)}"
            )

        if len(
            selected_keys
        ) == 0:
            raise ValueError(
                "loss_keys must not be empty"
            )

    total_loss = torch.stack(
        [
            full_losses[key]
            for key
            in selected_keys
        ]
    ).sum()

    return MTMLossOutput(
        total_loss=total_loss,
        full_losses=full_losses,
        masked_losses=(
            masked_losses
        ),
        visible_losses=(
            visible_losses
        ),
    )