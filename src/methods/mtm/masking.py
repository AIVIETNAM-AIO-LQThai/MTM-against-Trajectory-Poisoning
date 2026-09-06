from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch


REFERENCE_MASK_RATIOS = (
    0.5,
    0.6,
    0.7,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
)

REFERENCE_MODE_ORDER = (
    "states",
    "returns",
    "actions",
)

REFERENCE_MODE_WEIGHTS = (
    0.2,
    0.1,
    0.7,
)


@dataclass(frozen=True)
class ReferenceAutoMaskSample:
    """
    One sampled reference AUTO_MASK realization.

    Convention:
        1 = visible / conditioned
        0 = hidden / reconstructed
    """

    masks: dict[str, torch.Tensor]
    selected_mode: str
    selected_position: int


def _validate_mask_ratios(
    mask_ratios: float | Sequence[float],
) -> None:
    if isinstance(
        mask_ratios,
        (list, tuple, np.ndarray),
    ):
        ratios = [
            float(x)
            for x in mask_ratios
        ]

        if len(ratios) == 0:
            raise ValueError(
                "mask_ratios must not be empty"
            )
    else:
        ratios = [
            float(mask_ratios)
        ]

    for ratio in ratios:
        if not np.isfinite(ratio):
            raise ValueError(
                "mask ratio must be finite"
            )

        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(
                "mask ratio must be between "
                "0 and 1"
            )


def _validate_mode_weights(
    mode_weights: Sequence[float],
) -> tuple[float, ...]:
    weights = tuple(
        float(x)
        for x in mode_weights
    )

    if len(weights) != len(
        REFERENCE_MODE_ORDER
    ):
        raise ValueError(
            "mode_weights must contain exactly "
            "three values for "
            "states, returns, actions"
        )

    if any(
        weight < 0.0
        or not np.isfinite(weight)
        for weight in weights
    ):
        raise ValueError(
            "mode_weights must be finite "
            "and non-negative"
        )

    if not np.isclose(
        sum(weights),
        1.0,
    ):
        raise ValueError(
            "mode_weights must sum to 1"
        )

    return weights


def _choice(
    values,
    *,
    rng: np.random.RandomState | None,
    p=None,
):
    if rng is None:
        return np.random.choice(
            values,
            p=p,
        )

    return rng.choice(
        values,
        p=p,
    )


def _randint(
    low: int,
    high: int,
    *,
    rng: np.random.RandomState | None,
) -> int:
    if rng is None:
        return int(
            np.random.randint(
                low,
                high,
            )
        )

    return int(
        rng.randint(
            low,
            high,
        )
    )


def _shuffle(
    values: np.ndarray,
    *,
    rng: np.random.RandomState | None,
) -> None:
    if rng is None:
        np.random.shuffle(
            values
        )
    else:
        rng.shuffle(
            values
        )


def create_reference_full_random_mask(
    data_shape: tuple[int, int],
    *,
    traj_length: int,
    mask_ratios: float
    | Sequence[float] = REFERENCE_MASK_RATIOS,
    device: str | torch.device = "cpu",
    rng: np.random.RandomState | None = None,
) -> torch.Tensor:
    """
    Reproduce official MTM create_full_random_mask().

    data_shape:
        (tokens_per_timestep, feature_dim)

    Important naming mismatch in the reference implementation:

        mask value 1 = visible
        mask value 0 = hidden

    and:

        int(L * P * mask_ratio)

    entries are initialized to ONE.

    Therefore the official `mask_ratio` controls the initial
    visible-token count, not the hidden-token count.
    """
    if traj_length <= 0:
        raise ValueError(
            "traj_length must be positive"
        )

    if len(data_shape) != 2:
        raise ValueError(
            "data_shape must be "
            "(tokens_per_timestep, feature_dim)"
        )

    tokens_per_timestep = int(
        data_shape[0]
    )

    feature_dim = int(
        data_shape[1]
    )

    if tokens_per_timestep <= 0:
        raise ValueError(
            "tokens_per_timestep must be positive"
        )

    if feature_dim <= 0:
        raise ValueError(
            "feature_dim must be positive"
        )

    _validate_mask_ratios(
        mask_ratios
    )

    if isinstance(
        mask_ratios,
        (list, tuple, np.ndarray),
    ):
        chosen_ratio = float(
            _choice(
                mask_ratios,
                rng=rng,
            )
        )
    else:
        chosen_ratio = float(
            mask_ratios
        )

    total_tokens = (
        traj_length
        * tokens_per_timestep
    )

    visible_count = int(
        total_tokens
        * chosen_ratio
    )

    random_mask = np.concatenate(
        [
            np.ones(
                visible_count,
                dtype=np.float64,
            ),
            np.zeros(
                total_tokens
                - visible_count,
                dtype=np.float64,
            ),
        ]
    )

    _shuffle(
        random_mask,
        rng=rng,
    )

    random_mask = (
        random_mask.reshape(
            traj_length,
            tokens_per_timestep,
        )
    )

    # torch.from_numpy preserves the reference float64
    # NumPy mask dtype.
    return torch.from_numpy(
        random_mask
    ).to(
        device
    )


def apply_reference_auto_frontier(
    masks: Mapping[
        str,
        torch.Tensor,
    ],
    *,
    selected_mode: str,
    selected_position: int,
) -> dict[str, torch.Tensor]:
    """
    Apply the official AUTO_MASK future-masking rule.

    Reference modality order:

        states -> returns -> actions

    At the randomly selected position:

    modalities BEFORE selected_mode:
        may remain visible through selected_position;
        future starts at selected_position + 1.

    selected_mode and modalities AFTER it:
        are hidden starting at selected_position.
    """
    if selected_mode not in (
        REFERENCE_MODE_ORDER
    ):
        raise ValueError(
            f"unknown selected_mode: "
            f"{selected_mode}"
        )

    if len(masks) == 0:
        raise ValueError(
            "masks must not be empty"
        )

    result = {
        key: value.clone()
        for key, value
        in masks.items()
    }

    trajectory_lengths = {
        int(mask.shape[0])
        for mask in result.values()
    }

    if len(trajectory_lengths) != 1:
        raise ValueError(
            "all masks must have the "
            "same trajectory length"
        )

    traj_length = next(
        iter(
            trajectory_lengths
        )
    )

    if (
        selected_position < 0
        or selected_position
        >= traj_length
    ):
        raise ValueError(
            "selected_position is out of range"
        )

    reached_selected_mode = False

    for mode in REFERENCE_MODE_ORDER:
        if mode == selected_mode:
            reached_selected_mode = True

        if mode not in result:
            continue

        if reached_selected_mode:
            # Selected mode and all later modalities:
            # current position is hidden too.
            result[
                mode
            ][
                selected_position:
            ] = 0
        else:
            # Earlier modalities may remain visible
            # at the selected position.
            result[
                mode
            ][
                selected_position + 1:
            ] = 0

    return result


def sample_reference_auto_mask(
    data_shapes: Mapping[
        str,
        tuple[int, int],
    ],
    *,
    traj_length: int,
    mask_ratios: float
    | Sequence[float] = REFERENCE_MASK_RATIOS,
    mode_weights: Sequence[
        float
    ] = REFERENCE_MODE_WEIGHTS,
    device: str | torch.device = "cpu",
    rng: np.random.RandomState | None = None,
) -> ReferenceAutoMaskSample:
    """
    Reproduce official MTM AUTO_MASK sampling while optionally
    accepting an explicit legacy NumPy RandomState for tests.

    When rng=None, this uses NumPy's global RNG just like the
    reference implementation.

    Dictionary insertion order matters because the official
    implementation samples one initial random mask for each
    modality while iterating over data_shapes.
    """
    if traj_length <= 0:
        raise ValueError(
            "traj_length must be positive"
        )

    _validate_mask_ratios(
        mask_ratios
    )

    weights = (
        _validate_mode_weights(
            mode_weights
        )
    )

    selected_mode = str(
        _choice(
            REFERENCE_MODE_ORDER,
            rng=rng,
            p=weights,
        )
    )

    selected_position = (
        _randint(
            0,
            traj_length,
            rng=rng,
        )
    )

    masks: dict[
        str,
        torch.Tensor,
    ] = {}

    # Preserve dictionary insertion order.
    #
    # Official d4rl_cont tokenizer order is:
    #
    # states
    # actions
    # returns
    #
    # The exact RNG stream therefore depends on this order.
    for key, data_shape in (
        data_shapes.items()
    ):
        masks[key] = (
            create_reference_full_random_mask(
                data_shape,
                traj_length=traj_length,
                mask_ratios=mask_ratios,
                device=device,
                rng=rng,
            )
        )

    masks = (
        apply_reference_auto_frontier(
            masks,
            selected_mode=(
                selected_mode
            ),
            selected_position=(
                selected_position
            ),
        )
    )

    return ReferenceAutoMaskSample(
        masks=masks,
        selected_mode=selected_mode,
        selected_position=(
            selected_position
        ),
    )


def create_reference_auto_mask(
    data_shapes: Mapping[
        str,
        tuple[int, int],
    ],
    *,
    traj_length: int,
    mask_ratios: float
    | Sequence[float] = REFERENCE_MASK_RATIOS,
    mode_weights: Sequence[
        float
    ] = REFERENCE_MODE_WEIGHTS,
    device: str | torch.device = "cpu",
    rng: np.random.RandomState | None = None,
) -> dict[str, torch.Tensor]:
    """
    Convenience interface matching what the MTM model needs:
    a dictionary of modality masks.
    """
    return sample_reference_auto_mask(
        data_shapes,
        traj_length=traj_length,
        mask_ratios=mask_ratios,
        mode_weights=mode_weights,
        device=device,
        rng=rng,
    ).masks