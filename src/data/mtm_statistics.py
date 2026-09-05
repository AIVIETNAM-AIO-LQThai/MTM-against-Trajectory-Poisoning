from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.trajectories import (
    TrajectorySlice,
)


@dataclass(frozen=True)
class DataStatistics:
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray


def _as_feature_matrix(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """
    Convert a modality to [N, D].

    Accepts:
        [N]
        [N, D]
    """
    values = np.asarray(values)

    if values.ndim == 1:
        return values[:, None]

    if values.ndim == 2:
        return values

    raise ValueError(
        f"{name} must have shape [N] or [N, D], "
        f"got {values.shape}"
    )


def pad_completed_trajectories(
    values: np.ndarray,
    trajectories: list[TrajectorySlice]
    | tuple[TrajectorySlice, ...],
    *,
    max_path_length: int = 1000,
    name: str = "values",
) -> np.ndarray:
    """
    Reproduce the zero-padded segmented representation used
    by the official MTM SequenceDataset.

    Output:
        [num_trajectories, max_path_length, feature_dim]
    """
    values = _as_feature_matrix(
        values,
        name=name,
    )

    trajectories = tuple(
        trajectories
    )

    if max_path_length <= 0:
        raise ValueError(
            "max_path_length must be positive"
        )

    if len(trajectories) == 0:
        raise ValueError(
            "at least one trajectory is required"
        )

    feature_dim = values.shape[1]

    padded = np.zeros(
        (
            len(trajectories),
            max_path_length,
            feature_dim,
        ),
        dtype=values.dtype,
    )

    for trajectory_index, trajectory in enumerate(
        trajectories
    ):
        if trajectory.start < 0:
            raise ValueError(
                "trajectory start must be non-negative"
            )

        if trajectory.end > len(values):
            raise ValueError(
                "trajectory exceeds modality array"
            )

        trajectory_length = (
            trajectory.end
            - trajectory.start
        )

        if trajectory_length <= 0:
            raise ValueError(
                "trajectory must contain at least "
                "one transition"
            )

        if trajectory_length > max_path_length:
            raise ValueError(
                f"trajectory length {trajectory_length} "
                f"exceeds max_path_length "
                f"{max_path_length}"
            )

        padded[
            trajectory_index,
            :trajectory_length,
        ] = values[
            trajectory.start:
            trajectory.end
        ]

    return padded


def statistics_from_padded(
    padded: np.ndarray,
) -> DataStatistics:
    """
    Match SequenceDataset.trajectory_statistics():

        mean(axis=(0, 1))
        std(axis=(0, 1))
        min(axis=(0, 1))
        max(axis=(0, 1))
    """
    padded = np.asarray(
        padded
    )

    if padded.ndim != 3:
        raise ValueError(
            "padded modality must have shape "
            "[num_trajectories, max_path_length, feature_dim]"
        )

    return DataStatistics(
        mean=padded.mean(
            axis=(0, 1)
        ),
        std=padded.std(
            axis=(0, 1)
        ),
        min=padded.min(
            axis=(0, 1)
        ),
        max=padded.max(
            axis=(0, 1)
        ),
    )


def compute_reference_mtm_statistics(
    observations: np.ndarray,
    actions: np.ndarray,
    returns: np.ndarray,
    train_trajectories: list[TrajectorySlice]
    | tuple[TrajectorySlice, ...],
    *,
    max_path_length: int = 1000,
) -> dict[str, DataStatistics]:
    """
    Compute the training statistics used by the reference
    continuous MTM tokenizers.

    Modeled modalities for the official d4rl_cont setup:

        states
        actions
        returns

    Rewards are deliberately NOT included here because the
    official d4rl_cont tokenizer configuration does not create
    a reward tokenizer.
    """
    modalities = {
        "states": observations,
        "actions": actions,
        "returns": returns,
    }

    result: dict[
        str,
        DataStatistics,
    ] = {}

    for name, values in modalities.items():
        padded = (
            pad_completed_trajectories(
                values,
                train_trajectories,
                max_path_length=(
                    max_path_length
                ),
                name=name,
            )
        )

        result[name] = (
            statistics_from_padded(
                padded
            )
        )

    return result