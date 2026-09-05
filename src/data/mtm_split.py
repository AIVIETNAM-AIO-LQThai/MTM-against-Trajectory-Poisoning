from __future__ import annotations

from dataclasses import dataclass

from src.data.trajectories import TrajectorySlice


@dataclass(frozen=True)
class MTMTrajectorySplit:
    """
    Reference MTM trajectory-level train/validation split.

    The official D4RL dataset code splits trajectories in their
    existing order. It does NOT shuffle them before splitting.
    """

    train_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]

    train_trajectories: tuple[TrajectorySlice, ...]
    validation_trajectories: tuple[TrajectorySlice, ...]


def reference_trajectory_split(
    trajectories: list[TrajectorySlice]
    | tuple[TrajectorySlice, ...],
    *,
    train_fraction: float = 0.95,
) -> MTMTrajectorySplit:
    """
    Reproduce the official MTM/JAXRL trajectory-level split.

    For N trajectories:

        train_size = int(train_fraction * N)

    Training receives the first train_size trajectories.
    Validation receives the remaining trajectories.

    No shuffling is performed.
    """
    trajectories = tuple(trajectories)

    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            "train_fraction must be strictly between 0 and 1"
        )

    num_trajectories = len(trajectories)

    if num_trajectories < 2:
        raise ValueError(
            "At least two trajectories are required "
            "for a train/validation split"
        )

    train_size = int(
        train_fraction * num_trajectories
    )

    if train_size <= 0:
        raise ValueError(
            "train split would be empty"
        )

    if train_size >= num_trajectories:
        raise ValueError(
            "validation split would be empty"
        )

    train_ids = tuple(
        range(train_size)
    )

    validation_ids = tuple(
        range(
            train_size,
            num_trajectories,
        )
    )

    return MTMTrajectorySplit(
        train_ids=train_ids,
        validation_ids=validation_ids,
        train_trajectories=trajectories[
            :train_size
        ],
        validation_trajectories=trajectories[
            train_size:
        ],
    )