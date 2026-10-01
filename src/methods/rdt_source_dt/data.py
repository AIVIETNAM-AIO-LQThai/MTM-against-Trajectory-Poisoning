from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from src.data.trajectories import (
    find_completed_trajectories,
)


def discounted_cumsum(
    values: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    values = np.asarray(
        values
    )

    if values.ndim != 1:
        raise ValueError(
            "discounted_cumsum expects 1D input"
        )

    if len(
        values
    ) == 0:
        return values.copy()

    result = np.zeros_like(
        values
    )

    result[
        -1
    ] = values[
        -1
    ]

    for index in reversed(
        range(
            len(
                values
            )
            - 1
        )
    ):
        result[
            index
        ] = (
            values[
                index
            ]
            + gamma
            * result[
                index
                + 1
            ]
        )

    return result


def pad_right(
    array: np.ndarray,
    *,
    length: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    array = np.asarray(
        array
    )

    pad_size = (
        length
        - array.shape[
            0
        ]
    )

    if pad_size <= 0:
        return array

    pad_width = [
        (
            0,
            0,
        )
        for _ in range(
            array.ndim
        )
    ]

    pad_width[
        0
    ] = (
        0,
        pad_size,
    )

    return np.pad(
        array,
        pad_width=(
            pad_width
        ),
        mode="constant",
        constant_values=(
            fill_value
        ),
    )


@dataclass(frozen=True)
class SourceTrajectory:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    returns: np.ndarray

    @property
    def length(
        self,
    ) -> int:
        return int(
            self.actions.shape[
                0
            ]
        )


def load_source_trajectories(
    path: Path,
    *,
    expected_num_trajectories: int,
    expected_num_transitions: int,
    expected_trailing_transitions: int,
) -> list[SourceTrajectory]:
    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with h5py.File(
        path,
        "r",
    ) as handle:
        observations = (
            handle[
                "observations"
            ][:]
        )

        actions = (
            handle[
                "actions"
            ][:]
        )

        rewards = (
            handle[
                "rewards"
            ][:]
        )

        terminals = (
            handle[
                "terminals"
            ][:]
            .astype(
                bool
            )
        )

        timeouts = (
            handle[
                "timeouts"
            ][:]
            .astype(
                bool
            )
        )

    slices, trailing = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    used = sum(
        trajectory.length
        for trajectory
        in slices
    )

    if (
        len(
            slices
        )
        != expected_num_trajectories
        or used
        != expected_num_transitions
        or trailing
        != expected_trailing_transitions
    ):
        raise RuntimeError(
            "source-compatible dataset contract mismatch: "
            f"trajectories={len(slices)} "
            f"transitions={used} "
            f"trailing={trailing}"
        )

    trajectories: list[
        SourceTrajectory
    ] = []

    for trajectory in slices:
        obs = np.asarray(
            observations[
                trajectory.start:
                trajectory.end
            ],
            dtype=np.float32,
        )

        act = np.asarray(
            actions[
                trajectory.start:
                trajectory.end
            ],
            dtype=np.float32,
        )

        rew = np.asarray(
            rewards[
                trajectory.start:
                trajectory.end
            ],
            dtype=np.float32,
        )

        ret = discounted_cumsum(
            rew,
            gamma=1.0,
        ).astype(
            np.float32,
            copy=False,
        )

        trajectories.append(
            SourceTrajectory(
                observations=(
                    obs
                ),
                actions=act,
                rewards=rew,
                returns=ret,
            )
        )

    return trajectories


class SourceSequenceDataset:
    """
    Sequence sampler matching the public RDT vanilla-DT implementation.

    Important differences from this repository's older DT batcher:
    - global NumPy RNG;
    - right padding;
    - zero action padding;
    - no state normalization;
    - padded timesteps keep increasing;
    - RTG is multiplied by 0.001.
    """

    def __init__(
        self,
        trajectories: list[
            SourceTrajectory
        ],
        *,
        seq_len: int = 20,
        episode_len: int = 1000,
        reward_scale: float = 0.001,
    ) -> None:
        if not trajectories:
            raise ValueError(
                "no trajectories"
            )

        self.trajectories = (
            trajectories
        )

        self.seq_len = (
            int(
                seq_len
            )
        )

        self.episode_len = (
            int(
                episode_len
            )
        )

        self.reward_scale = (
            float(
                reward_scale
            )
        )

        lengths = np.asarray(
            [
                trajectory.length
                for trajectory
                in trajectories
            ],
            dtype=np.float64,
        )

        if np.any(
            lengths <= 0
        ):
            raise ValueError(
                "empty trajectory"
            )

        self.sample_prob = (
            lengths
            / lengths.sum()
        )

    def prepare_sample(
        self,
        trajectory_index: int,
        start_index: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        trajectory = (
            self.trajectories[
                int(
                    trajectory_index
                )
            ]
        )

        if not (
            0
            <= start_index
            < trajectory.length
        ):
            raise ValueError(
                "invalid start index"
            )

        end = (
            start_index
            + self.seq_len
        )

        states = (
            trajectory.observations[
                start_index:end
            ]
        )

        actions = (
            trajectory.actions[
                start_index:end
            ]
        )

        returns = (
            trajectory.returns[
                start_index:end
            ]
            .reshape(
                -1,
                1,
            )
        )

        time_steps = np.arange(
            start_index,
            start_index
            + self.seq_len,
            dtype=np.int64,
        )

        valid_length = (
            states.shape[
                0
            ]
        )

        mask = np.hstack(
            [
                np.ones(
                    valid_length,
                    dtype=np.float32,
                ),
                np.zeros(
                    self.seq_len
                    - valid_length,
                    dtype=np.float32,
                ),
            ]
        )

        states = pad_right(
            states,
            length=(
                self.seq_len
            ),
            fill_value=0.0,
        )

        actions = pad_right(
            actions,
            length=(
                self.seq_len
            ),
            fill_value=0.0,
        )

        returns = pad_right(
            returns,
            length=(
                self.seq_len
            ),
            fill_value=0.0,
        )

        returns = (
            returns
            * self.reward_scale
        )

        return (
            states.astype(
                np.float32,
                copy=False,
            ),
            actions.astype(
                np.float32,
                copy=False,
            ),
            returns.astype(
                np.float32,
                copy=False,
            ),
            time_steps,
            mask,
        )

    def get_batch(
        self,
        batch_size: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        trajectory_ids = (
            np.random.choice(
                np.arange(
                    len(
                        self.trajectories
                    )
                ),
                size=(
                    batch_size
                ),
                p=self.sample_prob,
                replace=True,
            )
        )

        states = []
        actions = []
        returns = []
        time_steps = []
        masks = []
        starts = []

        for trajectory_id in (
            trajectory_ids
        ):
            trajectory = (
                self.trajectories[
                    int(
                        trajectory_id
                    )
                ]
            )

            start_index = int(
                np.random.randint(
                    0,
                    trajectory.length,
                )
            )

            (
                state,
                action,
                return_to_go,
                timestep,
                mask,
            ) = self.prepare_sample(
                int(
                    trajectory_id
                ),
                start_index,
            )

            states.append(
                state
            )

            actions.append(
                action
            )

            returns.append(
                return_to_go
            )

            time_steps.append(
                timestep
            )

            masks.append(
                mask
            )

            starts.append(
                start_index
            )

        return (
            torch.tensor(
                np.asarray(
                    states
                )
            ),
            torch.tensor(
                np.asarray(
                    actions
                )
            ),
            torch.tensor(
                np.asarray(
                    returns
                )
            ),
            torch.tensor(
                np.asarray(
                    time_steps
                )
            ),
            torch.tensor(
                np.asarray(
                    masks
                )
            ),
            torch.tensor(
                np.asarray(
                    trajectory_ids,
                    dtype=np.int64,
                )
            ),
            torch.tensor(
                np.asarray(
                    starts,
                    dtype=np.int64,
                )
            ),
        )
