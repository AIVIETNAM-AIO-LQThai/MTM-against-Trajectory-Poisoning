from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.mtm_targets import (
    build_reference_mtm_returns,
)

from src.data.mtm_windows import (
    MTMWindowDataset,
)


@dataclass(frozen=True)
class MTMSample:
    """
    Complete project-aligned standalone MTM sample.

    Modalities follow the official continuous D4RL MTM setup:

        states
        actions
        rewards
        returns

    Metadata is retained for later poisoning/mechanism audits.
    """

    trajectory_id: int
    local_start: int
    global_start: int
    global_end: int

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    returns: np.ndarray

    @property
    def length(self) -> int:
        return self.global_end - self.global_start


class ReferenceMTMDataset:
    """
    Project-aligned standalone MTM dataset.

    Important distinction
    ---------------------
    Dataset identity and trajectory boundaries come from the
    frozen Group-1 project contract.

    Reward-to-return target semantics follow the official MTM
    SequenceDataset implementation.

    This therefore reproduces MTM target semantics while keeping
    the project dataset universe fixed.
    """

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        timeouts: np.ndarray,
        *,
        trajectory_length: int = 4,
        max_path_length: int = 1000,
        discount: float = 1.5,
    ) -> None:
        observations = np.asarray(
            observations
        )

        actions = np.asarray(
            actions
        )

        rewards = np.asarray(
            rewards
        )

        terminals = np.asarray(
            terminals,
            dtype=bool,
        )

        timeouts = np.asarray(
            timeouts,
            dtype=bool,
        )

        if rewards.ndim == 1:
            reward_vector = rewards

        elif (
            rewards.ndim == 2
            and rewards.shape[1] == 1
        ):
            reward_vector = rewards[:, 0]

        else:
            raise ValueError(
                "rewards must have shape [N] "
                "or [N, 1]"
            )

        n = len(observations)

        if not (
            len(actions)
            == len(reward_vector)
            == len(terminals)
            == len(timeouts)
            == n
        ):
            raise ValueError(
                "dataset arrays have "
                "inconsistent lengths"
            )

        self.window_dataset = (
            MTMWindowDataset(
                observations,
                actions,
                terminals,
                timeouts,
                trajectory_length=(
                    trajectory_length
                ),
            )
        )

        self.rewards = (
            reward_vector.astype(
                np.float32,
                copy=False,
            )[:, None]
        )

        self.returns = (
            build_reference_mtm_returns(
                reward_vector,
                self.window_dataset.trajectories,
                max_path_length=max_path_length,
                discount=discount,
            )
        )

        self.trajectory_length = int(
            trajectory_length
        )

        self.max_path_length = int(
            max_path_length
        )

        self.discount = float(
            discount
        )

    def __len__(self) -> int:
        return len(
            self.window_dataset
        )

    @property
    def trajectories(self):
        return (
            self.window_dataset.trajectories
        )

    @property
    def trailing_transitions(self) -> int:
        return (
            self.window_dataset
            .trailing_transitions
        )

    @property
    def num_completed_trajectories(
        self,
    ) -> int:
        return (
            self.window_dataset
            .num_completed_trajectories
        )

    @property
    def num_used_transitions(
        self,
    ) -> int:
        return (
            self.window_dataset
            .num_used_transitions
        )

    def __getitem__(
        self,
        index: int,
    ) -> MTMSample:
        window = (
            self.window_dataset[index]
        )

        rewards = self.rewards[
            window.global_start:
            window.global_end
        ]

        returns = self.returns[
            window.global_start:
            window.global_end
        ]

        if np.isnan(returns).any():
            raise RuntimeError(
                "MTM window unexpectedly contains "
                "unfinished trailing transitions"
            )

        return MTMSample(
            trajectory_id=(
                window.trajectory_id
            ),
            local_start=(
                window.local_start
            ),
            global_start=(
                window.global_start
            ),
            global_end=(
                window.global_end
            ),
            states=window.states,
            actions=window.actions,
            rewards=rewards,
            returns=returns,
        )