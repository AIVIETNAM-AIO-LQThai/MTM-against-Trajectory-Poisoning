from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.trajectories import (
    TrajectorySlice,
    find_completed_trajectories,
)


@dataclass(frozen=True)
class MTMWindow:
    """
    One fixed-length clean trajectory window.

    Arrays are slices/views of the original dataset arrays.
    No normalization, masking, or return construction happens here.
    """

    trajectory_id: int

    # Position inside the completed trajectory.
    local_start: int

    # Positions in the original raw HDF5 dataset.
    global_start: int
    global_end: int  # exclusive

    states: np.ndarray
    actions: np.ndarray

    @property
    def length(self) -> int:
        return self.global_end - self.global_start


class MTMWindowDataset:
    """
    Project-aligned clean MTM trajectory-window adapter.

    Important invariants
    --------------------
    1. Uses Group-1 `find_completed_trajectories`.
    2. Never crosses terminal/timeout boundaries.
    3. Excludes any unfinished trailing fragment.
    4. Does NOT perform normalization.
    5. Does NOT compute DT RTG.
    6. Does NOT generate MTM masks.
    7. Does NOT copy every possible window into memory.

    A dataset index is mapped lazily to a fixed-length window.
    """

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        terminals: np.ndarray,
        timeouts: np.ndarray,
        *,
        trajectory_length: int,
    ) -> None:
        observations = np.asarray(observations)
        actions = np.asarray(actions)
        terminals = np.asarray(terminals, dtype=bool)
        timeouts = np.asarray(timeouts, dtype=bool)

        if observations.ndim != 2:
            raise ValueError(
                "observations must be a 2D array "
                f"[N, state_dim], got {observations.shape}"
            )

        if actions.ndim != 2:
            raise ValueError(
                "actions must be a 2D array "
                f"[N, action_dim], got {actions.shape}"
            )

        num_transitions = len(observations)

        if not (
            len(actions)
            == len(terminals)
            == len(timeouts)
            == num_transitions
        ):
            raise ValueError(
                "observations, actions, terminals, and timeouts "
                "must have equal first-dimension length"
            )

        if trajectory_length <= 0:
            raise ValueError(
                "trajectory_length must be positive"
            )

        self.observations = observations
        self.actions = actions
        self.terminals = terminals
        self.timeouts = timeouts

        self.trajectory_length = int(
            trajectory_length
        )

        (
            self.trajectories,
            self.trailing_transitions,
        ) = find_completed_trajectories(
            terminals,
            timeouts,
        )

        # Number of valid fixed-length windows contributed by
        # each completed trajectory.
        #
        # For trajectory length T and window length L:
        #
        #     count = T - L + 1
        #
        # when T >= L, otherwise 0.
        self._window_counts = np.asarray(
            [
                max(
                    0,
                    trajectory.length
                    - self.trajectory_length
                    + 1,
                )
                for trajectory in self.trajectories
            ],
            dtype=np.int64,
        )

        # Cumulative counts allow O(log num_trajectories)
        # mapping from dataset index -> trajectory.
        self._cumulative_window_counts = np.cumsum(
            self._window_counts,
            dtype=np.int64,
        )

        if len(self._cumulative_window_counts) == 0:
            self._num_windows = 0
        else:
            self._num_windows = int(
                self._cumulative_window_counts[-1]
            )

    def __len__(self) -> int:
        return self._num_windows

    @property
    def num_completed_trajectories(self) -> int:
        return len(self.trajectories)

    @property
    def num_used_transitions(self) -> int:
        return int(
            sum(
                trajectory.length
                for trajectory in self.trajectories
            )
        )

    def _locate_window(
        self,
        index: int,
    ) -> tuple[int, int]:
        """
        Return:
            trajectory_id,
            local_start
        """
        if not isinstance(index, (int, np.integer)):
            raise TypeError(
                "MTM window index must be an integer"
            )

        index = int(index)

        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(
                f"MTM window index {index} out of range "
                f"for dataset of length {len(self)}"
            )

        trajectory_id = int(
            np.searchsorted(
                self._cumulative_window_counts,
                index,
                side="right",
            )
        )

        previous_cumulative = (
            0
            if trajectory_id == 0
            else int(
                self._cumulative_window_counts[
                    trajectory_id - 1
                ]
            )
        )

        local_start = (
            index - previous_cumulative
        )

        return trajectory_id, local_start

    def __getitem__(
        self,
        index: int,
    ) -> MTMWindow:
        (
            trajectory_id,
            local_start,
        ) = self._locate_window(index)

        trajectory: TrajectorySlice = (
            self.trajectories[trajectory_id]
        )

        global_start = (
            trajectory.start + local_start
        )

        global_end = (
            global_start
            + self.trajectory_length
        )

        # This must always hold if indexing is correct.
        if global_end > trajectory.end:
            raise RuntimeError(
                "Internal MTM window indexing crossed "
                "a trajectory boundary"
            )

        states = self.observations[
            global_start:global_end
        ]

        actions = self.actions[
            global_start:global_end
        ]

        if len(states) != self.trajectory_length:
            raise RuntimeError(
                "Incorrect state-window length"
            )

        if len(actions) != self.trajectory_length:
            raise RuntimeError(
                "Incorrect action-window length"
            )

        return MTMWindow(
            trajectory_id=trajectory_id,
            local_start=local_start,
            global_start=global_start,
            global_end=global_end,
            states=states,
            actions=actions,
        )