from __future__ import annotations

import numpy as np

from src.data.trajectories import TrajectorySlice


def compute_state_statistics(
    observations: np.ndarray,
    trajectories: list[TrajectorySlice],
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Decision Transformer state-normalization statistics
    from completed training trajectories only.

    The five trailing unfinished Walker2d transitions are excluded.

    The reference DT convention is:

        state_mean = mean(states)
        state_std  = std(states) + 1e-6

    Parameters
    ----------
    observations:
        Array with shape (N, state_dim).

    trajectories:
        Completed trajectory slices returned by
        find_completed_trajectories().

    epsilon:
        Small value added to standard deviation.

    Returns
    -------
    state_mean:
        Shape (state_dim,)

    state_std:
        Shape (state_dim,)
    """
    observations = np.asarray(observations)

    if observations.ndim != 2:
        raise ValueError(
            "observations must have shape (N, state_dim), "
            f"got {observations.shape}"
        )
    if len(trajectories) == 0:
        raise ValueError(
            "No completed trajectories were provided."
        )

    # Trajectories are contiguous from transition 0 through
    # the end of the final completed trajectory.
    expected_start = 0

    for trajectory in trajectories:
        if trajectory.start != expected_start:
            raise ValueError(
                "Trajectory slices are not contiguous: "
                f"expected start {expected_start}, "
                f"got {trajectory.start}"
            )

        expected_start = trajectory.end

    used_transitions = expected_start
    states = observations[:used_transitions]

    if not np.isfinite(states).all():
        raise ValueError(
            "Training observations contain NaN or Inf."
        )

    # Intentionally preserve NumPy's reference behavior rather
    # than forcing a different accumulation dtype.
    state_mean = np.mean(
        states,
        axis=0,
    )
    state_std = (
        np.std(
            states,
            axis=0,
        )
        + epsilon
    )

    return state_mean, state_std


def normalize_states(
    states: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
) -> np.ndarray:
    """
    Normalize states using previously fitted statistics.

        normalized = (state - mean) / std
    """
    states = np.asarray(states)
    state_mean = np.asarray(state_mean)
    state_std = np.asarray(state_std)

    if states.shape[-1] != state_mean.shape[0]:
        raise ValueError(
            "State dimension does not match mean dimension."
        )
    if state_mean.shape != state_std.shape:
        raise ValueError(
            "state_mean and state_std must have identical shapes."
        )
    if np.any(state_std <= 0):
        raise ValueError(
            "state_std must be strictly positive."
        )

    return (
        states - state_mean
    ) / state_std