from __future__ import annotations

import numpy as np

from src.data.trajectories import TrajectorySlice


def _as_reward_vector(
    rewards: np.ndarray,
) -> np.ndarray:
    """
    Convert rewards to shape [T].

    Accepts either:
        [T]
    or:
        [T, 1]
    """
    rewards = np.asarray(rewards)

    if rewards.ndim == 1:
        return rewards

    if (
        rewards.ndim == 2
        and rewards.shape[1] == 1
    ):
        return rewards[:, 0]

    raise ValueError(
        "rewards must have shape [T] or [T, 1], "
        f"got {rewards.shape}"
    )


def compute_reference_mtm_returns(
    rewards: np.ndarray,
    *,
    max_path_length: int = 1000,
    discount: float = 1.5,
) -> np.ndarray:
    """
    Reproduce the future-value construction used by the
    official facebookresearch/mtm SequenceDataset.

    Important
    ---------
    This is NOT Decision Transformer RTG.

    Official MTM semantics:

    If discount <= 1:
        V_t =
            r_{t+1}
            + discount * r_{t+2}
            + discount^2 * r_{t+3}
            + ...

    So the CURRENT reward r_t is excluded.

    If discount > 1:
        the official implementation switches to an
        "average future reward" mode:

        V_t =
            sum_{k=t+1}^{T-1} r_k
            / (max_path_length - t)

    Note that the denominator uses max_path_length,
    not the actual remaining trajectory length.

    This mirrors the official implementation's
    zero-padded segmented representation.

    Returns
    -------
    values:
        float64 array with shape [T, 1].
    """
    rewards = _as_reward_vector(
        rewards
    )

    trajectory_length = len(rewards)

    if max_path_length <= 0:
        raise ValueError(
            "max_path_length must be positive"
        )

    if trajectory_length > max_path_length:
        raise ValueError(
            "trajectory length exceeds "
            f"max_path_length: "
            f"{trajectory_length} > "
            f"{max_path_length}"
        )

    if not np.isfinite(discount):
        raise ValueError(
            "discount must be finite"
        )

    rewards_f64 = rewards.astype(
        np.float64,
        copy=False,
    )

    values = np.zeros(
        trajectory_length,
        dtype=np.float64,
    )

    if trajectory_length == 0:
        return values[:, None]

    # --------------------------------------------------------
    # Official special case:
    #
    # discount > 1
    #
    # becomes undiscounted average-future-reward mode.
    # --------------------------------------------------------
    if discount > 1.0:
        future_sum = 0.0

        for t in range(
            trajectory_length - 1,
            -1,
            -1,
        ):
            divisor = (
                max_path_length - t
            )

            values[t] = (
                future_sum / divisor
            )

            future_sum += float(
                rewards_f64[t]
            )

        return values[:, None]

    # --------------------------------------------------------
    # Standard discounted future-value mode.
    #
    # V_t excludes r_t.
    #
    # V_t = r_{t+1} + gamma V_{t+1}
    # --------------------------------------------------------
    values[
        trajectory_length - 1
    ] = 0.0

    for t in range(
        trajectory_length - 2,
        -1,
        -1,
    ):
        values[t] = (
            rewards_f64[t + 1]
            + discount
            * values[t + 1]
        )

    return values[:, None]


def build_reference_mtm_returns(
    rewards: np.ndarray,
    trajectories: list[TrajectorySlice],
    *,
    max_path_length: int = 1000,
    discount: float = 1.5,
) -> np.ndarray:
    """
    Construct reference-MTM return targets for all completed
    trajectories in a raw dataset.

    Completed trajectory positions receive finite values.

    Unfinished trailing transitions remain NaN intentionally.
    This prevents them from being accidentally used by later
    MTM code.
    """
    rewards = _as_reward_vector(
        rewards
    )

    values = np.full(
        (len(rewards), 1),
        np.nan,
        dtype=np.float64,
    )

    previous_end = 0

    for trajectory in trajectories:
        if trajectory.start < 0:
            raise ValueError(
                "trajectory start must be non-negative"
            )

        if trajectory.end > len(rewards):
            raise ValueError(
                "trajectory exceeds reward array"
            )

        if trajectory.start > trajectory.end:
            raise ValueError(
                "trajectory start exceeds end"
            )

        if trajectory.start < previous_end:
            raise ValueError(
                "trajectories overlap or are out of order"
            )

        trajectory_rewards = rewards[
            trajectory.start:
            trajectory.end
        ]

        trajectory_values = (
            compute_reference_mtm_returns(
                trajectory_rewards,
                max_path_length=max_path_length,
                discount=discount,
            )
        )

        values[
            trajectory.start:
            trajectory.end
        ] = trajectory_values

        previous_end = trajectory.end

    return values