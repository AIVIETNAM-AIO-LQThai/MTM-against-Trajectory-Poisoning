from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np

from src.data.rtg import compute_rtg
from src.data.trajectories import TrajectorySlice


@dataclass
class DTBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    rtg: np.ndarray
    timesteps: np.ndarray
    attention_mask: np.ndarray

    # Audit/debug information.
    trajectory_indices: np.ndarray
    start_indices: np.ndarray


def trajectory_sampling_probabilities(
    trajectories: list[TrajectorySlice],
) -> np.ndarray:
    """
    Reference DT samples trajectories proportional to
    trajectory length.
    """

    if len(trajectories) == 0:
        raise ValueError("No trajectories supplied.")

    lengths = np.asarray(
        [trajectory.length for trajectory in trajectories],
        dtype=np.float64,
    )

    if np.any(lengths <= 0):
        raise ValueError(
            "All trajectories must contain at least one transition."
        )

    return lengths / lengths.sum()


def sample_dt_batch(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    trajectories: list[TrajectorySlice],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    *,
    batch_size: int = 64,
    context_length: int = 20,
    max_ep_len: int = 1000,
    rtg_scale: float = 1000.0,
    np_rng: np.random.Generator,
    py_rng: Random,
) -> DTBatch:
    """
    Construct a Decision Transformer training batch following
    the original DT D4RL preprocessing convention.

    Sequence convention
    -------------------
    K = context_length

    states:
        (B, K, state_dim)

    actions:
        (B, K, action_dim)

    rewards:
        (B, K, 1)

    dones:
        (B, K)

    rtg:
        (B, K + 1, 1)

    timesteps:
        (B, K)

    attention_mask:
        (B, K)

    Padding is added on the LEFT.
    """

    observations = np.asarray(observations)
    actions = np.asarray(actions)
    rewards = np.asarray(rewards)
    terminals = np.asarray(terminals, dtype=bool)

    state_mean = np.asarray(state_mean)
    state_std = np.asarray(state_std)

    if observations.ndim != 2:
        raise ValueError(
            f"observations must be 2D, got {observations.shape}"
        )

    if actions.ndim != 2:
        raise ValueError(
            f"actions must be 2D, got {actions.shape}"
        )

    n = len(observations)

    if not (
        len(actions)
        == len(rewards)
        == len(terminals)
        == n
    ):
        raise ValueError(
            "Dataset arrays have inconsistent lengths."
        )

    state_dim = observations.shape[1]
    action_dim = actions.shape[1]

    if state_mean.shape != (state_dim,):
        raise ValueError(
            f"state_mean must have shape ({state_dim},)"
        )

    if state_std.shape != (state_dim,):
        raise ValueError(
            f"state_std must have shape ({state_dim},)"
        )

    if np.any(state_std <= 0):
        raise ValueError(
            "state_std must be strictly positive."
        )

    if context_length <= 0:
        raise ValueError(
            "context_length must be positive."
        )

    if rtg_scale <= 0:
        raise ValueError(
            "rtg_scale must be positive."
        )

    # --------------------------------------------------------
    # Reference DT:
    # sample trajectories proportional to trajectory length.
    # --------------------------------------------------------

    p_sample = trajectory_sampling_probabilities(
        trajectories
    )

    sampled_trajectory_indices = np_rng.choice(
        np.arange(len(trajectories)),
        size=batch_size,
        replace=True,
        p=p_sample,
    )

    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_dones = []
    batch_rtg = []
    batch_timesteps = []
    batch_masks = []

    sampled_start_indices = []

    for trajectory_index in sampled_trajectory_indices:
        trajectory = trajectories[
            int(trajectory_index)
        ]

        trajectory_length = trajectory.length

        # Reference implementation:
        #
        # si = random.randint(
        #     0,
        #     traj_length - 1
        # )
        local_start = py_rng.randint(
            0,
            trajectory_length - 1,
        )

        sampled_start_indices.append(
            local_start
        )

        global_start = (
            trajectory.start + local_start
        )

        global_end = min(
            global_start + context_length,
            trajectory.end,
        )

        # ----------------------------------------------------
        # Raw sequence.
        # ----------------------------------------------------

        s = observations[
            global_start:global_end
        ]

        a = actions[
            global_start:global_end
        ]

        r = rewards[
            global_start:global_end
        ]

        d = terminals[
            global_start:global_end
        ]

        tlen = len(s)

        if tlen <= 0:
            raise RuntimeError(
                "Sampled an empty trajectory sequence."
            )

        # ----------------------------------------------------
        # Local episode timesteps.
        #
        # These are NOT global HDF5 row numbers.
        # ----------------------------------------------------

        timesteps = np.arange(
            local_start,
            local_start + tlen,
            dtype=np.int64,
        )

        timesteps[
            timesteps >= max_ep_len
        ] = max_ep_len - 1

        # ----------------------------------------------------
        # Return-to-go.
        #
        # Important:
        # RTG is calculated to the END OF THE TRAJECTORY,
        # not merely to the end of this K-step context.
        # ----------------------------------------------------

        remaining_rewards = rewards[
            global_start:trajectory.end
        ]

        rtg = compute_rtg(
            remaining_rewards,
            gamma=1.0,
        )

        rtg = rtg[
            :tlen + 1
        ]

        # If the sampled context reaches the end of the
        # trajectory, there is no R_{t+1} after its final
        # action. The original DT code appends zero.
        if len(rtg) <= tlen:
            rtg = np.concatenate(
                [
                    rtg,
                    np.zeros(
                        1,
                        dtype=rtg.dtype,
                    ),
                ],
                axis=0,
            )

        # RTG must now contain tlen + 1 entries.
        assert len(rtg) == tlen + 1

        # ----------------------------------------------------
        # LEFT padding.
        # ----------------------------------------------------

        pad_len = (
            context_length - tlen
        )

        # Reference DT pads raw states with zero BEFORE
        # applying normalization.
        s = np.concatenate(
            [
                np.zeros(
                    (pad_len, state_dim),
                    dtype=observations.dtype,
                ),
                s,
            ],
            axis=0,
        )

        s = (
            s - state_mean
        ) / state_std

        # Actions use -10 as the padding sentinel in the
        # original implementation.
        a = np.concatenate(
            [
                np.full(
                    (pad_len, action_dim),
                    -10.0,
                    dtype=actions.dtype,
                ),
                a,
            ],
            axis=0,
        )

        r = np.concatenate(
            [
                np.zeros(
                    pad_len,
                    dtype=rewards.dtype,
                ),
                r,
            ],
            axis=0,
        )

        # Padding value 2 distinguishes padded done entries.
        d = np.concatenate(
            [
                np.full(
                    pad_len,
                    2,
                    dtype=np.int64,
                ),
                d.astype(np.int64),
            ],
            axis=0,
        )

        # RTG gets only K - tlen padding positions.
        #
        # Since rtg already has tlen + 1 values:
        #
        #     (K - tlen) + (tlen + 1)
        #            = K + 1
        rtg = np.concatenate(
            [
                np.zeros(
                    pad_len,
                    dtype=rtg.dtype,
                ),
                rtg,
            ],
            axis=0,
        )

        rtg = rtg / rtg_scale

        timesteps = np.concatenate(
            [
                np.zeros(
                    pad_len,
                    dtype=np.int64,
                ),
                timesteps,
            ],
            axis=0,
        )

        attention_mask = np.concatenate(
            [
                np.zeros(
                    pad_len,
                    dtype=np.float32,
                ),
                np.ones(
                    tlen,
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )

        # ----------------------------------------------------
        # Shape checks for one sample.
        # ----------------------------------------------------

        assert s.shape == (
            context_length,
            state_dim,
        )

        assert a.shape == (
            context_length,
            action_dim,
        )

        assert r.shape == (
            context_length,
        )

        assert d.shape == (
            context_length,
        )

        assert rtg.shape == (
            context_length + 1,
        )

        assert timesteps.shape == (
            context_length,
        )

        assert attention_mask.shape == (
            context_length,
        )

        batch_states.append(
            s.astype(
                np.float32,
                copy=False,
            )
        )

        batch_actions.append(
            a.astype(
                np.float32,
                copy=False,
            )
        )

        batch_rewards.append(
            r.astype(
                np.float32,
                copy=False,
            )[:, None]
        )

        batch_dones.append(
            d
        )

        batch_rtg.append(
            rtg.astype(
                np.float32,
                copy=False,
            )[:, None]
        )

        batch_timesteps.append(
            timesteps
        )

        batch_masks.append(
            attention_mask
        )

    return DTBatch(
        states=np.stack(
            batch_states,
            axis=0,
        ),
        actions=np.stack(
            batch_actions,
            axis=0,
        ),
        rewards=np.stack(
            batch_rewards,
            axis=0,
        ),
        dones=np.stack(
            batch_dones,
            axis=0,
        ),
        rtg=np.stack(
            batch_rtg,
            axis=0,
        ),
        timesteps=np.stack(
            batch_timesteps,
            axis=0,
        ),
        attention_mask=np.stack(
            batch_masks,
            axis=0,
        ),
        trajectory_indices=np.asarray(
            sampled_trajectory_indices,
            dtype=np.int64,
        ),
        start_indices=np.asarray(
            sampled_start_indices,
            dtype=np.int64,
        ),
    )