from pathlib import Path
from random import Random

import h5py
import numpy as np

from src.data.rtg import compute_rtg
from src.data.batching import sample_dt_batch
from src.data.trajectories import (
    find_completed_trajectories,
)


DATASET_PATH = Path(
    "data/raw/walker2d-medium-v2/"
    "walker2d_medium-v2.hdf5"
)

NORMALIZATION_PATH = Path(
    "data/metadata/"
    "walker2d_medium_normalization.npz"
)


def load_data():
    with h5py.File(DATASET_PATH, "r") as f:
        observations = f["observations"][:]
        actions = f["actions"][:]
        rewards = f["rewards"][:]

        terminals = (
            f["terminals"][:]
            .astype(bool)
        )

        timeouts = (
            f["timeouts"][:]
            .astype(bool)
        )

    trajectories, trailing = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    with np.load(NORMALIZATION_PATH) as f:
        state_mean = f["state_mean"]
        state_std = f["state_std"]

    return (
        observations,
        actions,
        rewards,
        terminals,
        trajectories,
        trailing,
        state_mean,
        state_std,
    )


def test_dt_batch_shapes():
    (
        observations,
        actions,
        rewards,
        terminals,
        trajectories,
        trailing,
        state_mean,
        state_std,
    ) = load_data()

    assert trailing == 5

    batch = sample_dt_batch(
        observations,
        actions,
        rewards,
        terminals,
        trajectories,
        state_mean,
        state_std,
        batch_size=8,
        context_length=20,
        max_ep_len=1000,
        rtg_scale=1000.0,
        np_rng=np.random.default_rng(123),
        py_rng=Random(123),
    )

    assert batch.states.shape == (
        8,
        20,
        17,
    )

    assert batch.actions.shape == (
        8,
        20,
        6,
    )

    assert batch.rewards.shape == (
        8,
        20,
        1,
    )

    assert batch.dones.shape == (
        8,
        20,
    )

    assert batch.rtg.shape == (
        8,
        21,
        1,
    )

    assert batch.timesteps.shape == (
        8,
        20,
    )

    assert batch.attention_mask.shape == (
        8,
        20,
    )

    assert batch.trajectory_indices.shape == (
        8,
    )

    assert batch.start_indices.shape == (
        8,
    )

def test_dt_batch_sequence_alignment():
    (
        observations, actions, rewards,
        terminals, trajectories, trailing,
        state_mean, state_std,
    ) = load_data()

    batch = sample_dt_batch(
        observations, actions, rewards,
        terminals, trajectories,
        state_mean, state_std,
        batch_size=32, context_length=20, max_ep_len=1000,
        rtg_scale=1000.0,
        np_rng=np.random.default_rng(456),
        py_rng=Random(456),
    )

    K = 20

    for b in range(32):
        trajectory_index = int(
            batch.trajectory_indices[b]
        )
        local_start = int(
            batch.start_indices[b]
        )
        trajectory = trajectories[
            trajectory_index
        ]
        global_start = (
            trajectory.start
            + local_start
        )
        global_end = min(
            global_start + K,
            trajectory.end,
        )
        tlen = global_end - global_start
        pad_len = K - tlen

        # --------------------------------------------------
        # 1. Sequence must never cross trajectory boundary.
        # --------------------------------------------------
        assert global_start >= trajectory.start
        assert global_end <= trajectory.end
        assert tlen >= 1
        assert tlen <= K

        # --------------------------------------------------
        # 2. Attention mask must correspond exactly to
        #    left padding + valid sequence.
        # --------------------------------------------------
        expected_mask = np.concatenate(
            [
                np.zeros(
                    pad_len,
                    dtype=np.float32,
                ),
                np.ones(
                    tlen,
                    dtype=np.float32,
                ),
            ]
        )

        np.testing.assert_array_equal(
            batch.attention_mask[b],
            expected_mask,
        )

        # Valid model positions.
        valid_slice = slice(
            pad_len,
            K,
        )

        # --------------------------------------------------
        # 3. States must correspond to exact raw states,
        #    with the frozen normalization applied.
        # --------------------------------------------------
        raw_states = observations[
            global_start:global_end
        ]
        expected_states = (
            raw_states - state_mean
        ) / state_std

        np.testing.assert_allclose(
            batch.states[
                b,
                valid_slice,
            ],
            expected_states,
            rtol=1e-6,
            atol=1e-6,
        )

        # --------------------------------------------------
        # 4. Actions must align with the SAME timestep.
        # --------------------------------------------------
        expected_actions = actions[
            global_start:global_end
        ]

        np.testing.assert_array_equal(
            batch.actions[
                b,
                valid_slice,
            ],
            expected_actions,
        )

        # --------------------------------------------------
        # 5. Rewards must align with the SAME timestep.
        # --------------------------------------------------
        expected_rewards = rewards[
            global_start:global_end
        ]

        np.testing.assert_array_equal(
            batch.rewards[
                b,
                valid_slice,
                0,
            ],
            expected_rewards,
        )

        # --------------------------------------------------
        # 6. Done flags must align with the raw transition.
        # --------------------------------------------------
        expected_dones = terminals[
            global_start:global_end
        ].astype(np.int64)

        np.testing.assert_array_equal(
            batch.dones[
                b,
                valid_slice,
            ],
            expected_dones,
        )

        # --------------------------------------------------
        # 7. Timesteps are LOCAL episode timesteps,
        #    not global HDF5 indices.
        # --------------------------------------------------
        expected_timesteps = np.arange(
            local_start,
            local_start + tlen,
            dtype=np.int64,
        )
        expected_timesteps[
            expected_timesteps >= 1000
        ] = 999

        np.testing.assert_array_equal(
            batch.timesteps[
                b,
                valid_slice,
            ],
            expected_timesteps,
        )

        # --------------------------------------------------
        # 8. RTG must be calculated to the trajectory end,
        #    NOT only to the end of the K-step window.
        # --------------------------------------------------
        remaining_rewards = rewards[
            global_start:trajectory.end
        ]

        expected_rtg = compute_rtg(
            remaining_rewards,
            gamma=1.0,
        )
        expected_rtg = expected_rtg[
            :tlen + 1
        ]

        # If the sampled sequence reaches the trajectory end,
        # append the reference DT zero continuation.
        if len(expected_rtg) <= tlen:
            expected_rtg = np.concatenate(
                [
                    expected_rtg,
                    np.zeros(1,dtype=expected_rtg.dtype),
                ]
            )

        assert len(expected_rtg) == tlen + 1

        expected_rtg = (
            expected_rtg / 1000.0
        )

        # RTG has K+1 entries.
        #
        # pad_len zeros
        # +
        # tlen+1 real RTG entries
        # =
        # K+1
        actual_rtg = batch.rtg[
            b,
            pad_len:pad_len + tlen + 1,
            0,
        ]

        np.testing.assert_allclose(
            actual_rtg, expected_rtg,
            rtol=1e-6, atol=1e-6,
        )