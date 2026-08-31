import numpy as np
from src.data.rtg import compute_rtg
from src.data.trajectories import find_completed_trajectories

def test_rtg_simple_sequence():
    rewards = np.array(
        [2.0, 3.0, 4.0, 1.0],
        dtype=np.float32,
    )

    expected = np.array(
        [10.0, 8.0, 5.0, 1.0],
        dtype=np.float32,
    )

    actual = compute_rtg(
        rewards,
        gamma=1.0,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-6,
    )

def test_rtg_last_value_equals_last_reward():
    rewards = np.array(
        [1.2, -0.5, 3.7],
        dtype=np.float32,
    )

    rtg = compute_rtg(rewards)
    assert np.isclose(
        rtg[-1],
        rewards[-1],
    )

def test_rtg_recurrence():
    rewards = np.array(
        [0.5, 1.0, -0.5, 4.0],
        dtype=np.float32,
    )

    rtg = compute_rtg(rewards)
    np.testing.assert_allclose(
        rtg[:-1],
        rewards[:-1] + rtg[1:],
        rtol=0.0,
        atol=1e-6,
    )

from pathlib import Path

import h5py


DATASET_PATH = Path(
    "data/raw/walker2d-medium-v2/"
    "walker2d_medium-v2.hdf5"
)


def test_rtg_on_all_completed_walker2d_trajectories():
    assert DATASET_PATH.exists(), (
        f"Dataset not found: {DATASET_PATH}"
    )

    with h5py.File(DATASET_PATH, "r") as f:
        rewards = f["rewards"][:]
        terminals = f["terminals"][:].astype(bool)
        timeouts = f["timeouts"][:].astype(bool)

    trajectories, trailing = find_completed_trajectories(
        terminals, timeouts,
    )

    for trajectory_index, trajectory in enumerate(trajectories):
        trajectory_rewards = rewards[
            trajectory.start:trajectory.end
        ]

        rtg = compute_rtg(
            trajectory_rewards,
            gamma=1.0,
        )

        # --------------------------------------
        # Property 1:
        # R_0 equals total trajectory return
        # --------------------------------------
        expected_return = np.sum(
            trajectory_rewards,
            dtype=np.float64,
        )

        assert np.isclose(
            float(rtg[0]),
            expected_return,
            rtol=1e-5,
            atol=1e-2,
        ), (
            f"RTG mismatch in trajectory {trajectory_index}: "
            f"rtg[0]={float(rtg[0])}, "
            f"sum_rewards={expected_return}, "
            f"difference={abs(float(rtg[0]) - expected_return)}"
        )

        # --------------------------------------
        # Property 2:
        # R_t = r_t + R_{t+1}
        # --------------------------------------
        if len(rtg) > 1:
            np.testing.assert_allclose(
                rtg[:-1],
                trajectory_rewards[:-1] + rtg[1:],
                rtol=1e-6,
                atol=1e-5,
            )

        # --------------------------------------
        # Property 3:
        # final RTG = final reward
        # --------------------------------------
        assert np.isclose(
            rtg[-1],
            trajectory_rewards[-1],
            rtol=0.0,
            atol=1e-6,
        )

    # Reference-compatible trajectory count.
    assert len(trajectories) == 1190

    # Five raw transitions remain after the final
    # completed trajectory and are not used by DT.
    assert trailing == 5

    # Completed trajectories use exactly 999,995 transitions.
    used_transitions = sum(
        trajectory.length
        for trajectory in trajectories
    )

    assert used_transitions == 999_995