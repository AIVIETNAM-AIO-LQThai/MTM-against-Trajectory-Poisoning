from pathlib import Path

import h5py
import numpy as np

from src.data.normalization import (
    compute_state_statistics,
    normalize_states,
)
from src.data.trajectories import (
    find_completed_trajectories,
)


DATASET_PATH = Path(
    "data/raw/walker2d-medium-v2/"
    "walker2d_medium-v2.hdf5"
)


def test_normalization_on_simple_data():
    states = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=np.float32,
    )

    # One completed trajectory containing all 3 rows.
    terminals = np.array(
        [False, False, True]
    )

    timeouts = np.array(
        [False, False, False]
    )

    trajectories, trailing = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    assert trailing == 0

    mean, std = compute_state_statistics(
        states,
        trajectories,
    )

    expected_mean = np.mean(
        states,
        axis=0,
    )

    expected_std = (
        np.std(
            states,
            axis=0,
        )
        + 1e-6
    )

    np.testing.assert_allclose(
        mean,
        expected_mean,
        rtol=0.0,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        std,
        expected_std,
        rtol=0.0,
        atol=1e-7,
    )


def test_walker2d_state_statistics():
    assert DATASET_PATH.exists()

    with h5py.File(DATASET_PATH, "r") as f:
        observations = f["observations"][:]
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

    state_mean, state_std = (
        compute_state_statistics(
            observations,
            trajectories,
        )
    )

    # Walker2d state dimensionality.
    assert state_mean.shape == (17,)
    assert state_std.shape == (17,)

    assert np.isfinite(state_mean).all()
    assert np.isfinite(state_std).all()

    assert np.all(
        state_std > 0
    )

    # Our audited dataset structure.
    assert len(trajectories) == 1190
    assert trailing == 5

    used_transitions = sum(
        trajectory.length
        for trajectory in trajectories
    )

    assert used_transitions == 999_995

    training_states = observations[:used_transitions]

    reference_mean = np.mean(
        training_states,
        axis=0,
    )
    reference_std = (
        np.std(
            training_states,
            axis=0,
        )
        + 1e-6
    )

    np.testing.assert_array_equal(
        state_mean,
        reference_mean,
    )

    np.testing.assert_array_equal(
        state_std,
        reference_std,
    )

def test_normalized_training_states_are_centered():
    with h5py.File(DATASET_PATH, "r") as f:
        observations = f["observations"][:]
        terminals = (
            f["terminals"][:]
            .astype(bool)
        )
        timeouts = (
            f["timeouts"][:]
            .astype(bool)
        )

    trajectories, _ = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    state_mean, state_std = (
        compute_state_statistics(
            observations,
            trajectories,
        )
    )

    used_transitions = sum(
        trajectory.length
        for trajectory in trajectories
    )

    states = observations[
        :used_transitions
    ]

    normalized = normalize_states(
        states,
        state_mean,
        state_std,
    )

    normalized_mean = np.mean(
        normalized.astype(np.float64),
        axis=0,
    )

    normalized_std = np.std(
        normalized.astype(np.float64),
        axis=0,
    )

    print("\nnormalized_mean:")
    print(normalized_mean)

    print("\nmax |normalized_mean|:")
    print(np.max(np.abs(normalized_mean)))

    print("\nnormalized_std:")
    print(normalized_std)

    print("\nmax |normalized_std - 1|:")
    print(np.max(np.abs(normalized_std - 1.0)))

    # ---------------------------------------------------------
    # More important correctness check:
    #
    # The statistics must exactly match the reference NumPy
    # computation over the SAME 999,995 training states.
    # ---------------------------------------------------------

    reference_mean = np.mean(
        states,
        axis=0,
    )

    reference_std = (
        np.std(
            states,
            axis=0,
        )
        + 1e-6
    )

    np.testing.assert_array_equal(
        state_mean,
        reference_mean,
    )

    np.testing.assert_array_equal(
        state_std,
        reference_std,
    )

    # Transformation matches exact normalization formula.
    expected_normalized = (
        states - reference_mean
    ) / reference_std

    np.testing.assert_array_equal(
        normalized,
        expected_normalized,
    )

    assert np.isfinite(normalized).all()

def test_normalization_is_reusable():
    """
    The same saved mean/std must produce the same output every time.
    This matters because evaluation must never refit statistics.
    """

    with h5py.File(DATASET_PATH, "r") as f:
        observations = f["observations"][:]
        terminals = (
            f["terminals"][:]
            .astype(bool)
        )
        timeouts = (
            f["timeouts"][:]
            .astype(bool)
        )

    trajectories, _ = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    state_mean, state_std = (
        compute_state_statistics(
            observations,
            trajectories,
        )
    )

    sample = observations[:100]

    normalized_a = normalize_states(
        sample,
        state_mean,
        state_std,
    )

    normalized_b = normalize_states(
        sample,
        state_mean,
        state_std,
    )

    np.testing.assert_array_equal(
        normalized_a,
        normalized_b,
    )