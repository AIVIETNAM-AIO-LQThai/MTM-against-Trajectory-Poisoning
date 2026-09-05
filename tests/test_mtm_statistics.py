import numpy as np
import pytest

from src.data.mtm_statistics import (
    compute_reference_mtm_statistics,
    pad_completed_trajectories,
    statistics_from_padded,
)

from src.data.trajectories import (
    TrajectorySlice,
)


def test_padding_matches_reference_layout():
    values = np.asarray(
        [
            [1.0],
            [2.0],
            [10.0],
            [20.0],
            [30.0],
        ],
        dtype=np.float32,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=2,
        ),
        TrajectorySlice(
            start=2,
            end=5,
        ),
    ]

    actual = (
        pad_completed_trajectories(
            values,
            trajectories,
            max_path_length=4,
        )
    )

    expected = np.asarray(
        [
            [
                [1.0],
                [2.0],
                [0.0],
                [0.0],
            ],
            [
                [10.0],
                [20.0],
                [30.0],
                [0.0],
            ],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_statistics_include_padding():
    padded = np.asarray(
        [
            [
                [1.0],
                [2.0],
                [0.0],
                [0.0],
            ],
            [
                [10.0],
                [20.0],
                [30.0],
                [0.0],
            ],
        ],
        dtype=np.float32,
    )

    actual = (
        statistics_from_padded(
            padded
        )
    )

    np.testing.assert_allclose(
        actual.mean,
        padded.mean(
            axis=(0, 1)
        ),
    )

    np.testing.assert_allclose(
        actual.std,
        padded.std(
            axis=(0, 1)
        ),
    )

    np.testing.assert_allclose(
        actual.min,
        padded.min(
            axis=(0, 1)
        ),
    )

    np.testing.assert_allclose(
        actual.max,
        padded.max(
            axis=(0, 1)
        ),
    )

    valid_only_mean = np.asarray(
        [1.0, 2.0, 10.0, 20.0, 30.0]
    ).mean()

    # This proves zero padding actually changes the statistic.
    assert not np.isclose(
        actual.mean[0],
        valid_only_mean,
    )


def test_reference_statistics_return_expected_modalities():
    observations = np.arange(
        10,
        dtype=np.float32,
    ).reshape(
        5,
        2,
    )

    actions = np.arange(
        5,
        dtype=np.float32,
    )[:, None]

    returns = np.asarray(
        [
            4.0,
            0.0,
            8.0,
            3.0,
            0.0,
        ],
        dtype=np.float64,
    )[:, None]

    trajectories = [
        TrajectorySlice(
            start=0,
            end=2,
        ),
        TrajectorySlice(
            start=2,
            end=5,
        ),
    ]

    stats = (
        compute_reference_mtm_statistics(
            observations,
            actions,
            returns,
            trajectories,
            max_path_length=4,
        )
    )

    assert set(stats.keys()) == {
        "states",
        "actions",
        "returns",
    }

    assert stats[
        "states"
    ].mean.shape == (2,)

    assert stats[
        "actions"
    ].mean.shape == (1,)

    assert stats[
        "returns"
    ].mean.shape == (1,)


def test_statistics_use_training_trajectories_only():
    observations = np.asarray(
        [
            [1.0],
            [2.0],
            [1000.0],
            [2000.0],
        ],
        dtype=np.float32,
    )

    actions = observations.copy()

    returns = observations.astype(
        np.float64
    )

    train_trajectories = [
        TrajectorySlice(
            start=0,
            end=2,
        )
    ]

    stats = (
        compute_reference_mtm_statistics(
            observations,
            actions,
            returns,
            train_trajectories,
            max_path_length=2,
        )
    )

    # Validation-like huge values must not leak in.
    np.testing.assert_allclose(
        stats["states"].mean,
        np.asarray([1.5]),
    )


def test_too_long_trajectory_raises():
    values = np.zeros(
        (5, 1),
        dtype=np.float32,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=5,
        )
    ]

    with pytest.raises(ValueError):
        pad_completed_trajectories(
            values,
            trajectories,
            max_path_length=4,
        )