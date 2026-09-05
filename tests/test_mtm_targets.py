import numpy as np
import pytest

from src.data.mtm_targets import (
    build_reference_mtm_returns,
    compute_reference_mtm_returns,
)

from src.data.trajectories import (
    TrajectorySlice,
)


def test_reference_average_mode_equal_path_length():
    rewards = np.asarray(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float32,
    )

    actual = compute_reference_mtm_returns(
        rewards,
        max_path_length=4,
        discount=1.5,
    )

    # Official average-mode behavior:
    #
    # t=0:
    #   (2 + 3 + 4) / 4 = 2.25
    #
    # t=1:
    #   (3 + 4) / 3 = 7/3
    #
    # t=2:
    #   4 / 2 = 2
    #
    # t=3:
    #   0 / 1 = 0

    expected = np.asarray(
        [
            2.25,
            7.0 / 3.0,
            2.0,
            0.0,
        ],
        dtype=np.float64,
    )[:, None]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_reference_average_mode_uses_max_path_length():
    rewards = np.asarray(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float32,
    )

    actual = compute_reference_mtm_returns(
        rewards,
        max_path_length=6,
        discount=1.5,
    )

    # Note:
    # denominator uses max_path_length - t,
    # NOT remaining valid trajectory length.
    #
    # t=0: (2+3+4)/6 = 1.5
    # t=1: (3+4)/5   = 1.4
    # t=2: 4/4       = 1
    # t=3: 0/3       = 0

    expected = np.asarray(
        [
            1.5,
            1.4,
            1.0,
            0.0,
        ],
        dtype=np.float64,
    )[:, None]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_reference_discounted_mode_excludes_current_reward():
    rewards = np.asarray(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float32,
    )

    actual = compute_reference_mtm_returns(
        rewards,
        max_path_length=4,
        discount=0.5,
    )

    # t=0:
    #   2 + .5*3 + .25*4 = 4.5
    #
    # t=1:
    #   3 + .5*4 = 5
    #
    # t=2:
    #   4
    #
    # t=3:
    #   0

    expected = np.asarray(
        [
            4.5,
            5.0,
            4.0,
            0.0,
        ],
        dtype=np.float64,
    )[:, None]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_reference_discount_one_is_future_sum():
    rewards = np.asarray(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float32,
    )

    actual = compute_reference_mtm_returns(
        rewards,
        max_path_length=4,
        discount=1.0,
    )

    expected = np.asarray(
        [
            9.0,
            7.0,
            4.0,
            0.0,
        ],
        dtype=np.float64,
    )[:, None]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_completed_trajectory_builder_leaves_tail_nan():
    rewards = np.asarray(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            10.0,
            20.0,
            30.0,
            99.0,
            99.0,
        ],
        dtype=np.float32,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=4,
        ),
        TrajectorySlice(
            start=4,
            end=7,
        ),
    ]

    actual = build_reference_mtm_returns(
        rewards,
        trajectories,
        max_path_length=4,
        discount=1.0,
    )

    # First trajectory:
    # [1,2,3,4] ->
    # [9,7,4,0]
    np.testing.assert_allclose(
        actual[0:4, 0],
        np.asarray(
            [9.0, 7.0, 4.0, 0.0]
        ),
    )

    # Second trajectory:
    # [10,20,30] ->
    # [50,30,0]
    np.testing.assert_allclose(
        actual[4:7, 0],
        np.asarray(
            [50.0, 30.0, 0.0]
        ),
    )

    # Unfinished trailing transitions are deliberately invalid.
    assert np.isnan(
        actual[7:, 0]
    ).all()


def test_two_dimensional_reward_input_supported():
    rewards = np.asarray(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float32,
    )

    actual = compute_reference_mtm_returns(
        rewards,
        max_path_length=3,
        discount=1.0,
    )

    expected = np.asarray(
        [
            [5.0],
            [3.0],
            [0.0],
        ]
    )

    np.testing.assert_allclose(
        actual,
        expected,
    )


def test_trajectory_longer_than_max_path_length_raises():
    rewards = np.ones(
        5,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        compute_reference_mtm_returns(
            rewards,
            max_path_length=4,
            discount=1.5,
        )


def test_invalid_reward_shape_raises():
    rewards = np.zeros(
        (3, 2),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        compute_reference_mtm_returns(
            rewards
        )