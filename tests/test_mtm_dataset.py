import numpy as np

from src.data.mtm_dataset import (
    ReferenceMTMDataset,
)


def make_dataset():
    """
    Completed trajectory 0:
        indices 0..4
        rewards [1,2,3,4,5]

    Completed trajectory 1:
        indices 5..8
        rewards [10,20,30,40]

    Trailing unfinished:
        indices 9..10
    """

    num_transitions = 11

    observations = np.arange(
        num_transitions * 2,
        dtype=np.float32,
    ).reshape(
        num_transitions,
        2,
    )

    actions = np.arange(
        num_transitions,
        dtype=np.float32,
    )[:, None]

    rewards = np.asarray(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            10.0,
            20.0,
            30.0,
            40.0,
            99.0,
            99.0,
        ],
        dtype=np.float32,
    )

    terminals = np.zeros(
        num_transitions,
        dtype=bool,
    )

    timeouts = np.zeros(
        num_transitions,
        dtype=bool,
    )

    terminals[4] = True
    timeouts[8] = True

    return (
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
    )


def test_reference_mtm_dataset_structure():
    (
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
    ) = make_dataset()

    dataset = ReferenceMTMDataset(
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
        trajectory_length=3,
        max_path_length=5,
        discount=1.5,
    )

    assert (
        dataset.num_completed_trajectories
        == 2
    )

    assert (
        dataset.num_used_transitions
        == 9
    )

    assert (
        dataset.trailing_transitions
        == 2
    )

    # trajectory lengths 5 and 4,
    # window length 3:
    #
    # 5 -> 3 windows
    # 4 -> 2 windows
    assert len(dataset) == 5


def test_reference_mtm_sample_modalities():
    (
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
    ) = make_dataset()

    dataset = ReferenceMTMDataset(
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
        trajectory_length=3,
        max_path_length=5,
        discount=1.5,
    )

    sample = dataset[0]

    assert sample.states.shape == (
        3,
        2,
    )

    assert sample.actions.shape == (
        3,
        1,
    )

    assert sample.rewards.shape == (
        3,
        1,
    )

    assert sample.returns.shape == (
        3,
        1,
    )


def test_first_window_values_are_correct():
    (
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
    ) = make_dataset()

    dataset = ReferenceMTMDataset(
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
        trajectory_length=3,
        max_path_length=5,
        discount=1.5,
    )

    sample = dataset[0]

    np.testing.assert_array_equal(
        sample.rewards[:, 0],
        np.asarray(
            [1.0, 2.0, 3.0]
        ),
    )

    # Full first trajectory:
    #
    # [1,2,3,4,5]
    #
    # Reference average mode:
    #
    # t0 = (2+3+4+5)/5 = 2.8
    # t1 = (3+4+5)/4   = 3.0
    # t2 = (4+5)/3     = 3.0
    # t3 = 5/2         = 2.5
    # t4 = 0
    #
    # Window 0 takes t=0..2.

    np.testing.assert_allclose(
        sample.returns[:, 0],
        np.asarray(
            [
                2.8,
                3.0,
                3.0,
            ]
        ),
        atol=1e-12,
    )


def test_second_trajectory_uses_its_own_future():
    (
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
    ) = make_dataset()

    dataset = ReferenceMTMDataset(
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
        trajectory_length=3,
        max_path_length=5,
        discount=1.5,
    )

    # Windows:
    #
    # trajectory 0 -> indices 0,1,2
    # trajectory 1 -> dataset windows 3,4
    sample = dataset[3]

    assert sample.trajectory_id == 1

    np.testing.assert_array_equal(
        sample.rewards[:, 0],
        np.asarray(
            [10.0, 20.0, 30.0]
        ),
    )

    # Second trajectory:
    #
    # [10,20,30,40]
    #
    # t0 = (20+30+40)/5 = 18
    # t1 = (30+40)/4    = 17.5
    # t2 = 40/3
    # t3 = 0

    np.testing.assert_allclose(
        sample.returns[:, 0],
        np.asarray(
            [
                18.0,
                17.5,
                40.0 / 3.0,
            ]
        ),
        atol=1e-12,
    )


def test_trailing_rewards_do_not_affect_completed_returns():
    (
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
    ) = make_dataset()

    dataset_a = ReferenceMTMDataset(
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
        trajectory_length=3,
        max_path_length=5,
        discount=1.5,
    )

    changed_rewards = rewards.copy()

    changed_rewards[9:] = (
        1_000_000.0
    )

    dataset_b = ReferenceMTMDataset(
        observations,
        actions,
        changed_rewards,
        terminals,
        timeouts,
        trajectory_length=3,
        max_path_length=5,
        discount=1.5,
    )

    for i in range(len(dataset_a)):
        np.testing.assert_allclose(
            dataset_a[i].returns,
            dataset_b[i].returns,
        )


def test_dataset_never_exposes_nan_returns():
    (
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
    ) = make_dataset()

    dataset = ReferenceMTMDataset(
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
        trajectory_length=3,
        max_path_length=5,
        discount=1.5,
    )

    for i in range(len(dataset)):
        assert not np.isnan(
            dataset[i].returns
        ).any()