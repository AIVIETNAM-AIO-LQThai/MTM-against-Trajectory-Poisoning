import numpy as np

from src.data.mtm_batching import (
    build_split_window_ranges,
    sample_mtm_batch,
)

from src.data.mtm_dataset import (
    ReferenceMTMDataset,
)

from src.data.mtm_split import (
    reference_trajectory_split,
)


def make_dataset():
    """
    Four completed trajectories, each length 5.

    Window length 3.

    Each trajectory therefore contributes:

        5 - 3 + 1 = 3 windows

    Total = 12 windows.

    With train_fraction=0.5:

        train trajectories = 0,1
        validation = 2,3

        train windows = 6
        validation windows = 6
    """

    n = 20

    observations = np.arange(
        n * 2,
        dtype=np.float32,
    ).reshape(
        n,
        2,
    )

    actions = np.arange(
        n,
        dtype=np.float32,
    )[:, None]

    rewards = np.arange(
        1,
        n + 1,
        dtype=np.float32,
    )

    terminals = np.zeros(
        n,
        dtype=bool,
    )

    timeouts = np.zeros(
        n,
        dtype=bool,
    )

    terminals[
        [
            4,
            9,
            14,
            19,
        ]
    ] = True

    return ReferenceMTMDataset(
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
        trajectory_length=3,
        max_path_length=5,
        discount=1.0,
    )


def test_split_window_ranges():
    dataset = make_dataset()

    split = (
        reference_trajectory_split(
            dataset.trajectories,
            train_fraction=0.5,
        )
    )

    ranges = (
        build_split_window_ranges(
            dataset,
            split,
        )
    )

    assert len(dataset) == 12

    assert ranges.train.start == 0
    assert ranges.train.end == 6
    assert ranges.train.count == 6

    assert (
        ranges.validation.start
        == 6
    )

    assert (
        ranges.validation.end
        == 12
    )

    assert (
        ranges.validation.count
        == 6
    )


def test_train_sampling_never_uses_validation_trajectory():
    dataset = make_dataset()

    split = (
        reference_trajectory_split(
            dataset.trajectories,
            train_fraction=0.5,
        )
    )

    ranges = (
        build_split_window_ranges(
            dataset,
            split,
        )
    )

    batch = sample_mtm_batch(
        dataset,
        ranges.train,
        batch_size=100,
        rng=np.random.RandomState(
            0
        ),
    )

    assert set(
        batch.trajectory_ids.tolist()
    ).issubset(
        {
            0,
            1,
        }
    )


def test_validation_sampling_never_uses_training_trajectory():
    dataset = make_dataset()

    split = (
        reference_trajectory_split(
            dataset.trajectories,
            train_fraction=0.5,
        )
    )

    ranges = (
        build_split_window_ranges(
            dataset,
            split,
        )
    )

    batch = sample_mtm_batch(
        dataset,
        ranges.validation,
        batch_size=100,
        rng=np.random.RandomState(
            0
        ),
    )

    assert set(
        batch.trajectory_ids.tolist()
    ).issubset(
        {
            2,
            3,
        }
    )


def test_batch_shapes():
    dataset = make_dataset()

    split = (
        reference_trajectory_split(
            dataset.trajectories,
            train_fraction=0.5,
        )
    )

    ranges = (
        build_split_window_ranges(
            dataset,
            split,
        )
    )

    batch = sample_mtm_batch(
        dataset,
        ranges.train,
        batch_size=7,
        rng=np.random.RandomState(
            5
        ),
    )

    assert batch.states.shape == (
        7,
        3,
        2,
    )

    assert batch.actions.shape == (
        7,
        3,
        1,
    )

    assert batch.returns.shape == (
        7,
        3,
        1,
    )

    assert (
        batch.dataset_indices.shape
        == (7,)
    )

    assert (
        batch.trajectory_ids.shape
        == (7,)
    )

    assert (
        batch.global_starts.shape
        == (7,)
    )


def test_sampling_is_deterministic_for_same_rng_seed():
    dataset = make_dataset()

    split = (
        reference_trajectory_split(
            dataset.trajectories,
            train_fraction=0.5,
        )
    )

    ranges = (
        build_split_window_ranges(
            dataset,
            split,
        )
    )

    batch_a = sample_mtm_batch(
        dataset,
        ranges.train,
        batch_size=20,
        rng=np.random.RandomState(
            123
        ),
    )

    batch_b = sample_mtm_batch(
        dataset,
        ranges.train,
        batch_size=20,
        rng=np.random.RandomState(
            123
        ),
    )

    np.testing.assert_array_equal(
        batch_a.dataset_indices,
        batch_b.dataset_indices,
    )

    np.testing.assert_array_equal(
        batch_a.states,
        batch_b.states,
    )

    np.testing.assert_array_equal(
        batch_a.actions,
        batch_b.actions,
    )

    np.testing.assert_array_equal(
        batch_a.returns,
        batch_b.returns,
    )