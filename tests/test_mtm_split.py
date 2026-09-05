import pytest

from src.data.mtm_split import (
    reference_trajectory_split,
)

from src.data.trajectories import (
    TrajectorySlice,
)


def make_trajectories(
    count: int,
) -> list[TrajectorySlice]:
    return [
        TrajectorySlice(
            start=i * 10,
            end=(i + 1) * 10,
        )
        for i in range(count)
    ]


def test_reference_split_uses_first_trajectories():
    trajectories = make_trajectories(
        20
    )

    split = reference_trajectory_split(
        trajectories,
        train_fraction=0.75,
    )

    assert len(
        split.train_trajectories
    ) == 15

    assert len(
        split.validation_trajectories
    ) == 5

    assert split.train_ids == tuple(
        range(15)
    )

    assert split.validation_ids == tuple(
        range(15, 20)
    )

    assert (
        split.train_trajectories[-1]
        == trajectories[14]
    )

    assert (
        split.validation_trajectories[0]
        == trajectories[15]
    )


def test_reference_walker2d_split_count():
    # We don't need the real HDF5 for this arithmetic test.
    trajectories = make_trajectories(
        1190
    )

    split = reference_trajectory_split(
        trajectories,
        train_fraction=0.95,
    )

    assert len(
        split.train_trajectories
    ) == 1130

    assert len(
        split.validation_trajectories
    ) == 60

    assert split.train_ids[0] == 0
    assert split.train_ids[-1] == 1129

    assert (
        split.validation_ids[0]
        == 1130
    )

    assert (
        split.validation_ids[-1]
        == 1189
    )


def test_reference_split_does_not_reorder():
    trajectories = [
        TrajectorySlice(
            start=100,
            end=110,
        ),
        TrajectorySlice(
            start=20,
            end=30,
        ),
        TrajectorySlice(
            start=900,
            end=910,
        ),
        TrajectorySlice(
            start=50,
            end=60,
        ),
    ]

    split = reference_trajectory_split(
        trajectories,
        train_fraction=0.5,
    )

    # Deliberately weird raw ordering.
    # Reference split must preserve it.
    assert (
        split.train_trajectories
        == tuple(trajectories[:2])
    )

    assert (
        split.validation_trajectories
        == tuple(trajectories[2:])
    )


@pytest.mark.parametrize(
    "train_fraction",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_invalid_fraction_raises(
    train_fraction,
):
    trajectories = make_trajectories(
        10
    )

    with pytest.raises(ValueError):
        reference_trajectory_split(
            trajectories,
            train_fraction=train_fraction,
        )


def test_too_few_trajectories_raises():
    trajectories = make_trajectories(
        1
    )

    with pytest.raises(ValueError):
        reference_trajectory_split(
            trajectories,
        )