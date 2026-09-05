import numpy as np
import pytest

from src.data.mtm_windows import MTMWindowDataset


def make_tiny_dataset():
    """
    Synthetic layout:

    completed trajectory 0:
        raw indices 0..4
        length = 5

    completed trajectory 1:
        raw indices 5..8
        length = 4

    unfinished trailing fragment:
        raw indices 9..10
        length = 2

    With trajectory_length = 3:

    traj 0 contributes:
        [0,1,2]
        [1,2,3]
        [2,3,4]

    traj 1 contributes:
        [5,6,7]
        [6,7,8]

    unfinished raw indices 9,10 contribute nothing.

    Total expected windows = 5.
    """

    num_transitions = 11

    observations = np.arange(
        num_transitions * 2,
        dtype=np.float32,
    ).reshape(num_transitions, 2)

    actions = (
        100
        + np.arange(
            num_transitions,
            dtype=np.float32,
        )
    )[:, None]

    terminals = np.zeros(
        num_transitions,
        dtype=bool,
    )

    timeouts = np.zeros(
        num_transitions,
        dtype=bool,
    )

    # First completed trajectory ends at raw index 4.
    terminals[4] = True

    # Second completed trajectory ends at raw index 8.
    timeouts[8] = True

    # Raw indices 9 and 10 intentionally have no boundary.
    # They form an unfinished trailing fragment.

    return (
        observations,
        actions,
        terminals,
        timeouts,
    )


def test_mtm_window_dataset_metadata():
    (
        observations,
        actions,
        terminals,
        timeouts,
    ) = make_tiny_dataset()

    dataset = MTMWindowDataset(
        observations,
        actions,
        terminals,
        timeouts,
        trajectory_length=3,
    )

    assert dataset.num_completed_trajectories == 2
    assert dataset.num_used_transitions == 9
    assert dataset.trailing_transitions == 2
    assert len(dataset) == 5


def test_mtm_windows_have_expected_indices():
    (
        observations,
        actions,
        terminals,
        timeouts,
    ) = make_tiny_dataset()

    dataset = MTMWindowDataset(
        observations,
        actions,
        terminals,
        timeouts,
        trajectory_length=3,
    )

    expected = [
        # trajectory_id, local_start,
        # global_start, global_end
        (0, 0, 0, 3),
        (0, 1, 1, 4),
        (0, 2, 2, 5),
        (1, 0, 5, 8),
        (1, 1, 6, 9),
    ]

    actual = []

    for i in range(len(dataset)):
        window = dataset[i]

        actual.append(
            (
                window.trajectory_id,
                window.local_start,
                window.global_start,
                window.global_end,
            )
        )

    assert actual == expected


def test_mtm_windows_preserve_state_action_alignment():
    (
        observations,
        actions,
        terminals,
        timeouts,
    ) = make_tiny_dataset()

    dataset = MTMWindowDataset(
        observations,
        actions,
        terminals,
        timeouts,
        trajectory_length=3,
    )

    for i in range(len(dataset)):
        window = dataset[i]

        np.testing.assert_array_equal(
            window.states,
            observations[
                window.global_start:
                window.global_end
            ],
        )

        np.testing.assert_array_equal(
            window.actions,
            actions[
                window.global_start:
                window.global_end
            ],
        )

        assert window.states.shape == (3, 2)
        assert window.actions.shape == (3, 1)
        assert window.length == 3


def test_mtm_windows_never_use_unfinished_tail():
    (
        observations,
        actions,
        terminals,
        timeouts,
    ) = make_tiny_dataset()

    dataset = MTMWindowDataset(
        observations,
        actions,
        terminals,
        timeouts,
        trajectory_length=3,
    )

    for i in range(len(dataset)):
        window = dataset[i]

        # Raw indices 9 and 10 are unfinished.
        assert window.global_end <= 9


def test_mtm_windows_never_cross_completed_boundaries():
    (
        observations,
        actions,
        terminals,
        timeouts,
    ) = make_tiny_dataset()

    dataset = MTMWindowDataset(
        observations,
        actions,
        terminals,
        timeouts,
        trajectory_length=3,
    )

    for i in range(len(dataset)):
        window = dataset[i]

        trajectory = dataset.trajectories[
            window.trajectory_id
        ]

        assert window.global_start >= trajectory.start
        assert window.global_end <= trajectory.end


def test_negative_index_matches_python_sequence_semantics():
    (
        observations,
        actions,
        terminals,
        timeouts,
    ) = make_tiny_dataset()

    dataset = MTMWindowDataset(
        observations,
        actions,
        terminals,
        timeouts,
        trajectory_length=3,
    )

    first_from_end = dataset[-1]
    explicit_last = dataset[len(dataset) - 1]

    assert (
        first_from_end.trajectory_id
        == explicit_last.trajectory_id
    )

    assert (
        first_from_end.global_start
        == explicit_last.global_start
    )

    assert (
        first_from_end.global_end
        == explicit_last.global_end
    )


def test_out_of_range_index_raises():
    (
        observations,
        actions,
        terminals,
        timeouts,
    ) = make_tiny_dataset()

    dataset = MTMWindowDataset(
        observations,
        actions,
        terminals,
        timeouts,
        trajectory_length=3,
    )

    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]

    with pytest.raises(IndexError):
        _ = dataset[-len(dataset) - 1]


def test_invalid_trajectory_length_raises():
    (
        observations,
        actions,
        terminals,
        timeouts,
    ) = make_tiny_dataset()

    with pytest.raises(ValueError):
        MTMWindowDataset(
            observations,
            actions,
            terminals,
            timeouts,
            trajectory_length=0,
        )


def test_trajectory_too_short_contributes_no_windows():
    observations = np.arange(
        4 * 2,
        dtype=np.float32,
    ).reshape(4, 2)

    actions = np.arange(
        4,
        dtype=np.float32,
    )[:, None]

    terminals = np.zeros(
        4,
        dtype=bool,
    )

    timeouts = np.zeros(
        4,
        dtype=bool,
    )

    terminals[1] = True
    terminals[3] = True

    # Two completed trajectories, each length 2.
    # Requested MTM length is 3.
    dataset = MTMWindowDataset(
        observations,
        actions,
        terminals,
        timeouts,
        trajectory_length=3,
    )

    assert dataset.num_completed_trajectories == 2
    assert dataset.trailing_transitions == 0
    assert len(dataset) == 0