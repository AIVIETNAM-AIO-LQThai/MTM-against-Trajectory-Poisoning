import numpy as np
import pytest

from src.attacks.csdpc.patterns import (
    count_pattern_frequencies,
    deduplicate_consecutive,
    iter_sequence_windows,
)
from src.data.trajectories import TrajectorySlice


def test_deduplicate_consecutive_labels():
    result = deduplicate_consecutive(
        [1, 1, 1, 4, 4]
    )

    assert result == (1, 4)


def test_deduplication_preserves_nonconsecutive_repetition():
    result = deduplicate_consecutive(
        [1, 1, 4, 1, 1]
    )

    assert result == (1, 4, 1)


def test_windowing_happens_before_deduplication():
    labels = np.array(
        [1, 1, 1, 4, 4, 7],
        dtype=np.int64,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=6,
        )
    ]

    windows = list(
        iter_sequence_windows(
            labels,
            trajectories,
            sequence_length=5,
        )
    )

    assert len(windows) == 2

    assert windows[0].raw_cluster_labels == (
        1,
        1,
        1,
        4,
        4,
    )
    assert windows[0].pattern == (1, 4)

    assert windows[1].raw_cluster_labels == (
        1,
        1,
        4,
        4,
        7,
    )
    assert windows[1].pattern == (
        1,
        4,
        7,
    )


def test_windows_never_cross_trajectory_boundaries():
    labels = np.array(
        [
            0,
            0,
            1,
            2,
            3,
            3,
            4,
        ],
        dtype=np.int64,
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

    windows = list(
        iter_sequence_windows(
            labels,
            trajectories,
            sequence_length=3,
        )
    )

    actual_indices = [
        window.transition_indices
        for window in windows
    ]

    assert actual_indices == [
        (0, 1, 2),
        (1, 2, 3),
        (4, 5, 6),
    ]

    assert (2, 3, 4) not in actual_indices
    assert (3, 4, 5) not in actual_indices


def test_raw_window_always_has_exact_sequence_length():
    labels = np.array(
        [0, 0, 1, 1, 2, 2],
        dtype=np.int64,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=6,
        )
    ]

    windows = list(
        iter_sequence_windows(
            labels,
            trajectories,
            sequence_length=5,
        )
    )

    assert len(windows) == 2

    for window in windows:
        assert len(
            window.raw_cluster_labels
        ) == 5

        assert len(
            window.transition_indices
        ) == 5

        assert (
            window.global_end
            - window.global_start
        ) == 5


def test_short_trajectory_produces_no_window():
    labels = np.array(
        [1, 2, 3, 4],
        dtype=np.int64,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=4,
        )
    ]

    windows = list(
        iter_sequence_windows(
            labels,
            trajectories,
            sequence_length=5,
        )
    )

    assert windows == []


def test_pattern_frequency_is_occurrence_count():
    labels = np.array(
        [1, 1, 2, 1, 1, 2],
        dtype=np.int64,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=6,
        )
    ]

    windows = iter_sequence_windows(
        labels,
        trajectories,
        sequence_length=3,
    )

    counts = count_pattern_frequencies(
        windows
    )

    assert counts[(1, 2)] == 2
    assert counts[(1, 2, 1)] == 1
    assert counts[(2, 1)] == 1
    assert sum(counts.values()) == 4


def test_invalid_trajectory_beyond_label_array_raises():
    labels = np.array(
        [1, 2, 3],
        dtype=np.int64,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=4,
        )
    ]

    with pytest.raises(
        ValueError,
        match="extends beyond",
    ):
        list(
            iter_sequence_windows(
                labels,
                trajectories,
                sequence_length=2,
            )
        )