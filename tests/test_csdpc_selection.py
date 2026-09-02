from collections import Counter

import numpy as np
import pytest

from src.attacks.csdpc.patterns import (
    iter_sequence_windows,
)
from src.attacks.csdpc.selection import (
    compute_transition_budget,
    select_rare_nonoverlapping_windows,
)
from src.data.trajectories import (
    TrajectorySlice,
)


def _make_windows(
    labels,
    *,
    sequence_length=3,
):
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=len(labels),
        )
    ]

    return list(
        iter_sequence_windows(
            labels,
            trajectories,
            sequence_length=sequence_length,
        )
    )


def test_transition_budget_uses_floor():
    assert compute_transition_budget(
        num_transitions=1000,
        rho=0.01,
    ) == 10

    assert compute_transition_budget(
        num_transitions=999,
        rho=0.01,
    ) == 9


def test_zero_budget_selects_nothing():
    windows = _make_windows(
        [0, 1, 2, 3, 4]
    )

    counts = Counter(
        window.pattern
        for window in windows
    )

    result = (
        select_rare_nonoverlapping_windows(
            windows,
            counts,
            transition_budget=0,
        )
    )

    assert result.selected_windows == ()
    assert (
        result.actual_transition_budget
        == 0
    )


def test_selection_never_exceeds_budget():
    windows = _make_windows(
        [0, 1, 2, 3, 4, 5, 6]
    )

    counts = Counter(
        window.pattern
        for window in windows
    )

    result = (
        select_rare_nonoverlapping_windows(
            windows,
            counts,
            transition_budget=5,
        )
    )

    # Windows have length 3.
    # We can select one complete window,
    # but not two.
    assert (
        result.actual_transition_budget
        == 3
    )

    assert (
        result.actual_transition_budget
        <= 5
    )


def test_selected_windows_do_not_overlap():
    windows = _make_windows(
        [0, 1, 2, 3, 4, 5, 6],
        sequence_length=3,
    )

    counts = Counter(
        window.pattern
        for window in windows
    )

    result = (
        select_rare_nonoverlapping_windows(
            windows,
            counts,
            transition_budget=6,
        )
    )

    used = set()

    for window in result.selected_windows:
        indices = set(
            window.transition_indices
        )

        assert used.isdisjoint(
            indices
        )

        used.update(indices)

    assert (
        result.actual_transition_budget
        == len(used)
    )


def test_rare_pattern_is_selected_first():
    labels = np.asarray(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            2,
        ],
        dtype=np.int64,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=len(labels),
        )
    ]

    windows = list(
        iter_sequence_windows(
            labels,
            trajectories,
            sequence_length=3,
        )
    )

    counts = Counter(
        window.pattern
        for window in windows
    )

    result = (
        select_rare_nonoverlapping_windows(
            windows,
            counts,
            transition_budget=3,
        )
    )

    assert len(
        result.selected_windows
    ) == 1

    selected_pattern = (
        result.selected_windows[
            0
        ].source_pattern
    )

    minimum_frequency = min(
        counts.values()
    )

    assert (
        counts[selected_pattern]
        == minimum_frequency
    )


def test_pattern_tie_break_is_lexicographic():
    labels = np.asarray(
        [0, 1, 2, 3, 4, 5],
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
            sequence_length=2,
        )
    )

    counts = Counter(
        window.pattern
        for window in windows
    )

    # Every pattern occurs once.
    result = (
        select_rare_nonoverlapping_windows(
            windows,
            counts,
            transition_budget=2,
        )
    )

    assert (
        result.selected_windows[
            0
        ].source_pattern
        == min(counts)
    )


def test_invalid_rho_is_rejected():
    with pytest.raises(ValueError):
        compute_transition_budget(
            num_transitions=100,
            rho=-0.01,
        )

    with pytest.raises(ValueError):
        compute_transition_budget(
            num_transitions=100,
            rho=1.01,
        )