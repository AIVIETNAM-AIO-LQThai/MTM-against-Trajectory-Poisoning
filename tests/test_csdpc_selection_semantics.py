from __future__ import annotations

import numpy as np

from scripts.audit_csdpc_selection_semantics import (
    DiagnosticSelection,
    _build_pattern_index,
    _canonical_selection,
    _select_occurrence_ranked_overlap_allowed,
    _select_pattern_type_atomic_prefix,
    _summarize_selection,
)
from src.attacks.csdpc.types import (
    SequenceWindow,
)


def _window(
    start,
    pattern,
    *,
    trajectory_id=0,
    length=5,
):
    pattern = tuple(
        pattern
    )

    raw_label = (
        int(
            pattern[0]
        )
        if pattern
        else 0
    )

    return SequenceWindow(
        trajectory_id=int(
            trajectory_id
        ),
        global_start=int(
            start
        ),
        global_end=int(
            start + length
        ),
        transition_indices=tuple(
            range(
                start,
                start + length,
            )
        ),
        raw_cluster_labels=tuple(
            raw_label
            for _ in range(
                length
            )
        ),
        pattern=pattern,
    )


def test_pattern_ranking_is_frequency_then_lexicographic():
    windows = (
        _window(
            0,
            (2,),
        ),
        _window(
            5,
            (1,),
        ),
        _window(
            10,
            (3,),
        ),
        _window(
            15,
            (3,),
        ),
    )

    counts = {
        (2,): 1,
        (1,): 1,
        (3,): 2,
    }

    _, ranked = (
        _build_pattern_index(
            windows,
            counts,
        )
    )

    assert ranked == (
        (1,),
        (2,),
        (3,),
    )


def test_s1_can_use_overlapping_occurrences():
    windows = (
        _window(
            0,
            (0,),
        ),
        _window(
            2,
            (1,),
        ),
    )

    counts = {
        (0,): 1,
        (1,): 1,
    }

    (
        occurrences,
        ranked,
    ) = _build_pattern_index(
        windows,
        counts,
    )

    selection = (
        _select_occurrence_ranked_overlap_allowed(
            occurrences_by_pattern=(
                occurrences
            ),
            ranked_patterns=ranked,
            transition_budget=7,
        )
    )

    assert len(
        selection.selected_windows
    ) == 2

    assert len(
        selection.unique_transition_indices
    ) == 7


def test_canonical_selector_rejects_overlap():
    windows = (
        _window(
            0,
            (0,),
        ),
        _window(
            2,
            (1,),
        ),
    )

    counts = {
        (0,): 1,
        (1,): 1,
    }

    selection = _canonical_selection(
        windows=windows,
        pattern_frequencies=counts,
        transition_budget=10,
    )

    assert len(
        selection.selected_windows
    ) == 1

    assert len(
        selection.unique_transition_indices
    ) == 5

    assert (
        selection.overlap_rejection_count
        == 1
    )


def test_s2_selects_pattern_type_atomically():
    windows = (
        _window(
            0,
            (0,),
        ),
        _window(
            5,
            (0,),
        ),

        _window(
            10,
            (1,),
        ),
        _window(
            15,
            (1,),
        ),
        _window(
            20,
            (1,),
        ),
    )

    counts = {
        (0,): 2,
        (1,): 3,
    }

    (
        occurrences,
        ranked,
    ) = _build_pattern_index(
        windows,
        counts,
    )

    selection = (
        _select_pattern_type_atomic_prefix(
            occurrences_by_pattern=(
                occurrences
            ),
            ranked_patterns=ranked,
            transition_budget=10,
        )
    )

    assert len(
        selection.selected_windows
    ) == 2

    selected_patterns = {
        tuple(
            window.pattern
        )
        for window
        in selection.selected_windows
    }

    assert selected_patterns == {
        (0,)
    }

    assert len(
        selection.unique_transition_indices
    ) == 10


def test_s2_stops_at_first_pattern_type_that_does_not_fit():
    windows = (
        _window(
            0,
            (0,),
        ),

        _window(
            5,
            (1,),
        ),
        _window(
            10,
            (1,),
        ),

        _window(
            20,
            (2,),
        ),
        _window(
            25,
            (2,),
        ),
        _window(
            30,
            (2,),
        ),
    )

    counts = {
        (0,): 1,
        (1,): 2,
        (2,): 3,
    }

    (
        occurrences,
        ranked,
    ) = _build_pattern_index(
        windows,
        counts,
    )

    selection = (
        _select_pattern_type_atomic_prefix(
            occurrences_by_pattern=(
                occurrences
            ),
            ranked_patterns=ranked,
            transition_budget=12,
        )
    )

    # Pattern (0,) consumes five transitions.
    #
    # Complete pattern (1,) would require another
    # ten, so the rare-pattern prefix must stop.
    #
    # Pattern (2,) must NOT be considered afterward.
    assert len(
        selection.selected_windows
    ) == 1

    assert (
        selection.stopped_before_pattern_rank
        == 1
    )

    assert (
        selection.stopped_before_pattern
        == (1,)
    )


def test_s2_summary_has_full_completion():
    windows = (
        _window(
            0,
            (0,),
        ),
        _window(
            5,
            (0,),
        ),
        _window(
            10,
            (1,),
        ),
        _window(
            15,
            (1,),
        ),
        _window(
            20,
            (1,),
        ),
    )

    counts = {
        (0,): 2,
        (1,): 3,
    }

    (
        occurrences,
        ranked,
    ) = _build_pattern_index(
        windows,
        counts,
    )

    selection = (
        _select_pattern_type_atomic_prefix(
            occurrences_by_pattern=(
                occurrences
            ),
            ranked_patterns=ranked,
            transition_budget=10,
        )
    )

    metrics = _summarize_selection(
        selection=selection,
        pattern_frequencies=counts,
        total_window_count=5,
        num_transitions=25,
        transition_budget=10,
    )

    assert np.isclose(
        metrics[
            "fully_selected_source_pattern_fraction"
        ],
        1.0,
    )

    assert np.isclose(
        metrics[
            "selected_source_occurrence_completion_fraction"
        ],
        1.0,
    )

    assert (
        metrics[
            "partially_selected_source_pattern_type_count"
        ]
        == 0
    )


def test_overlap_metrics_use_unique_transition_footprint():
    windows = (
        _window(
            0,
            (0,),
        ),
        _window(
            2,
            (1,),
        ),
    )

    selection = DiagnosticSelection(
        selected_windows=windows,
        unique_transition_indices=tuple(
            range(
                7
            )
        ),
    )

    metrics = _summarize_selection(
        selection=selection,
        pattern_frequencies={
            (0,): 1,
            (1,): 1,
        },
        total_window_count=2,
        num_transitions=20,
        transition_budget=7,
    )

    assert (
        metrics[
            "selected_window_transition_slot_count"
        ]
        == 10
    )

    assert (
        metrics[
            "unique_transition_footprint"
        ]
        == 7
    )

    assert np.isclose(
        metrics[
            "overlap_reuse_fraction"
        ],
        0.3,
    )

    assert (
        metrics[
            "max_transition_selection_multiplicity"
        ]
        == 2
    )