from __future__ import annotations

import numpy as np

from scripts.audit_csdpc_overlap_conflict import (
    WindowProposal,
    _compute_conflict_metrics,
    _is_prefix,
)
from src.attacks.csdpc.types import (
    SelectedWindow,
)
from src.attacks.csdpc.patterns import (
    deduplicate_consecutive,
)


def _window(
    start,
    end,
    pattern=(0,),
):
    return SelectedWindow(
        trajectory_id=0,
        global_start=start,
        global_end=end,
        source_pattern=tuple(pattern),
    )


def _proposal(
    labels,
    *,
    source_frequency=1,
    target_frequency=2,
):
    return WindowProposal(
        raw_target_labels=tuple(
            labels
        ),
        target_pattern=tuple(
            deduplicate_consecutive(
                labels
            )
        ),
        source_frequency=(
            source_frequency
        ),
        target_frequency=(
            target_frequency
        ),
        candidate_index=0,
    )


def test_prefix_check():
    a = _window(
        0,
        5,
        (1,),
    )

    b = _window(
        5,
        10,
        (2,),
    )

    c = _window(
        10,
        15,
        (3,),
    )

    assert _is_prefix(
        (a, b),
        (a, b, c),
    )

    assert not _is_prefix(
        (b, a),
        (a, b, c),
    )


def test_overlap_with_same_label_has_no_conflict():
    windows = (
        _window(
            0,
            3,
        ),
        _window(
            2,
            5,
        ),
    )

    proposals = (
        _proposal(
            (1, 2, 3)
        ),
        _proposal(
            (3, 4, 5)
        ),
    )

    m = _compute_conflict_metrics(
        selected_windows=windows,
        proposals=proposals,
    )

    assert (
        m[
            "unique_transition_footprint"
        ]
        == 5
    )

    assert (
        m[
            "selected_window_transition_slot_count"
        ]
        == 6
    )

    assert np.isclose(
        m[
            "overlap_reuse_fraction"
        ],
        1.0 / 6.0,
    )

    assert np.isclose(
        m[
            "overlapped_unique_transition_fraction"
        ],
        1.0 / 5.0,
    )

    assert np.isclose(
        m[
            "conflicting_overlap_transition_fraction"
        ],
        0.0,
    )

    assert np.isclose(
        m[
            "window_touching_label_conflict_fraction"
        ],
        0.0,
    )


def test_overlap_with_different_labels_is_conflict():
    windows = (
        _window(
            0,
            3,
        ),
        _window(
            2,
            5,
        ),
    )

    proposals = (
        _proposal(
            (1, 2, 3)
        ),
        _proposal(
            (7, 4, 5)
        ),
    )

    m = _compute_conflict_metrics(
        selected_windows=windows,
        proposals=proposals,
    )

    assert np.isclose(
        m[
            "conflicting_overlap_transition_fraction"
        ],
        1.0,
    )

    assert np.isclose(
        m[
            "conflict_transition_fraction_of_unique_footprint"
        ],
        1.0 / 5.0,
    )

    assert np.isclose(
        m[
            "pairwise_label_disagreement_fraction_on_overlaps"
        ],
        1.0,
    )

    assert np.isclose(
        m[
            "window_touching_label_conflict_fraction"
        ],
        1.0,
    )

    assert np.isclose(
        m[
            "window_not_touching_label_conflict_fraction"
        ],
        0.0,
    )


def test_three_way_overlap_pairwise_disagreement():
    windows = (
        _window(
            0,
            1,
        ),
        _window(
            0,
            1,
        ),
        _window(
            0,
            1,
        ),
    )

    proposals = (
        _proposal(
            (2,)
        ),
        _proposal(
            (2,)
        ),
        _proposal(
            (5,)
        ),
    )

    m = _compute_conflict_metrics(
        selected_windows=windows,
        proposals=proposals,
    )

    # Three pairs:
    # 2-vs-2 agrees;
    # two 2-vs-5 pairs disagree.
    assert np.isclose(
        m[
            "pairwise_label_disagreement_fraction_on_overlaps"
        ],
        2.0 / 3.0,
    )

    assert np.isclose(
        m[
            "mean_distinct_target_labels_on_overlapped_transitions"
        ],
        2.0,
    )

    assert (
        m[
            "max_distinct_target_labels_on_any_transition"
        ]
        == 2
    )


def test_nonoverlap_has_zero_conflict_metrics():
    windows = (
        _window(
            0,
            2,
        ),
        _window(
            2,
            4,
        ),
    )

    proposals = (
        _proposal(
            (1, 1)
        ),
        _proposal(
            (2, 2)
        ),
    )

    m = _compute_conflict_metrics(
        selected_windows=windows,
        proposals=proposals,
    )

    assert np.isclose(
        m[
            "overlapped_unique_transition_fraction"
        ],
        0.0,
    )

    assert np.isclose(
        m[
            "conflicting_overlap_transition_fraction"
        ],
        0.0,
    )

    assert np.isclose(
        m[
            "window_touching_label_conflict_fraction"
        ],
        0.0,
    )

    assert np.isclose(
        m[
            "window_not_touching_label_conflict_fraction"
        ],
        1.0,
    )