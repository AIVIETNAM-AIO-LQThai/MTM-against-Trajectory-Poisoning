from __future__ import annotations

import numpy as np

from scripts.audit_csdpc_candidate_search_sensitivity import (
    CandidateChoice,
    MAX_EXTENSION_SIZE,
    _build_poison_labels,
    _extension_seed,
    _select_nested_choices,
    _selections_form_prefix,
)
from src.attacks.csdpc.types import (
    SelectedWindow,
)


def _baseline():
    return CandidateChoice(
        target_pattern=(0,),
        target_frequency=2,
        candidate_index=7,
        total_linf_perturbation=5.0,
        labels=(0,),
    )


def test_extension_seed_is_deterministic_and_window_specific():
    first = _extension_seed(
        attack_seed=0,
        trajectory_id=10,
        global_start=100,
    )

    second = _extension_seed(
        attack_seed=0,
        trajectory_id=10,
        global_start=100,
    )

    different = _extension_seed(
        attack_seed=0,
        trajectory_id=10,
        global_start=101,
    )

    assert first == second
    assert first != different


def test_nested_candidate_frequency_is_monotonic():
    patterns = [
        (1,)
        for _ in range(
            MAX_EXTENSION_SIZE
        )
    ]

    frequencies = np.ones(
        MAX_EXTENSION_SIZE,
        dtype=np.int64,
    )

    costs = np.ones(
        MAX_EXTENSION_SIZE,
        dtype=np.float64,
    )

    labels = np.ones(
        (
            MAX_EXTENSION_SIZE,
            1,
        ),
        dtype=np.int64,
    )

    # Within the first 400 extensions:
    frequencies[
        200
    ] = 5

    patterns[
        200
    ] = (5,)

    labels[
        200,
        0,
    ] = 5

    # Only available to C1000:
    frequencies[
        700
    ] = 10

    patterns[
        700
    ] = (7,)

    labels[
        700,
        0,
    ] = 7

    result = _select_nested_choices(
        baseline=_baseline(),
        extra_patterns=patterns,
        extra_frequencies=frequencies,
        extra_costs=costs,
        extra_labels=labels,
    )

    assert (
        result[
            100
        ].target_frequency
        == 2
    )

    assert (
        result[
            500
        ].target_frequency
        == 5
    )

    assert (
        result[
            1000
        ].target_frequency
        == 10
    )


def test_equal_frequency_can_win_by_lower_cost():
    patterns = [
        (1,)
        for _ in range(
            MAX_EXTENSION_SIZE
        )
    ]

    frequencies = np.ones(
        MAX_EXTENSION_SIZE,
        dtype=np.int64,
    )

    costs = np.full(
        MAX_EXTENSION_SIZE,
        20.0,
        dtype=np.float64,
    )

    labels = np.ones(
        (
            MAX_EXTENSION_SIZE,
            1,
        ),
        dtype=np.int64,
    )

    # Same frequency as canonical baseline,
    # but lower perturbation cost.
    frequencies[
        0
    ] = 2

    costs[
        0
    ] = 1.0

    result = _select_nested_choices(
        baseline=_baseline(),
        extra_patterns=patterns,
        extra_frequencies=frequencies,
        extra_costs=costs,
        extra_labels=labels,
    )

    assert (
        result[
            500
        ].target_frequency
        == 2
    )

    assert (
        result[
            500
        ].candidate_index
        == 100
    )


def test_selection_prefix_check():
    first = SelectedWindow(
        trajectory_id=0,
        global_start=0,
        global_end=5,
        source_pattern=(1,),
    )

    second = SelectedWindow(
        trajectory_id=0,
        global_start=10,
        global_end=15,
        source_pattern=(2,),
    )

    third = SelectedWindow(
        trajectory_id=1,
        global_start=20,
        global_end=25,
        source_pattern=(3,),
    )

    assert _selections_form_prefix(
        (
            first,
            second,
        ),
        (
            first,
            second,
            third,
        ),
    )

    assert not _selections_form_prefix(
        (
            second,
            first,
        ),
        (
            first,
            second,
            third,
        ),
    )


def test_build_poison_labels_modifies_only_selected_windows():
    clean = np.asarray(
        [
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
        ],
        dtype=np.int64,
    )

    windows = [
        SelectedWindow(
            trajectory_id=0,
            global_start=0,
            global_end=2,
            source_pattern=(0,),
        ),
        SelectedWindow(
            trajectory_id=0,
            global_start=4,
            global_end=6,
            source_pattern=(1,),
        ),
    ]

    choices = [
        CandidateChoice(
            target_pattern=(2,),
            target_frequency=10,
            candidate_index=100,
            total_linf_perturbation=1.0,
            labels=(2, 2),
        ),
        CandidateChoice(
            target_pattern=(3,),
            target_frequency=20,
            candidate_index=101,
            total_linf_perturbation=1.0,
            labels=(3, 3),
        ),
    ]

    poisoned = _build_poison_labels(
        clean_labels=clean,
        selected_windows=windows,
        choices=choices,
    )

    np.testing.assert_array_equal(
        poisoned,
        np.asarray(
            [
                2,
                2,
                0,
                0,
                3,
                3,
                1,
                1,
            ],
            dtype=np.int64,
        ),
    )