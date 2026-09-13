from __future__ import annotations

import numpy as np

from scripts.audit_csdpc_overlap_resolution import (
    ProposalSlot,
    _choose_slot,
    _pairwise_resolution_metrics,
    _resolve_requirements,
)


def _slot(
    window_id,
    label,
    value,
):
    return ProposalSlot(
        window_id=int(
            window_id
        ),
        target_label=int(
            label
        ),
        observation=np.asarray(
            [
                float(
                    value
                )
            ],
            dtype=np.float64,
        ),
        action=np.asarray(
            [
                float(
                    value
                )
            ],
            dtype=np.float64,
        ),
    )


def test_first_rare_wins():
    requirements = [
        _slot(
            0,
            2,
            0.1,
        ),
        _slot(
            3,
            7,
            0.2,
        ),
    ]

    chosen = _choose_slot(
        requirements,
        rule_id=(
            "R0_FIRST_RARE_WINS"
        ),
    )

    assert (
        chosen.window_id
        == 0
    )

    assert (
        chosen.target_label
        == 2
    )


def test_last_wins():
    requirements = [
        _slot(
            0,
            2,
            0.1,
        ),
        _slot(
            3,
            7,
            0.2,
        ),
    ]

    chosen = _choose_slot(
        requirements,
        rule_id=(
            "R1_LAST_WINS"
        ),
    )

    assert (
        chosen.window_id
        == 3
    )

    assert (
        chosen.target_label
        == 7
    )


def test_mode_label_uses_majority():
    requirements = [
        _slot(
            0,
            2,
            0.1,
        ),
        _slot(
            1,
            7,
            0.2,
        ),
        _slot(
            2,
            7,
            0.3,
        ),
    ]

    chosen = _choose_slot(
        requirements,
        rule_id=(
            "R2_MODE_LABEL_FIRST"
        ),
    )

    assert (
        chosen.target_label
        == 7
    )

    # Earliest window supporting label 7.
    assert (
        chosen.window_id
        == 1
    )


def test_mode_label_tie_uses_earliest_supporting_window():
    requirements = [
        _slot(
            0,
            5,
            0.1,
        ),
        _slot(
            1,
            2,
            0.2,
        ),
    ]

    chosen = _choose_slot(
        requirements,
        rule_id=(
            "R2_MODE_LABEL_FIRST"
        ),
    )

    assert (
        chosen.target_label
        == 5
    )

    assert (
        chosen.window_id
        == 0
    )


def test_all_rules_agree_without_label_conflict():
    requirements = [
        _slot(
            0,
            4,
            0.1,
        ),
        _slot(
            1,
            4,
            0.2,
        ),
        _slot(
            2,
            4,
            0.3,
        ),
    ]

    for rule in [
        "R0_FIRST_RARE_WINS",
        "R1_LAST_WINS",
        "R2_MODE_LABEL_FIRST",
    ]:
        chosen = _choose_slot(
            requirements,
            rule_id=rule,
        )

        assert (
            chosen.target_label
            == 4
        )


def test_resolve_requirements_preserves_footprint():
    requirements = {
        10: [
            _slot(
                0,
                1,
                0.1,
            ),
        ],
        11: [
            _slot(
                0,
                2,
                0.2,
            ),
            _slot(
                1,
                3,
                0.3,
            ),
        ],
    }

    resolved = (
        _resolve_requirements(
            requirements,
            rule_id=(
                "R0_FIRST_RARE_WINS"
            ),
        )
    )

    assert set(
        resolved
    ) == {
        10,
        11,
    }

    assert (
        resolved[
            10
        ].target_label
        == 1
    )

    assert (
        resolved[
            11
        ].target_label
        == 2
    )


def test_pairwise_resolution_metrics_separate_label_and_row_difference():
    left = {
        0: _slot(
            0,
            2,
            0.1,
        ),
        1: _slot(
            0,
            3,
            0.2,
        ),
    }

    right = {
        # Same label but different continuous row.
        0: _slot(
            1,
            2,
            0.9,
        ),

        # Different label and different row.
        1: _slot(
            1,
            7,
            0.8,
        ),
    }

    metrics = (
        _pairwise_resolution_metrics(
            left=left,
            right=right,
        )
    )

    assert np.isclose(
        metrics[
            "merged_target_label_disagreement_fraction"
        ],
        0.5,
    )

    assert np.isclose(
        metrics[
            "merged_continuous_row_disagreement_fraction"
        ],
        1.0,
    )