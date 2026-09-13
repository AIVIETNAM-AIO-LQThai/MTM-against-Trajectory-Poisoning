from __future__ import annotations

import numpy as np

from scripts.audit_csdpc_dedup_metric_reconciliation import (
    _compute_metrics,
)
from src.data.trajectories import (
    TrajectorySlice,
)


def test_metric_reconciliation_simple_case():
    labels = np.asarray(
        [
            1,
            1,
            2,
            2,
            3,
        ],
        dtype=np.int64,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=5,
        )
    ]

    metrics = _compute_metrics(
        labels,
        trajectories,
        sequence_length=5,
    )

    assert (
        metrics[
            "window_count"
        ]
        == 1
    )

    assert (
        metrics[
            "raw_distinct_sequence_type_count"
        ]
        == 1
    )

    assert (
        metrics[
            "deduplicated_distinct_pattern_type_count"
        ]
        == 1
    )

    assert np.isclose(
        metrics[
            "canonical_distinct_type_reduction_fraction"
        ],
        0.0,
    )

    assert np.isclose(
        metrics[
            "window_instance_changed_fraction"
        ],
        1.0,
    )

    assert np.isclose(
        metrics[
            "raw_distinct_sequence_type_changed_fraction"
        ],
        1.0,
    )

    # 5 labels -> 3 labels.
    assert np.isclose(
        metrics[
            "total_label_token_reduction_fraction"
        ],
        0.4,
    )

    assert np.isclose(
        metrics[
            "average_deduplicated_pattern_length"
        ],
        3.0,
    )


def test_no_duplicate_case():
    labels = np.asarray(
        [
            0,
            1,
            2,
            3,
            4,
        ],
        dtype=np.int64,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=5,
        )
    ]

    metrics = _compute_metrics(
        labels,
        trajectories,
        sequence_length=5,
    )

    assert np.isclose(
        metrics[
            "window_instance_changed_fraction"
        ],
        0.0,
    )

    assert np.isclose(
        metrics[
            "raw_distinct_sequence_type_changed_fraction"
        ],
        0.0,
    )

    assert np.isclose(
        metrics[
            "total_label_token_reduction_fraction"
        ],
        0.0,
    )

    assert np.isclose(
        metrics[
            "full_length_pattern_occurrence_fraction"
        ],
        1.0,
    )

    assert np.isclose(
        metrics[
            "full_length_distinct_pattern_fraction"
        ],
        1.0,
    )


def test_metric_windows_respect_trajectory_boundaries():
    labels = np.asarray(
        [
            0,
            0,
            1,
            1,
            2,
            2,
        ],
        dtype=np.int64,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=3,
        ),
        TrajectorySlice(
            start=3,
            end=6,
        ),
    ]

    metrics = _compute_metrics(
        labels,
        trajectories,
        sequence_length=2,
    )

    assert (
        metrics[
            "window_count"
        ]
        == 4
    )