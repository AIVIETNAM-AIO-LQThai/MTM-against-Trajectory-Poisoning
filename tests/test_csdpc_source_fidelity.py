from __future__ import annotations

import numpy as np

from scripts.audit_csdpc_source_fidelity import (
    _adjacent_same_cluster_fraction,
    _summarize_patterns,
    _zscore_per_dimension,
)
from src.data.trajectories import (
    TrajectorySlice,
)


def test_zscore_per_dimension_preserves_dtype():
    features = np.asarray(
        [
            [1.0, 10.0, 5.0],
            [3.0, 20.0, 5.0],
            [5.0, 30.0, 5.0],
        ],
        dtype=np.float32,
    )

    standardized, details = (
        _zscore_per_dimension(
            features
        )
    )

    assert (
        standardized.dtype
        == np.float32
    )

    np.testing.assert_allclose(
        standardized[
            :,
            :2,
        ].mean(
            axis=0
        ),
        np.zeros(
            2
        ),
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        standardized[
            :,
            :2,
        ].std(
            axis=0,
            ddof=0,
        ),
        np.ones(
            2
        ),
        atol=1.0e-6,
    )

    np.testing.assert_array_equal(
        standardized[
            :,
            2
        ],
        np.zeros(
            3,
            dtype=np.float32,
        ),
    )

    assert (
        details[
            "ddof"
        ]
        == 0
    )

    assert (
        details[
            "zero_variance_dimension_count"
        ]
        == 1
    )


def test_adjacent_same_cluster_respects_boundaries():
    labels = np.asarray(
        [
            0,
            0,
            1,
            1,
            1,
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

    fraction, compared = (
        _adjacent_same_cluster_fraction(
            labels,
            trajectories,
        )
    )

    assert compared == 4

    assert np.isclose(
        fraction,
        0.5,
    )


def test_pattern_summary_deduplicates_after_windowing():
    labels = np.asarray(
        [
            1,
            1,
            2,
            0,
            0,
            0,
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

    metrics, integrity = (
        _summarize_patterns(
            labels,
            trajectories,
            sequence_length=3,
        )
    )

    assert (
        metrics[
            "raw_distinct_pattern_count"
        ]
        == 2
    )

    assert (
        metrics[
            "deduplicated_distinct_pattern_count"
        ]
        == 2
    )

    assert np.isclose(
        metrics[
            "distinct_pattern_reduction_fraction"
        ],
        0.0,
    )

    assert np.isclose(
        metrics[
            "dedup_affected_window_fraction"
        ],
        1.0,
    )

    assert np.isclose(
        metrics[
            "average_deduplicated_pattern_length"
        ],
        1.5,
    )

    assert (
        integrity[
            "expected_window_count"
        ]
        == 2
    )

    assert (
        integrity[
            "observed_window_count"
        ]
        == 2
    )

    assert (
        integrity[
            "window_count_matches"
        ]
        is True
    )


def test_windows_never_cross_trajectory_boundary():
    labels = np.asarray(
        [
            0,
            1,
            2,
            3,
            4,
            5,
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

    metrics, integrity = (
        _summarize_patterns(
            labels,
            trajectories,
            sequence_length=2,
        )
    )

    # Each length-3 trajectory has exactly
    # two legal length-2 windows.
    assert (
        integrity[
            "observed_window_count"
        ]
        == 4
    )

    assert (
        integrity[
            "expected_window_count"
        ]
        == 4
    )

    # All labels differ, so deduplication
    # should affect no window.
    assert np.isclose(
        metrics[
            "dedup_affected_window_fraction"
        ],
        0.0,
    )