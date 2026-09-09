from __future__ import annotations

import numpy as np

from scripts.audit_csdpc_dedup_order import (
    _audit_d0_canonical,
    _audit_d3_trajectory_dedup,
    _canonical_raw_baseline,
    _compress_trajectory_labels,
    _iter_compressed_windows,
)
from src.data.trajectories import (
    TrajectorySlice,
)


def test_trajectory_compression_is_per_trajectory():
    labels = np.asarray(
        [
            1, 1, 2,
            2, 2, 3,
        ],
        dtype=np.int64,
    )

    first = TrajectorySlice(
        start=0,
        end=3,
    )

    second = TrajectorySlice(
        start=3,
        end=6,
    )

    assert (
        _compress_trajectory_labels(
            labels,
            first,
        )
        == (1, 2)
    )

    assert (
        _compress_trajectory_labels(
            labels,
            second,
        )
        == (2, 3)
    )

    # The two boundary-adjacent 2 labels
    # must NOT collapse across trajectories.
    assert (
        len(
            _compress_trajectory_labels(
                labels,
                first,
            )
        )
        + len(
            _compress_trajectory_labels(
                labels,
                second,
            )
        )
        == 4
    )


def test_compressed_windows_use_compressed_positions():
    compressed = (
        0,
        1,
        2,
        3,
        4,
        5,
    )

    windows = list(
        _iter_compressed_windows(
            compressed,
            sequence_length=5,
        )
    )

    assert windows == [
        (
            0,
            1,
            2,
            3,
            4,
        ),
        (
            1,
            2,
            3,
            4,
            5,
        ),
    ]


def test_d0_matches_window_then_deduplicate():
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

    raw = (
        _canonical_raw_baseline(
            labels,
            trajectories,
            sequence_length=5,
        )
    )

    result = (
        _audit_d0_canonical(
            labels,
            trajectories,
            sequence_length=5,
            raw_baseline=raw,
        )
    )

    assert (
        result[
            "original_transition_label_count"
        ]
        == 5
    )

    assert (
        result[
            "compressed_label_count"
        ]
        == 5
    )

    assert np.isclose(
        result[
            "compressed_label_fraction"
        ],
        1.0,
    )

    assert (
        result[
            "raw_window_count"
        ]
        == 1
    )

    assert (
        result[
            "diagnostic_window_count"
        ]
        == 1
    )

    assert (
        result[
            "raw_distinct_pattern_count"
        ]
        == 1
    )

    assert (
        result[
            "diagnostic_distinct_pattern_count"
        ]
        == 1
    )

    assert np.isclose(
        result[
            "distinct_pattern_reduction_fraction"
        ],
        0.0,
    )

    # [1,1,2,2,3] -> [1,2,3]
    assert np.isclose(
        result[
            "average_pattern_length"
        ],
        3.0,
    )


def test_d3_deduplicates_before_windowing():
    labels = np.asarray(
        [
            0, 0,
            1, 1,
            2, 2,
            3, 3,
            4, 4,
            5, 5,
        ],
        dtype=np.int64,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=12,
        )
    ]

    raw = (
        _canonical_raw_baseline(
            labels,
            trajectories,
            sequence_length=5,
        )
    )

    result = (
        _audit_d3_trajectory_dedup(
            labels,
            trajectories,
            sequence_length=5,
            raw_baseline=raw,
        )
    )

    # 12 original labels compress to:
    # [0,1,2,3,4,5]
    assert (
        result[
            "original_transition_label_count"
        ]
        == 12
    )

    assert (
        result[
            "compressed_label_count"
        ]
        == 6
    )

    assert np.isclose(
        result[
            "compressed_label_fraction"
        ],
        0.5,
    )

    # Six compressed labels produce two
    # compressed length-5 windows.
    assert (
        result[
            "diagnostic_window_count"
        ]
        == 2
    )

    assert (
        result[
            "diagnostic_distinct_pattern_count"
        ]
        == 2
    )

    assert np.isclose(
        result[
            "average_pattern_length"
        ],
        5.0,
    )


def test_d3_never_joins_compressed_trajectories():
    labels = np.asarray(
        [
            0, 0, 1, 1,
            2, 2, 3, 3,
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
            end=8,
        ),
    ]

    raw = (
        _canonical_raw_baseline(
            labels,
            trajectories,
            sequence_length=2,
        )
    )

    result = (
        _audit_d3_trajectory_dedup(
            labels,
            trajectories,
            sequence_length=2,
            raw_baseline=raw,
        )
    )

    # Each trajectory independently becomes
    # length 2, hence one diagnostic window each.
    assert (
        result[
            "diagnostic_window_count"
        ]
        == 2
    )

    assert (
        result[
            "compressed_label_count"
        ]
        == 4
    )