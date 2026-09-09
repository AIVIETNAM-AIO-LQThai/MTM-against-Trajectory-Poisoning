from __future__ import annotations

import numpy as np

from scripts.audit_csdpc_sequence_enumeration import (
    DEFAULT_CONFIG,
    _expected_window_count,
    _iter_canonical_windows,
    _iter_stride_windows,
    _load_config,
    _summarize_windows,
)
from src.data.trajectories import (
    TrajectorySlice,
)


def test_frozen_config_is_accepted():
    config = _load_config(
        DEFAULT_CONFIG
    )

    assert (
        config[
            "experiment"
        ]
        == "CSDPC_SEQUENCE_ENUMERATION_DIAGNOSTIC"
    )

    assert (
        config[
            "num_clusters"
        ]
        == 8
    )

    assert (
        config[
            "sequence_length"
        ]
        == 5
    )


def test_stride_one_matches_canonical_enumeration():
    labels = np.asarray(
        [
            0,
            0,
            1,
            2,
            2,
            3,
            4,
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
            end=8,
        ),
    ]

    canonical = list(
        _iter_canonical_windows(
            labels,
            trajectories,
            sequence_length=3,
        )
    )

    stride_one = list(
        _iter_stride_windows(
            labels,
            trajectories,
            sequence_length=3,
            stride=1,
            trajectory_offset=0,
        )
    )

    canonical_signature = [
        (
            window[
                "trajectory_id"
            ],
            window[
                "global_start"
            ],
            window[
                "raw_pattern"
            ],
            window[
                "deduplicated_pattern"
            ],
        )
        for window
        in canonical
    ]

    stride_signature = [
        (
            window[
                "trajectory_id"
            ],
            window[
                "global_start"
            ],
            window[
                "raw_pattern"
            ],
            window[
                "deduplicated_pattern"
            ],
        )
        for window
        in stride_one
    ]

    assert (
        canonical_signature
        == stride_signature
    )


def test_stride_offset_is_trajectory_relative():
    labels = np.arange(
        12,
        dtype=np.int64,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=6,
        ),
        TrajectorySlice(
            start=6,
            end=12,
        ),
    ]

    windows = list(
        _iter_stride_windows(
            labels,
            trajectories,
            sequence_length=3,
            stride=3,
            trajectory_offset=1,
        )
    )

    starts = [
        window[
            "global_start"
        ]
        for window
        in windows
    ]

    # Offset 1 is applied independently
    # relative to each trajectory start:
    #
    # trajectory 0 -> start 1
    # trajectory 1 -> start 7
    assert starts == [
        1,
        7,
    ]


def test_stride_windows_never_cross_trajectory_boundary():
    labels = np.arange(
        10,
        dtype=np.int64,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=5,
        ),
        TrajectorySlice(
            start=5,
            end=10,
        ),
    ]

    windows = list(
        _iter_stride_windows(
            labels,
            trajectories,
            sequence_length=4,
            stride=5,
            trajectory_offset=0,
        )
    )

    assert len(
        windows
    ) == 2

    assert (
        windows[
            0
        ][
            "global_start"
        ]
        == 0
    )

    assert (
        windows[
            0
        ][
            "global_end"
        ]
        == 4
    )

    assert (
        windows[
            1
        ][
            "global_start"
        ]
        == 5
    )

    assert (
        windows[
            1
        ][
            "global_end"
        ]
        == 9
    )


def test_deduplication_occurs_after_selected_window():
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

    windows = list(
        _iter_stride_windows(
            labels,
            trajectories,
            sequence_length=5,
            stride=5,
            trajectory_offset=0,
        )
    )

    assert len(
        windows
    ) == 1

    assert (
        windows[
            0
        ][
            "raw_pattern"
        ]
        == (
            1,
            1,
            2,
            2,
            3,
        )
    )

    assert (
        windows[
            0
        ][
            "deduplicated_pattern"
        ]
        == (
            1,
            2,
            3,
        )
    )


def test_expected_window_count_matches_enumerator():
    labels = np.arange(
        23,
        dtype=np.int64,
    )

    trajectories = [
        TrajectorySlice(
            start=0,
            end=11,
        ),
        TrajectorySlice(
            start=11,
            end=23,
        ),
    ]

    for offset in range(
        5
    ):
        windows = list(
            _iter_stride_windows(
                labels,
                trajectories,
                sequence_length=5,
                stride=5,
                trajectory_offset=offset,
            )
        )

        expected = (
            _expected_window_count(
                trajectories,
                sequence_length=5,
                stride=5,
                trajectory_offset=(
                    offset
                ),
            )
        )

        assert len(
            windows
        ) == expected


def test_summary_metrics_are_correct():
    labels = np.asarray(
        [
            0,
            0,
            1,
            1,
            2,
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
            end=10,
        )
    ]

    metrics = _summarize_windows(
        _iter_stride_windows(
            labels,
            trajectories,
            sequence_length=5,
            stride=5,
            trajectory_offset=0,
        )
    )

    assert (
        metrics[
            "window_count"
        ]
        == 2
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
        0.5,
    )

    # First pattern:
    # [0,0,1,1,2] -> [0,1,2] = length 3
    #
    # Second:
    # [0,1,2,3,4] -> unchanged = length 5
    #
    # Mean = 4.
    assert np.isclose(
        metrics[
            "average_deduplicated_pattern_length"
        ],
        4.0,
    )