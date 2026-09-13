from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import h5py
import numpy as np

from src.attacks.csdpc.clustering import (
    build_raw_decision_units,
    fit_kmeans_decision_units,
)
from src.attacks.csdpc.metadata import (
    sha256_file,
    write_metadata_json,
)
from src.attacks.csdpc.patterns import (
    deduplicate_consecutive,
    iter_sequence_windows,
)
from src.data.trajectories import (
    TrajectorySlice,
    find_completed_trajectories,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = (
    ROOT
    / "data"
    / "raw"
    / "walker2d-medium-v2"
    / "walker2d_medium-v2.hdf5"
)

DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "gates"
    / "csdpc_sequence_enumeration_diagnostic.json"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "csdpc_sequence_enumeration"
)

EXPECTED_DATASET_SHA256 = (
    "cf00f43add04c17fdfc2958dd581dea"
    "0851b2e5bedbe6fda073758a8f841aeda"
)

REQUIRED_METRICS = {
    "window_count",
    "raw_distinct_pattern_count",
    "deduplicated_distinct_pattern_count",
    "distinct_pattern_reduction_fraction",
    "dedup_affected_window_fraction",
    "average_deduplicated_pattern_length",
}

EXPECTED_VARIANTS = {
    "D0_OVERLAPPING_STRIDE_1": {
        "window_stride": 1,
        "trajectory_offset": 0,
    },
    "D4_NONOVERLAPPING_STRIDE_5_OFFSET_0": {
        "window_stride": 5,
        "trajectory_offset": 0,
    },
}

EXPECTED_OFFSETS = [
    0,
    1,
    2,
    3,
    4,
]


def _portable_path(
    path: Path,
) -> str:
    path = path.resolve()

    try:
        return str(
            path.relative_to(
                ROOT.resolve()
            )
        )
    except ValueError:
        return str(path)


def _load_config(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = json.load(
            handle
        )

    if (
        config.get("experiment")
        != "CSDPC_SEQUENCE_ENUMERATION_DIAGNOSTIC"
    ):
        raise ValueError(
            "unexpected experiment identifier"
        )

    if (
        config.get("status")
        != "DIAGNOSTIC_ONLY"
    ):
        raise ValueError(
            "sequence-enumeration experiment "
            "must remain DIAGNOSTIC_ONLY"
        )

    if not bool(
        config.get(
            "canonical_attack_is_unchanged",
            False,
        )
    ):
        raise ValueError(
            "canonical_attack_is_unchanged "
            "must be true"
        )

    if int(
        config.get(
            "num_clusters",
            -1,
        )
    ) != 8:
        raise ValueError(
            "frozen diagnostic requires k=8"
        )

    if int(
        config.get(
            "sequence_length",
            -1,
        )
    ) != 5:
        raise ValueError(
            "frozen diagnostic requires "
            "sequence_length=5"
        )

    metrics = set(
        config.get(
            "metrics",
            [],
        )
    )

    if metrics != REQUIRED_METRICS:
        raise ValueError(
            "frozen metric set changed"
        )

    observed_variants = {}

    for variant in config.get(
        "diagnostic_variants",
        [],
    ):
        variant_id = str(
            variant[
                "id"
            ]
        )

        observed_variants[
            variant_id
        ] = {
            "window_stride": int(
                variant[
                    "window_stride"
                ]
            ),
            "trajectory_offset": int(
                variant[
                    "trajectory_offset"
                ]
            ),
        }

    if (
        observed_variants
        != EXPECTED_VARIANTS
    ):
        raise ValueError(
            "frozen D0/D4 definitions changed"
        )

    sensitivity = config.get(
        "offset_sensitivity",
        {},
    )

    if not bool(
        sensitivity.get(
            "enabled",
            False,
        )
    ):
        raise ValueError(
            "offset sensitivity must remain enabled"
        )

    if int(
        sensitivity.get(
            "stride",
            -1,
        )
    ) != 5:
        raise ValueError(
            "offset sensitivity stride "
            "must remain 5"
        )

    offsets = [
        int(value)
        for value in sensitivity.get(
            "offsets",
            [],
        )
    ]

    if offsets != EXPECTED_OFFSETS:
        raise ValueError(
            "frozen offset set changed"
        )

    if not bool(
        sensitivity.get(
            "report_all_offsets",
            False,
        )
    ):
        raise ValueError(
            "all offsets must be reported"
        )

    if not bool(
        sensitivity.get(
            "do_not_select_best_offset",
            False,
        )
    ):
        raise ValueError(
            "best-offset selection must "
            "remain forbidden"
        )

    canonical = config.get(
        "canonical_reference",
        {},
    )

    if (
        canonical.get(
            "feature_preprocessing"
        )
        != "none"
    ):
        raise ValueError(
            "canonical feature preprocessing changed"
        )

    if (
        canonical.get(
            "deduplication_order"
        )
        != "window_then_deduplicate"
    ):
        raise ValueError(
            "canonical deduplication order changed"
        )

    if int(
        canonical.get(
            "window_stride",
            -1,
        )
    ) != 1:
        raise ValueError(
            "canonical stride changed"
        )

    if not bool(
        canonical.get(
            "respect_episode_boundaries",
            False,
        )
    ):
        raise ValueError(
            "canonical trajectory-boundary "
            "rule changed"
        )

    return config


def _load_required_arrays(
    path: Path,
):
    with h5py.File(
        path,
        "r",
    ) as handle:
        required = (
            "observations",
            "actions",
            "terminals",
            "timeouts",
        )

        missing = [
            key
            for key in required
            if key not in handle
        ]

        if missing:
            raise KeyError(
                "dataset missing required fields: "
                + ", ".join(
                    missing
                )
            )

        observations = np.asarray(
            handle[
                "observations"
            ]
        )

        actions = np.asarray(
            handle[
                "actions"
            ]
        )

        terminals = np.asarray(
            handle[
                "terminals"
            ]
        )

        timeouts = np.asarray(
            handle[
                "timeouts"
            ]
        )

    return (
        observations,
        actions,
        terminals,
        timeouts,
    )


def _iter_stride_windows(
    labels: np.ndarray,
    trajectories: list[
        TrajectorySlice
    ],
    *,
    sequence_length: int,
    stride: int,
    trajectory_offset: int,
):
    """
    Enumerate trajectory-safe fixed-length windows
    with a specified stride and trajectory-relative
    starting offset.

    trajectory_offset is interpreted independently
    relative to the start of each completed trajectory.

    For example, stride=5 and trajectory_offset=2
    examines local starts:

        2, 7, 12, ...

    within every completed trajectory.

    Deduplication is always applied only after an
    original-position window has been constructed.
    """

    labels = np.asarray(
        labels
    )

    if labels.ndim != 1:
        raise ValueError(
            "labels must be 1D"
        )

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive"
        )

    if stride <= 0:
        raise ValueError(
            "stride must be positive"
        )

    if trajectory_offset < 0:
        raise ValueError(
            "trajectory_offset cannot be negative"
        )

    if trajectory_offset >= stride:
        raise ValueError(
            "trajectory_offset must be "
            "smaller than stride"
        )

    num_labels = len(
        labels
    )

    for (
        trajectory_id,
        trajectory,
    ) in enumerate(
        trajectories
    ):
        trajectory_start = int(
            trajectory.start
        )

        trajectory_end = int(
            trajectory.end
        )

        if (
            trajectory_start < 0
            or trajectory_end
            < trajectory_start
            or trajectory_end
            > num_labels
        ):
            raise ValueError(
                "invalid trajectory slice"
            )

        first_start = (
            trajectory_start
            + trajectory_offset
        )

        last_start = (
            trajectory_end
            - sequence_length
        )

        if (
            first_start
            > last_start
        ):
            continue

        for global_start in range(
            first_start,
            last_start + 1,
            stride,
        ):
            global_end = (
                global_start
                + sequence_length
            )

            raw_pattern = tuple(
                int(label)
                for label
                in labels[
                    global_start:
                    global_end
                ]
            )

            deduplicated_pattern = (
                deduplicate_consecutive(
                    raw_pattern
                )
            )

            yield {
                "trajectory_id": int(
                    trajectory_id
                ),
                "global_start": int(
                    global_start
                ),
                "global_end": int(
                    global_end
                ),
                "raw_pattern": (
                    raw_pattern
                ),
                "deduplicated_pattern": (
                    deduplicated_pattern
                ),
            }


def _iter_canonical_windows(
    labels: np.ndarray,
    trajectories: list[
        TrajectorySlice
    ],
    *,
    sequence_length: int,
):
    """
    Adapter around the actual canonical window
    implementation. D0 uses this function directly
    instead of reimplementing stride=1 behavior.
    """

    for window in (
        iter_sequence_windows(
            labels,
            trajectories,
            sequence_length=(
                sequence_length
            ),
        )
    ):
        yield {
            "trajectory_id": int(
                window.trajectory_id
            ),
            "global_start": int(
                window.global_start
            ),
            "global_end": int(
                window.global_end
            ),
            "raw_pattern": tuple(
                int(value)
                for value
                in window.raw_cluster_labels
            ),
            "deduplicated_pattern": tuple(
                int(value)
                for value
                in window.pattern
            ),
        }


def _summarize_windows(
    windows,
) -> dict:
    raw_patterns = set()
    deduplicated_patterns = set()

    window_count = 0
    dedup_affected_count = 0
    total_deduplicated_length = 0

    for window in windows:
        raw_pattern = tuple(
            window[
                "raw_pattern"
            ]
        )

        deduplicated_pattern = tuple(
            window[
                "deduplicated_pattern"
            ]
        )

        raw_patterns.add(
            raw_pattern
        )

        deduplicated_patterns.add(
            deduplicated_pattern
        )

        window_count += 1

        total_deduplicated_length += len(
            deduplicated_pattern
        )

        if (
            raw_pattern
            != deduplicated_pattern
        ):
            dedup_affected_count += 1

    raw_distinct_count = len(
        raw_patterns
    )

    deduplicated_distinct_count = len(
        deduplicated_patterns
    )

    distinct_reduction = (
        1.0
        - (
            float(
                deduplicated_distinct_count
            )
            / float(
                raw_distinct_count
            )
        )
        if raw_distinct_count
        else 0.0
    )

    affected_fraction = (
        float(
            dedup_affected_count
        )
        / float(
            window_count
        )
        if window_count
        else 0.0
    )

    average_length = (
        float(
            total_deduplicated_length
        )
        / float(
            window_count
        )
        if window_count
        else 0.0
    )

    return {
        "window_count": int(
            window_count
        ),
        "raw_distinct_pattern_count": int(
            raw_distinct_count
        ),
        "deduplicated_distinct_pattern_count": int(
            deduplicated_distinct_count
        ),
        "distinct_pattern_reduction_fraction": float(
            distinct_reduction
        ),
        "dedup_affected_window_fraction": float(
            affected_fraction
        ),
        "average_deduplicated_pattern_length": float(
            average_length
        ),
    }


def _expected_window_count(
    trajectories: list[
        TrajectorySlice
    ],
    *,
    sequence_length: int,
    stride: int,
    trajectory_offset: int,
) -> int:
    total = 0

    for trajectory in trajectories:
        first_start = (
            int(
                trajectory.start
            )
            + trajectory_offset
        )

        last_start = (
            int(
                trajectory.end
            )
            - sequence_length
        )

        if first_start > last_start:
            continue

        total += (
            (
                last_start
                - first_start
            )
            // stride
            + 1
        )

    return int(
        total
    )


def _run_seed(
    *,
    seed: int,
    config: dict,
    raw_features: np.ndarray,
    trajectories: list[
        TrajectorySlice
    ],
):
    num_clusters = int(
        config[
            "num_clusters"
        ]
    )

    sequence_length = int(
        config[
            "sequence_length"
        ]
    )

    print(
        "Fitting shared canonical raw KMeans:",
        f"seed={seed},",
        f"k={num_clusters}",
    )

    (
        _,
        clustering,
    ) = fit_kmeans_decision_units(
        raw_features,
        num_clusters=(
            num_clusters
        ),
        seed=seed,
    )

    labels = np.asarray(
        clustering.labels,
        dtype=np.int64,
    )

    d0_metrics = _summarize_windows(
        _iter_canonical_windows(
            labels,
            trajectories,
            sequence_length=(
                sequence_length
            ),
        )
    )

    d0_expected_windows = (
        _expected_window_count(
            trajectories,
            sequence_length=(
                sequence_length
            ),
            stride=1,
            trajectory_offset=0,
        )
    )

    if (
        d0_metrics[
            "window_count"
        ]
        != d0_expected_windows
    ):
        raise RuntimeError(
            "canonical stride-1 window "
            "count integrity failure"
        )

    offset_metrics = {}

    for offset in EXPECTED_OFFSETS:
        metrics = _summarize_windows(
            _iter_stride_windows(
                labels,
                trajectories,
                sequence_length=(
                    sequence_length
                ),
                stride=5,
                trajectory_offset=(
                    offset
                ),
            )
        )

        expected = _expected_window_count(
            trajectories,
            sequence_length=(
                sequence_length
            ),
            stride=5,
            trajectory_offset=(
                offset
            ),
        )

        if (
            metrics[
                "window_count"
            ]
            != expected
        ):
            raise RuntimeError(
                "stride-5 offset "
                f"{offset} window-count "
                "integrity failure"
            )

        offset_metrics[
            str(
                offset
            )
        ] = metrics

    d4_metrics = dict(
        offset_metrics[
            "0"
        ]
    )

    return {
        "clustering": {
            "method": "kmeans",
            "feature_preprocessing": (
                "none"
            ),
            "num_clusters": int(
                num_clusters
            ),
            "attack_seed": int(
                seed
            ),
            "inertia": float(
                clustering.inertia
            ),
            "n_iter": int(
                clustering.n_iter
            ),
            "shared_by_all_variants": True,
        },
        "variants": {
            "D0_OVERLAPPING_STRIDE_1": {
                "definition": {
                    "window_stride": 1,
                    "trajectory_offset": 0,
                    "sequence_length": int(
                        sequence_length
                    ),
                    "offset_semantics": (
                        "trajectory_relative"
                    ),
                    "deduplication_order": (
                        "window_then_deduplicate"
                    ),
                },
                "metrics": (
                    d0_metrics
                ),
            },
            "D4_NONOVERLAPPING_STRIDE_5_OFFSET_0": {
                "definition": {
                    "window_stride": 5,
                    "trajectory_offset": 0,
                    "sequence_length": int(
                        sequence_length
                    ),
                    "offset_semantics": (
                        "trajectory_relative"
                    ),
                    "deduplication_order": (
                        "window_then_deduplicate"
                    ),
                },
                "metrics": (
                    d4_metrics
                ),
            },
        },
        "offset_sensitivity": {
            "stride": 5,
            "offset_semantics": (
                "trajectory_relative"
            ),
            "do_not_select_best_offset": True,
            "offsets": {
                str(offset): {
                    "definition": {
                        "window_stride": 5,
                        "trajectory_offset": int(
                            offset
                        ),
                        "sequence_length": int(
                            sequence_length
                        ),
                        "deduplication_order": (
                            "window_then_deduplicate"
                        ),
                    },
                    "metrics": (
                        offset_metrics[
                            str(
                                offset
                            )
                        ]
                    ),
                }
                for offset
                in EXPECTED_OFFSETS
            },
        },
    }


def _aggregate_metric_block(
    *,
    seed_records: dict,
    seeds,
    extractor,
    metric_names,
):
    aggregate = {}

    for metric in metric_names:
        values = np.asarray(
            [
                extractor(
                    seed_records[
                        str(seed)
                    ],
                    metric,
                )
                for seed in seeds
            ],
            dtype=np.float64,
        )

        aggregate[
            metric
        ] = {
            "values_by_seed": {
                str(seed): float(
                    value
                )
                for seed, value
                in zip(
                    seeds,
                    values,
                )
            },
            "mean": float(
                np.mean(
                    values
                )
            ),
            "std": float(
                np.std(
                    values,
                    ddof=0,
                )
            ),
            "min": float(
                np.min(
                    values
                )
            ),
            "max": float(
                np.max(
                    values
                )
            ),
        }

    return aggregate


def _build_summary(
    *,
    config: dict,
    seed_records: dict,
):
    seeds = [
        int(seed)
        for seed
        in config[
            "attack_seeds"
        ]
    ]

    metric_names = list(
        config[
            "metrics"
        ]
    )

    aggregate_variants = {}

    for variant in (
        config[
            "diagnostic_variants"
        ]
    ):
        variant_id = str(
            variant[
                "id"
            ]
        )

        aggregate_variants[
            variant_id
        ] = _aggregate_metric_block(
            seed_records=(
                seed_records
            ),
            seeds=seeds,
            metric_names=(
                metric_names
            ),
            extractor=(
                lambda record,
                metric,
                variant_id=variant_id:
                record[
                    "variants"
                ][
                    variant_id
                ][
                    "metrics"
                ][
                    metric
                ]
            ),
        )

    aggregate_offsets = {}

    for offset in EXPECTED_OFFSETS:
        offset_key = str(
            offset
        )

        aggregate_offsets[
            offset_key
        ] = _aggregate_metric_block(
            seed_records=(
                seed_records
            ),
            seeds=seeds,
            metric_names=(
                metric_names
            ),
            extractor=(
                lambda record,
                metric,
                offset_key=offset_key:
                record[
                    "offset_sensitivity"
                ][
                    "offsets"
                ][
                    offset_key
                ][
                    "metrics"
                ][
                    metric
                ]
            ),
        )

    return {
        "schema_version": (
            "csdpc-sequence-enumeration-summary-v1"
        ),
        "experiment": config[
            "experiment"
        ],
        "status": config[
            "status"
        ],
        "canonical_attack_is_unchanged": True,
        "attack_seeds": seeds,
        "aggregate_variants": (
            aggregate_variants
        ),
        "aggregate_offset_sensitivity": (
            aggregate_offsets
        ),
        "interpretation": {
            "offset_semantics": (
                "Each offset is measured relative "
                "to the start of every completed "
                "trajectory."
            ),
            "all_offsets_must_be_reported": True,
            "best_offset_selection_forbidden": True,
            "canonical_gate_b_unchanged": True,
        },
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen clean-data-only "
            "CSDPC sequence-enumeration "
            "source-fidelity diagnostic."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    config = _load_config(
        args.config
    )

    dataset_path = (
        args.dataset.resolve()
    )

    config_path = (
        args.config.resolve()
    )

    output_root = (
        args.output_root.resolve()
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"dataset not found: {dataset_path}"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"config not found: {config_path}"
        )

    dataset_sha256 = sha256_file(
        dataset_path
    )

    if (
        dataset_sha256
        != EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "frozen clean dataset SHA256 mismatch\n"
            f"expected: {EXPECTED_DATASET_SHA256}\n"
            f"actual:   {dataset_sha256}"
        )

    print(
        "Frozen clean dataset SHA256: PASS"
    )

    config_sha256 = sha256_file(
        config_path
    )

    (
        observations,
        actions,
        terminals,
        timeouts,
    ) = _load_required_arrays(
        dataset_path
    )

    num_transitions = len(
        observations
    )

    if (
        len(
            actions
        )
        != num_transitions
        or len(
            terminals
        )
        != num_transitions
        or len(
            timeouts
        )
        != num_transitions
    ):
        raise RuntimeError(
            "dataset transition counts differ"
        )

    trajectories, trailing = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    raw_features = (
        build_raw_decision_units(
            observations,
            actions,
        )
    )

    completed_transition_count = int(
        sum(
            int(
                trajectory.length
            )
            for trajectory
            in trajectories
        )
    )

    dataset_record = {
        "name": config[
            "dataset"
        ],
        "path": _portable_path(
            dataset_path
        ),
        "sha256": (
            dataset_sha256
        ),
        "num_transitions": int(
            num_transitions
        ),
        "completed_trajectory_count": int(
            len(
                trajectories
            )
        ),
        "completed_transition_count": (
            completed_transition_count
        ),
        "trailing_transition_count": int(
            trailing
        ),
        "state_dim": int(
            observations.shape[
                1
            ]
        ),
        "action_dim": int(
            actions.shape[
                1
            ]
        ),
        "decision_unit_dim": int(
            raw_features.shape[
                1
            ]
        ),
    }

    protocol_record = {
        "config_path": _portable_path(
            config_path
        ),
        "config_sha256": (
            config_sha256
        ),
        "offset_semantics": (
            "trajectory_relative"
        ),
        "canonical_attack_is_unchanged": True,
    }

    seed_records = {}

    for seed in config[
        "attack_seeds"
    ]:
        seed = int(
            seed
        )

        print()
        print(
            "=" * 72
        )
        print(
            "CSDPC SEQUENCE ENUMERATION "
            f"DIAGNOSTIC — SEED {seed}"
        )
        print(
            "=" * 72
        )

        result = _run_seed(
            seed=seed,
            config=config,
            raw_features=(
                raw_features
            ),
            trajectories=(
                trajectories
            ),
        )

        seed_record = {
            "schema_version": (
                "csdpc-sequence-enumeration-seed-v1"
            ),
            "experiment": config[
                "experiment"
            ],
            "status": config[
                "status"
            ],
            "canonical_attack_is_unchanged": True,
            "seed": int(
                seed
            ),
            "dataset": (
                dataset_record
            ),
            "protocol": (
                protocol_record
            ),
            **result,
        }

        output_path = (
            output_root
            / f"seed_{seed}.json"
        )

        write_metadata_json(
            output_path,
            seed_record,
        )

        seed_records[
            str(
                seed
            )
        ] = seed_record

        print(
            "Wrote:",
            output_path,
        )

        gc.collect()

    summary = _build_summary(
        config=config,
        seed_records=(
            seed_records
        ),
    )

    summary[
        "dataset"
    ] = dataset_record

    summary[
        "protocol"
    ] = protocol_record

    summary[
        "seed_result_files"
    ] = {
        str(seed): _portable_path(
            output_root
            / f"seed_{seed}.json"
        )
        for seed
        in config[
            "attack_seeds"
        ]
    }

    summary_path = (
        output_root
        / "summary.json"
    )

    write_metadata_json(
        summary_path,
        summary,
    )

    print()
    print(
        "=" * 72
    )
    print(
        "SEQUENCE ENUMERATION "
        "DIAGNOSTIC COMPLETE"
    )
    print(
        "=" * 72
    )
    print(
        "Summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()