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
    / "csdpc_dedup_order_diagnostic.json"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "csdpc_dedup_order"
)

EXPECTED_DATASET_SHA256 = (
    "cf00f43add04c17fdfc2958dd581dea"
    "0851b2e5bedbe6fda073758a8f841aeda"
)

REQUIRED_METRICS = {
    "original_transition_label_count",
    "compressed_label_count",
    "compressed_label_fraction",
    "raw_window_count",
    "diagnostic_window_count",
    "raw_distinct_pattern_count",
    "diagnostic_distinct_pattern_count",
    "distinct_pattern_reduction_fraction",
    "average_pattern_length",
}

EXPECTED_VARIANTS = {
    "D0_CANONICAL": "window_then_deduplicate",
    "D3_TRAJECTORY_DEDUP_BEFORE_WINDOW": (
        "trajectory_deduplicate_then_window"
    ),
}


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
        != "CSDPC_DEDUP_ORDER_DIAGNOSTIC"
    ):
        raise ValueError(
            "unexpected experiment identifier"
        )

    if (
        config.get("status")
        != "DIAGNOSTIC_ONLY"
    ):
        raise ValueError(
            "dedup-order experiment must remain "
            "DIAGNOSTIC_ONLY"
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

    if int(
        config.get(
            "num_clusters",
            -1,
        )
    ) != 8:
        raise ValueError(
            "frozen diagnostic requires k=8"
        )

    variants = config.get(
        "diagnostic_variants",
        [],
    )

    observed_variants = {
        str(
            variant["id"]
        ): str(
            variant[
                "deduplication_order"
            ]
        )
        for variant
        in variants
    }

    if (
        observed_variants
        != EXPECTED_VARIANTS
    ):
        raise ValueError(
            "frozen D0/D3 definitions changed"
        )

    for variant in variants:
        if (
            int(
                variant[
                    "sequence_length"
                ]
            )
            != 5
        ):
            raise ValueError(
                "frozen diagnostic requires "
                "sequence_length=5"
            )

        if (
            str(
                variant[
                    "feature_preprocessing"
                ]
            )
            != "none"
        ):
            raise ValueError(
                "frozen diagnostic requires "
                "raw unscaled features"
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


def _completed_label_count(
    trajectories: list[
        TrajectorySlice
    ],
) -> int:
    return int(
        sum(
            int(
                trajectory.length
            )
            for trajectory
            in trajectories
        )
    )


def _canonical_raw_baseline(
    labels: np.ndarray,
    trajectories: list[
        TrajectorySlice
    ],
    *,
    sequence_length: int,
):
    """
    Canonical pre-deduplication reference.

    Windows are formed from original decision positions,
    independently within each completed trajectory.
    """

    raw_patterns = set()
    raw_window_count = 0

    for window in (
        iter_sequence_windows(
            labels,
            trajectories,
            sequence_length=(
                sequence_length
            ),
        )
    ):
        raw_window_count += 1

        raw_patterns.add(
            tuple(
                window.raw_cluster_labels
            )
        )

    expected_window_count = sum(
        max(
            0,
            int(
                trajectory.length
            )
            - sequence_length
            + 1,
        )
        for trajectory
        in trajectories
    )

    if (
        raw_window_count
        != expected_window_count
    ):
        raise RuntimeError(
            "canonical raw window-count "
            "integrity failure"
        )

    return {
        "raw_window_count": int(
            raw_window_count
        ),
        "raw_distinct_pattern_count": int(
            len(
                raw_patterns
            )
        ),
    }


def _audit_d0_canonical(
    labels: np.ndarray,
    trajectories: list[
        TrajectorySlice
    ],
    *,
    sequence_length: int,
    raw_baseline: dict,
):
    """
    Canonical interpretation:

        original trajectory
        -> original-position length-L window
        -> consecutive-label deduplication
    """

    diagnostic_patterns = set()

    diagnostic_window_count = 0
    total_pattern_length = 0

    for window in (
        iter_sequence_windows(
            labels,
            trajectories,
            sequence_length=(
                sequence_length
            ),
        )
    ):
        pattern = tuple(
            window.pattern
        )

        diagnostic_patterns.add(
            pattern
        )

        diagnostic_window_count += 1

        total_pattern_length += len(
            pattern
        )

    original_label_count = (
        _completed_label_count(
            trajectories
        )
    )

    compressed_label_count = (
        original_label_count
    )

    diagnostic_distinct_count = (
        len(
            diagnostic_patterns
        )
    )

    raw_distinct_count = int(
        raw_baseline[
            "raw_distinct_pattern_count"
        ]
    )

    reduction = (
        1.0
        - (
            float(
                diagnostic_distinct_count
            )
            / float(
                raw_distinct_count
            )
        )
        if raw_distinct_count
        else 0.0
    )

    average_pattern_length = (
        float(
            total_pattern_length
        )
        / float(
            diagnostic_window_count
        )
        if diagnostic_window_count
        else 0.0
    )

    return {
        "original_transition_label_count": int(
            original_label_count
        ),
        "compressed_label_count": int(
            compressed_label_count
        ),
        "compressed_label_fraction": 1.0,
        "raw_window_count": int(
            raw_baseline[
                "raw_window_count"
            ]
        ),
        "diagnostic_window_count": int(
            diagnostic_window_count
        ),
        "raw_distinct_pattern_count": int(
            raw_distinct_count
        ),
        "diagnostic_distinct_pattern_count": int(
            diagnostic_distinct_count
        ),
        "distinct_pattern_reduction_fraction": float(
            reduction
        ),
        "average_pattern_length": float(
            average_pattern_length
        ),
    }


def _compress_trajectory_labels(
    labels: np.ndarray,
    trajectory: TrajectorySlice,
) -> tuple[int, ...]:
    """
    Deduplicate one completed trajectory only.

    No label from one trajectory may affect
    compression of another trajectory.
    """

    start = int(
        trajectory.start
    )

    end = int(
        trajectory.end
    )

    return deduplicate_consecutive(
        labels[
            start:end
        ]
    )


def _iter_compressed_windows(
    compressed_labels: tuple[
        int,
        ...
    ],
    *,
    sequence_length: int,
):
    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive"
        )

    if (
        len(
            compressed_labels
        )
        < sequence_length
    ):
        return

    last_start = (
        len(
            compressed_labels
        )
        - sequence_length
    )

    for start in range(
        last_start
        + 1
    ):
        yield tuple(
            compressed_labels[
                start:
                start
                + sequence_length
            ]
        )


def _audit_d3_trajectory_dedup(
    labels: np.ndarray,
    trajectories: list[
        TrajectorySlice
    ],
    *,
    sequence_length: int,
    raw_baseline: dict,
):
    """
    Alternative source interpretation:

        original trajectory labels
        -> deduplicate entire trajectory
        -> form length-L windows on compressed labels

    This function is diagnostic only. It does not
    alter canonical CSDPC attack code or datasets.
    """

    original_label_count = (
        _completed_label_count(
            trajectories
        )
    )

    compressed_label_count = 0

    diagnostic_window_count = 0

    diagnostic_patterns = set()

    total_pattern_length = 0

    for trajectory in trajectories:
        compressed = (
            _compress_trajectory_labels(
                labels,
                trajectory,
            )
        )

        compressed_label_count += len(
            compressed
        )

        for pattern in (
            _iter_compressed_windows(
                compressed,
                sequence_length=(
                    sequence_length
                ),
            )
        ):
            diagnostic_window_count += 1

            diagnostic_patterns.add(
                pattern
            )

            total_pattern_length += len(
                pattern
            )

    compressed_fraction = (
        float(
            compressed_label_count
        )
        / float(
            original_label_count
        )
        if original_label_count
        else 0.0
    )

    raw_distinct_count = int(
        raw_baseline[
            "raw_distinct_pattern_count"
        ]
    )

    diagnostic_distinct_count = (
        len(
            diagnostic_patterns
        )
    )

    reduction = (
        1.0
        - (
            float(
                diagnostic_distinct_count
            )
            / float(
                raw_distinct_count
            )
        )
        if raw_distinct_count
        else 0.0
    )

    average_pattern_length = (
        float(
            total_pattern_length
        )
        / float(
            diagnostic_window_count
        )
        if diagnostic_window_count
        else 0.0
    )

    return {
        "original_transition_label_count": int(
            original_label_count
        ),
        "compressed_label_count": int(
            compressed_label_count
        ),
        "compressed_label_fraction": float(
            compressed_fraction
        ),
        "raw_window_count": int(
            raw_baseline[
                "raw_window_count"
            ]
        ),
        "diagnostic_window_count": int(
            diagnostic_window_count
        ),
        "raw_distinct_pattern_count": int(
            raw_distinct_count
        ),
        "diagnostic_distinct_pattern_count": int(
            diagnostic_distinct_count
        ),
        "distinct_pattern_reduction_fraction": float(
            reduction
        ),
        "average_pattern_length": float(
            average_pattern_length
        ),
    }


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

    sequence_length = 5

    print(
        "Fitting canonical raw KMeans:",
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

    raw_baseline = (
        _canonical_raw_baseline(
            labels,
            trajectories,
            sequence_length=(
                sequence_length
            ),
        )
    )

    d0_metrics = (
        _audit_d0_canonical(
            labels,
            trajectories,
            sequence_length=(
                sequence_length
            ),
            raw_baseline=(
                raw_baseline
            ),
        )
    )

    d3_metrics = (
        _audit_d3_trajectory_dedup(
            labels,
            trajectories,
            sequence_length=(
                sequence_length
            ),
            raw_baseline=(
                raw_baseline
            ),
        )
    )

    if (
        d0_metrics[
            "raw_window_count"
        ]
        != d3_metrics[
            "raw_window_count"
        ]
    ):
        raise RuntimeError(
            "D0/D3 raw baseline window "
            "counts differ"
        )

    if (
        d0_metrics[
            "raw_distinct_pattern_count"
        ]
        != d3_metrics[
            "raw_distinct_pattern_count"
        ]
    ):
        raise RuntimeError(
            "D0/D3 raw distinct baselines differ"
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
            "shared_by_d0_and_d3": True,
        },
        "variants": {
            "D0_CANONICAL": {
                "definition": {
                    "sequence_length": 5,
                    "feature_preprocessing": (
                        "none"
                    ),
                    "deduplication_order": (
                        "window_then_deduplicate"
                    ),
                },
                "metrics": (
                    d0_metrics
                ),
            },
            "D3_TRAJECTORY_DEDUP_BEFORE_WINDOW": {
                "definition": {
                    "sequence_length": 5,
                    "feature_preprocessing": (
                        "none"
                    ),
                    "deduplication_order": (
                        "trajectory_deduplicate_then_window"
                    ),
                },
                "metrics": (
                    d3_metrics
                ),
            },
        },
    }


def _build_summary(
    *,
    config: dict,
    seed_records: dict,
):
    aggregate = {}

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

        aggregate[
            variant_id
        ] = {}

        for metric in config[
            "metrics"
        ]:
            values = np.asarray(
                [
                    seed_records[
                        str(seed)
                    ][
                        "variants"
                    ][
                        variant_id
                    ][
                        "metrics"
                    ][
                        metric
                    ]
                    for seed
                    in config[
                        "attack_seeds"
                    ]
                ],
                dtype=np.float64,
            )

            aggregate[
                variant_id
            ][
                metric
            ] = {
                "values_by_seed": {
                    str(seed): float(
                        value
                    )
                    for seed, value
                    in zip(
                        config[
                            "attack_seeds"
                        ],
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

    return {
        "schema_version": (
            "csdpc-dedup-order-summary-v1"
        ),
        "experiment": config[
            "experiment"
        ],
        "status": config[
            "status"
        ],
        "canonical_attack_is_unchanged": True,
        "attack_seeds": [
            int(seed)
            for seed
            in config[
                "attack_seeds"
            ]
        ],
        "metric_semantics": {
            "original_transition_label_count": (
                "cluster labels belonging to completed "
                "trajectories only; trailing incomplete "
                "fragment excluded"
            ),
            "compressed_label_count": (
                "label-stream length used before windowing; "
                "D0 equals original completed labels, D3 "
                "equals trajectory-wise consecutively "
                "deduplicated labels"
            ),
            "compressed_label_fraction": (
                "compressed_label_count divided by "
                "original_transition_label_count"
            ),
            "raw_window_count": (
                "canonical length-5 original-position "
                "window count"
            ),
            "diagnostic_window_count": (
                "window count after applying each "
                "diagnostic interpretation"
            ),
            "distinct_pattern_reduction_fraction": (
                "1 - diagnostic_distinct_pattern_count / "
                "raw_distinct_pattern_count"
            ),
        },
        "aggregate_metrics": (
            aggregate
        ),
        "interpretation_rule": (
            "D3 is source-fidelity diagnosis only and "
            "cannot replace canonical CSDPC or Gate B."
        ),
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen clean-data-only "
            "CSDPC deduplication-order diagnostic."
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

    output_root = (
        args.output_root.resolve()
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"dataset not found: {dataset_path}"
        )

    actual_sha256 = (
        sha256_file(
            dataset_path
        )
    )

    if (
        actual_sha256
        != EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "frozen clean dataset SHA256 mismatch\n"
            f"expected: {EXPECTED_DATASET_SHA256}\n"
            f"actual:   {actual_sha256}"
        )

    print(
        "Frozen clean dataset SHA256: PASS"
    )

    (
        observations,
        actions,
        terminals,
        timeouts,
    ) = _load_required_arrays(
        dataset_path
    )

    if (
        len(
            observations
        )
        != len(
            actions
        )
        or len(
            observations
        )
        != len(
            terminals
        )
        or len(
            observations
        )
        != len(
            timeouts
        )
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

    dataset_record = {
        "name": config[
            "dataset"
        ],
        "path": _portable_path(
            dataset_path
        ),
        "sha256": actual_sha256,
        "num_transitions": int(
            len(
                observations
            )
        ),
        "completed_trajectory_count": int(
            len(
                trajectories
            )
        ),
        "completed_transition_count": int(
            _completed_label_count(
                trajectories
            )
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
            "CSDPC DEDUP ORDER "
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
                "csdpc-dedup-order-seed-v1"
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

        print(
            "Wrote:",
            output_path,
        )

        seed_records[
            str(
                seed
            )
        ] = seed_record

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
        "DEDUP ORDER DIAGNOSTIC COMPLETE"
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