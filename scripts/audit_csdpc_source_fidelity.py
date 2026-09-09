from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
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
    / "csdpc_source_fidelity_diagnostics.json"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "csdpc_source_fidelity"
)

EXPECTED_DATASET_SHA256 = (
    "cf00f43add04c17fdfc2958dd581dea"
    "0851b2e5bedbe6fda073758a8f841aeda"
)

REQUIRED_METRICS = {
    "raw_distinct_pattern_count",
    "deduplicated_distinct_pattern_count",
    "distinct_pattern_reduction_fraction",
    "dedup_affected_window_fraction",
    "average_deduplicated_pattern_length",
    "adjacent_same_cluster_fraction",
    "cluster_occupancy",
}

SUPPORTED_PREPROCESSING = {
    "none",
    "zscore_per_dimension",
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
        config = json.load(handle)

    if (
        config.get("experiment")
        != "CSDPC_SOURCE_FIDELITY_DIAGNOSTICS"
    ):
        raise ValueError(
            "unexpected diagnostic experiment identifier"
        )

    if (
        config.get("status")
        != "DIAGNOSTIC_ONLY"
    ):
        raise ValueError(
            "source-fidelity diagnostic must remain DIAGNOSTIC_ONLY"
        )

    if not bool(
        config.get(
            "canonical_attack_is_unchanged",
            False,
        )
    ):
        raise ValueError(
            "canonical_attack_is_unchanged must be true"
        )

    metrics = set(
        config.get(
            "metrics",
            [],
        )
    )

    if metrics != REQUIRED_METRICS:
        raise ValueError(
            "frozen source-fidelity metric set changed"
        )

    variants = config.get(
        "diagnostic_variants",
        [],
    )

    if not variants:
        raise ValueError(
            "diagnostic_variants cannot be empty"
        )

    variant_ids = [
        str(variant["id"])
        for variant in variants
    ]

    if len(variant_ids) != len(
        set(variant_ids)
    ):
        raise ValueError(
            "diagnostic variant IDs must be unique"
        )

    for variant in variants:
        preprocessing = str(
            variant[
                "feature_preprocessing"
            ]
        )

        if (
            preprocessing
            not in SUPPORTED_PREPROCESSING
        ):
            raise ValueError(
                "unsupported preprocessing: "
                f"{preprocessing}"
            )

        sequence_length = int(
            variant[
                "sequence_length"
            ]
        )

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive"
            )

        if not bool(
            variant[
                "respect_episode_boundaries"
            ]
        ):
            raise ValueError(
                "this diagnostic requires "
                "trajectory-safe windows"
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


def _zscore_per_dimension(
    features: np.ndarray,
):
    """
    Diagnostic D2 preprocessing.

    Operational clarification frozen in code before
    observing D0/D1/D2 real-data results:

    - statistics are computed over the complete clean
      decision-unit matrix;
    - one mean/std is computed per feature dimension;
    - population standard deviation is used (ddof=0);
    - zero-variance dimensions use scale 1.0;
    - standardized data are cast back to the input dtype
      so KMeans numerical dtype does not become an
      additional diagnostic variable.
    """

    features = np.asarray(
        features
    )

    if features.ndim != 2:
        raise ValueError(
            "features must be 2D"
        )

    if not np.isfinite(
        features
    ).all():
        raise ValueError(
            "features contain NaN or Inf"
        )

    mean = np.mean(
        features,
        axis=0,
        dtype=np.float64,
    )

    std = np.std(
        features,
        axis=0,
        dtype=np.float64,
        ddof=0,
    )

    zero_variance = (
        std == 0.0
    )

    safe_std = std.copy()

    safe_std[
        zero_variance
    ] = 1.0

    standardized_float64 = (
        features.astype(
            np.float64,
            copy=False,
        )
        - mean[
            None,
            :
        ]
    )

    standardized_float64 /= (
        safe_std[
            None,
            :
        ]
    )

    standardized = np.asarray(
        standardized_float64,
        dtype=features.dtype,
    )

    details = {
        "method": (
            "zscore_per_dimension"
        ),
        "statistics_source": (
            "complete_clean_decision_unit_matrix"
        ),
        "ddof": 0,
        "zero_variance_scale": 1.0,
        "zero_variance_dimension_count": int(
            np.sum(
                zero_variance
            )
        ),
        "input_dtype": str(
            features.dtype
        ),
        "output_dtype": str(
            standardized.dtype
        ),
        "mean": [
            float(value)
            for value in mean
        ],
        "std": [
            float(value)
            for value in std
        ],
    }

    return (
        standardized,
        details,
    )


def _prepare_feature_views(
    raw_features: np.ndarray,
    variants,
):
    required = {
        str(
            variant[
                "feature_preprocessing"
            ]
        )
        for variant
        in variants
    }

    views = {}
    details = {}

    if "none" in required:
        views[
            "none"
        ] = raw_features

        details[
            "none"
        ] = {
            "method": "none",
            "input_dtype": str(
                raw_features.dtype
            ),
            "output_dtype": str(
                raw_features.dtype
            ),
        }

    if (
        "zscore_per_dimension"
        in required
    ):
        (
            standardized,
            standardization_details,
        ) = _zscore_per_dimension(
            raw_features
        )

        views[
            "zscore_per_dimension"
        ] = standardized

        details[
            "zscore_per_dimension"
        ] = (
            standardization_details
        )

    return (
        views,
        details,
    )


def _adjacent_same_cluster_fraction(
    labels: np.ndarray,
    trajectories: list[
        TrajectorySlice
    ],
):
    """
    Fraction of adjacent decision-unit pairs with
    identical cluster IDs, considering completed
    trajectories only and never comparing across
    trajectory boundaries.
    """

    labels = np.asarray(
        labels
    )

    same = 0
    compared = 0

    for trajectory in trajectories:
        start = int(
            trajectory.start
        )

        end = int(
            trajectory.end
        )

        if (
            end
            - start
            < 2
        ):
            continue

        trajectory_labels = (
            labels[
                start:end
            ]
        )

        same += int(
            np.sum(
                trajectory_labels[
                    1:
                ]
                == trajectory_labels[
                    :-1
                ]
            )
        )

        compared += int(
            len(
                trajectory_labels
            )
            - 1
        )

    fraction = (
        float(same)
        / float(compared)
        if compared
        else 0.0
    )

    return (
        fraction,
        compared,
    )


def _cluster_occupancy(
    labels: np.ndarray,
    *,
    num_clusters: int,
):
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    counts = np.bincount(
        labels,
        minlength=num_clusters,
    )

    if len(counts) != num_clusters:
        raise RuntimeError(
            "cluster label outside expected range"
        )

    total = int(
        np.sum(
            counts
        )
    )

    fractions = (
        counts.astype(
            np.float64
        )
        / float(total)
        if total
        else np.zeros(
            num_clusters,
            dtype=np.float64,
        )
    )

    return {
        "counts": {
            str(cluster_id): int(
                counts[
                    cluster_id
                ]
            )
            for cluster_id
            in range(
                num_clusters
            )
        },
        "fractions": {
            str(cluster_id): float(
                fractions[
                    cluster_id
                ]
            )
            for cluster_id
            in range(
                num_clusters
            )
        },
    }


def _summarize_patterns(
    labels: np.ndarray,
    trajectories: list[
        TrajectorySlice
    ],
    *,
    sequence_length: int,
):
    raw_patterns = set()
    deduplicated_patterns = set()

    total_windows = 0
    affected_windows = 0
    total_deduplicated_length = 0

    expected_windows = sum(
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

    for window in (
        iter_sequence_windows(
            labels,
            trajectories,
            sequence_length=(
                sequence_length
            ),
        )
    ):
        total_windows += 1

        raw_pattern = tuple(
            window.raw_cluster_labels
        )

        deduplicated_pattern = tuple(
            window.pattern
        )

        raw_patterns.add(
            raw_pattern
        )

        deduplicated_patterns.add(
            deduplicated_pattern
        )

        total_deduplicated_length += (
            len(
                deduplicated_pattern
            )
        )

        if (
            raw_pattern
            != deduplicated_pattern
        ):
            affected_windows += 1

    if (
        total_windows
        != expected_windows
    ):
        raise RuntimeError(
            "window-count integrity failure: "
            f"expected={expected_windows}, "
            f"observed={total_windows}"
        )

    raw_count = len(
        raw_patterns
    )

    deduplicated_count = len(
        deduplicated_patterns
    )

    distinct_reduction = (
        1.0
        - (
            float(
                deduplicated_count
            )
            / float(
                raw_count
            )
        )
        if raw_count
        else 0.0
    )

    affected_fraction = (
        float(
            affected_windows
        )
        / float(
            total_windows
        )
        if total_windows
        else 0.0
    )

    average_length = (
        float(
            total_deduplicated_length
        )
        / float(
            total_windows
        )
        if total_windows
        else 0.0
    )

    metrics = {
        "raw_distinct_pattern_count": int(
            raw_count
        ),
        "deduplicated_distinct_pattern_count": int(
            deduplicated_count
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

    integrity = {
        "expected_window_count": int(
            expected_windows
        ),
        "observed_window_count": int(
            total_windows
        ),
        "window_count_matches": bool(
            total_windows
            == expected_windows
        ),
    }

    return (
        metrics,
        integrity,
    )


def _run_seed(
    *,
    seed: int,
    config: dict,
    feature_views: dict,
    preprocessing_details: dict,
    trajectories,
):
    num_clusters = int(
        config[
            "num_clusters"
        ]
    )

    variants = config[
        "diagnostic_variants"
    ]

    clustering_cache = {}
    variant_records = {}

    for variant in variants:
        variant_id = str(
            variant[
                "id"
            ]
        )

        preprocessing = str(
            variant[
                "feature_preprocessing"
            ]
        )

        sequence_length = int(
            variant[
                "sequence_length"
            ]
        )

        if (
            preprocessing
            not in clustering_cache
        ):
            print(
                "Fitting KMeans:",
                f"seed={seed},",
                f"preprocessing={preprocessing},",
                f"k={num_clusters}",
            )

            (
                _,
                clustering,
            ) = (
                fit_kmeans_decision_units(
                    feature_views[
                        preprocessing
                    ],
                    num_clusters=(
                        num_clusters
                    ),
                    seed=seed,
                )
            )

            clustering_cache[
                preprocessing
            ] = clustering

        clustering = (
            clustering_cache[
                preprocessing
            ]
        )

        labels = np.asarray(
            clustering.labels,
            dtype=np.int64,
        )

        (
            pattern_metrics,
            integrity,
        ) = _summarize_patterns(
            labels,
            trajectories,
            sequence_length=(
                sequence_length
            ),
        )

        (
            adjacent_fraction,
            adjacent_pair_count,
        ) = (
            _adjacent_same_cluster_fraction(
                labels,
                trajectories,
            )
        )

        metrics = {
            **pattern_metrics,
            "adjacent_same_cluster_fraction": float(
                adjacent_fraction
            ),
            "cluster_occupancy": (
                _cluster_occupancy(
                    labels,
                    num_clusters=(
                        num_clusters
                    ),
                )
            ),
        }

        variant_records[
            variant_id
        ] = {
            "definition": {
                "feature_preprocessing": (
                    preprocessing
                ),
                "sequence_length": int(
                    sequence_length
                ),
                "respect_episode_boundaries": True,
            },
            "preprocessing": (
                preprocessing_details[
                    preprocessing
                ]
            ),
            "clustering": {
                "method": "kmeans",
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
                "shared_cache_key": (
                    preprocessing
                ),
            },
            "metrics": metrics,
            "integrity": {
                **integrity,
                "adjacent_pair_count": int(
                    adjacent_pair_count
                ),
            },
        }

    return variant_records


def _build_summary(
    *,
    config: dict,
    seed_records: dict,
):
    aggregate = {}

    scalar_metrics = [
        metric
        for metric
        in config[
            "metrics"
        ]
        if (
            metric
            != "cluster_occupancy"
        )
    ]

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

        for metric in scalar_metrics:
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
            "csdpc-source-fidelity-summary-v1"
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
        "aggregate_scalar_metrics": (
            aggregate
        ),
        "cluster_occupancy_note": (
            "Cluster IDs are arbitrary across independent "
            "KMeans seeds, so cluster occupancy is preserved "
            "per seed and is not averaged by cluster ID."
        ),
        "interpretation_rule": (
            "No diagnostic variant replaces the canonical "
            "Gate-B result."
        ),
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen clean-data-only CSDPC "
            "source-fidelity diagnostics."
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

    num_transitions = int(
        observations.shape[
            0
        ]
    )

    if (
        actions.shape[
            0
        ]
        != num_transitions
    ):
        raise RuntimeError(
            "observation/action counts differ"
        )

    if (
        len(
            terminals
        )
        != num_transitions
        or len(
            timeouts
        )
        != num_transitions
    ):
        raise RuntimeError(
            "trajectory-boundary arrays differ "
            "from transition count"
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

    (
        feature_views,
        preprocessing_details,
    ) = _prepare_feature_views(
        raw_features,
        config[
            "diagnostic_variants"
        ],
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
            num_transitions
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
        "completed_trajectory_count": int(
            len(
                trajectories
            )
        ),
        "trailing_transition_count": int(
            trailing
        ),
    }

    seed_records = {}

    for seed in config[
        "attack_seeds"
    ]:
        seed = int(seed)

        print()
        print(
            "=" * 72
        )
        print(
            "SOURCE FIDELITY DIAGNOSTIC "
            f"SEED {seed}"
        )
        print(
            "=" * 72
        )

        variants = _run_seed(
            seed=seed,
            config=config,
            feature_views=(
                feature_views
            ),
            preprocessing_details=(
                preprocessing_details
            ),
            trajectories=trajectories,
        )

        seed_record = {
            "schema_version": (
                "csdpc-source-fidelity-seed-v1"
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
            "variants": (
                variants
            ),
        }

        output_path = (
            output_root
            / (
                f"seed_{seed}.json"
            )
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
            / (
                f"seed_{seed}.json"
            )
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
        "SOURCE FIDELITY DIAGNOSTIC COMPLETE"
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