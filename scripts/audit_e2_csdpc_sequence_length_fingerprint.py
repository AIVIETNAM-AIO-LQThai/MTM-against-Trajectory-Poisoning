from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.audit_csdpc_dedup_metric_reconciliation import (
    EXPECTED_METRICS,
    _compute_metrics,
)
from src.attacks.csdpc.clustering import (
    build_raw_decision_units,
    fit_kmeans_decision_units,
)
from src.attacks.csdpc.metadata import (
    sha256_file,
    write_metadata_json,
)
from src.data.trajectories import (
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
    / "postfinal_controls"
    / "e2_csdpc_sequence_length_fingerprint.json"
)

DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "postfinal_controls"
    / "e2_csdpc_sequence_length_fingerprint.json"
)


def _parse_args():
    parser = argparse.ArgumentParser()

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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def _load_config(path):
    config = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    if (
        config.get("experiment")
        != "E2_CSDPC_SEQUENCE_LENGTH_FINGERPRINT"
    ):
        raise ValueError(
            "unexpected E2 experiment identifier"
        )

    if (
        config.get("status")
        != "POSTFINAL_DIAGNOSTIC_ONLY"
    ):
        raise ValueError(
            "E2 diagnostic status changed"
        )

    if not config.get(
        "canonical_attack_is_unchanged",
        False,
    ):
        raise ValueError(
            "canonical attack boundary changed"
        )

    if int(config["num_clusters"]) != 8:
        raise ValueError(
            "E2 must keep k = 8"
        )

    seeds = [
        int(x)
        for x in config["attack_seeds"]
    ]

    if seeds != [0, 1, 2]:
        raise ValueError(
            "E2 attack seeds must remain [0, 1, 2]"
        )

    lengths = [
        int(x)
        for x in config[
            "actual_sequence_lengths"
        ]
    ]

    if lengths != list(range(2, 11)):
        raise ValueError(
            "E2 must report every integer L from 2 through 10"
        )

    pipeline = config["pipeline"]

    expected_pipeline = {
        "decision_unit": "concat(state, action)",
        "feature_preprocessing": "none",
        "kmeans_init": "k-means++",
        "kmeans_n_init": 10,
        "kmeans_max_iter": 300,
        "kmeans_tol": 0.0001,
        "kmeans_algorithm": "lloyd",
        "window_stride": 1,
        "respect_episode_boundaries": True,
        "deduplication_order": "window_then_deduplicate",
        "deduplication": (
            "remove_consecutive_duplicate_cluster_labels"
        ),
    }

    if pipeline != expected_pipeline:
        raise ValueError(
            "E2 canonical pipeline definition changed"
        )

    reference = config["source_reference"]

    if not np.isclose(
        float(reference["reference_fraction"]),
        0.8,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "source reference changed"
        )

    if bool(
        reference["is_acceptance_gate"]
    ):
        raise ValueError(
            "source reference must not become a gate"
        )

    return config


def _load_dataset(path):
    with h5py.File(path, "r") as handle:
        return (
            np.asarray(
                handle["observations"]
            ),
            np.asarray(
                handle["actions"]
            ),
            np.asarray(
                handle["terminals"]
            ),
            np.asarray(
                handle["timeouts"]
            ),
        )


def _check_regression_anchors(
    config,
    seed_records,
):
    checks = []

    for anchor in config[
        "historical_regression_anchors"
    ]:
        seed = int(
            anchor["attack_seed"]
        )

        length = int(
            anchor[
                "actual_sequence_length"
            ]
        )

        expected = float(
            anchor[
                "canonical_distinct_type_reduction_fraction"
            ]
        )

        tolerance = float(
            anchor[
                "absolute_tolerance"
            ]
        )

        actual = float(
            seed_records[str(seed)][
                str(length)
            ]["metrics"][
                "canonical_distinct_type_reduction_fraction"
            ]
        )

        error = abs(
            actual - expected
        )

        passed = bool(
            error <= tolerance
        )

        checks.append(
            {
                "attack_seed": seed,
                "actual_sequence_length": length,
                "expected": expected,
                "actual": actual,
                "absolute_error": float(
                    error
                ),
                "absolute_tolerance": tolerance,
                "passed": passed,
                "origin": anchor["origin"],
            }
        )

    all_passed = all(
        row["passed"]
        for row in checks
    )

    return {
        "all_passed": bool(
            all_passed
        ),
        "checks": checks,
    }


def _aggregate_by_length(
    config,
    seed_records,
):
    result = {}

    for length in config[
        "actual_sequence_lengths"
    ]:
        length = int(length)

        aggregate_metrics = {}

        for metric in sorted(
            EXPECTED_METRICS
        ):
            values = np.asarray(
                [
                    seed_records[
                        str(seed)
                    ][str(length)][
                        "metrics"
                    ][metric]
                    for seed
                    in config[
                        "attack_seeds"
                    ]
                ],
                dtype=np.float64,
            )

            aggregate_metrics[
                metric
            ] = {
                "values_by_seed": {
                    str(seed): float(value)
                    for seed, value
                    in zip(
                        config[
                            "attack_seeds"
                        ],
                        values,
                    )
                },
                "mean": float(
                    values.mean()
                ),
                "std": float(
                    values.std(
                        ddof=0
                    )
                ),
                "min": float(
                    values.min()
                ),
                "max": float(
                    values.max()
                ),
            }

        result[str(length)] = {
            "actual_sequence_length": (
                length
            ),
            "endpoint_span_l": (
                length - 1
            ),
            "metrics": aggregate_metrics,
        }

    return result


def _reference_proximity(
    config,
    aggregate_by_length,
):
    reference = float(
        config[
            "source_reference"
        ]["reference_fraction"]
    )

    rows = []

    for length in config[
        "actual_sequence_lengths"
    ]:
        length = int(length)

        mean_reduction = float(
            aggregate_by_length[
                str(length)
            ]["metrics"][
                "canonical_distinct_type_reduction_fraction"
            ]["mean"]
        )

        rows.append(
            {
                "actual_sequence_length": (
                    length
                ),
                "endpoint_span_l": (
                    length - 1
                ),
                "mean_reduction_fraction": (
                    mean_reduction
                ),
                "absolute_distance_to_reference": float(
                    abs(
                        mean_reduction
                        - reference
                    )
                ),
            }
        )

    nearest = min(
        rows,
        key=lambda row: (
            row[
                "absolute_distance_to_reference"
            ],
            row[
                "actual_sequence_length"
            ],
        ),
    )

    means = [
        row[
            "mean_reduction_fraction"
        ]
        for row in rows
    ]

    monotonic_nondecreasing = all(
        b >= a
        for a, b
        in zip(
            means,
            means[1:],
        )
    )

    return {
        "reference_fraction": (
            reference
        ),
        "reference_is_approximate": bool(
            config[
                "source_reference"
            ]["approximate"]
        ),
        "reference_is_acceptance_gate": False,
        "rows": rows,
        "nearest_predeclared_length": (
            nearest
        ),
        "mean_curve_monotonic_nondecreasing": bool(
            monotonic_nondecreasing
        ),
        "interpretation_boundary": (
            "nearest_predeclared_length is descriptive only "
            "and must not be promoted to a new attack setting"
        ),
    }


def main():
    args = _parse_args()

    config = _load_config(
        args.config
    )

    dataset_path = (
        args.dataset.resolve()
    )

    output_path = (
        args.output.resolve()
    )

    actual_sha = sha256_file(
        dataset_path
    )

    expected_sha = str(
        config["dataset_sha256"]
    )

    if actual_sha != expected_sha:
        raise RuntimeError(
            "frozen clean dataset SHA256 mismatch: "
            f"expected={expected_sha}, "
            f"actual={actual_sha}"
        )

    print(
        "Frozen clean dataset SHA256: PASS"
    )

    (
        observations,
        actions,
        terminals,
        timeouts,
    ) = _load_dataset(
        dataset_path
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

    seed_records = {}

    print()
    print("=" * 120)
    print(
        "E2 - CSDPC SEQUENCE-LENGTH SOURCE FINGERPRINT"
    )
    print("=" * 120)
    print(
        "seed  L span_l    windows    raw_types  dedup_types  reduction   avg_dedup_len"
    )

    for seed in config[
        "attack_seeds"
    ]:
        seed = int(seed)

        (
            _,
            clustering,
        ) = fit_kmeans_decision_units(
            raw_features,
            num_clusters=int(
                config[
                    "num_clusters"
                ]
            ),
            seed=seed,
        )

        labels = np.asarray(
            clustering.labels,
            dtype=np.int64,
        )

        per_length = {}

        for length in config[
            "actual_sequence_lengths"
        ]:
            length = int(length)

            metrics = _compute_metrics(
                labels,
                trajectories,
                sequence_length=length,
            )

            record = {
                "actual_sequence_length": (
                    length
                ),
                "endpoint_span_l": (
                    length - 1
                ),
                "metrics": metrics,
            }

            per_length[
                str(length)
            ] = record

            print(
                f"{seed:4d} "
                f"{length:2d} "
                f"{length - 1:6d} "
                f"{metrics['window_count']:10d} "
                f"{metrics['raw_distinct_sequence_type_count']:12d} "
                f"{metrics['deduplicated_distinct_pattern_type_count']:12d} "
                f"{metrics['canonical_distinct_type_reduction_fraction']:10.6f} "
                f"{metrics['average_deduplicated_pattern_length']:13.6f}"
            )

            gc.collect()

        seed_records[
            str(seed)
        ] = {
            "seed": seed,
            "clustering": {
                "num_clusters": int(
                    config[
                        "num_clusters"
                    ]
                ),
                "inertia": float(
                    clustering.inertia
                ),
                "n_iter": int(
                    clustering.n_iter
                ),
            },
            **per_length,
        }

        gc.collect()

    regression = (
        _check_regression_anchors(
            config,
            seed_records,
        )
    )

    print()
    print(
        "Historical regression anchors:",
        "PASS"
        if regression[
            "all_passed"
        ]
        else "FAIL",
    )

    for row in regression[
        "checks"
    ]:
        print(
            "  "
            f"seed={row['attack_seed']} "
            f"L={row['actual_sequence_length']} "
            f"expected={row['expected']:.8f} "
            f"actual={row['actual']:.8f} "
            f"error={row['absolute_error']:.3e}"
        )

    if not regression[
        "all_passed"
    ]:
        raise RuntimeError(
            "E2 historical regression anchor failed; "
            "do not interpret the length sweep"
        )

    aggregate = (
        _aggregate_by_length(
            config,
            seed_records,
        )
    )

    proximity = (
        _reference_proximity(
            config,
            aggregate,
        )
    )

    summary = {
        "schema_version": (
            "e2-csdpc-sequence-length-fingerprint-v1"
        ),
        "experiment": (
            config["experiment"]
        ),
        "status": (
            config["status"]
        ),
        "canonical_attack_is_unchanged": True,
        "dataset": {
            "name": (
                config["dataset"]
            ),
            "sha256": actual_sha,
            "num_transitions": int(
                len(observations)
            ),
            "completed_trajectory_count": int(
                len(trajectories)
            ),
            "trailing_transition_count": int(
                trailing
            ),
        },
        "frozen_pipeline": (
            config["pipeline"]
        ),
        "num_clusters": int(
            config["num_clusters"]
        ),
        "attack_seeds": [
            int(x)
            for x in config[
                "attack_seeds"
            ]
        ],
        "actual_sequence_lengths": [
            int(x)
            for x in config[
                "actual_sequence_lengths"
            ]
        ],
        "metric_semantics": {
            "canonical_distinct_type_reduction_fraction": (
                "1 - unique deduplicated pattern types "
                "/ unique raw cluster-label sequence types"
            ),
            "endpoint_span_l": (
                "notation-only view equal to actual_sequence_length - 1"
            ),
        },
        "historical_regression": (
            regression
        ),
        "seed_records": (
            seed_records
        ),
        "aggregate_by_length": (
            aggregate
        ),
        "source_reference_proximity": (
            proximity
        ),
        "claim_boundary": [
            "No poisoned dataset was generated.",
            "No learner was trained or evaluated.",
            "The closest length to the source reference is descriptive only.",
            "Canonical Gate B and Group-2 closure remain unchanged."
        ],
    }

    write_metadata_json(
        output_path,
        summary,
    )

    print()
    print("=" * 120)
    print("E2 AGGREGATE CURVE")
    print("=" * 120)
    print(
        " L span_l   mean_reduction       std   distance_to_0.80"
    )

    for row in proximity[
        "rows"
    ]:
        length = int(
            row[
                "actual_sequence_length"
            ]
        )

        metric = (
            aggregate[
                str(length)
            ]["metrics"][
                "canonical_distinct_type_reduction_fraction"
            ]
        )

        print(
            f"{length:2d} "
            f"{length - 1:6d} "
            f"{metric['mean']:16.8f} "
            f"{metric['std']:9.8f} "
            f"{row['absolute_distance_to_reference']:18.8f}"
        )

    nearest = proximity[
        "nearest_predeclared_length"
    ]

    print()
    print(
        "Nearest predeclared length to descriptive 0.80 reference:"
    )
    print(
        json.dumps(
            nearest,
            indent=2,
        )
    )
    print()
    print(
        "IMPORTANT: this is descriptive only; "
        "do not promote it to a new attack setting."
    )
    print()
    print(
        "output ->",
        output_path,
    )


if __name__ == "__main__":
    main()
