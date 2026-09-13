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
    iter_sequence_windows,
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
    / "gates"
    / "csdpc_dedup_metric_reconciliation.json"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "csdpc_dedup_metric_reconciliation"
)

EXPECTED_DATASET_SHA256 = (
    "cf00f43add04c17fdfc2958dd581dea"
    "0851b2e5bedbe6fda073758a8f841aeda"
)

EXPECTED_METRICS = {
    "window_count",
    "raw_distinct_sequence_type_count",
    "deduplicated_distinct_pattern_type_count",
    "canonical_distinct_type_reduction_fraction",
    "window_instance_changed_fraction",
    "raw_distinct_sequence_type_changed_fraction",
    "total_label_token_reduction_fraction",
    "average_deduplicated_pattern_length",
    "full_length_pattern_occurrence_fraction",
    "full_length_distinct_pattern_fraction",
    "non_equivalent_total_window_denominator_reduction_fraction",
}


def _load_config(path: Path) -> dict:
    config = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if (
        config.get("experiment")
        != "CSDPC_DEDUP_METRIC_RECONCILIATION"
    ):
        raise ValueError(
            "unexpected experiment identifier"
        )

    if config.get("status") != "DIAGNOSTIC_ONLY":
        raise ValueError(
            "diagnostic status changed"
        )

    if not config.get(
        "canonical_attack_is_unchanged",
        False,
    ):
        raise ValueError(
            "canonical attack must remain unchanged"
        )

    if int(config["num_clusters"]) != 8:
        raise ValueError("k must remain 8")

    if int(config["sequence_length"]) != 5:
        raise ValueError(
            "sequence length must remain 5"
        )

    pipeline = config["pipeline"]

    expected_pipeline = {
        "feature_preprocessing": "none",
        "window_stride": 1,
        "respect_episode_boundaries": True,
        "deduplication_order": (
            "window_then_deduplicate"
        ),
    }

    if pipeline != expected_pipeline:
        raise ValueError(
            "canonical pipeline definition changed"
        )

    if set(config["metrics"]) != EXPECTED_METRICS:
        raise ValueError(
            "frozen metric set changed"
        )

    return config


def _load_dataset(path: Path):
    with h5py.File(path, "r") as handle:
        return (
            np.asarray(handle["observations"]),
            np.asarray(handle["actions"]),
            np.asarray(handle["terminals"]),
            np.asarray(handle["timeouts"]),
        )


def _compute_metrics(
    labels,
    trajectories,
    *,
    sequence_length: int,
):
    raw_types = set()
    dedup_types = set()

    raw_type_changed = {}

    window_count = 0
    changed_windows = 0

    total_raw_tokens = 0
    total_dedup_tokens = 0

    full_length_occurrences = 0

    for window in iter_sequence_windows(
        labels,
        trajectories,
        sequence_length=sequence_length,
    ):
        raw = tuple(
            window.raw_cluster_labels
        )

        dedup = tuple(
            window.pattern
        )

        raw_types.add(raw)
        dedup_types.add(dedup)

        changed = raw != dedup

        previous = raw_type_changed.get(
            raw
        )

        if previous is not None:
            if previous != changed:
                raise RuntimeError(
                    "identical raw sequence type "
                    "mapped inconsistently"
                )
        else:
            raw_type_changed[
                raw
            ] = changed

        window_count += 1

        if changed:
            changed_windows += 1

        total_raw_tokens += len(raw)
        total_dedup_tokens += len(dedup)

        if len(dedup) == sequence_length:
            full_length_occurrences += 1

    raw_distinct = len(raw_types)
    dedup_distinct = len(dedup_types)

    changed_raw_types = sum(
        bool(value)
        for value
        in raw_type_changed.values()
    )

    full_length_distinct = sum(
        1
        for pattern
        in dedup_types
        if len(pattern) == sequence_length
    )

    canonical_reduction = (
        1.0
        - dedup_distinct
        / raw_distinct
        if raw_distinct
        else 0.0
    )

    window_changed_fraction = (
        changed_windows
        / window_count
        if window_count
        else 0.0
    )

    raw_type_changed_fraction = (
        changed_raw_types
        / raw_distinct
        if raw_distinct
        else 0.0
    )

    token_reduction = (
        1.0
        - total_dedup_tokens
        / total_raw_tokens
        if total_raw_tokens
        else 0.0
    )

    average_length = (
        total_dedup_tokens
        / window_count
        if window_count
        else 0.0
    )

    full_length_occurrence_fraction = (
        full_length_occurrences
        / window_count
        if window_count
        else 0.0
    )

    full_length_distinct_fraction = (
        full_length_distinct
        / dedup_distinct
        if dedup_distinct
        else 0.0
    )

    non_equivalent_window_denominator = (
        1.0
        - dedup_distinct
        / window_count
        if window_count
        else 0.0
    )

    return {
        "window_count": int(
            window_count
        ),
        "raw_distinct_sequence_type_count": int(
            raw_distinct
        ),
        "deduplicated_distinct_pattern_type_count": int(
            dedup_distinct
        ),
        "canonical_distinct_type_reduction_fraction": float(
            canonical_reduction
        ),
        "window_instance_changed_fraction": float(
            window_changed_fraction
        ),
        "raw_distinct_sequence_type_changed_fraction": float(
            raw_type_changed_fraction
        ),
        "total_label_token_reduction_fraction": float(
            token_reduction
        ),
        "average_deduplicated_pattern_length": float(
            average_length
        ),
        "full_length_pattern_occurrence_fraction": float(
            full_length_occurrence_fraction
        ),
        "full_length_distinct_pattern_fraction": float(
            full_length_distinct_fraction
        ),
        "non_equivalent_total_window_denominator_reduction_fraction": float(
            non_equivalent_window_denominator
        ),
        "counts": {
            "changed_window_instances": int(
                changed_windows
            ),
            "changed_raw_sequence_types": int(
                changed_raw_types
            ),
            "total_raw_label_tokens": int(
                total_raw_tokens
            ),
            "total_deduplicated_label_tokens": int(
                total_dedup_tokens
            ),
            "full_length_pattern_occurrences": int(
                full_length_occurrences
            ),
            "full_length_distinct_patterns": int(
                full_length_distinct
            ),
        },
    }


def _aggregate(
    config,
    seed_records,
):
    aggregate = {}

    for metric in config["metrics"]:
        values = np.asarray(
            [
                seed_records[
                    str(seed)
                ]["metrics"][metric]
                for seed
                in config["attack_seeds"]
            ],
            dtype=np.float64,
        )

        aggregate[metric] = {
            "values_by_seed": {
                str(seed): float(value)
                for seed, value
                in zip(
                    config["attack_seeds"],
                    values,
                )
            },
            "mean": float(
                values.mean()
            ),
            "std": float(
                values.std(ddof=0)
            ),
            "min": float(
                values.min()
            ),
            "max": float(
                values.max()
            ),
        }

    return aggregate


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

    actual_sha = sha256_file(
        dataset_path
    )

    if (
        actual_sha
        != EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "frozen clean dataset SHA256 mismatch"
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

    for seed in config[
        "attack_seeds"
    ]:
        seed = int(seed)

        print()
        print(
            "=" * 72
        )
        print(
            "DEDUP METRIC RECONCILIATION "
            f"— SEED {seed}"
        )
        print(
            "=" * 72
        )

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

        metrics = _compute_metrics(
            np.asarray(
                clustering.labels,
                dtype=np.int64,
            ),
            trajectories,
            sequence_length=int(
                config[
                    "sequence_length"
                ]
            ),
        )

        record = {
            "schema_version": (
                "csdpc-dedup-metric-reconciliation-seed-v1"
            ),
            "experiment": config[
                "experiment"
            ],
            "status": config[
                "status"
            ],
            "canonical_attack_is_unchanged": True,
            "seed": seed,
            "dataset_sha256": actual_sha,
            "completed_trajectory_count": int(
                len(
                    trajectories
                )
            ),
            "trailing_transition_count": int(
                trailing
            ),
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
            "metrics": metrics,
        }

        output_path = (
            output_root
            / f"seed_{seed}.json"
        )

        write_metadata_json(
            output_path,
            record,
        )

        seed_records[
            str(seed)
        ] = record

        gc.collect()

    summary = {
        "schema_version": (
            "csdpc-dedup-metric-reconciliation-summary-v1"
        ),
        "experiment": config[
            "experiment"
        ],
        "status": config[
            "status"
        ],
        "canonical_attack_is_unchanged": True,
        "dataset_sha256": actual_sha,
        "attack_seeds": [
            int(seed)
            for seed
            in config[
                "attack_seeds"
            ]
        ],
        "metric_semantics": {
            "canonical_distinct_type_reduction_fraction": (
                "1 - number of unique deduplicated "
                "patterns / number of unique raw "
                "cluster-label sequences"
            ),
            "window_instance_changed_fraction": (
                "fraction of window occurrences whose "
                "label tuple changes after deduplication"
            ),
            "raw_distinct_sequence_type_changed_fraction": (
                "fraction of unique raw cluster-label "
                "sequence types altered by deduplication"
            ),
            "total_label_token_reduction_fraction": (
                "fraction of cluster-label positions "
                "removed across all window occurrences"
            ),
            "non_equivalent_total_window_denominator_reduction_fraction": (
                "1 - unique deduplicated pattern types / "
                "total window occurrences; reported only "
                "to expose denominator ambiguity and is "
                "not treated as the source statistic"
            ),
        },
        "aggregate_metrics": _aggregate(
            config,
            seed_records,
        ),
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
        "Summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()