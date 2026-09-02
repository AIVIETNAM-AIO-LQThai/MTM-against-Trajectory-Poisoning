from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import yaml

from src.attacks.csdpc.clustering import (
    build_raw_decision_units,
    fit_kmeans_decision_units,
)
from src.attacks.csdpc.patterns import iter_sequence_windows
from src.data.trajectories import find_completed_trajectories


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
    / "attacks"
    / "csdpc_walker2d_medium.yaml"
)

DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "metadata"
    / "csdpc"
    / "walker2d_medium_pattern_audit_seed0.json"
)

EXPECTED_DATASET_SHA256 = (
    "cf00f43add04c17fdfc2958dd581dea0851b2e5bedbe6fda073758a8f841aeda"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def pattern_record(pattern, count):
    return {
        "pattern": list(pattern),
        "count": int(count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit CSDPC clustering and sequence-pattern extraction "
            "on the frozen Walker2d-medium-v2 dataset."
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    config_path = args.config.resolve()
    output_path = args.output.resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"dataset not found: {dataset_path}"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"config not found: {config_path}"
        )

    print("Checking frozen dataset SHA256...")

    dataset_sha256 = sha256_file(dataset_path)

    if dataset_sha256 != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Frozen dataset SHA256 mismatch.\n"
            f"Expected: {EXPECTED_DATASET_SHA256}\n"
            f"Actual:   {dataset_sha256}"
        )

    print("Dataset hash: PASS")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    attack_config = config["attack"]

    num_clusters = int(
        attack_config["clustering"]["num_clusters"]
    )

    sequence_length = int(
        attack_config["pattern"]["sequence_length"]
    )

    print("Loading Walker2d dataset...")

    with h5py.File(dataset_path, "r") as dataset:
        required_keys = {
            "observations",
            "actions",
            "terminals",
            "timeouts",
        }

        missing = required_keys.difference(dataset.keys())

        if missing:
            raise KeyError(
                "Dataset is missing required keys: "
                + ", ".join(sorted(missing))
            )

        observations = np.asarray(
            dataset["observations"]
        )

        actions = np.asarray(
            dataset["actions"]
        )

        terminals = np.asarray(
            dataset["terminals"]
        )

        timeouts = np.asarray(
            dataset["timeouts"]
        )

    num_transitions = observations.shape[0]
    state_dim = observations.shape[1]
    action_dim = actions.shape[1]

    if actions.shape[0] != num_transitions:
        raise RuntimeError(
            "Observation/action transition counts differ"
        )

    if len(terminals) != num_transitions:
        raise RuntimeError(
            "terminal count differs from transition count"
        )

    if len(timeouts) != num_transitions:
        raise RuntimeError(
            "timeout count differs from transition count"
        )

    print("Building raw (state, action) decision units...")

    features = build_raw_decision_units(
        observations,
        actions,
    )

    print(
        f"Decision-unit matrix: {features.shape}"
    )

    print("Finding completed trajectories...")

    trajectories, trailing_transitions = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    expected_windows = sum(
        max(
            0,
            trajectory.length
            - sequence_length
            + 1,
        )
        for trajectory in trajectories
    )

    print(
        f"Completed trajectories: {len(trajectories)}"
    )
    print(
        f"Trailing transitions: {trailing_transitions}"
    )

    print(
        f"Fitting KMeans: k={num_clusters}, seed={args.seed}"
    )

    _, clustering = fit_kmeans_decision_units(
        features,
        num_clusters=num_clusters,
        seed=args.seed,
    )

    print(
        f"KMeans inertia: {clustering.inertia:.6f}"
    )
    print(
        f"KMeans iterations: {clustering.n_iter}"
    )

    raw_sequence_counts = Counter()
    pattern_counts = Counter()
    pattern_length_counts = Counter()
    cluster_occupancy_counts = Counter(
        int(label)
        for label in clustering.labels
    )

    windows_with_deduplication = 0
    total_deduplicated_length = 0

    total_windows = 0

    print(
        "Extracting trajectory-safe sequence windows..."
    )

    windows = iter_sequence_windows(
        clustering.labels,
        trajectories,
        sequence_length=sequence_length,
    )

    for window in windows:
        total_windows += 1

        raw_sequence_counts[
            window.raw_cluster_labels
        ] += 1

        pattern_counts[
            window.pattern
        ] += 1

        pattern_length_counts[
            len(window.pattern)
        ] += 1

        total_deduplicated_length += len(
            window.pattern
        )

        if (
            len(window.pattern)
            < len(window.raw_cluster_labels)
        ):
            windows_with_deduplication += 1

    if total_windows != expected_windows:
        raise RuntimeError(
            "Window-count integrity check failed: "
            f"expected {expected_windows}, "
            f"observed {total_windows}"
        )

    unique_raw_sequences = len(
        raw_sequence_counts
    )

    unique_patterns = len(
        pattern_counts
    )

    if unique_raw_sequences > 0:
        distinct_pattern_reduction = (
            1.0
            - unique_patterns
            / unique_raw_sequences
        )
        if total_windows > 0:
            deduplicated_window_fraction = (
                windows_with_deduplication
                / total_windows
            )

            average_pattern_length = (
                total_deduplicated_length
                / total_windows
            )
        else:
            deduplicated_window_fraction = 0.0
            average_pattern_length = 0.0
    else:
        distinct_pattern_reduction = 0.0

    rarest = sorted(
        pattern_counts.items(),
        key=lambda item: (
            item[1],
            item[0],
        ),
    )[:10]

    most_common = sorted(
        pattern_counts.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )[:10]

    print()
    print("========== CSDPC PATTERN AUDIT ==========")
    print(f"Transitions: {num_transitions}")
    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(
        f"Decision-unit shape: {features.shape}"
    )
    print(
        f"Completed trajectories: {len(trajectories)}"
    )
    print(
        f"Trailing transitions: {trailing_transitions}"
    )
    print(f"k: {num_clusters}")
    print(
        f"KMeans inertia: {clustering.inertia:.6f}"
    )
    print(
        f"KMeans iterations: {clustering.n_iter}"
    )
    print(
        f"Sequence length: {sequence_length}"
    )
    print(f"Total windows: {total_windows}")
    print(
        "Unique raw cluster sequences: "
        f"{unique_raw_sequences}"
    )
    print(
        "Unique deduplicated patterns: "
        f"{unique_patterns}"
    )
    print(
        "Distinct-pattern reduction: "
        f"{100.0 * distinct_pattern_reduction:.2f}%"
    )
    print(
        "Windows with deduplication: "
        f"{windows_with_deduplication}"
    )

    print(
        "Fraction with deduplication: "
        f"{100.0 * deduplicated_window_fraction:.2f}%"
    )

    print(
        "Average deduplicated pattern length: "
        f"{average_pattern_length:.4f}"
    )

    print("Cluster occupancy:")

    for cluster_id in sorted(
        cluster_occupancy_counts
    ):
        print(
            f"  cluster {cluster_id}: "
            f"{cluster_occupancy_counts[cluster_id]}"
        )

    print("Pattern length distribution:")

    for length in sorted(
        pattern_length_counts
    ):
        print(
            f"  length {length}: "
            f"{pattern_length_counts[length]}"
        )

    print("10 rarest patterns:")

    for pattern, count in rarest:
        print(
            f"  {pattern}: {count}"
        )

    print("10 most common patterns:")

    for pattern, count in most_common:
        print(
            f"  {pattern}: {count}"
        )

    audit = {
        "dataset": {
            "name": config["dataset"]["name"],
            "path": str(dataset_path),
            "sha256": dataset_sha256,
            "transitions": int(
                num_transitions
            ),
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "decision_unit_dim": int(
                features.shape[1]
            ),
        },
        "trajectory_audit": {
            "completed_trajectories": int(
                len(trajectories)
            ),
            "trailing_transitions": int(
                trailing_transitions
            ),
        },
        "clustering": {
            "method": "kmeans",
            "num_clusters": int(
                num_clusters
            ),
            "attack_seed": int(
                args.seed
            ),
            "inertia": float(
                clustering.inertia
            ),
            "n_iter": int(
                clustering.n_iter
            ),
        },
        "pattern_extraction": {
            "sequence_length": int(
                sequence_length
            ),
            "deduplicate_after_windowing": True,
            "respect_episode_boundaries": True,
            "frequency_definition": (
                "occurrence_count"
            ),
            "expected_windows": int(
                expected_windows
            ),
            "total_windows": int(
                total_windows
            ),
            "unique_raw_cluster_sequences": int(
                unique_raw_sequences
            ),
            "unique_deduplicated_patterns": int(
                unique_patterns
            ),
            "distinct_pattern_reduction_fraction": float(
                distinct_pattern_reduction
            ),
            "pattern_length_distribution": {
                str(length): int(count)
                for length, count
                in sorted(
                    pattern_length_counts.items()
                )
            },
            "windows_with_deduplication": int(
                windows_with_deduplication
            ),
            "deduplicated_window_fraction": float(
                deduplicated_window_fraction
            ),
            "average_deduplicated_pattern_length": float(
                average_pattern_length
            ),
        },
        "cluster_occupancy": {
            str(cluster_id): int(count)
            for cluster_id, count
            in sorted(
                cluster_occupancy_counts.items()
            )
        },
        "rarest_patterns": [
            pattern_record(
                pattern,
                count,
            )
            for pattern, count in rarest
        ],
        "most_common_patterns": [
            pattern_record(
                pattern,
                count,
            )
            for pattern, count
            in most_common
        ],
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            indent=2,
            sort_keys=True,
        )

    print()
    print(
        f"Saved audit: {output_path}"
    )


if __name__ == "__main__":
    main()