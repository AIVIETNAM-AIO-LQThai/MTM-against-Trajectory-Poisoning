from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.attacks.csdpc.attack import prepare_csdpc_attack
from src.attacks.csdpc.clustering import build_raw_decision_units
from src.attacks.csdpc.patterns import (
    deduplicate_consecutive,
    iter_sequence_windows,
)
from src.attacks.csdpc.selection import (
    compute_transition_budget,
    select_rare_nonoverlapping_windows,
)
from src.data.hdf5_io import load_hdf5_dataset
from src.data.trajectories import find_completed_trajectories


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CLEAN = (
    ROOT
    / "data"
    / "raw"
    / "walker2d-medium-v2"
    / "walker2d_medium-v2.hdf5"
)

DEFAULT_POISON_ROOT = (
    ROOT
    / "data"
    / "poisoned"
    / "csdpc"
    / "walker2d-medium-v2"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "csdpc_mechanism"
    / "canonical_k8"
)


def _rho_code(rho):
    percent = int(round(float(rho) * 100.0))

    reconstructed = percent / 100.0

    if not np.isclose(
        rho,
        reconstructed,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "rho must be a whole percentage for canonical artifacts"
        )

    return f"{percent:03d}"


def _predict_labels(
    dataset,
    kmeans_model,
):
    features = build_raw_decision_units(
        np.asarray(dataset["observations"]),
        np.asarray(dataset["actions"]),
    )

    centers = np.asarray(
        kmeans_model.cluster_centers_
    )

    if features.dtype != centers.dtype:
        features = np.asarray(
            features,
            dtype=centers.dtype,
        )

    return np.asarray(
        kmeans_model.predict(features),
        dtype=np.int64,
    )


def _safe_fraction(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


def _audit_one_rho(
    *,
    clean_dataset,
    poisoned_dataset,
    prepared,
    trajectories,
    clean_counts,
    raw_distinct_pattern_count,
    dedup_affected_window_fraction,
    rho,
):
    poison_labels = _predict_labels(
        poisoned_dataset,
        prepared.clustering_model,
    )

    poison_counts = Counter(
        window.pattern
        for window
        in iter_sequence_windows(
            poison_labels,
            trajectories,
            sequence_length=prepared.sequence_length,
        )
    )

    transition_budget = (
        compute_transition_budget(
            num_transitions=prepared.num_transitions,
            rho=rho,
        )
    )

    selection = (
        select_rare_nonoverlapping_windows(
            prepared.windows,
            clean_counts,
            transition_budget=transition_budget,
        )
    )

    selected_windows = (
        selection.selected_windows
    )

    selected_indices = sorted(
        {
            index
            for window in selected_windows
            for index in range(
                window.global_start,
                window.global_end,
            )
        }
    )

    selected_indices_array = np.asarray(
        selected_indices,
        dtype=np.int64,
    )

    clean_observations = np.asarray(
        clean_dataset["observations"]
    )

    clean_actions = np.asarray(
        clean_dataset["actions"]
    )

    poisoned_observations = np.asarray(
        poisoned_dataset["observations"]
    )

    poisoned_actions = np.asarray(
        poisoned_dataset["actions"]
    )

    observation_changed = np.any(
        clean_observations
        != poisoned_observations,
        axis=1,
    )

    action_changed = np.any(
        clean_actions
        != poisoned_actions,
        axis=1,
    )

    actual_modified_indices = np.flatnonzero(
        observation_changed
        | action_changed
    ).astype(
        np.int64
    )

    selected_indices_match_dataset_diff = bool(
        np.array_equal(
            selected_indices_array,
            actual_modified_indices,
        )
    )

    clean_labels = np.asarray(
        prepared.clustering.labels,
        dtype=np.int64,
    )

    if len(selected_indices_array):
        cluster_label_change_fraction = float(
            np.mean(
                clean_labels[
                    selected_indices_array
                ]
                != poison_labels[
                    selected_indices_array
                ]
            )
        )
    else:
        cluster_label_change_fraction = 0.0

    pattern_changed = 0
    frequency_improved = 0
    changed_and_frequency_improved = 0
    changed_to_new_pattern = 0

    selected_source_patterns = set()

    for window in selected_windows:
        source_pattern = (
            window.source_pattern
        )

        selected_source_patterns.add(
            source_pattern
        )

        poisoned_pattern = (
            deduplicate_consecutive(
                poison_labels[
                    window.global_start:
                    window.global_end
                ]
            )
        )

        source_frequency = int(
            clean_counts.get(
                source_pattern,
                0,
            )
        )

        target_frequency = int(
            clean_counts.get(
                poisoned_pattern,
                0,
            )
        )

        changed = (
            poisoned_pattern
            != source_pattern
        )

        improved = (
            target_frequency
            > source_frequency
        )

        if changed:
            pattern_changed += 1

        if improved:
            frequency_improved += 1

        if changed and improved:
            changed_and_frequency_improved += 1

        if (
            changed
            and poisoned_pattern
            not in clean_counts
        ):
            changed_to_new_pattern += 1

    selected_window_count = len(
        selected_windows
    )

    selected_source_pattern_count = len(
        selected_source_patterns
    )

    eradicated_source_patterns = sum(
        1
        for pattern
        in selected_source_patterns
        if poison_counts.get(
            pattern,
            0,
        )
        == 0
    )

    selected_source_mass_before = sum(
        int(
            clean_counts.get(
                pattern,
                0,
            )
        )
        for pattern
        in selected_source_patterns
    )

    selected_source_mass_after = sum(
        int(
            poison_counts.get(
                pattern,
                0,
            )
        )
        for pattern
        in selected_source_patterns
    )

    if selected_source_mass_before:
        selected_source_mass_reduction = (
            1.0
            - (
                float(
                    selected_source_mass_after
                )
                / float(
                    selected_source_mass_before
                )
            )
        )
    else:
        selected_source_mass_reduction = 0.0

    clean_pattern_types = set(
        clean_counts
    )

    poison_pattern_types = set(
        poison_counts
    )

    removed_pattern_types = (
        clean_pattern_types
        - poison_pattern_types
    )

    new_pattern_types = (
        poison_pattern_types
        - clean_pattern_types
    )

    clean_distinct_pattern_count = len(
        clean_pattern_types
    )

    poison_distinct_pattern_count = len(
        poison_pattern_types
    )

    if clean_distinct_pattern_count:
        clean_to_poison_distinct_reduction = (
            1.0
            - (
                float(
                    poison_distinct_pattern_count
                )
                / float(
                    clean_distinct_pattern_count
                )
            )
        )
    else:
        clean_to_poison_distinct_reduction = 0.0

    if raw_distinct_pattern_count:
        clean_dedup_reduction = (
            1.0
            - (
                float(
                    clean_distinct_pattern_count
                )
                / float(
                    raw_distinct_pattern_count
                )
            )
        )
    else:
        clean_dedup_reduction = 0.0

    if len(selected_indices_array):
        modified_actions = (
            poisoned_actions[
                selected_indices_array
            ]
        )

        rows_at_bound = np.any(
            np.isclose(
                np.abs(
                    modified_actions
                ),
                1.0,
                rtol=0.0,
                atol=1.0e-7,
            ),
            axis=1,
        )

        action_rows_at_bound_fraction = float(
            np.mean(
                rows_at_bound
            )
        )
    else:
        action_rows_at_bound_fraction = 0.0

    return {
        "schema_version": (
            "csdpc-mechanism-audit-v1"
        ),
        "attack_seed": int(
            prepared.attack_seed
        ),
        "rho": float(
            rho
        ),
        "num_clusters": int(
            prepared.clustering.centers.shape[0]
        ),
        "sequence_length": int(
            prepared.sequence_length
        ),
        "eta": float(
            prepared.eta
        ),
        "num_candidates": int(
            prepared.num_candidates
        ),

        "budget": {
            "requested_transition_budget": int(
                transition_budget
            ),
            "selected_transition_count": int(
                len(
                    selected_indices_array
                )
            ),
            "dataset_diff_transition_count": int(
                len(
                    actual_modified_indices
                )
            ),
            "selected_indices_match_dataset_diff": (
                selected_indices_match_dataset_diff
            ),
            "selection_skipped_overlap_windows": int(
                selection.skipped_overlap_windows
            ),
        },

        "clean_pattern_baseline": {
            "window_count": int(
                len(
                    prepared.windows
                )
            ),
            "raw_distinct_pattern_count": int(
                raw_distinct_pattern_count
            ),
            "deduplicated_distinct_pattern_count": int(
                clean_distinct_pattern_count
            ),
            "clean_dedup_distinct_pattern_reduction": float(
                clean_dedup_reduction
            ),
            "dedup_affected_window_fraction": float(
                dedup_affected_window_fraction
            ),
        },

        "local_attack_effect": {
            "selected_window_count": int(
                selected_window_count
            ),
            "selected_window_pattern_change_fraction": (
                _safe_fraction(
                    pattern_changed,
                    selected_window_count,
                )
            ),
            "selected_window_pattern_unchanged_fraction": (
                _safe_fraction(
                    selected_window_count
                    - pattern_changed,
                    selected_window_count,
                )
            ),
            "selected_window_frequency_improvement_fraction": (
                _safe_fraction(
                    frequency_improved,
                    selected_window_count,
                )
            ),
            "selected_window_frequency_improvement_given_changed": (
                _safe_fraction(
                    changed_and_frequency_improved,
                    pattern_changed,
                )
            ),
            "changed_to_new_pattern_fraction_given_changed": (
                _safe_fraction(
                    changed_to_new_pattern,
                    pattern_changed,
                )
            ),
            "modified_transition_cluster_label_change_fraction": (
                cluster_label_change_fraction
            ),
        },

        "targeted_pattern_effect": {
            "selected_source_pattern_type_count": int(
                selected_source_pattern_count
            ),
            "eradicated_selected_source_pattern_type_count": int(
                eradicated_source_patterns
            ),
            "selected_source_pattern_eradication_fraction": (
                _safe_fraction(
                    eradicated_source_patterns,
                    selected_source_pattern_count,
                )
            ),
            "selected_source_occurrence_mass_before": int(
                selected_source_mass_before
            ),
            "selected_source_occurrence_mass_after": int(
                selected_source_mass_after
            ),
            "selected_source_occurrence_mass_reduction_fraction": float(
                selected_source_mass_reduction
            ),
        },

        "global_pattern_effect": {
            "clean_distinct_pattern_count": int(
                clean_distinct_pattern_count
            ),
            "poison_distinct_pattern_count": int(
                poison_distinct_pattern_count
            ),
            "clean_to_poison_distinct_pattern_reduction_fraction": float(
                clean_to_poison_distinct_reduction
            ),
            "removed_pattern_type_count": int(
                len(
                    removed_pattern_types
                )
            ),
            "new_pattern_type_count": int(
                len(
                    new_pattern_types
                )
            ),
        },

        "action_bound_diagnostic": {
            "modified_action_rows_at_bound_fraction": float(
                action_rows_at_bound_fraction
            ),
            "note": (
                "This is an action-bound saturation diagnostic, "
                "not the exact fraction of candidates that were clipped."
            ),
        },
    }


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_CLEAN,
    )

    parser.add_argument(
        "--poison-root",
        type=Path,
        default=DEFAULT_POISON_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--num-clusters",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--num-candidates",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--rhos",
        type=float,
        nargs="+",
        default=[
            0.01,
            0.05,
        ],
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    print(
        "Loading clean dataset:"
    )
    print(
        args.dataset
    )

    clean_dataset = (
        load_hdf5_dataset(
            args.dataset
        )
    )

    print(
        "Preparing clean CSDPC context "
        f"for seed={args.seed}, "
        f"k={args.num_clusters}..."
    )

    prepared = (
        prepare_csdpc_attack(
            clean_dataset,
            attack_seed=args.seed,
            num_clusters=args.num_clusters,
            sequence_length=args.sequence_length,
            eta=args.eta,
            num_candidates=args.num_candidates,
        )
    )

    trajectories, trailing = (
        find_completed_trajectories(
            clean_dataset[
                "terminals"
            ],
            clean_dataset[
                "timeouts"
            ],
        )
    )

    clean_counts = Counter(
        prepared.pattern_frequencies
    )

    raw_distinct_patterns = {
        window.raw_cluster_labels
        for window
        in prepared.windows
    }

    raw_distinct_pattern_count = len(
        raw_distinct_patterns
    )

    dedup_affected_window_count = sum(
        1
        for window
        in prepared.windows
        if tuple(
            window.raw_cluster_labels
        )
        != tuple(
            window.pattern
        )
    )

    dedup_affected_window_fraction = (
        _safe_fraction(
            dedup_affected_window_count,
            len(
                prepared.windows
            ),
        )
    )

    print(
        "Completed trajectories:",
        len(
            trajectories
        ),
    )

    print(
        "Trailing transitions:",
        trailing,
    )

    print(
        "Clean windows:",
        len(
            prepared.windows
        ),
    )

    print(
        "Raw distinct patterns:",
        raw_distinct_pattern_count,
    )

    print(
        "Deduplicated distinct patterns:",
        len(
            clean_counts
        ),
    )

    for rho in args.rhos:
        code = _rho_code(
            rho
        )

        poison_path = (
            args.poison_root
            / (
                f"rho_{code}"
                f"_seed_{args.seed}.hdf5"
            )
        )

        if not poison_path.exists():
            raise FileNotFoundError(
                poison_path
            )

        print()
        print(
            "=" * 70
        )
        print(
            f"Auditing seed={args.seed}, "
            f"rho={rho:.2%}"
        )
        print(
            poison_path
        )
        print(
            "=" * 70
        )

        poisoned_dataset = (
            load_hdf5_dataset(
                poison_path
            )
        )

        result = _audit_one_rho(
            clean_dataset=clean_dataset,
            poisoned_dataset=poisoned_dataset,
            prepared=prepared,
            trajectories=trajectories,
            clean_counts=clean_counts,
            raw_distinct_pattern_count=(
                raw_distinct_pattern_count
            ),
            dedup_affected_window_fraction=(
                dedup_affected_window_fraction
            ),
            rho=rho,
        )

        output_path = (
            args.output_root
            / (
                f"rho_{code}"
                f"_seed_{args.seed}.json"
            )
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
        )

        print(
            "Wrote:",
            output_path
        )


if __name__ == "__main__":
    main()