from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.audit_csdpc_selection_semantics import (
    _build_pattern_index,
    _select_pattern_type_atomic_prefix,
)
from src.attacks.csdpc.attack import (
    prepare_csdpc_attack,
)
from src.attacks.csdpc.clustering import (
    build_raw_decision_units,
)
from src.attacks.csdpc.metadata import (
    sha256_file,
    write_metadata_json,
)
from src.attacks.csdpc.patterns import (
    deduplicate_consecutive,
)
from src.attacks.csdpc.perturbation import (
    perturb_selected_window,
)
from src.attacks.csdpc.selection import (
    compute_transition_budget,
)
from src.attacks.csdpc.types import (
    SelectedWindow,
)
from src.data.hdf5_io import (
    load_hdf5_dataset,
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
    / "csdpc_overlap_conflict_diagnostic.json"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "csdpc_overlap_conflict"
)

EXPECTED_DATASET_SHA256 = (
    "cf00f43add04c17fdfc2958dd581dea"
    "0851b2e5bedbe6fda073758a8f841aeda"
)

SCALAR_METRICS = (
    "selected_window_count",
    "unique_transition_footprint",
    "selected_window_transition_slot_count",
    "overlap_reuse_fraction",

    "overlapped_unique_transition_fraction",
    "window_touching_overlap_fraction",

    "conflicting_overlap_transition_fraction",
    "conflict_transition_fraction_of_unique_footprint",
    "proposal_slot_conflict_fraction",
    "pairwise_label_disagreement_fraction_on_overlaps",

    "window_touching_label_conflict_fraction",
    "window_not_touching_label_conflict_fraction",

    "mean_proposal_multiplicity_on_overlapped_transitions",
    "mean_distinct_target_labels_on_overlapped_transitions",
    "max_distinct_target_labels_on_any_transition",

    "independent_proposal_pattern_change_fraction",
    "independent_proposal_frequency_improvement_fraction",
    "mean_source_pattern_frequency",
    "mean_target_pattern_frequency",
)


@dataclass(frozen=True)
class WindowProposal:
    raw_target_labels: tuple[int, ...]
    target_pattern: tuple[int, ...]
    source_frequency: int
    target_frequency: int
    candidate_index: int


def _safe_fraction(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


def _mean(values):
    if not values:
        return 0.0

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def _rho_key(rho):
    return f"{float(rho):.2f}"


def _load_config(path: Path):
    x = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if (
        x.get("experiment")
        != "CSDPC_OVERLAP_TARGET_CONFLICT_DIAGNOSTIC"
    ):
        raise ValueError(
            "unexpected experiment"
        )

    if (
        x.get("status")
        != "DIAGNOSTIC_ONLY"
    ):
        raise ValueError(
            "diagnostic status changed"
        )

    if not x.get(
        "canonical_attack_is_unchanged",
        False,
    ):
        raise ValueError(
            "canonical attack must remain unchanged"
        )

    if (
        x.get("selection_reference")
        != "S2_PATTERN_TYPE_ATOMIC_PREFIX"
    ):
        raise ValueError(
            "selection reference changed"
        )

    if int(
        x["num_clusters"]
    ) != 8:
        raise ValueError(
            "k must remain 8"
        )

    if int(
        x["sequence_length"]
    ) != 5:
        raise ValueError(
            "sequence length must remain 5"
        )

    if not np.isclose(
        float(x["eta"]),
        0.05,
    ):
        raise ValueError(
            "eta must remain 0.05"
        )

    if int(
        x["num_candidates"]
    ) != 100:
        raise ValueError(
            "candidate count must remain 100"
        )

    if list(
        x["attack_seeds"]
    ) != [0, 1, 2]:
        raise ValueError(
            "seed set changed"
        )

    if list(
        x["poison_rates"]
    ) != [0.01, 0.05]:
        raise ValueError(
            "rho set changed"
        )

    if set(
        x["metrics"]
    ) != set(
        SCALAR_METRICS
    ):
        raise ValueError(
            "metric set changed"
        )

    return x


def _to_selected_window(window):
    return SelectedWindow(
        trajectory_id=int(
            window.trajectory_id
        ),
        global_start=int(
            window.global_start
        ),
        global_end=int(
            window.global_end
        ),
        source_pattern=tuple(
            int(value)
            for value
            in window.pattern
        ),
    )


def _window_key(window):
    return (
        int(window.trajectory_id),
        int(window.global_start),
        int(window.global_end),
        tuple(
            int(v)
            for v
            in window.source_pattern
        ),
    )


def _is_prefix(
    smaller,
    larger,
):
    small = [
        _window_key(w)
        for w in smaller
    ]

    large = [
        _window_key(w)
        for w in larger
    ]

    return (
        small
        == large[
            :len(small)
        ]
    )


def _predict_window_labels(
    perturbed,
    model,
):
    features = build_raw_decision_units(
        np.asarray(
            perturbed.observations
        ),
        np.asarray(
            perturbed.actions
        ),
    )

    centers = np.asarray(
        model.cluster_centers_
    )

    features = np.asarray(
        features,
        dtype=centers.dtype,
    )

    return tuple(
        int(value)
        for value
        in model.predict(
            features
        )
    )


def _generate_proposals(
    *,
    selected_windows,
    clean_dataset,
    prepared,
):
    observations = np.asarray(
        clean_dataset[
            "observations"
        ]
    )

    actions = np.asarray(
        clean_dataset[
            "actions"
        ]
    )

    rng = np.random.default_rng(
        prepared.attack_seed
    )

    proposals = []

    for position, window in enumerate(
        selected_windows,
        start=1,
    ):
        perturbed = perturb_selected_window(
            observations,
            actions,
            window,
            kmeans_model=(
                prepared.clustering_model
            ),
            clean_pattern_frequencies=(
                prepared.pattern_frequencies
            ),
            eta=prepared.eta,
            num_candidates=100,
            rng=rng,
            action_low=-1.0,
            action_high=1.0,
        )

        labels = _predict_window_labels(
            perturbed,
            prepared.clustering_model,
        )

        pattern = deduplicate_consecutive(
            labels
        )

        if (
            tuple(pattern)
            != tuple(
                perturbed.target_pattern
            )
        ):
            raise RuntimeError(
                "predicted labels do not reproduce "
                "PerturbedWindow target pattern"
            )

        proposals.append(
            WindowProposal(
                raw_target_labels=tuple(
                    labels
                ),
                target_pattern=tuple(
                    perturbed.target_pattern
                ),
                source_frequency=int(
                    perturbed.source_frequency
                ),
                target_frequency=int(
                    perturbed.target_frequency
                ),
                candidate_index=int(
                    perturbed.candidate_index
                ),
            )
        )

        if position % 500 == 0:
            print(
                "Independent C100 proposals:",
                position,
                "/",
                len(selected_windows),
            )

    return tuple(
        proposals
    )


def _compute_conflict_metrics(
    *,
    selected_windows,
    proposals,
):
    if (
        len(selected_windows)
        != len(proposals)
    ):
        raise ValueError(
            "window/proposal count mismatch"
        )

    transition_requirements = (
        defaultdict(list)
    )

    for window_id, (
        window,
        proposal,
    ) in enumerate(
        zip(
            selected_windows,
            proposals,
        )
    ):
        length = (
            int(window.global_end)
            - int(window.global_start)
        )

        if (
            len(
                proposal.raw_target_labels
            )
            != length
        ):
            raise RuntimeError(
                "proposal label length mismatch"
            )

        for local_index, target_label in (
            enumerate(
                proposal.raw_target_labels
            )
        ):
            global_index = (
                int(window.global_start)
                + local_index
            )

            transition_requirements[
                global_index
            ].append(
                (
                    int(window_id),
                    int(target_label),
                )
            )

    unique_transition_count = len(
        transition_requirements
    )

    selected_slot_count = sum(
        len(requirements)
        for requirements
        in transition_requirements.values()
    )

    overlap_indices = [
        index
        for index, requirements
        in transition_requirements.items()
        if len(requirements) > 1
    ]

    conflict_indices = [
        index
        for index
        in overlap_indices
        if len(
            {
                label
                for _, label
                in transition_requirements[
                    index
                ]
            }
        ) > 1
    ]

    windows_touching_overlap = set()
    windows_touching_conflict = set()

    overlap_multiplicities = []
    overlap_distinct_labels = []

    conflict_slot_count = 0

    total_overlap_pairs = 0
    disagreeing_overlap_pairs = 0

    for index in overlap_indices:
        requirements = (
            transition_requirements[
                index
            ]
        )

        multiplicity = len(
            requirements
        )

        labels = [
            label
            for _, label
            in requirements
        ]

        counts = Counter(
            labels
        )

        overlap_multiplicities.append(
            multiplicity
        )

        overlap_distinct_labels.append(
            len(counts)
        )

        for window_id, _ in requirements:
            windows_touching_overlap.add(
                window_id
            )

        total_pairs = (
            multiplicity
            * (
                multiplicity - 1
            )
            // 2
        )

        agreeing_pairs = sum(
            count
            * (
                count - 1
            )
            // 2
            for count
            in counts.values()
        )

        total_overlap_pairs += (
            total_pairs
        )

        disagreeing_overlap_pairs += (
            total_pairs
            - agreeing_pairs
        )

        if (
            len(counts)
            > 1
        ):
            conflict_slot_count += (
                multiplicity
            )

            for window_id, _ in requirements:
                windows_touching_conflict.add(
                    window_id
                )

    selected_window_count = len(
        selected_windows
    )

    pattern_changed = sum(
        proposal.target_pattern
        != window.source_pattern
        for window, proposal
        in zip(
            selected_windows,
            proposals,
        )
    )

    frequency_improved = sum(
        proposal.target_frequency
        > proposal.source_frequency
        for proposal
        in proposals
    )

    source_frequencies = [
        proposal.source_frequency
        for proposal
        in proposals
    ]

    target_frequencies = [
        proposal.target_frequency
        for proposal
        in proposals
    ]

    overlap_reuse_fraction = (
        1.0
        - _safe_fraction(
            unique_transition_count,
            selected_slot_count,
        )
        if selected_slot_count
        else 0.0
    )

    return {
        "selected_window_count": int(
            selected_window_count
        ),

        "unique_transition_footprint": int(
            unique_transition_count
        ),

        "selected_window_transition_slot_count": int(
            selected_slot_count
        ),

        "overlap_reuse_fraction": float(
            overlap_reuse_fraction
        ),

        "overlapped_unique_transition_fraction": (
            _safe_fraction(
                len(overlap_indices),
                unique_transition_count,
            )
        ),

        "window_touching_overlap_fraction": (
            _safe_fraction(
                len(
                    windows_touching_overlap
                ),
                selected_window_count,
            )
        ),

        "conflicting_overlap_transition_fraction": (
            _safe_fraction(
                len(conflict_indices),
                len(overlap_indices),
            )
        ),

        "conflict_transition_fraction_of_unique_footprint": (
            _safe_fraction(
                len(conflict_indices),
                unique_transition_count,
            )
        ),

        "proposal_slot_conflict_fraction": (
            _safe_fraction(
                conflict_slot_count,
                selected_slot_count,
            )
        ),

        "pairwise_label_disagreement_fraction_on_overlaps": (
            _safe_fraction(
                disagreeing_overlap_pairs,
                total_overlap_pairs,
            )
        ),

        "window_touching_label_conflict_fraction": (
            _safe_fraction(
                len(
                    windows_touching_conflict
                ),
                selected_window_count,
            )
        ),

        "window_not_touching_label_conflict_fraction": (
            _safe_fraction(
                selected_window_count
                - len(
                    windows_touching_conflict
                ),
                selected_window_count,
            )
        ),

        "mean_proposal_multiplicity_on_overlapped_transitions": (
            _mean(
                overlap_multiplicities
            )
        ),

        "mean_distinct_target_labels_on_overlapped_transitions": (
            _mean(
                overlap_distinct_labels
            )
        ),

        "max_distinct_target_labels_on_any_transition": int(
            max(
                overlap_distinct_labels,
                default=1,
            )
        ),

        "independent_proposal_pattern_change_fraction": (
            _safe_fraction(
                pattern_changed,
                selected_window_count,
            )
        ),

        "independent_proposal_frequency_improvement_fraction": (
            _safe_fraction(
                frequency_improved,
                selected_window_count,
            )
        ),

        "mean_source_pattern_frequency": (
            _mean(
                source_frequencies
            )
        ),

        "mean_target_pattern_frequency": (
            _mean(
                target_frequencies
            )
        ),
    }


def _run_seed(
    *,
    seed,
    config,
    clean_dataset,
):
    prepared = prepare_csdpc_attack(
        clean_dataset,
        attack_seed=int(seed),
        num_clusters=int(
            config[
                "num_clusters"
            ]
        ),
        sequence_length=int(
            config[
                "sequence_length"
            ]
        ),
        eta=float(
            config[
                "eta"
            ]
        ),
        num_candidates=int(
            config[
                "num_candidates"
            ]
        ),
    )

    (
        occurrences,
        ranked_patterns,
    ) = _build_pattern_index(
        prepared.windows,
        prepared.pattern_frequencies,
    )

    selections = {}

    for rho in config[
        "poison_rates"
    ]:
        rho = float(rho)

        budget = compute_transition_budget(
            num_transitions=(
                prepared.num_transitions
            ),
            rho=rho,
        )

        diagnostic = (
            _select_pattern_type_atomic_prefix(
                occurrences_by_pattern=(
                    occurrences
                ),
                ranked_patterns=(
                    ranked_patterns
                ),
                transition_budget=(
                    budget
                ),
            )
        )

        selections[rho] = tuple(
            _to_selected_window(
                window
            )
            for window
            in diagnostic.selected_windows
        )

    if not _is_prefix(
        selections[0.01],
        selections[0.05],
    ):
        raise RuntimeError(
            "rho=0.01 S2 selection is not "
            "an exact prefix of rho=0.05"
        )

    print(
        "S2 rho-prefix check: PASS"
    )

    max_windows = (
        selections[
            0.05
        ]
    )

    proposals = _generate_proposals(
        selected_windows=max_windows,
        clean_dataset=clean_dataset,
        prepared=prepared,
    )

    rho_results = {}

    for rho in config[
        "poison_rates"
    ]:
        rho = float(rho)

        windows = selections[
            rho
        ]

        rho_proposals = proposals[
            :len(windows)
        ]

        metrics = (
            _compute_conflict_metrics(
                selected_windows=windows,
                proposals=rho_proposals,
            )
        )

        rho_results[
            _rho_key(rho)
        ] = {
            "metrics": metrics
        }

    return {
        "schema_version": (
            "csdpc-overlap-conflict-seed-v1"
        ),
        "experiment": config[
            "experiment"
        ],
        "status": config[
            "status"
        ],
        "canonical_attack_is_unchanged": True,
        "seed": int(seed),
        "rho_results": rho_results,
    }


def _build_summary(
    *,
    config,
    seed_records,
):
    seeds = [
        int(seed)
        for seed
        in config[
            "attack_seeds"
        ]
    ]

    rho_results = {}

    for rho in config[
        "poison_rates"
    ]:
        rho_key = _rho_key(
            rho
        )

        metric_summary = {}

        for metric in SCALAR_METRICS:
            values = np.asarray(
                [
                    seed_records[
                        str(seed)
                    ][
                        "rho_results"
                    ][
                        rho_key
                    ][
                        "metrics"
                    ][
                        metric
                    ]
                    for seed
                    in seeds
                ],
                dtype=np.float64,
            )

            metric_summary[
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
                    np.mean(values)
                ),
                "std": float(
                    np.std(
                        values,
                        ddof=0,
                    )
                ),
                "min": float(
                    np.min(values)
                ),
                "max": float(
                    np.max(values)
                ),
            }

        rho_results[
            rho_key
        ] = {
            "metrics": metric_summary
        }

    return {
        "schema_version": (
            "csdpc-overlap-conflict-summary-v1"
        ),
        "experiment": config[
            "experiment"
        ],
        "status": config[
            "status"
        ],
        "canonical_attack_is_unchanged": True,
        "attack_seeds": seeds,
        "rho_results": rho_results,
        "interpretation_rules": {
            "no_conflict_resolution_tested": True,
            "no_poisoned_dataset_generated": True,
            "no_learner_trained": True,
            "canonical_gate_b_unchanged": True,
        },
    }


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

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--summarize-only",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    config = _load_config(
        args.config
    )

    output_root = (
        args.output_root.resolve()
    )

    if args.summarize_only:
        seed_records = {}

        for seed in config[
            "attack_seeds"
        ]:
            path = (
                output_root
                / f"seed_{seed}.json"
            )

            seed_records[
                str(seed)
            ] = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        summary = _build_summary(
            config=config,
            seed_records=seed_records,
        )

        summary_path = (
            output_root
            / "summary.json"
        )

        write_metadata_json(
            summary_path,
            summary,
        )

        print(
            "Summary:",
            summary_path,
        )

        return

    dataset_path = (
        args.dataset.resolve()
    )

    actual_sha = sha256_file(
        dataset_path
    )

    if (
        actual_sha
        != EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "frozen dataset SHA256 mismatch"
        )

    print(
        "Frozen clean dataset SHA256: PASS"
    )

    clean_dataset = (
        load_hdf5_dataset(
            dataset_path
        )
    )

    if args.seed is None:
        seeds = [
            int(seed)
            for seed
            in config[
                "attack_seeds"
            ]
        ]
    else:
        if (
            args.seed
            not in config[
                "attack_seeds"
            ]
        ):
            raise ValueError(
                "requested seed not frozen"
            )

        seeds = [
            int(args.seed)
        ]

    for seed in seeds:
        print()
        print("=" * 72)
        print(
            "CSDPC OVERLAP TARGET CONFLICT "
            f"— SEED {seed}"
        )
        print("=" * 72)

        record = _run_seed(
            seed=seed,
            config=config,
            clean_dataset=clean_dataset,
        )

        output_path = (
            output_root
            / f"seed_{seed}.json"
        )

        write_metadata_json(
            output_path,
            record,
        )

        print(
            "Wrote:",
            output_path,
        )

        gc.collect()


if __name__ == "__main__":
    main()