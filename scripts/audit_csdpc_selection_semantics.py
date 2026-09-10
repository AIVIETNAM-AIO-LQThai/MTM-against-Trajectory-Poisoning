from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.attacks.csdpc.attack import (
    prepare_csdpc_attack,
)
from src.attacks.csdpc.metadata import (
    sha256_file,
    write_metadata_json,
)
from src.attacks.csdpc.selection import (
    compute_transition_budget,
    select_rare_nonoverlapping_windows,
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
    / "csdpc_selection_semantics_diagnostic.json"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "csdpc_selection_semantics"
)

EXPECTED_DATASET_SHA256 = (
    "cf00f43add04c17fdfc2958dd581dea"
    "0851b2e5bedbe6fda073758a8f841aeda"
)

VARIANT_IDS = (
    "S0_CANONICAL_NONOVERLAP_OCCURRENCE",
    "S1_OCCURRENCE_RANKED_OVERLAP_ALLOWED",
    "S2_PATTERN_TYPE_ATOMIC_PREFIX",
)

SCALAR_METRICS = (
    "selected_window_count",
    "selected_window_fraction_of_all_windows",

    "selected_source_pattern_type_count",
    "selected_source_pattern_type_fraction_of_all_clean_types",

    "requested_transition_budget",
    "unique_transition_footprint",
    "unique_transition_budget_utilization",
    "unique_transition_fraction_of_dataset",

    "selected_window_transition_slot_count",
    "overlap_reuse_fraction",
    "overlapped_unique_transition_fraction",
    "max_transition_selection_multiplicity",
    "overlap_rejection_count",

    "selected_source_total_clean_occurrences",
    "selected_source_total_clean_occurrence_mass_fraction_of_all_windows",

    "fully_selected_source_pattern_type_count",
    "partially_selected_source_pattern_type_count",
    "fully_selected_source_pattern_fraction",
    "selected_source_occurrence_completion_fraction",

    "mean_selected_source_pattern_frequency_by_type",
    "max_selected_source_pattern_frequency",
)


@dataclass(frozen=True)
class DiagnosticSelection:
    selected_windows: tuple
    unique_transition_indices: tuple[int, ...]
    overlap_rejection_count: int = 0
    budget_rejection_count: int = 0
    stopped_before_pattern_rank: int | None = None
    stopped_before_pattern: tuple[int, ...] | None = None


def _safe_fraction(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return float(
        numerator
    ) / float(
        denominator
    )


def _mean(
    values,
):
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


def _rho_key(
    rho: float,
):
    return f"{float(rho):.2f}"


def _load_config(
    path: Path,
):
    config = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if (
        config.get("experiment")
        != "CSDPC_SELECTION_SEMANTICS_DIAGNOSTIC"
    ):
        raise ValueError(
            "unexpected experiment identifier"
        )

    if (
        config.get("status")
        != "DIAGNOSTIC_ONLY"
    ):
        raise ValueError(
            "selection-semantics experiment "
            "must remain DIAGNOSTIC_ONLY"
        )

    if not bool(
        config.get(
            "canonical_attack_is_unchanged",
            False,
        )
    ):
        raise ValueError(
            "canonical attack must remain unchanged"
        )

    if int(
        config["num_clusters"]
    ) != 8:
        raise ValueError(
            "frozen diagnostic requires k=8"
        )

    if int(
        config["sequence_length"]
    ) != 5:
        raise ValueError(
            "frozen diagnostic requires "
            "sequence_length=5"
        )

    if list(
        config["attack_seeds"]
    ) != [
        0,
        1,
        2,
    ]:
        raise ValueError(
            "frozen attack-seed set changed"
        )

    if list(
        config["poison_rates"]
    ) != [
        0.01,
        0.05,
    ]:
        raise ValueError(
            "frozen poison-rate set changed"
        )

    observed_ids = tuple(
        variant["id"]
        for variant
        in config["variants"]
    )

    if observed_ids != VARIANT_IDS:
        raise ValueError(
            "frozen variant definitions changed"
        )

    if set(
        config["metrics"]
    ) != set(
        SCALAR_METRICS
    ):
        raise ValueError(
            "frozen metric set changed"
        )

    return config


def _window_key(
    window,
):
    return (
        int(
            window.trajectory_id
        ),
        int(
            window.global_start
        ),
        int(
            window.global_end
        ),
        tuple(
            int(value)
            for value
            in window.pattern
        ),
    )


def _build_pattern_index(
    windows,
    pattern_frequencies,
):
    occurrences = defaultdict(
        list
    )

    for window in windows:
        pattern = tuple(
            window.pattern
        )

        occurrences[
            pattern
        ].append(
            window
        )

    if set(
        occurrences
    ) != set(
        pattern_frequencies
    ):
        raise RuntimeError(
            "pattern index does not match "
            "clean pattern-frequency keys"
        )

    for pattern, windows_for_pattern in (
        occurrences.items()
    ):
        expected = int(
            pattern_frequencies[
                pattern
            ]
        )

        if (
            len(
                windows_for_pattern
            )
            != expected
        ):
            raise RuntimeError(
                "pattern occurrence count mismatch"
            )

    ranked_patterns = tuple(
        sorted(
            pattern_frequencies,
            key=lambda pattern: (
                int(
                    pattern_frequencies[
                        pattern
                    ]
                ),
                tuple(
                    pattern
                ),
            ),
        )
    )

    return (
        dict(
            occurrences
        ),
        ranked_patterns,
    )


def _canonical_selection(
    *,
    windows,
    pattern_frequencies,
    transition_budget: int,
):
    result = (
        select_rare_nonoverlapping_windows(
            windows,
            pattern_frequencies,
            transition_budget=(
                transition_budget
            ),
        )
    )

    unique_indices = sorted(
        {
            int(index)
            for window
            in result.selected_windows
            for index
            in window.transition_indices
        }
    )

    if (
        len(
            unique_indices
        )
        != result.actual_transition_budget
    ):
        raise RuntimeError(
            "canonical unique-transition "
            "accounting mismatch"
        )

    return DiagnosticSelection(
        selected_windows=tuple(
            result.selected_windows
        ),
        unique_transition_indices=tuple(
            unique_indices
        ),
        overlap_rejection_count=int(
            result.skipped_overlap_windows
        ),
    )


def _select_occurrence_ranked_overlap_allowed(
    *,
    occurrences_by_pattern,
    ranked_patterns,
    transition_budget: int,
):
    if transition_budget < 0:
        raise ValueError(
            "transition budget cannot be negative"
        )

    used = set()
    selected = []

    budget_rejections = 0

    for pattern in ranked_patterns:
        occurrences = (
            occurrences_by_pattern[
                pattern
            ]
        )

        for window in occurrences:
            if (
                len(
                    used
                )
                == transition_budget
            ):
                return DiagnosticSelection(
                    selected_windows=tuple(
                        selected
                    ),
                    unique_transition_indices=tuple(
                        sorted(
                            used
                        )
                    ),
                    budget_rejection_count=(
                        budget_rejections
                    ),
                )

            indices = set(
                int(index)
                for index
                in window.transition_indices
            )

            additional = (
                indices
                - used
            )

            if (
                len(
                    used
                )
                + len(
                    additional
                )
                > transition_budget
            ):
                budget_rejections += 1
                continue

            selected.append(
                window
            )

            used.update(
                indices
            )

    return DiagnosticSelection(
        selected_windows=tuple(
            selected
        ),
        unique_transition_indices=tuple(
            sorted(
                used
            )
        ),
        budget_rejection_count=(
            budget_rejections
        ),
    )


def _select_pattern_type_atomic_prefix(
    *,
    occurrences_by_pattern,
    ranked_patterns,
    transition_budget: int,
):
    """
    Source-semantics diagnostic.

    Pattern types are considered in ascending clean
    occurrence-frequency order.

    A pattern type is accepted only if ALL of its
    clean window occurrences can be included without
    exceeding the unique-transition budget.

    The selected set remains a strict rare-pattern
    prefix: if the next complete type cannot fit,
    selection stops rather than skipping forward to
    less-rare types.
    """

    if transition_budget < 0:
        raise ValueError(
            "transition budget cannot be negative"
        )

    used = set()
    selected = []

    for rank, pattern in enumerate(
        ranked_patterns
    ):
        if (
            len(
                used
            )
            == transition_budget
        ):
            break

        occurrences = (
            occurrences_by_pattern[
                pattern
            ]
        )

        pattern_indices = {
            int(index)
            for window
            in occurrences
            for index
            in window.transition_indices
        }

        additional = (
            pattern_indices
            - used
        )

        if (
            len(
                used
            )
            + len(
                additional
            )
            > transition_budget
        ):
            return DiagnosticSelection(
                selected_windows=tuple(
                    selected
                ),
                unique_transition_indices=tuple(
                    sorted(
                        used
                    )
                ),
                stopped_before_pattern_rank=int(
                    rank
                ),
                stopped_before_pattern=tuple(
                    int(value)
                    for value
                    in pattern
                ),
            )

        selected.extend(
            occurrences
        )

        used.update(
            pattern_indices
        )

    return DiagnosticSelection(
        selected_windows=tuple(
            selected
        ),
        unique_transition_indices=tuple(
            sorted(
                used
            )
        ),
    )


def _selection_multiplicity(
    selected_windows,
):
    multiplicity = Counter()

    for window in selected_windows:
        for index in (
            window.transition_indices
        ):
            multiplicity[
                int(index)
            ] += 1

    return multiplicity


def _summarize_selection(
    *,
    selection: DiagnosticSelection,
    pattern_frequencies,
    total_window_count: int,
    num_transitions: int,
    transition_budget: int,
):
    selected_windows = (
        selection.selected_windows
    )

    multiplicity = (
        _selection_multiplicity(
            selected_windows
        )
    )

    unique_indices = set(
        selection.unique_transition_indices
    )

    if set(
        multiplicity
    ) != unique_indices:
        raise RuntimeError(
            "selection unique-transition "
            "footprint mismatch"
        )

    unique_transition_footprint = len(
        unique_indices
    )

    if (
        unique_transition_footprint
        > transition_budget
    ):
        raise RuntimeError(
            "selection exceeds transition budget"
        )

    selected_window_count = len(
        selected_windows
    )

    selected_slot_count = int(
        sum(
            int(
                window.global_end
            )
            - int(
                window.global_start
            )
            for window
            in selected_windows
        )
    )

    if selected_slot_count:
        overlap_reuse_fraction = (
            1.0
            - float(
                unique_transition_footprint
            )
            / float(
                selected_slot_count
            )
        )
    else:
        overlap_reuse_fraction = 0.0

    overlapped_unique_count = sum(
        count > 1
        for count
        in multiplicity.values()
    )

    overlapped_unique_fraction = (
        _safe_fraction(
            overlapped_unique_count,
            unique_transition_footprint,
        )
    )

    max_multiplicity = max(
        multiplicity.values(),
        default=0,
    )

    selected_occurrence_counts = Counter(
        tuple(
            window.source_pattern
            if hasattr(
                window,
                "source_pattern",
            )
            else window.pattern
        )
        for window
        in selected_windows
    )

    selected_patterns = set(
        selected_occurrence_counts
    )

    selected_pattern_count = len(
        selected_patterns
    )

    total_clean_occurrences = sum(
        int(
            pattern_frequencies[
                pattern
            ]
        )
        for pattern
        in selected_patterns
    )

    fully_selected_count = sum(
        1
        for pattern
        in selected_patterns
        if (
            selected_occurrence_counts[
                pattern
            ]
            == int(
                pattern_frequencies[
                    pattern
                ]
            )
        )
    )

    partially_selected_count = (
        selected_pattern_count
        - fully_selected_count
    )

    pattern_type_frequencies = [
        int(
            pattern_frequencies[
                pattern
            ]
        )
        for pattern
        in selected_patterns
    ]

    return {
        "selected_window_count": int(
            selected_window_count
        ),

        "selected_window_fraction_of_all_windows": (
            _safe_fraction(
                selected_window_count,
                total_window_count,
            )
        ),

        "selected_source_pattern_type_count": int(
            selected_pattern_count
        ),

        "selected_source_pattern_type_fraction_of_all_clean_types": (
            _safe_fraction(
                selected_pattern_count,
                len(
                    pattern_frequencies
                ),
            )
        ),

        "requested_transition_budget": int(
            transition_budget
        ),

        "unique_transition_footprint": int(
            unique_transition_footprint
        ),

        "unique_transition_budget_utilization": (
            _safe_fraction(
                unique_transition_footprint,
                transition_budget,
            )
        ),

        "unique_transition_fraction_of_dataset": (
            _safe_fraction(
                unique_transition_footprint,
                num_transitions,
            )
        ),

        "selected_window_transition_slot_count": int(
            selected_slot_count
        ),

        "overlap_reuse_fraction": float(
            overlap_reuse_fraction
        ),

        "overlapped_unique_transition_fraction": float(
            overlapped_unique_fraction
        ),

        "max_transition_selection_multiplicity": int(
            max_multiplicity
        ),

        "overlap_rejection_count": int(
            selection.overlap_rejection_count
        ),

        "selected_source_total_clean_occurrences": int(
            total_clean_occurrences
        ),

        "selected_source_total_clean_occurrence_mass_fraction_of_all_windows": (
            _safe_fraction(
                total_clean_occurrences,
                total_window_count,
            )
        ),

        "fully_selected_source_pattern_type_count": int(
            fully_selected_count
        ),

        "partially_selected_source_pattern_type_count": int(
            partially_selected_count
        ),

        "fully_selected_source_pattern_fraction": (
            _safe_fraction(
                fully_selected_count,
                selected_pattern_count,
            )
        ),

        "selected_source_occurrence_completion_fraction": (
            _safe_fraction(
                selected_window_count,
                total_clean_occurrences,
            )
        ),

        "mean_selected_source_pattern_frequency_by_type": (
            _mean(
                pattern_type_frequencies
            )
        ),

        "max_selected_source_pattern_frequency": int(
            max(
                pattern_type_frequencies,
                default=0,
            )
        ),
    }


def _run_seed(
    *,
    seed: int,
    config,
    clean_dataset,
):
    prepared = prepare_csdpc_attack(
        clean_dataset,
        attack_seed=seed,
        num_clusters=int(
            config["num_clusters"]
        ),
        sequence_length=int(
            config["sequence_length"]
        ),
        eta=0.05,
        num_candidates=100,
    )

    (
        occurrences_by_pattern,
        ranked_patterns,
    ) = _build_pattern_index(
        prepared.windows,
        prepared.pattern_frequencies,
    )

    rho_results = {}

    for rho in config[
        "poison_rates"
    ]:
        rho = float(
            rho
        )

        transition_budget = (
            compute_transition_budget(
                num_transitions=(
                    prepared.num_transitions
                ),
                rho=rho,
            )
        )

        s0 = _canonical_selection(
            windows=(
                prepared.windows
            ),
            pattern_frequencies=(
                prepared.pattern_frequencies
            ),
            transition_budget=(
                transition_budget
            ),
        )

        s1 = (
            _select_occurrence_ranked_overlap_allowed(
                occurrences_by_pattern=(
                    occurrences_by_pattern
                ),
                ranked_patterns=(
                    ranked_patterns
                ),
                transition_budget=(
                    transition_budget
                ),
            )
        )

        s2 = (
            _select_pattern_type_atomic_prefix(
                occurrences_by_pattern=(
                    occurrences_by_pattern
                ),
                ranked_patterns=(
                    ranked_patterns
                ),
                transition_budget=(
                    transition_budget
                ),
            )
        )

        variant_selections = {
            "S0_CANONICAL_NONOVERLAP_OCCURRENCE": s0,
            "S1_OCCURRENCE_RANKED_OVERLAP_ALLOWED": s1,
            "S2_PATTERN_TYPE_ATOMIC_PREFIX": s2,
        }

        variants = {}

        for (
            variant_id,
            selection,
        ) in variant_selections.items():
            metrics = (
                _summarize_selection(
                    selection=selection,
                    pattern_frequencies=(
                        prepared.pattern_frequencies
                    ),
                    total_window_count=len(
                        prepared.windows
                    ),
                    num_transitions=(
                        prepared.num_transitions
                    ),
                    transition_budget=(
                        transition_budget
                    ),
                )
            )

            variants[
                variant_id
            ] = {
                "metrics": metrics,
                "diagnostics": {
                    "budget_rejection_count": int(
                        selection.budget_rejection_count
                    ),
                    "stopped_before_pattern_rank": (
                        selection.stopped_before_pattern_rank
                    ),
                    "stopped_before_pattern": (
                        list(
                            selection.stopped_before_pattern
                        )
                        if (
                            selection.stopped_before_pattern
                            is not None
                        )
                        else None
                    ),
                },
            }

        if (
            variants[
                "S0_CANONICAL_NONOVERLAP_OCCURRENCE"
            ][
                "metrics"
            ][
                "unique_transition_footprint"
            ]
            != transition_budget
        ):
            raise RuntimeError(
                "canonical selection did not "
                "fill frozen transition budget"
            )

        s2_metrics = variants[
            "S2_PATTERN_TYPE_ATOMIC_PREFIX"
        ][
            "metrics"
        ]

        if (
            s2_metrics[
                "selected_source_pattern_type_count"
            ]
            > 0
        ):
            if not np.isclose(
                s2_metrics[
                    "fully_selected_source_pattern_fraction"
                ],
                1.0,
            ):
                raise RuntimeError(
                    "S2 contains a partially "
                    "selected pattern type"
                )

            if not np.isclose(
                s2_metrics[
                    "selected_source_occurrence_completion_fraction"
                ],
                1.0,
            ):
                raise RuntimeError(
                    "S2 occurrence completion "
                    "is not exactly 1"
                )

        rho_results[
            _rho_key(
                rho
            )
        ] = {
            "requested_transition_budget": int(
                transition_budget
            ),
            "variants": variants,
        }

    return {
        "schema_version": (
            "csdpc-selection-semantics-seed-v1"
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
        "clustering": {
            "num_clusters": int(
                config["num_clusters"]
            ),
            "sequence_length": int(
                config["sequence_length"]
            ),
            "inertia": float(
                prepared.clustering.inertia
            ),
            "n_iter": int(
                prepared.clustering.n_iter
            ),
        },
        "clean_pattern_baseline": {
            "window_count": int(
                len(
                    prepared.windows
                )
            ),
            "distinct_pattern_type_count": int(
                len(
                    prepared.pattern_frequencies
                )
            ),
        },
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

    rho_summary = {}

    for rho in config[
        "poison_rates"
    ]:
        rho_key = _rho_key(
            float(
                rho
            )
        )

        variants_summary = {}

        for variant_id in (
            VARIANT_IDS
        ):
            metric_summary = {}

            for metric in (
                SCALAR_METRICS
            ):
                values = np.asarray(
                    [
                        seed_records[
                            str(seed)
                        ][
                            "rho_results"
                        ][
                            rho_key
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

            variants_summary[
                variant_id
            ] = metric_summary

        rho_summary[
            rho_key
        ] = {
            "variants": (
                variants_summary
            )
        }

    return {
        "schema_version": (
            "csdpc-selection-semantics-summary-v1"
        ),
        "experiment": config[
            "experiment"
        ],
        "status": config[
            "status"
        ],
        "canonical_attack_is_unchanged": True,
        "attack_seeds": seeds,
        "rho_results": rho_summary,
        "interpretation_rules": {
            "S0_is_canonical": True,
            "S1_and_S2_are_diagnostic_only": True,
            "overlapping_windows_are_not_yet_a_poisoning_rule": True,
            "no_learner_result_used": True,
            "canonical_gate_b_unchanged": True,
        },
    }


def _load_seed_records(
    *,
    output_root: Path,
    seeds,
):
    records = {}

    for seed in seeds:
        path = (
            output_root
            / f"seed_{seed}.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"missing seed result: {path}"
            )

        records[
            str(
                seed
            )
        ] = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    return records


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
        records = _load_seed_records(
            output_root=output_root,
            seeds=config[
                "attack_seeds"
            ],
        )

        summary = _build_summary(
            config=config,
            seed_records=records,
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

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"dataset not found: {dataset_path}"
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
                "requested seed is not frozen"
            )

        seeds = [
            int(
                args.seed
            )
        ]

    for seed in seeds:
        print()
        print(
            "=" * 72
        )
        print(
            "CSDPC SELECTION SEMANTICS "
            f"— SEED {seed}"
        )
        print(
            "=" * 72
        )

        record = _run_seed(
            seed=seed,
            config=config,
            clean_dataset=(
                clean_dataset
            ),
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