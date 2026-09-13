from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from src.attacks.csdpc.attack import prepare_csdpc_attack
from src.attacks.csdpc.clustering import build_raw_decision_units
from src.attacks.csdpc.metadata import (
    sha256_file,
    write_metadata_json,
)
from src.attacks.csdpc.patterns import deduplicate_consecutive
from src.attacks.csdpc.selection import (
    compute_transition_budget,
    select_rare_nonoverlapping_windows,
)
from src.data.hdf5_io import load_hdf5_dataset


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
    / "csdpc_reachability_oracle.json"
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
    / "csdpc_reachability_oracle"
)

EXPECTED_DATASET_SHA256 = (
    "cf00f43add04c17fdfc2958dd581dea"
    "0851b2e5bedbe6fda073758a8f841aeda"
)

RHO_CODES = {
    0.01: "001",
    0.05: "005",
}

SCALAR_METRICS = (
    "selected_window_count",
    "selected_transition_count",
    "reachable_cluster_count_mean",
    "reachable_cluster_count_max",
    "multi_cluster_reachable_transition_fraction",
    "source_only_reachable_transition_fraction",
    "reachable_pattern_count_mean",
    "reachable_pattern_count_max",
    "oracle_pattern_change_possible_fraction",
    "oracle_frequency_improvement_possible_fraction",
    "actual_frequency_improvement_fraction",
    "actual_attains_oracle_best_frequency_fraction",
    "oracle_search_gap_fraction",
    "actual_pattern_reachable_fraction",
    "mean_source_pattern_frequency",
    "mean_actual_target_frequency",
    "mean_oracle_best_frequency",
    "mean_oracle_minus_actual_frequency",
)


def _load_config(path: Path) -> dict:
    config = json.loads(
        path.read_text(encoding="utf-8")
    )

    if (
        config.get("experiment")
        != "CSDPC_REACHABILITY_ORACLE_DIAGNOSTIC"
    ):
        raise ValueError(
            "unexpected experiment identifier"
        )

    if config.get("status") != "DIAGNOSTIC_ONLY":
        raise ValueError(
            "oracle must remain DIAGNOSTIC_ONLY"
        )

    if not config.get(
        "canonical_attack_is_unchanged",
        False,
    ):
        raise ValueError(
            "canonical attack must remain unchanged"
        )

    if int(config["num_clusters"]) != 8:
        raise ValueError("oracle requires k=8")

    if int(config["sequence_length"]) != 5:
        raise ValueError(
            "oracle requires sequence_length=5"
        )

    if not np.isclose(
        float(config["eta"]),
        0.05,
    ):
        raise ValueError(
            "oracle requires eta=0.05"
        )

    if list(config["attack_seeds"]) != [
        0, 1, 2
    ]:
        raise ValueError(
            "frozen seed set changed"
        )

    if list(config["poison_rates"]) != [
        0.01, 0.05
    ]:
        raise ValueError(
            "frozen poison-rate set changed"
        )

    if set(config["metrics"]) != set(
        SCALAR_METRICS
    ):
        raise ValueError(
            "frozen metric set changed"
        )

    return config


def _predict_labels(
    dataset,
    model,
):
    features = build_raw_decision_units(
        np.asarray(
            dataset["observations"]
        ),
        np.asarray(
            dataset["actions"]
        ),
    )

    centers = np.asarray(
        model.cluster_centers_
    )

    features = np.asarray(
        features,
        dtype=centers.dtype,
    )

    return np.asarray(
        model.predict(features),
        dtype=np.int64,
    )


def _transition_box(
    observation,
    action,
    *,
    eta: float,
    action_low: float,
    action_high: float,
):
    observation = np.asarray(
        observation,
        dtype=np.float64,
    )

    action = np.asarray(
        action,
        dtype=np.float64,
    )

    state_scale = float(
        eta
        * np.max(
            np.abs(observation)
        )
    )

    action_scale = float(
        eta
        * np.max(
            np.abs(action)
        )
    )

    state_lower = (
        observation
        - state_scale
    )

    state_upper = (
        observation
        + state_scale
    )

    action_lower = np.maximum(
        action - action_scale,
        action_low,
    )

    action_upper = np.minimum(
        action + action_scale,
        action_high,
    )

    lower = np.concatenate(
        [
            state_lower,
            action_lower,
        ]
    )

    upper = np.concatenate(
        [
            state_upper,
            action_upper,
        ]
    )

    if np.any(
        lower > upper
    ):
        raise RuntimeError(
            "invalid perturbation box"
        )

    return lower, upper


def _linear_min_over_box(
    coefficients,
    lower,
    upper,
) -> float:
    coefficients = np.asarray(
        coefficients,
        dtype=np.float64,
    )

    point = np.where(
        coefficients >= 0.0,
        lower,
        upper,
    )

    return float(
        np.dot(
            coefficients,
            point,
        )
    )


def _cluster_reachable(
    *,
    target_cluster: int,
    source_cluster: int,
    centers,
    lower,
    upper,
) -> bool:
    """
    Test whether the perturbation box intersects
    the target cluster's closed Euclidean Voronoi cell.

    Closed-cell reachability is deliberately an upper
    bound around exact sklearn tie handling.
    """

    if (
        target_cluster
        == source_cluster
    ):
        return True

    centers = np.asarray(
        centers,
        dtype=np.float64,
    )

    target = centers[
        target_cluster
    ]

    source = centers[
        source_cluster
    ]

    # Cheap necessary condition:
    # target must at least be able to beat the
    # currently assigned source cluster.
    coefficients = (
        2.0
        * (
            source
            - target
        )
    )

    rhs = float(
        np.dot(
            source,
            source,
        )
        - np.dot(
            target,
            target,
        )
    )

    minimum = (
        _linear_min_over_box(
            coefficients,
            lower,
            upper,
        )
    )

    tolerance = (
        1.0e-10
        * (
            1.0
            + abs(rhs)
        )
    )

    if minimum > (
        rhs + tolerance
    ):
        return False

    rows = []
    bounds_rhs = []

    for cluster_id, center in enumerate(
        centers
    ):
        if (
            cluster_id
            == target_cluster
        ):
            continue

        rows.append(
            2.0
            * (
                center
                - target
            )
        )

        bounds_rhs.append(
            float(
                np.dot(
                    center,
                    center,
                )
                - np.dot(
                    target,
                    target,
                )
            )
        )

    result = linprog(
        c=np.zeros(
            centers.shape[1],
            dtype=np.float64,
        ),
        A_ub=np.asarray(
            rows,
            dtype=np.float64,
        ),
        b_ub=np.asarray(
            bounds_rhs,
            dtype=np.float64,
        ),
        bounds=[
            (
                float(low),
                float(high),
            )
            for low, high
            in zip(
                lower,
                upper,
            )
        ],
        method="highs",
    )

    if result.success:
        return True

    # HiGHS status 2 = infeasible.
    if result.status == 2:
        return False

    raise RuntimeError(
        "unexpected linprog failure: "
        f"status={result.status}, "
        f"message={result.message}"
    )


def _reachable_clusters(
    *,
    observation,
    action,
    source_cluster: int,
    centers,
    eta: float,
    action_low: float,
    action_high: float,
):
    lower, upper = (
        _transition_box(
            observation,
            action,
            eta=eta,
            action_low=action_low,
            action_high=action_high,
        )
    )

    reachable = []

    for target_cluster in range(
        len(centers)
    ):
        if _cluster_reachable(
            target_cluster=(
                target_cluster
            ),
            source_cluster=(
                source_cluster
            ),
            centers=centers,
            lower=lower,
            upper=upper,
        ):
            reachable.append(
                int(
                    target_cluster
                )
            )

    if (
        source_cluster
        not in reachable
    ):
        raise RuntimeError(
            "source cluster not reachable"
        )

    return tuple(
        reachable
    )


def _reachable_patterns(
    reachable_label_sets,
):
    """
    Dynamic enumeration of unique deduplicated
    patterns, avoiding redundant raw Cartesian
    sequences where consecutive labels repeat.
    """

    patterns = {
        tuple()
    }

    for reachable_labels in (
        reachable_label_sets
    ):
        next_patterns = set()

        for pattern in patterns:
            for label in (
                reachable_labels
            ):
                label = int(
                    label
                )

                if (
                    pattern
                    and pattern[-1]
                    == label
                ):
                    candidate = pattern
                else:
                    candidate = (
                        pattern
                        + (
                            label,
                        )
                    )

                next_patterns.add(
                    candidate
                )

        patterns = (
            next_patterns
        )

    return patterns


def _rho_code(rho: float) -> str:
    for known, code in (
        RHO_CODES.items()
    ):
        if np.isclose(
            rho,
            known,
            rtol=0.0,
            atol=1.0e-12,
        ):
            return code

    raise ValueError(
        f"unsupported rho: {rho}"
    )


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


def _fraction(
    numerator,
    denominator,
):
    if not denominator:
        return 0.0

    return float(
        numerator
        / denominator
    )


def _audit_rho(
    *,
    clean_dataset,
    poison_dataset,
    prepared,
    selection,
    reachability_cache,
    action_low: float,
    action_high: float,
):
    clean_counts = (
        prepared.pattern_frequencies
    )

    poison_labels = _predict_labels(
        poison_dataset,
        prepared.clustering_model,
    )

    selected_windows = (
        selection.selected_windows
    )

    selected_indices = sorted(
        {
            index
            for window
            in selected_windows
            for index
            in window.transition_indices
        }
    )

    reachable_counts = [
        len(
            reachability_cache[
                index
            ]
        )
        for index
        in selected_indices
    ]

    pattern_counts = []

    source_frequencies = []
    actual_frequencies = []
    oracle_frequencies = []

    oracle_pattern_change_possible = 0
    oracle_improvement_possible = 0
    actual_improvement = 0
    actual_attains_oracle = 0
    oracle_search_gap = 0
    actual_pattern_reachable = 0

    for window in selected_windows:
        source_pattern = (
            window.source_pattern
        )

        source_frequency = int(
            clean_counts.get(
                source_pattern,
                0,
            )
        )

        actual_pattern = (
            deduplicate_consecutive(
                poison_labels[
                    window.global_start:
                    window.global_end
                ]
            )
        )

        actual_frequency = int(
            clean_counts.get(
                actual_pattern,
                0,
            )
        )

        reachable_label_sets = [
            reachability_cache[
                index
            ]
            for index
            in window.transition_indices
        ]

        reachable_patterns = (
            _reachable_patterns(
                reachable_label_sets
            )
        )

        if source_pattern not in (
            reachable_patterns
        ):
            raise RuntimeError(
                "source pattern missing from "
                "oracle reachable set"
            )

        if actual_pattern in (
            reachable_patterns
        ):
            actual_pattern_reachable += 1

        oracle_best_frequency = max(
            int(
                clean_counts.get(
                    pattern,
                    0,
                )
            )
            for pattern
            in reachable_patterns
        )

        if (
            oracle_best_frequency
            < source_frequency
        ):
            raise RuntimeError(
                "oracle worse than source pattern"
            )

        changed_possible = any(
            pattern
            != source_pattern
            for pattern
            in reachable_patterns
        )

        oracle_improved = (
            oracle_best_frequency
            > source_frequency
        )

        actual_improved = (
            actual_frequency
            > source_frequency
        )

        attained_oracle = (
            actual_frequency
            == oracle_best_frequency
        )

        search_gap = (
            oracle_improved
            and actual_frequency
            < oracle_best_frequency
        )

        oracle_pattern_change_possible += int(
            changed_possible
        )

        oracle_improvement_possible += int(
            oracle_improved
        )

        actual_improvement += int(
            actual_improved
        )

        actual_attains_oracle += int(
            attained_oracle
        )

        oracle_search_gap += int(
            search_gap
        )

        pattern_counts.append(
            len(
                reachable_patterns
            )
        )

        source_frequencies.append(
            source_frequency
        )

        actual_frequencies.append(
            actual_frequency
        )

        oracle_frequencies.append(
            oracle_best_frequency
        )

    window_count = len(
        selected_windows
    )

    transition_count = len(
        selected_indices
    )

    return {
        "selected_window_count": int(
            window_count
        ),
        "selected_transition_count": int(
            transition_count
        ),
        "reachable_cluster_count_mean": _mean(
            reachable_counts
        ),
        "reachable_cluster_count_max": int(
            max(
                reachable_counts,
                default=0,
            )
        ),
        "multi_cluster_reachable_transition_fraction": (
            _fraction(
                sum(
                    count > 1
                    for count
                    in reachable_counts
                ),
                transition_count,
            )
        ),
        "source_only_reachable_transition_fraction": (
            _fraction(
                sum(
                    count == 1
                    for count
                    in reachable_counts
                ),
                transition_count,
            )
        ),
        "reachable_pattern_count_mean": _mean(
            pattern_counts
        ),
        "reachable_pattern_count_max": int(
            max(
                pattern_counts,
                default=0,
            )
        ),
        "oracle_pattern_change_possible_fraction": (
            _fraction(
                oracle_pattern_change_possible,
                window_count,
            )
        ),
        "oracle_frequency_improvement_possible_fraction": (
            _fraction(
                oracle_improvement_possible,
                window_count,
            )
        ),
        "actual_frequency_improvement_fraction": (
            _fraction(
                actual_improvement,
                window_count,
            )
        ),
        "actual_attains_oracle_best_frequency_fraction": (
            _fraction(
                actual_attains_oracle,
                window_count,
            )
        ),
        "oracle_search_gap_fraction": (
            _fraction(
                oracle_search_gap,
                window_count,
            )
        ),
        "actual_pattern_reachable_fraction": (
            _fraction(
                actual_pattern_reachable,
                window_count,
            )
        ),
        "mean_source_pattern_frequency": _mean(
            source_frequencies
        ),
        "mean_actual_target_frequency": _mean(
            actual_frequencies
        ),
        "mean_oracle_best_frequency": _mean(
            oracle_frequencies
        ),
        "mean_oracle_minus_actual_frequency": _mean(
            [
                oracle
                - actual
                for oracle, actual
                in zip(
                    oracle_frequencies,
                    actual_frequencies,
                )
            ]
        ),
        "reachability_histograms": {
            "reachable_cluster_count": {
                str(count): int(
                    sum(
                        value == count
                        for value
                        in reachable_counts
                    )
                )
                for count
                in sorted(
                    set(
                        reachable_counts
                    )
                )
            }
        },
    }


def _aggregate(
    seed_records,
    *,
    rho: float,
):
    rho_key = (
        f"{rho:.2f}"
    )

    output = {}

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
                    metric
                ]
                for seed
                in sorted(
                    int(value)
                    for value
                    in seed_records
                )
            ],
            dtype=np.float64,
        )

        output[
            metric
        ] = {
            "values_by_seed": {
                str(seed): float(
                    value
                )
                for seed, value
                in zip(
                    sorted(
                        int(value)
                        for value
                        in seed_records
                    ),
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

    return output


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
        "--poison-root",
        type=Path,
        default=DEFAULT_POISON_ROOT,
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

    clean_dataset = (
        load_hdf5_dataset(
            dataset_path
        )
    )

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

    action_low = float(
        config[
            "perturbation_geometry"
        ][
            "action_low"
        ]
    )

    action_high = float(
        config[
            "perturbation_geometry"
        ][
            "action_high"
        ]
    )

    seed_records = {}

    for seed in config[
        "attack_seeds"
    ]:
        seed = int(seed)

        print()
        print("=" * 72)
        print(
            "CSDPC REACHABILITY ORACLE "
            f"— SEED {seed}"
        )
        print("=" * 72)

        prepared = prepare_csdpc_attack(
            clean_dataset,
            attack_seed=seed,
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
                    "num_candidates_reference"
                ]
            ),
        )

        selections = {}

        union_indices = set()

        for rho in config[
            "poison_rates"
        ]:
            rho = float(rho)

            budget = (
                compute_transition_budget(
                    num_transitions=(
                        prepared.num_transitions
                    ),
                    rho=rho,
                )
            )

            selection = (
                select_rare_nonoverlapping_windows(
                    prepared.windows,
                    prepared.pattern_frequencies,
                    transition_budget=budget,
                )
            )

            selections[rho] = (
                selection
            )

            for window in (
                selection.selected_windows
            ):
                union_indices.update(
                    window.transition_indices
                )

        union_indices = sorted(
            union_indices
        )

        print(
            "Unique selected transitions "
            "requiring oracle:",
            len(
                union_indices
            ),
        )

        centers = np.asarray(
            prepared.clustering.centers,
            dtype=np.float64,
        )

        clean_labels = np.asarray(
            prepared.clustering.labels,
            dtype=np.int64,
        )

        reachability_cache = {}

        for position, index in enumerate(
            union_indices,
            start=1,
        ):
            reachability_cache[
                index
            ] = _reachable_clusters(
                observation=(
                    observations[
                        index
                    ]
                ),
                action=(
                    actions[
                        index
                    ]
                ),
                source_cluster=int(
                    clean_labels[
                        index
                    ]
                ),
                centers=centers,
                eta=float(
                    prepared.eta
                ),
                action_low=(
                    action_low
                ),
                action_high=(
                    action_high
                ),
            )

            if (
                position % 1000
                == 0
            ):
                print(
                    "Oracle transitions:",
                    position,
                    "/",
                    len(
                        union_indices
                    ),
                )

        rho_results = {}

        for rho in config[
            "poison_rates"
        ]:
            rho = float(rho)

            code = _rho_code(
                rho
            )

            poison_path = (
                args.poison_root.resolve()
                / (
                    f"rho_{code}"
                    f"_seed_{seed}.hdf5"
                )
            )

            if not poison_path.exists():
                raise FileNotFoundError(
                    f"missing canonical poison: "
                    f"{poison_path}"
                )

            poison_dataset = (
                load_hdf5_dataset(
                    poison_path
                )
            )

            metrics = _audit_rho(
                clean_dataset=(
                    clean_dataset
                ),
                poison_dataset=(
                    poison_dataset
                ),
                prepared=prepared,
                selection=(
                    selections[
                        rho
                    ]
                ),
                reachability_cache=(
                    reachability_cache
                ),
                action_low=(
                    action_low
                ),
                action_high=(
                    action_high
                ),
            )

            rho_results[
                f"{rho:.2f}"
            ] = metrics

        seed_record = {
            "schema_version": (
                "csdpc-reachability-oracle-seed-v1"
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
            "rho_results": (
                rho_results
            ),
        }

        output_path = (
            args.output_root.resolve()
            / f"seed_{seed}.json"
        )

        write_metadata_json(
            output_path,
            seed_record,
        )

        seed_records[
            str(seed)
        ] = seed_record

        gc.collect()

    summary = {
        "schema_version": (
            "csdpc-reachability-oracle-summary-v1"
        ),
        "experiment": config[
            "experiment"
        ],
        "status": config[
            "status"
        ],
        "canonical_attack_is_unchanged": True,
        "dataset_sha256": actual_sha,
        "interpretation": {
            "closed_voronoi_is_upper_bound": True,
            "candidate_search_gap_definition": (
                "oracle can reach a higher clean "
                "pattern frequency than the realized "
                "canonical 100-candidate result"
            ),
        },
        "rho_results": {
            f"{float(rho):.2f}": _aggregate(
                seed_records,
                rho=float(rho),
            )
            for rho
            in config[
                "poison_rates"
            ]
        },
    }

    summary_path = (
        args.output_root.resolve()
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