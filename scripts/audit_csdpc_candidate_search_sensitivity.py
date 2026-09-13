from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.attacks.csdpc.attack import (
    apply_csdpc_attack,
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
    iter_sequence_windows,
)
from src.attacks.csdpc.selection import (
    compute_transition_budget,
    select_rare_nonoverlapping_windows,
)
from src.data.hdf5_io import (
    load_hdf5_dataset,
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
    / "csdpc_candidate_search_sensitivity.json"
)

DEFAULT_POISON_ROOT = (
    ROOT
    / "data"
    / "poisoned"
    / "csdpc"
    / "walker2d-medium-v2"
)

DEFAULT_ORACLE_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "csdpc_reachability_oracle"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "csdpc_candidate_search"
)

EXPECTED_DATASET_SHA256 = (
    "cf00f43add04c17fdfc2958dd581dea"
    "0851b2e5bedbe6fda073758a8f841aeda"
)

POOL_SIZES = (
    100,
    500,
    1000,
)

MAX_EXTENSION_SIZE = 900

POOL_PREFIXES = {
    100: 0,
    500: 400,
    1000: 900,
}

RHO_CODES = {
    0.01: "001",
    0.05: "005",
}

SCALAR_METRICS = (
    "selected_window_count",
    "selected_transition_count",
    "selected_window_pattern_change_fraction",
    "selected_window_frequency_improvement_fraction",
    "selected_window_frequency_improvement_given_changed",
    "modified_transition_cluster_label_change_fraction",
    "mean_source_pattern_frequency",
    "mean_target_pattern_frequency",
    "oracle_frequency_improvement_possible_fraction_reference",
    "binary_oracle_opportunity_capture_ratio",
    "oracle_mean_best_frequency_reference",
    "mean_frequency_gap_to_oracle_reference",
    "candidate_frequency_nonregression_fraction_vs_c100",
    "candidate_frequency_nonregression_fraction_vs_previous_pool",
    "strict_frequency_gain_fraction_vs_previous_pool",
    "selected_source_pattern_type_count",
    "fully_selected_source_pattern_fraction",
    "selected_source_occurrence_completion_fraction",
    "selected_source_pattern_eradication_fraction",
    "selected_source_occurrence_mass_reduction_fraction",
    "clean_to_poison_distinct_pattern_reduction_fraction",
    "removed_pattern_type_count",
    "new_pattern_type_count",
)


@dataclass(frozen=True)
class CandidateChoice:
    target_pattern: tuple
    target_frequency: int
    candidate_index: int
    total_linf_perturbation: float
    labels: tuple


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
) -> str:
    return f"{float(rho):.2f}"


def _rho_code(
    rho: float,
) -> str:
    for known_rho, code in (
        RHO_CODES.items()
    ):
        if np.isclose(
            rho,
            known_rho,
            rtol=0.0,
            atol=1.0e-12,
        ):
            return code

    raise ValueError(
        f"unsupported rho: {rho}"
    )


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
            in window.source_pattern
        ),
    )


def _selected_indices(
    windows,
):
    return sorted(
        {
            int(index)
            for window
            in windows
            for index
            in window.transition_indices
        }
    )


def _selections_form_prefix(
    smaller,
    larger,
) -> bool:
    smaller_keys = [
        _window_key(
            window
        )
        for window
        in smaller
    ]

    larger_keys = [
        _window_key(
            window
        )
        for window
        in larger
    ]

    return (
        smaller_keys
        == larger_keys[
            :len(
                smaller_keys
            )
        ]
    )


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
        != "CSDPC_CANDIDATE_SEARCH_SENSITIVITY"
    ):
        raise ValueError(
            "unexpected experiment identifier"
        )

    if (
        config.get("status")
        != "DIAGNOSTIC_ONLY"
    ):
        raise ValueError(
            "candidate-search sensitivity "
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
        config[
            "num_clusters"
        ]
    ) != 8:
        raise ValueError(
            "frozen diagnostic requires k=8"
        )

    if int(
        config[
            "sequence_length"
        ]
    ) != 5:
        raise ValueError(
            "frozen diagnostic requires "
            "sequence_length=5"
        )

    if not np.isclose(
        float(
            config[
                "eta"
            ]
        ),
        0.05,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "frozen diagnostic requires eta=0.05"
        )

    if tuple(
        int(value)
        for value
        in config[
            "candidate_pools"
        ]
    ) != POOL_SIZES:
        raise ValueError(
            "candidate-pool set changed"
        )

    if list(
        config[
            "attack_seeds"
        ]
    ) != [
        0,
        1,
        2,
    ]:
        raise ValueError(
            "attack-seed set changed"
        )

    if list(
        config[
            "poison_rates"
        ]
    ) != [
        0.01,
        0.05,
    ]:
        raise ValueError(
            "poison-rate set changed"
        )

    design = config[
        "search_design"
    ]

    if (
        design.get("id")
        != "NESTED_UNIFORM_EXTENSION_V1"
    ):
        raise ValueError(
            "search-design identifier changed"
        )

    if int(
        design[
            "canonical_prefix_size"
        ]
    ) != 100:
        raise ValueError(
            "canonical prefix size changed"
        )

    if int(
        design[
            "maximum_extension_size"
        ]
    ) != MAX_EXTENSION_SIZE:
        raise ValueError(
            "maximum extension size changed"
        )

    if not bool(
        design[
            "nested_frequency_nonregression_required"
        ]
    ):
        raise ValueError(
            "nested frequency nonregression "
            "must remain required"
        )

    if set(
        config[
            "metrics"
        ]
    ) != set(
        SCALAR_METRICS
    ):
        raise ValueError(
            "frozen metric set changed"
        )

    return config


def _extension_seed(
    *,
    attack_seed: int,
    trajectory_id: int,
    global_start: int,
) -> int:
    payload = (
        "candidate_search_extension_v1"
        f"|{int(attack_seed)}"
        f"|{int(trajectory_id)}"
        f"|{int(global_start)}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[
            :8
        ],
        byteorder="little",
        signed=False,
    )


def _relative_linf_scales(
    values: np.ndarray,
    eta: float,
):
    return (
        eta
        * np.max(
            np.abs(
                values
            ),
            axis=1,
            keepdims=True,
        )
    )


def _candidate_costs(
    *,
    source_observations,
    source_actions,
    candidate_observations,
    candidate_actions,
):
    state_deltas = (
        candidate_observations
        - source_observations[
            None,
            :,
            :,
        ]
    )

    action_deltas = (
        candidate_actions
        - source_actions[
            None,
            :,
            :,
        ]
    )

    state_cost = np.max(
        np.abs(
            state_deltas
        ),
        axis=2,
    )

    action_cost = np.max(
        np.abs(
            action_deltas
        ),
        axis=2,
    )

    return np.sum(
        state_cost
        + action_cost,
        axis=1,
    )


def _predict_window_labels(
    observations,
    actions,
    kmeans_model,
):
    features = np.concatenate(
        [
            np.asarray(
                observations
            ),
            np.asarray(
                actions
            ),
        ],
        axis=1,
    )

    centers = np.asarray(
        kmeans_model.cluster_centers_
    )

    prediction_features = np.asarray(
        features,
        dtype=centers.dtype,
    )

    return tuple(
        int(value)
        for value
        in kmeans_model.predict(
            prediction_features
        )
    )


def _canonical_choice(
    perturbed_window,
    *,
    kmeans_model,
):
    labels = _predict_window_labels(
        perturbed_window.observations,
        perturbed_window.actions,
        kmeans_model,
    )

    pattern = (
        deduplicate_consecutive(
            labels
        )
    )

    if (
        pattern
        != perturbed_window.target_pattern
    ):
        raise RuntimeError(
            "canonical PerturbedWindow target "
            "does not match predicted labels"
        )

    return CandidateChoice(
        target_pattern=tuple(
            int(value)
            for value
            in pattern
        ),
        target_frequency=int(
            perturbed_window.target_frequency
        ),
        candidate_index=int(
            perturbed_window.candidate_index
        ),
        total_linf_perturbation=float(
            perturbed_window.total_linf_perturbation
        ),
        labels=tuple(
            int(value)
            for value
            in labels
        ),
    )


def _select_nested_choices(
    *,
    baseline: CandidateChoice,
    extra_patterns,
    extra_frequencies,
    extra_costs,
    extra_labels,
):
    if len(
        extra_patterns
    ) != MAX_EXTENSION_SIZE:
        raise ValueError(
            "unexpected extension pattern count"
        )

    if len(
        extra_frequencies
    ) != MAX_EXTENSION_SIZE:
        raise ValueError(
            "unexpected extension frequency count"
        )

    if len(
        extra_costs
    ) != MAX_EXTENSION_SIZE:
        raise ValueError(
            "unexpected extension cost count"
        )

    if len(
        extra_labels
    ) != MAX_EXTENSION_SIZE:
        raise ValueError(
            "unexpected extension label count"
        )

    best = baseline

    best_score = (
        -int(
            best.target_frequency
        ),
        float(
            best.total_linf_perturbation
        ),
        int(
            best.candidate_index
        ),
    )

    outputs = {
        100: baseline
    }

    for extension_index in range(
        MAX_EXTENSION_SIZE
    ):
        global_candidate_index = (
            100
            + extension_index
        )

        candidate_score = (
            -int(
                extra_frequencies[
                    extension_index
                ]
            ),
            float(
                extra_costs[
                    extension_index
                ]
            ),
            int(
                global_candidate_index
            ),
        )

        if (
            candidate_score
            < best_score
        ):
            best = CandidateChoice(
                target_pattern=tuple(
                    int(value)
                    for value
                    in extra_patterns[
                        extension_index
                    ]
                ),
                target_frequency=int(
                    extra_frequencies[
                        extension_index
                    ]
                ),
                candidate_index=int(
                    global_candidate_index
                ),
                total_linf_perturbation=float(
                    extra_costs[
                        extension_index
                    ]
                ),
                labels=tuple(
                    int(value)
                    for value
                    in extra_labels[
                        extension_index
                    ]
                ),
            )

            best_score = (
                candidate_score
            )

        if (
            extension_index
            + 1
            == 400
        ):
            outputs[
                500
            ] = best

    outputs[
        1000
    ] = best

    if (
        outputs[
            500
        ].target_frequency
        < outputs[
            100
        ].target_frequency
    ):
        raise RuntimeError(
            "C500 frequency regressed below C100"
        )

    if (
        outputs[
            1000
        ].target_frequency
        < outputs[
            500
        ].target_frequency
    ):
        raise RuntimeError(
            "C1000 frequency regressed below C500"
        )

    return outputs


def _generate_nested_choices_for_window(
    *,
    clean_dataset,
    selected_window,
    canonical_perturbed_window,
    prepared,
    action_low: float,
    action_high: float,
):
    baseline = _canonical_choice(
        canonical_perturbed_window,
        kmeans_model=(
            prepared.clustering_model
        ),
    )

    start = int(
        selected_window.global_start
    )

    end = int(
        selected_window.global_end
    )

    source_observations = np.asarray(
        clean_dataset[
            "observations"
        ][start:end]
    )

    source_actions = np.asarray(
        clean_dataset[
            "actions"
        ][start:end]
    )

    sequence_length = (
        end
        - start
    )

    if (
        sequence_length
        != prepared.sequence_length
    ):
        raise RuntimeError(
            "unexpected selected-window length"
        )

    state_dim = int(
        source_observations.shape[
            1
        ]
    )

    action_dim = int(
        source_actions.shape[
            1
        ]
    )

    state_scales = (
        _relative_linf_scales(
            source_observations,
            prepared.eta,
        )
    )

    action_scales = (
        _relative_linf_scales(
            source_actions,
            prepared.eta,
        )
    )

    rng = np.random.default_rng(
        _extension_seed(
            attack_seed=(
                prepared.attack_seed
            ),
            trajectory_id=(
                selected_window.trajectory_id
            ),
            global_start=(
                selected_window.global_start
            ),
        )
    )

    # Candidate-major tensor ensures that
    # candidates 0:400 are an exact prefix
    # of candidates 0:900.
    noise = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(
            MAX_EXTENSION_SIZE,
            sequence_length,
            state_dim
            + action_dim,
        ),
    )

    state_noise = (
        noise[
            :,
            :,
            :state_dim,
        ]
    )

    action_noise = (
        noise[
            :,
            :,
            state_dim:,
        ]
    )

    candidate_observations = (
        source_observations[
            None,
            :,
            :,
        ]
        + state_noise
        * state_scales[
            None,
            :,
            :,
        ]
    )

    candidate_actions = (
        source_actions[
            None,
            :,
            :,
        ]
        + action_noise
        * action_scales[
            None,
            :,
            :,
        ]
    )

    candidate_actions = np.clip(
        candidate_actions,
        action_low,
        action_high,
    )

    costs = _candidate_costs(
        source_observations=(
            source_observations
        ),
        source_actions=(
            source_actions
        ),
        candidate_observations=(
            candidate_observations
        ),
        candidate_actions=(
            candidate_actions
        ),
    )

    decision_units = np.concatenate(
        [
            candidate_observations,
            candidate_actions,
        ],
        axis=2,
    )

    centers = np.asarray(
        prepared.clustering_model.cluster_centers_
    )

    prediction_units = np.asarray(
        decision_units,
        dtype=centers.dtype,
    )

    predicted_labels = np.asarray(
        prepared.clustering_model.predict(
            prediction_units.reshape(
                MAX_EXTENSION_SIZE
                * sequence_length,
                prediction_units.shape[
                    2
                ],
            )
        ),
        dtype=np.int64,
    ).reshape(
        MAX_EXTENSION_SIZE,
        sequence_length,
    )

    patterns = [
        deduplicate_consecutive(
            predicted_labels[
                index
            ]
        )
        for index
        in range(
            MAX_EXTENSION_SIZE
        )
    ]

    frequencies = np.asarray(
        [
            int(
                prepared.pattern_frequencies.get(
                    pattern,
                    0,
                )
            )
            for pattern
            in patterns
        ],
        dtype=np.int64,
    )

    return _select_nested_choices(
        baseline=baseline,
        extra_patterns=patterns,
        extra_frequencies=frequencies,
        extra_costs=costs,
        extra_labels=(
            predicted_labels
        ),
    )


def _build_poison_labels(
    *,
    clean_labels,
    selected_windows,
    choices,
):
    if (
        len(
            selected_windows
        )
        != len(
            choices
        )
    ):
        raise ValueError(
            "window/choice count mismatch"
        )

    poison_labels = np.asarray(
        clean_labels,
        dtype=np.int64,
    ).copy()

    used = set()

    for window, choice in zip(
        selected_windows,
        choices,
    ):
        start = int(
            window.global_start
        )

        end = int(
            window.global_end
        )

        if (
            len(
                choice.labels
            )
            != end - start
        ):
            raise RuntimeError(
                "choice label count does not "
                "match selected-window length"
            )

        indices = range(
            start,
            end,
        )

        if any(
            index in used
            for index
            in indices
        ):
            raise RuntimeError(
                "selected windows overlap"
            )

        used.update(
            range(
                start,
                end,
            )
        )

        poison_labels[
            start:end
        ] = np.asarray(
            choice.labels,
            dtype=np.int64,
        )

    return poison_labels


def _compute_condition_metrics(
    *,
    clean_labels,
    trajectories,
    selected_windows,
    choices,
    c100_choices,
    previous_choices,
    clean_counts,
    sequence_length: int,
    oracle_reference,
):
    if not (
        len(
            selected_windows
        )
        == len(
            choices
        )
        == len(
            c100_choices
        )
        == len(
            previous_choices
        )
    ):
        raise ValueError(
            "condition input lengths differ"
        )

    poison_labels = (
        _build_poison_labels(
            clean_labels=(
                clean_labels
            ),
            selected_windows=(
                selected_windows
            ),
            choices=choices,
        )
    )

    poison_counts = Counter(
        window.pattern
        for window
        in iter_sequence_windows(
            poison_labels,
            trajectories,
            sequence_length=(
                sequence_length
            ),
        )
    )

    selected_window_count = len(
        selected_windows
    )

    selected_indices = (
        _selected_indices(
            selected_windows
        )
    )

    selected_transition_count = len(
        selected_indices
    )

    source_frequencies = []

    target_frequencies = []

    pattern_changed = 0
    frequency_improved = 0
    changed_and_improved = 0

    for window, choice in zip(
        selected_windows,
        choices,
    ):
        source_pattern = tuple(
            window.source_pattern
        )

        source_frequency = int(
            clean_counts.get(
                source_pattern,
                0,
            )
        )

        target_frequency = int(
            choice.target_frequency
        )

        changed = (
            choice.target_pattern
            != source_pattern
        )

        improved = (
            target_frequency
            > source_frequency
        )

        pattern_changed += int(
            changed
        )

        frequency_improved += int(
            improved
        )

        changed_and_improved += int(
            changed
            and improved
        )

        source_frequencies.append(
            source_frequency
        )

        target_frequencies.append(
            target_frequency
        )

    if selected_indices:
        selected_indices_array = (
            np.asarray(
                selected_indices,
                dtype=np.int64,
            )
        )

        cluster_change_fraction = float(
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
        cluster_change_fraction = 0.0

    selected_source_patterns = {
        tuple(
            window.source_pattern
        )
        for window
        in selected_windows
    }

    selected_occurrence_counts = Counter(
        tuple(
            window.source_pattern
        )
        for window
        in selected_windows
    )

    selected_source_total_clean_occurrences = sum(
        int(
            clean_counts[
                pattern
            ]
        )
        for pattern
        in selected_source_patterns
    )

    fully_selected_source_pattern_count = sum(
        1
        for pattern
        in selected_source_patterns
        if (
            selected_occurrence_counts[
                pattern
            ]
            == clean_counts[
                pattern
            ]
        )
    )

    selected_source_pattern_count = len(
        selected_source_patterns
    )

    fully_selected_source_pattern_fraction = (
        _safe_fraction(
            fully_selected_source_pattern_count,
            selected_source_pattern_count,
        )
    )

    occurrence_completion_fraction = (
        _safe_fraction(
            selected_window_count,
            selected_source_total_clean_occurrences,
        )
    )

    eradicated_source_patterns = sum(
        1
        for pattern
        in selected_source_patterns
        if (
            poison_counts.get(
                pattern,
                0,
            )
            == 0
        )
    )

    source_mass_before = sum(
        int(
            clean_counts.get(
                pattern,
                0,
            )
        )
        for pattern
        in selected_source_patterns
    )

    source_mass_after = sum(
        int(
            poison_counts.get(
                pattern,
                0,
            )
        )
        for pattern
        in selected_source_patterns
    )

    source_mass_reduction = (
        1.0
        - float(
            source_mass_after
        )
        / float(
            source_mass_before
        )
        if source_mass_before
        else 0.0
    )

    clean_pattern_types = set(
        clean_counts
    )

    poison_pattern_types = set(
        poison_counts
    )

    clean_distinct = len(
        clean_pattern_types
    )

    poison_distinct = len(
        poison_pattern_types
    )

    global_distinct_reduction = (
        1.0
        - float(
            poison_distinct
        )
        / float(
            clean_distinct
        )
        if clean_distinct
        else 0.0
    )

    current_frequencies = np.asarray(
        [
            choice.target_frequency
            for choice
            in choices
        ],
        dtype=np.int64,
    )

    c100_frequencies = np.asarray(
        [
            choice.target_frequency
            for choice
            in c100_choices
        ],
        dtype=np.int64,
    )

    previous_frequencies = np.asarray(
        [
            choice.target_frequency
            for choice
            in previous_choices
        ],
        dtype=np.int64,
    )

    if len(
        current_frequencies
    ):
        nonregression_vs_c100 = float(
            np.mean(
                current_frequencies
                >= c100_frequencies
            )
        )

        nonregression_vs_previous = float(
            np.mean(
                current_frequencies
                >= previous_frequencies
            )
        )

        strict_gain_vs_previous = float(
            np.mean(
                current_frequencies
                > previous_frequencies
            )
        )
    else:
        nonregression_vs_c100 = 1.0
        nonregression_vs_previous = 1.0
        strict_gain_vs_previous = 0.0

    if (
        nonregression_vs_c100
        != 1.0
    ):
        raise RuntimeError(
            "nested search regressed "
            "relative to C100"
        )

    if (
        nonregression_vs_previous
        != 1.0
    ):
        raise RuntimeError(
            "nested search regressed "
            "relative to previous pool"
        )

    actual_improvement_fraction = (
        _safe_fraction(
            frequency_improved,
            selected_window_count,
        )
    )

    oracle_improvement_possible = float(
        oracle_reference[
            "oracle_frequency_improvement_possible_fraction"
        ]
    )

    binary_capture_ratio = (
        _safe_fraction(
            actual_improvement_fraction,
            oracle_improvement_possible,
        )
    )

    if (
        binary_capture_ratio
        > 1.0
        + 1.0e-10
    ):
        raise RuntimeError(
            "realized improvement exceeds "
            "oracle opportunity bound"
        )

    mean_target_frequency = _mean(
        target_frequencies
    )

    oracle_mean_best_frequency = float(
        oracle_reference[
            "mean_oracle_best_frequency"
        ]
    )

    mean_frequency_gap = (
        oracle_mean_best_frequency
        - mean_target_frequency
    )

    if (
        mean_frequency_gap
        < -1.0e-8
    ):
        raise RuntimeError(
            "mean realized frequency exceeds "
            "oracle upper bound"
        )

    return {
        "selected_window_count": int(
            selected_window_count
        ),

        "selected_transition_count": int(
            selected_transition_count
        ),

        "selected_window_pattern_change_fraction": (
            _safe_fraction(
                pattern_changed,
                selected_window_count,
            )
        ),

        "selected_window_frequency_improvement_fraction": (
            actual_improvement_fraction
        ),

        "selected_window_frequency_improvement_given_changed": (
            _safe_fraction(
                changed_and_improved,
                pattern_changed,
            )
        ),

        "modified_transition_cluster_label_change_fraction": (
            cluster_change_fraction
        ),

        "mean_source_pattern_frequency": (
            _mean(
                source_frequencies
            )
        ),

        "mean_target_pattern_frequency": (
            mean_target_frequency
        ),

        "oracle_frequency_improvement_possible_fraction_reference": (
            oracle_improvement_possible
        ),

        "binary_oracle_opportunity_capture_ratio": (
            binary_capture_ratio
        ),

        "oracle_mean_best_frequency_reference": (
            oracle_mean_best_frequency
        ),

        "mean_frequency_gap_to_oracle_reference": (
            mean_frequency_gap
        ),

        "candidate_frequency_nonregression_fraction_vs_c100": (
            nonregression_vs_c100
        ),

        "candidate_frequency_nonregression_fraction_vs_previous_pool": (
            nonregression_vs_previous
        ),

        "strict_frequency_gain_fraction_vs_previous_pool": (
            strict_gain_vs_previous
        ),

        "selected_source_pattern_type_count": int(
            selected_source_pattern_count
        ),

        "fully_selected_source_pattern_fraction": float(
            fully_selected_source_pattern_fraction
        ),

        "selected_source_occurrence_completion_fraction": float(
            occurrence_completion_fraction
        ),

        "selected_source_pattern_eradication_fraction": (
            _safe_fraction(
                eradicated_source_patterns,
                selected_source_pattern_count,
            )
        ),

        "selected_source_occurrence_mass_reduction_fraction": float(
            source_mass_reduction
        ),

        "clean_to_poison_distinct_pattern_reduction_fraction": float(
            global_distinct_reduction
        ),

        "removed_pattern_type_count": int(
            len(
                clean_pattern_types
                - poison_pattern_types
            )
        ),

        "new_pattern_type_count": int(
            len(
                poison_pattern_types
                - clean_pattern_types
            )
        ),
    }


def _load_oracle_seed(
    *,
    oracle_root: Path,
    seed: int,
):
    path = (
        oracle_root
        / f"seed_{seed}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"oracle seed result missing: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _canonical_selected_rows_match(
    *,
    canonical_dataset,
    regenerated_dataset,
    selected_windows,
):
    indices = _selected_indices(
        selected_windows
    )

    if not indices:
        return True

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    return bool(
        np.array_equal(
            np.asarray(
                canonical_dataset[
                    "observations"
                ]
            )[
                indices
            ],
            np.asarray(
                regenerated_dataset[
                    "observations"
                ]
            )[
                indices
            ],
        )
        and np.array_equal(
            np.asarray(
                canonical_dataset[
                    "actions"
                ]
            )[
                indices
            ],
            np.asarray(
                regenerated_dataset[
                    "actions"
                ]
            )[
                indices
            ],
        )
    )


def _run_seed(
    *,
    seed: int,
    config,
    clean_dataset,
    poison_root: Path,
    oracle_root: Path,
):
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
        num_candidates=100,
    )

    selections = {}

    for rho in config[
        "poison_rates"
    ]:
        rho = float(
            rho
        )

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
                transition_budget=(
                    budget
                ),
            )
        )

        selections[
            rho
        ] = selection

    rho_001_selection = (
        selections[
            0.01
        ].selected_windows
    )

    rho_005_selection = (
        selections[
            0.05
        ].selected_windows
    )

    selection_prefix_verified = (
        _selections_form_prefix(
            rho_001_selection,
            rho_005_selection,
        )
    )

    if not selection_prefix_verified:
        raise RuntimeError(
            "rho=0.01 selection is not an "
            "exact prefix of rho=0.05 selection"
        )

    print(
        "Selection-prefix check: PASS"
    )

    print(
        "Generating canonical C100 rho=0.05 "
        "reference in memory..."
    )

    canonical_max_result = (
        apply_csdpc_attack(
            clean_dataset,
            prepared,
            rho=0.05,
        )
    )

    if [
        _window_key(
            window
        )
        for window
        in canonical_max_result.selected_windows
    ] != [
        _window_key(
            window
        )
        for window
        in rho_005_selection
    ]:
        raise RuntimeError(
            "canonical apply_csdpc_attack selection "
            "does not match frozen selector"
        )

    max_windows = (
        rho_005_selection
    )

    canonical_perturbed = (
        canonical_max_result.perturbed_windows
    )

    if (
        len(
            max_windows
        )
        != len(
            canonical_perturbed
        )
    ):
        raise RuntimeError(
            "canonical perturbation count mismatch"
        )

    choices_by_pool = {
        pool_size: []
        for pool_size
        in POOL_SIZES
    }

    print(
        "Generating nested extension pool "
        f"for {len(max_windows)} windows..."
    )

    for position, (
        selected_window,
        perturbed_window,
    ) in enumerate(
        zip(
            max_windows,
            canonical_perturbed,
        ),
        start=1,
    ):
        nested_choices = (
            _generate_nested_choices_for_window(
                clean_dataset=(
                    clean_dataset
                ),
                selected_window=(
                    selected_window
                ),
                canonical_perturbed_window=(
                    perturbed_window
                ),
                prepared=prepared,
                action_low=-1.0,
                action_high=1.0,
            )
        )

        for pool_size in (
            POOL_SIZES
        ):
            choices_by_pool[
                pool_size
            ].append(
                nested_choices[
                    pool_size
                ]
            )

        if (
            position
            % 250
            == 0
        ):
            print(
                "Candidate-search windows:",
                position,
                "/",
                len(
                    max_windows
                ),
            )

    clean_labels = np.asarray(
        prepared.clustering.labels,
        dtype=np.int64,
    )

    terminals = np.asarray(
        clean_dataset[
            "terminals"
        ]
    )

    timeouts = np.asarray(
        clean_dataset[
            "timeouts"
        ]
    )

    trajectories, _ = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    clean_counts = (
        prepared.pattern_frequencies
    )

    oracle_seed = (
        _load_oracle_seed(
            oracle_root=(
                oracle_root
            ),
            seed=seed,
        )
    )

    rho_results = {}

    for rho in config[
        "poison_rates"
    ]:
        rho = float(
            rho
        )

        rho_key = _rho_key(
            rho
        )

        selection_windows = (
            selections[
                rho
            ].selected_windows
        )

        num_windows = len(
            selection_windows
        )

        oracle_reference = (
            oracle_seed[
                "rho_results"
            ][
                rho_key
            ]
        )

        canonical_path = (
            poison_root
            / (
                "rho_"
                + _rho_code(
                    rho
                )
                + f"_seed_{seed}.hdf5"
            )
        )

        if not canonical_path.exists():
            raise FileNotFoundError(
                "missing canonical C100 artifact: "
                f"{canonical_path}"
            )

        canonical_dataset = (
            load_hdf5_dataset(
                canonical_path
            )
        )

        c100_rows_match = (
            _canonical_selected_rows_match(
                canonical_dataset=(
                    canonical_dataset
                ),
                regenerated_dataset=(
                    canonical_max_result.poisoned_dataset
                ),
                selected_windows=(
                    selection_windows
                ),
            )
        )

        if not c100_rows_match:
            raise RuntimeError(
                "regenerated C100 selected rows "
                "do not match canonical artifact"
            )

        pool_results = {}

        previous_pool_size = 100

        for pool_size in (
            POOL_SIZES
        ):
            choices = (
                choices_by_pool[
                    pool_size
                ][
                    :num_windows
                ]
            )

            c100_choices = (
                choices_by_pool[
                    100
                ][
                    :num_windows
                ]
            )

            previous_choices = (
                choices_by_pool[
                    previous_pool_size
                ][
                    :num_windows
                ]
            )

            metrics = (
                _compute_condition_metrics(
                    clean_labels=(
                        clean_labels
                    ),
                    trajectories=(
                        trajectories
                    ),
                    selected_windows=(
                        selection_windows
                    ),
                    choices=choices,
                    c100_choices=(
                        c100_choices
                    ),
                    previous_choices=(
                        previous_choices
                    ),
                    clean_counts=(
                        clean_counts
                    ),
                    sequence_length=(
                        prepared.sequence_length
                    ),
                    oracle_reference=(
                        oracle_reference
                    ),
                )
            )

            if (
                pool_size
                == 100
            ):
                expected_improvement = float(
                    oracle_reference[
                        "actual_frequency_improvement_fraction"
                    ]
                )

                expected_target_mean = float(
                    oracle_reference[
                        "mean_actual_target_frequency"
                    ]
                )

                if not np.isclose(
                    metrics[
                        "selected_window_frequency_improvement_fraction"
                    ],
                    expected_improvement,
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    raise RuntimeError(
                        "C100 frequency-improvement "
                        "fraction does not match oracle "
                        "canonical reference"
                    )

                if not np.isclose(
                    metrics[
                        "mean_target_pattern_frequency"
                    ],
                    expected_target_mean,
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    raise RuntimeError(
                        "C100 mean target frequency "
                        "does not match oracle "
                        "canonical reference"
                    )

            pool_results[
                str(
                    pool_size
                )
            ] = metrics

            previous_pool_size = (
                pool_size
            )

        rho_results[
            rho_key
        ] = {
            "integrity": {
                "C100_selected_rows_match_canonical_artifact": (
                    c100_rows_match
                ),
                "C100_matches_oracle_reference": True,
            },
            "candidate_pools": (
                pool_results
            ),
        }

        del canonical_dataset
        gc.collect()

    return {
        "schema_version": (
            "csdpc-candidate-search-seed-v1"
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
        "integrity": {
            "rho_001_is_prefix_of_rho_005": bool(
                selection_prefix_verified
            ),
            "nested_frequency_nonregression": True,
        },
        "rho_results": (
            rho_results
        ),
    }


def _build_summary(
    *,
    config,
    seed_records,
):
    summary_rhos = {}

    seeds = [
        int(seed)
        for seed
        in config[
            "attack_seeds"
        ]
    ]

    for rho in config[
        "poison_rates"
    ]:
        rho_key = _rho_key(
            float(
                rho
            )
        )

        pool_summary = {}

        for pool_size in (
            POOL_SIZES
        ):
            pool_key = str(
                pool_size
            )

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
                            "candidate_pools"
                        ][
                            pool_key
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

            pool_summary[
                pool_key
            ] = metric_summary

        summary_rhos[
            rho_key
        ] = {
            "candidate_pools": (
                pool_summary
            )
        }

    return {
        "schema_version": (
            "csdpc-candidate-search-summary-v1"
        ),
        "experiment": config[
            "experiment"
        ],
        "status": config[
            "status"
        ],
        "canonical_attack_is_unchanged": True,
        "attack_seeds": seeds,
        "candidate_pools": list(
            POOL_SIZES
        ),
        "rho_results": (
            summary_rhos
        ),
        "interpretation_rules": {
            "C100_is_canonical_reference": True,
            "C500_and_C1000_are_diagnostic_only": True,
            "candidate_pools_are_nested": True,
            "no_learner_result_used_for_selection": True,
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
                "missing candidate-search seed result: "
                f"{path}"
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
        "--poison-root",
        type=Path,
        default=DEFAULT_POISON_ROOT,
    )

    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=DEFAULT_ORACLE_ROOT,
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

    if (
        args.summarize_only
    ):
        seed_records = (
            _load_seed_records(
                output_root=(
                    output_root
                ),
                seeds=(
                    config[
                        "attack_seeds"
                    ]
                ),
            )
        )

        summary = _build_summary(
            config=config,
            seed_records=(
                seed_records
            ),
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
                "requested seed is not in "
                "the frozen seed set"
            )

        seeds = [
            int(
                args.seed
            )
        ]

    dataset_path = (
        args.dataset.resolve()
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"dataset missing: {dataset_path}"
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

    for seed in seeds:
        print()
        print(
            "=" * 72
        )
        print(
            "CSDPC CANDIDATE SEARCH "
            f"SENSITIVITY — SEED {seed}"
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
            poison_root=(
                args.poison_root.resolve()
            ),
            oracle_root=(
                args.oracle_root.resolve()
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