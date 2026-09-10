from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.audit_csdpc_overlap_conflict import (
    WindowProposal as ConflictWindowProposal,
    _compute_conflict_metrics,
)
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
    iter_sequence_windows,
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
    / "csdpc_overlap_resolution_sensitivity.json"
)

DEFAULT_CONFLICT_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "csdpc_overlap_conflict"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "csdpc_overlap_resolution"
)

EXPECTED_DATASET_SHA256 = (
    "cf00f43add04c17fdfc2958dd581dea"
    "0851b2e5bedbe6fda073758a8f841aeda"
)

RULE_IDS = (
    "R0_FIRST_RARE_WINS",
    "R1_LAST_WINS",
    "R2_MODE_LABEL_FIRST",
)

SCALAR_METRICS = (
    "selected_window_count",
    "unique_transition_footprint",
    "actual_modified_transition_count",
    "actual_modified_fraction_of_selected_footprint",

    "proposal_slot_target_label_preservation_fraction",
    "window_exact_independent_target_pattern_preservation_fraction",

    "merged_window_pattern_change_fraction",
    "merged_window_frequency_improvement_fraction",
    "merged_frequency_improvement_given_changed",

    "modified_transition_cluster_label_change_fraction",

    "mean_source_pattern_frequency",
    "mean_independent_target_pattern_frequency",
    "mean_merged_target_pattern_frequency",
    "mean_frequency_delta_vs_independent",

    "merged_frequency_at_least_independent_fraction",
    "merged_frequency_strictly_better_than_independent_fraction",

    "selected_source_pattern_type_count",
    "selected_source_pattern_eradication_fraction",
    "selected_source_occurrence_mass_reduction_fraction",

    "clean_to_poison_distinct_pattern_reduction_fraction",
    "removed_pattern_type_count",
    "new_pattern_type_count",

    "state_perturbation_bound_valid_fraction",
    "action_perturbation_bound_valid_fraction",
    "action_rows_within_environment_bounds_fraction",
)

PAIRWISE_METRICS = (
    "merged_target_label_disagreement_fraction",
    "merged_continuous_row_disagreement_fraction",
)


@dataclass(frozen=True)
class ResolutionProposal:
    raw_target_labels: tuple[int, ...]
    target_pattern: tuple[int, ...]
    source_frequency: int
    target_frequency: int
    candidate_index: int
    observations: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True)
class ProposalSlot:
    window_id: int
    target_label: int
    observation: np.ndarray
    action: np.ndarray


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
        != "CSDPC_OVERLAP_RESOLUTION_SENSITIVITY"
    ):
        raise ValueError(
            "unexpected experiment identifier"
        )

    if (
        config.get("status")
        != "DIAGNOSTIC_ONLY"
    ):
        raise ValueError(
            "overlap-resolution experiment "
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

    if (
        config.get("selection_reference")
        != "S2_PATTERN_TYPE_ATOMIC_PREFIX"
    ):
        raise ValueError(
            "selection reference changed"
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

    if not np.isclose(
        float(
            config["eta"]
        ),
        0.05,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "frozen diagnostic requires eta=0.05"
        )

    if int(
        config["num_candidates"]
    ) != 100:
        raise ValueError(
            "frozen diagnostic requires C100"
        )

    if list(
        config["attack_seeds"]
    ) != [
        0,
        1,
        2,
    ]:
        raise ValueError(
            "frozen seed set changed"
        )

    if list(
        config["poison_rates"]
    ) != [
        0.01,
        0.05,
    ]:
        raise ValueError(
            "frozen rho set changed"
        )

    observed_rules = tuple(
        rule["id"]
        for rule
        in config["resolution_rules"]
    )

    if (
        observed_rules
        != RULE_IDS
    ):
        raise ValueError(
            "frozen resolution-rule set changed"
        )

    if (
        "metrics"
        in config
        and set(
            config["metrics"]
        )
        != set(
            SCALAR_METRICS
        )
    ):
        raise ValueError(
            "frozen scalar metric set changed"
        )

    return config


def _to_selected_window(
    window,
):
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


def _is_prefix(
    smaller,
    larger,
):
    small = [
        _window_key(
            window
        )
        for window
        in smaller
    ]

    large = [
        _window_key(
            window
        )
        for window
        in larger
    ]

    return (
        small
        == large[
            :len(
                small
            )
        ]
    )


def _predict_window_labels(
    perturbed,
    model,
):
    features = (
        build_raw_decision_units(
            np.asarray(
                perturbed.observations
            ),
            np.asarray(
                perturbed.actions
            ),
        )
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
        perturbed = (
            perturb_selected_window(
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
                num_candidates=(
                    prepared.num_candidates
                ),
                rng=rng,
                action_low=-1.0,
                action_high=1.0,
            )
        )

        labels = (
            _predict_window_labels(
                perturbed,
                prepared.clustering_model,
            )
        )

        pattern = (
            deduplicate_consecutive(
                labels
            )
        )

        if (
            tuple(pattern)
            != tuple(
                perturbed.target_pattern
            )
        ):
            raise RuntimeError(
                "proposal prediction does not "
                "reproduce target pattern"
            )

        proposals.append(
            ResolutionProposal(
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
                observations=np.asarray(
                    perturbed.observations
                ).copy(),
                actions=np.asarray(
                    perturbed.actions
                ).copy(),
            )
        )

        if (
            position
            % 500
            == 0
        ):
            print(
                "Independent C100 proposals:",
                position,
                "/",
                len(
                    selected_windows
                ),
            )

    return tuple(
        proposals
    )


def _as_conflict_proposals(
    proposals,
):
    return tuple(
        ConflictWindowProposal(
            raw_target_labels=(
                proposal.raw_target_labels
            ),
            target_pattern=(
                proposal.target_pattern
            ),
            source_frequency=int(
                proposal.source_frequency
            ),
            target_frequency=int(
                proposal.target_frequency
            ),
            candidate_index=int(
                proposal.candidate_index
            ),
        )
        for proposal
        in proposals
    )


def _verify_conflict_reference(
    *,
    selected_windows,
    proposals,
    reference,
):
    observed = (
        _compute_conflict_metrics(
            selected_windows=(
                selected_windows
            ),
            proposals=(
                _as_conflict_proposals(
                    proposals
                )
            ),
        )
    )

    for key, expected in (
        reference.items()
    ):
        if key not in observed:
            raise RuntimeError(
                "conflict-reference metric "
                f"missing from regenerated result: {key}"
            )

        actual = observed[
            key
        ]

        if not np.isclose(
            float(actual),
            float(expected),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "regenerated independent proposals "
                "do not reproduce conflict diagnostic: "
                f"{key}: actual={actual}, "
                f"expected={expected}"
            )


def _build_requirements(
    *,
    selected_windows,
    proposals,
):
    if (
        len(
            selected_windows
        )
        != len(
            proposals
        )
    ):
        raise ValueError(
            "window/proposal count mismatch"
        )

    requirements = defaultdict(
        list
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
        start = int(
            window.global_start
        )

        end = int(
            window.global_end
        )

        length = (
            end
            - start
        )

        if (
            len(
                proposal.raw_target_labels
            )
            != length
        ):
            raise RuntimeError(
                "raw target-label length mismatch"
            )

        if (
            proposal.observations.shape[
                0
            ]
            != length
        ):
            raise RuntimeError(
                "proposal observation length mismatch"
            )

        if (
            proposal.actions.shape[
                0
            ]
            != length
        ):
            raise RuntimeError(
                "proposal action length mismatch"
            )

        for local_index in range(
            length
        ):
            global_index = (
                start
                + local_index
            )

            requirements[
                global_index
            ].append(
                ProposalSlot(
                    window_id=int(
                        window_id
                    ),
                    target_label=int(
                        proposal.raw_target_labels[
                            local_index
                        ]
                    ),
                    observation=np.asarray(
                        proposal.observations[
                            local_index
                        ]
                    ).copy(),
                    action=np.asarray(
                        proposal.actions[
                            local_index
                        ]
                    ).copy(),
                )
            )

    return dict(
        requirements
    )


def _choose_slot(
    requirements,
    *,
    rule_id: str,
):
    if not requirements:
        raise ValueError(
            "cannot resolve empty requirement list"
        )

    ordered = sorted(
        requirements,
        key=lambda slot: (
            slot.window_id
        ),
    )

    if (
        rule_id
        == "R0_FIRST_RARE_WINS"
    ):
        return ordered[
            0
        ]

    if (
        rule_id
        == "R1_LAST_WINS"
    ):
        return ordered[
            -1
        ]

    if (
        rule_id
        == "R2_MODE_LABEL_FIRST"
    ):
        counts = Counter(
            slot.target_label
            for slot
            in ordered
        )

        labels = sorted(
            counts
        )

        best_label = min(
            labels,
            key=lambda label: (
                -counts[
                    label
                ],
                min(
                    slot.window_id
                    for slot
                    in ordered
                    if (
                        slot.target_label
                        == label
                    )
                ),
                label,
            ),
        )

        for slot in ordered:
            if (
                slot.target_label
                == best_label
            ):
                return slot

        raise RuntimeError(
            "mode label has no supporting slot"
        )

    raise ValueError(
        f"unknown resolution rule: {rule_id}"
    )


def _resolve_requirements(
    requirements,
    *,
    rule_id: str,
):
    return {
        int(index): _choose_slot(
            slots,
            rule_id=rule_id,
        )
        for index, slots
        in requirements.items()
    }


def _build_merged_dataset(
    *,
    clean_dataset,
    resolved,
):
    merged = {
        key: np.asarray(
            value
        ).copy()
        for key, value
        in clean_dataset.items()
    }

    for index, slot in (
        resolved.items()
    ):
        merged[
            "observations"
        ][
            index
        ] = (
            slot.observation
        )

        merged[
            "actions"
        ][
            index
        ] = (
            slot.action
        )

    return merged


def _predict_selected_labels(
    *,
    merged_dataset,
    resolved,
    model,
    clean_labels,
):
    indices = sorted(
        resolved
    )

    labels = np.asarray(
        clean_labels,
        dtype=np.int64,
    ).copy()

    if not indices:
        return (
            labels,
            tuple(),
        )

    index_array = np.asarray(
        indices,
        dtype=np.int64,
    )

    features = (
        build_raw_decision_units(
            np.asarray(
                merged_dataset[
                    "observations"
                ]
            )[
                index_array
            ],
            np.asarray(
                merged_dataset[
                    "actions"
                ]
            )[
                index_array
            ],
        )
    )

    centers = np.asarray(
        model.cluster_centers_
    )

    features = np.asarray(
        features,
        dtype=centers.dtype,
    )

    predicted = np.asarray(
        model.predict(
            features
        ),
        dtype=np.int64,
    )

    for index, label in zip(
        indices,
        predicted,
    ):
        expected = int(
            resolved[
                index
            ].target_label
        )

        if int(
            label
        ) != expected:
            raise RuntimeError(
                "merged continuous proposal does not "
                "reproduce selected target label: "
                f"transition={index}, "
                f"predicted={int(label)}, "
                f"expected={expected}"
            )

    labels[
        index_array
    ] = predicted

    return (
        labels,
        tuple(
            indices
        ),
    )


def _perturbation_integrity(
    *,
    clean_dataset,
    merged_dataset,
    selected_indices,
    eta: float,
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

    merged_observations = np.asarray(
        merged_dataset[
            "observations"
        ]
    )

    merged_actions = np.asarray(
        merged_dataset[
            "actions"
        ]
    )

    selected_indices = np.asarray(
        selected_indices,
        dtype=np.int64,
    )

    if len(
        selected_indices
    ) == 0:
        return {
            "actual_modified_transition_count": 0,
            "actual_modified_fraction_of_selected_footprint": 0.0,
            "state_perturbation_bound_valid_fraction": 1.0,
            "action_perturbation_bound_valid_fraction": 1.0,
            "action_rows_within_environment_bounds_fraction": 1.0,
            "nonselected_attack_rows_identical": True,
            "all_nonattack_arrays_identical": True,
        }

    clean_obs = (
        observations[
            selected_indices
        ]
    )

    clean_act = (
        actions[
            selected_indices
        ]
    )

    poison_obs = (
        merged_observations[
            selected_indices
        ]
    )

    poison_act = (
        merged_actions[
            selected_indices
        ]
    )

    obs_changed = np.any(
        clean_obs
        != poison_obs,
        axis=1,
    )

    act_changed = np.any(
        clean_act
        != poison_act,
        axis=1,
    )

    modified = (
        obs_changed
        | act_changed
    )

    allowed_state = (
        eta
        * np.max(
            np.abs(
                clean_obs
            ),
            axis=1,
        )
    )

    actual_state = np.max(
        np.abs(
            poison_obs
            - clean_obs
        ),
        axis=1,
    )

    state_valid = (
        actual_state
        <= allowed_state
        + 1.0e-10
    )

    allowed_action = (
        eta
        * np.max(
            np.abs(
                clean_act
            ),
            axis=1,
        )
    )

    actual_action = np.max(
        np.abs(
            poison_act
            - clean_act
        ),
        axis=1,
    )

    action_valid = (
        actual_action
        <= allowed_action
        + 1.0e-10
    )

    action_env_valid = (
        np.all(
            poison_act
            >= -1.0
            - 1.0e-10,
            axis=1,
        )
        & np.all(
            poison_act
            <= 1.0
            + 1.0e-10,
            axis=1,
        )
    )

    selected_mask = np.zeros(
        len(
            observations
        ),
        dtype=bool,
    )

    selected_mask[
        selected_indices
    ] = True

    nonselected_attack_rows_identical = bool(
        np.array_equal(
            observations[
                ~selected_mask
            ],
            merged_observations[
                ~selected_mask
            ],
        )
        and np.array_equal(
            actions[
                ~selected_mask
            ],
            merged_actions[
                ~selected_mask
            ],
        )
    )

    all_nonattack_arrays_identical = bool(
        all(
            np.array_equal(
                np.asarray(
                    clean_dataset[
                        key
                    ]
                ),
                np.asarray(
                    merged_dataset[
                        key
                    ]
                ),
            )
            for key
            in clean_dataset
            if key not in {
                "observations",
                "actions",
            }
        )
    )

    return {
        "actual_modified_transition_count": int(
            np.sum(
                modified
            )
        ),

        "actual_modified_fraction_of_selected_footprint": (
            float(
                np.mean(
                    modified
                )
            )
        ),

        "state_perturbation_bound_valid_fraction": float(
            np.mean(
                state_valid
            )
        ),

        "action_perturbation_bound_valid_fraction": float(
            np.mean(
                action_valid
            )
        ),

        "action_rows_within_environment_bounds_fraction": float(
            np.mean(
                action_env_valid
            )
        ),

        "nonselected_attack_rows_identical": (
            nonselected_attack_rows_identical
        ),

        "all_nonattack_arrays_identical": (
            all_nonattack_arrays_identical
        ),
    }


def _compute_rule_metrics(
    *,
    rule_id,
    clean_dataset,
    merged_dataset,
    clean_labels,
    merged_labels,
    selected_windows,
    proposals,
    clean_counts,
    trajectories,
    selected_indices,
    eta,
):
    if not (
        len(
            selected_windows
        )
        == len(
            proposals
        )
    ):
        raise ValueError(
            "window/proposal count mismatch"
        )

    poison_counts = Counter(
        window.pattern
        for window
        in iter_sequence_windows(
            merged_labels,
            trajectories,
            sequence_length=5,
        )
    )

    selected_window_count = len(
        selected_windows
    )

    pattern_changed = 0
    frequency_improved = 0
    changed_and_improved = 0

    exact_target_preserved = 0
    frequency_at_least_independent = 0
    frequency_strictly_better = 0

    source_frequencies = []
    independent_frequencies = []
    merged_frequencies = []

    proposal_slot_matches = 0
    proposal_slot_total = 0

    for window, proposal in zip(
        selected_windows,
        proposals,
    ):
        start = int(
            window.global_start
        )

        end = int(
            window.global_end
        )

        merged_raw_labels = tuple(
            int(value)
            for value
            in merged_labels[
                start:end
            ]
        )

        merged_pattern = (
            deduplicate_consecutive(
                merged_raw_labels
            )
        )

        source_pattern = tuple(
            window.source_pattern
        )

        source_frequency = int(
            clean_counts.get(
                source_pattern,
                0,
            )
        )

        merged_frequency = int(
            clean_counts.get(
                merged_pattern,
                0,
            )
        )

        independent_frequency = int(
            proposal.target_frequency
        )

        changed = (
            merged_pattern
            != source_pattern
        )

        improved = (
            merged_frequency
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

        exact_target_preserved += int(
            merged_pattern
            == proposal.target_pattern
        )

        frequency_at_least_independent += int(
            merged_frequency
            >= independent_frequency
        )

        frequency_strictly_better += int(
            merged_frequency
            > independent_frequency
        )

        source_frequencies.append(
            source_frequency
        )

        independent_frequencies.append(
            independent_frequency
        )

        merged_frequencies.append(
            merged_frequency
        )

        for (
            merged_label,
            independent_label,
        ) in zip(
            merged_raw_labels,
            proposal.raw_target_labels,
        ):
            proposal_slot_matches += int(
                merged_label
                == independent_label
            )

            proposal_slot_total += 1

    if selected_indices:
        index_array = np.asarray(
            selected_indices,
            dtype=np.int64,
        )

        transition_cluster_change = float(
            np.mean(
                clean_labels[
                    index_array
                ]
                != merged_labels[
                    index_array
                ]
            )
        )
    else:
        transition_cluster_change = 0.0

    selected_source_patterns = {
        tuple(
            window.source_pattern
        )
        for window
        in selected_windows
    }

    selected_source_pattern_count = len(
        selected_source_patterns
    )

    eradicated = sum(
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

    integrity = (
        _perturbation_integrity(
            clean_dataset=(
                clean_dataset
            ),
            merged_dataset=(
                merged_dataset
            ),
            selected_indices=(
                selected_indices
            ),
            eta=eta,
        )
    )

    if not integrity[
        "nonselected_attack_rows_identical"
    ]:
        raise RuntimeError(
            "non-selected observation/action "
            f"rows changed under {rule_id}"
        )

    if not integrity[
        "all_nonattack_arrays_identical"
    ]:
        raise RuntimeError(
            "non-attack arrays changed under "
            f"{rule_id}"
        )

    if not np.isclose(
        integrity[
            "state_perturbation_bound_valid_fraction"
        ],
        1.0,
    ):
        raise RuntimeError(
            "state perturbation bound violation"
        )

    if not np.isclose(
        integrity[
            "action_perturbation_bound_valid_fraction"
        ],
        1.0,
    ):
        raise RuntimeError(
            "action perturbation bound violation"
        )

    if not np.isclose(
        integrity[
            "action_rows_within_environment_bounds_fraction"
        ],
        1.0,
    ):
        raise RuntimeError(
            "action environment-bound violation"
        )

    frequency_deltas = [
        merged
        - independent
        for merged, independent
        in zip(
            merged_frequencies,
            independent_frequencies,
        )
    ]

    return {
        "selected_window_count": int(
            selected_window_count
        ),

        "unique_transition_footprint": int(
            len(
                selected_indices
            )
        ),

        "actual_modified_transition_count": int(
            integrity[
                "actual_modified_transition_count"
            ]
        ),

        "actual_modified_fraction_of_selected_footprint": float(
            integrity[
                "actual_modified_fraction_of_selected_footprint"
            ]
        ),

        "proposal_slot_target_label_preservation_fraction": (
            _safe_fraction(
                proposal_slot_matches,
                proposal_slot_total,
            )
        ),

        "window_exact_independent_target_pattern_preservation_fraction": (
            _safe_fraction(
                exact_target_preserved,
                selected_window_count,
            )
        ),

        "merged_window_pattern_change_fraction": (
            _safe_fraction(
                pattern_changed,
                selected_window_count,
            )
        ),

        "merged_window_frequency_improvement_fraction": (
            _safe_fraction(
                frequency_improved,
                selected_window_count,
            )
        ),

        "merged_frequency_improvement_given_changed": (
            _safe_fraction(
                changed_and_improved,
                pattern_changed,
            )
        ),

        "modified_transition_cluster_label_change_fraction": (
            transition_cluster_change
        ),

        "mean_source_pattern_frequency": (
            _mean(
                source_frequencies
            )
        ),

        "mean_independent_target_pattern_frequency": (
            _mean(
                independent_frequencies
            )
        ),

        "mean_merged_target_pattern_frequency": (
            _mean(
                merged_frequencies
            )
        ),

        "mean_frequency_delta_vs_independent": (
            _mean(
                frequency_deltas
            )
        ),

        "merged_frequency_at_least_independent_fraction": (
            _safe_fraction(
                frequency_at_least_independent,
                selected_window_count,
            )
        ),

        "merged_frequency_strictly_better_than_independent_fraction": (
            _safe_fraction(
                frequency_strictly_better,
                selected_window_count,
            )
        ),

        "selected_source_pattern_type_count": int(
            selected_source_pattern_count
        ),

        "selected_source_pattern_eradication_fraction": (
            _safe_fraction(
                eradicated,
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

        "state_perturbation_bound_valid_fraction": float(
            integrity[
                "state_perturbation_bound_valid_fraction"
            ]
        ),

        "action_perturbation_bound_valid_fraction": float(
            integrity[
                "action_perturbation_bound_valid_fraction"
            ]
        ),

        "action_rows_within_environment_bounds_fraction": float(
            integrity[
                "action_rows_within_environment_bounds_fraction"
            ]
        ),
    }


def _pairwise_resolution_metrics(
    *,
    left,
    right,
):
    if set(
        left
    ) != set(
        right
    ):
        raise ValueError(
            "resolution footprints differ"
        )

    indices = sorted(
        left
    )

    if not indices:
        return {
            "merged_target_label_disagreement_fraction": 0.0,
            "merged_continuous_row_disagreement_fraction": 0.0,
        }

    label_disagreement = 0
    row_disagreement = 0

    for index in indices:
        left_slot = left[
            index
        ]

        right_slot = right[
            index
        ]

        label_disagreement += int(
            left_slot.target_label
            != right_slot.target_label
        )

        same_observation = (
            np.array_equal(
                left_slot.observation,
                right_slot.observation,
            )
        )

        same_action = (
            np.array_equal(
                left_slot.action,
                right_slot.action,
            )
        )

        row_disagreement += int(
            not (
                same_observation
                and same_action
            )
        )

    return {
        "merged_target_label_disagreement_fraction": (
            _safe_fraction(
                label_disagreement,
                len(
                    indices
                ),
            )
        ),

        "merged_continuous_row_disagreement_fraction": (
            _safe_fraction(
                row_disagreement,
                len(
                    indices
                ),
            )
        ),
    }


def _load_conflict_seed(
    *,
    conflict_root,
    seed,
):
    path = (
        conflict_root
        / f"seed_{seed}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            "missing overlap-conflict reference: "
            f"{path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _run_seed(
    *,
    seed,
    config,
    clean_dataset,
    conflict_root,
):
    prepared = (
        prepare_csdpc_attack(
            clean_dataset,
            attack_seed=int(
                seed
            ),
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

        selections[
            rho
        ] = tuple(
            _to_selected_window(
                window
            )
            for window
            in diagnostic.selected_windows
        )

    if not _is_prefix(
        selections[
            0.01
        ],
        selections[
            0.05
        ],
    ):
        raise RuntimeError(
            "rho=0.01 S2 selection is not "
            "an exact rho=0.05 prefix"
        )

    print(
        "S2 rho-prefix check: PASS"
    )

    max_windows = (
        selections[
            0.05
        ]
    )

    print(
        "Regenerating independent C100 "
        "proposals for max-rho S2 selection..."
    )

    max_proposals = (
        _generate_proposals(
            selected_windows=(
                max_windows
            ),
            clean_dataset=(
                clean_dataset
            ),
            prepared=prepared,
        )
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

    clean_labels = np.asarray(
        prepared.clustering.labels,
        dtype=np.int64,
    )

    clean_counts = (
        prepared.pattern_frequencies
    )

    conflict_seed = (
        _load_conflict_seed(
            conflict_root=(
                conflict_root
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

        key = _rho_key(
            rho
        )

        windows = selections[
            rho
        ]

        proposals = max_proposals[
            :len(
                windows
            )
        ]

        reference = (
            conflict_seed[
                "rho_results"
            ][
                key
            ][
                "metrics"
            ]
        )

        _verify_conflict_reference(
            selected_windows=(
                windows
            ),
            proposals=(
                proposals
            ),
            reference=(
                reference
            ),
        )

        print(
            f"rho={key} conflict-reference "
            "reproduction: PASS"
        )

        requirements = (
            _build_requirements(
                selected_windows=(
                    windows
                ),
                proposals=(
                    proposals
                ),
            )
        )

        rule_metrics = {}
        resolved_by_rule = {}

        for rule_id in (
            RULE_IDS
        ):
            resolved = (
                _resolve_requirements(
                    requirements,
                    rule_id=(
                        rule_id
                    ),
                )
            )

            merged_dataset = (
                _build_merged_dataset(
                    clean_dataset=(
                        clean_dataset
                    ),
                    resolved=resolved,
                )
            )

            (
                merged_labels,
                selected_indices,
            ) = (
                _predict_selected_labels(
                    merged_dataset=(
                        merged_dataset
                    ),
                    resolved=resolved,
                    model=(
                        prepared.clustering_model
                    ),
                    clean_labels=(
                        clean_labels
                    ),
                )
            )

            metrics = (
                _compute_rule_metrics(
                    rule_id=rule_id,
                    clean_dataset=(
                        clean_dataset
                    ),
                    merged_dataset=(
                        merged_dataset
                    ),
                    clean_labels=(
                        clean_labels
                    ),
                    merged_labels=(
                        merged_labels
                    ),
                    selected_windows=(
                        windows
                    ),
                    proposals=(
                        proposals
                    ),
                    clean_counts=(
                        clean_counts
                    ),
                    trajectories=(
                        trajectories
                    ),
                    selected_indices=(
                        selected_indices
                    ),
                    eta=float(
                        config[
                            "eta"
                        ]
                    ),
                )
            )

            rule_metrics[
                rule_id
            ] = metrics

            resolved_by_rule[
                rule_id
            ] = resolved

            print(
                f"rho={key} {rule_id}: PASS"
            )

            del merged_dataset
            del merged_labels
            gc.collect()

        pairwise = {}

        pairs = (
            (
                "R0_FIRST_RARE_WINS",
                "R1_LAST_WINS",
            ),
            (
                "R0_FIRST_RARE_WINS",
                "R2_MODE_LABEL_FIRST",
            ),
            (
                "R1_LAST_WINS",
                "R2_MODE_LABEL_FIRST",
            ),
        )

        for left, right in pairs:
            pair_key = (
                left
                + "__VS__"
                + right
            )

            pairwise[
                pair_key
            ] = (
                _pairwise_resolution_metrics(
                    left=(
                        resolved_by_rule[
                            left
                        ]
                    ),
                    right=(
                        resolved_by_rule[
                            right
                        ]
                    ),
                )
            )

        rho_results[
            key
        ] = {
            "integrity": {
                "conflict_reference_reproduced": True,
                "same_selection_all_rules": True,
                "same_independent_proposals_all_rules": True,
                "no_hdf5_written": True,
            },
            "rules": (
                rule_metrics
            ),
            "pairwise_resolution": (
                pairwise
            ),
        }

    return {
        "schema_version": (
            "csdpc-overlap-resolution-seed-v1"
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
        "rho_results": (
            rho_results
        ),
    }


def _aggregate_values(
    *,
    seeds,
    getter,
):
    values = np.asarray(
        [
            getter(
                seed
            )
            for seed
            in seeds
        ],
        dtype=np.float64,
    )

    return {
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
        rho_key = (
            _rho_key(
                rho
            )
        )

        rules_summary = {}

        for rule_id in (
            RULE_IDS
        ):
            metric_summary = {}

            for metric in (
                SCALAR_METRICS
            ):
                metric_summary[
                    metric
                ] = (
                    _aggregate_values(
                        seeds=seeds,
                        getter=lambda seed,
                        metric=metric,
                        rule_id=rule_id: (
                            seed_records[
                                str(seed)
                            ][
                                "rho_results"
                            ][
                                rho_key
                            ][
                                "rules"
                            ][
                                rule_id
                            ][
                                metric
                            ]
                        ),
                    )
                )

            rules_summary[
                rule_id
            ] = (
                metric_summary
            )

        pairwise_keys = tuple(
            seed_records[
                str(
                    seeds[
                        0
                    ]
                )
            ][
                "rho_results"
            ][
                rho_key
            ][
                "pairwise_resolution"
            ].keys()
        )

        pairwise_summary = {}

        for pair_key in (
            pairwise_keys
        ):
            pairwise_summary[
                pair_key
            ] = {}

            for metric in (
                PAIRWISE_METRICS
            ):
                pairwise_summary[
                    pair_key
                ][
                    metric
                ] = (
                    _aggregate_values(
                        seeds=seeds,
                        getter=lambda seed,
                        pair_key=pair_key,
                        metric=metric: (
                            seed_records[
                                str(seed)
                            ][
                                "rho_results"
                            ][
                                rho_key
                            ][
                                "pairwise_resolution"
                            ][
                                pair_key
                            ][
                                metric
                            ]
                        ),
                    )
                )

        rho_results[
            rho_key
        ] = {
            "rules": (
                rules_summary
            ),
            "pairwise_resolution": (
                pairwise_summary
            ),
        }

    return {
        "schema_version": (
            "csdpc-overlap-resolution-summary-v1"
        ),
        "experiment": config[
            "experiment"
        ],
        "status": config[
            "status"
        ],
        "canonical_attack_is_unchanged": True,
        "attack_seeds": seeds,
        "resolution_rules": list(
            RULE_IDS
        ),
        "rho_results": (
            rho_results
        ),
        "interpretation_rules": {
            "R0_is_predeclared_primary_convention": True,
            "R1_and_R2_are_sensitivity_controls": True,
            "no_rule_selected_from_learner_performance": True,
            "no_poisoned_hdf5_written": True,
            "canonical_gate_b_unchanged": True,
        },
    }


def _load_seed_records(
    *,
    output_root,
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
                f"missing resolution seed result: {path}"
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
        "--conflict-root",
        type=Path,
        default=DEFAULT_CONFLICT_ROOT,
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

        summary = (
            _build_summary(
                config=config,
                seed_records=(
                    seed_records
                ),
            )
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
            f"dataset missing: {dataset_path}"
        )

    actual_sha = (
        sha256_file(
            dataset_path
        )
    )

    if (
        actual_sha
        != EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "frozen clean dataset "
            "SHA256 mismatch"
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
                "requested seed not in "
                "frozen seed set"
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
            "CSDPC OVERLAP RESOLUTION "
            f"SENSITIVITY — SEED {seed}"
        )
        print(
            "=" * 72
        )

        record = (
            _run_seed(
                seed=seed,
                config=config,
                clean_dataset=(
                    clean_dataset
                ),
                conflict_root=(
                    args.conflict_root.resolve()
                ),
            )
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