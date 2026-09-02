from __future__ import annotations

from collections import Counter
from typing import Mapping

import numpy as np

from src.data.trajectories import (
    find_completed_trajectories,
)

from .clustering import (
    build_raw_decision_units,
    fit_kmeans_decision_units,
)
from .patterns import (
    iter_sequence_windows,
)
from .perturbation import (
    perturb_selected_window,
)
from .selection import (
    compute_transition_budget,
    select_rare_nonoverlapping_windows,
)
from .types import (
    PreparedCSDPC, CSDPCAttackResult,
)


def prepare_csdpc_attack(
    clean_dataset: Mapping[str, np.ndarray],
    *,
    attack_seed: int,
    num_clusters: int,
    sequence_length: int,
    eta: float,
    num_candidates: int,
) -> PreparedCSDPC:
    """
    Prepare the clean-data structures required by CSDPC.

    This stage does not mutate or poison the dataset.

    One prepared object should be reused for all poison rates
    belonging to the same attack seed.
    """

    required_keys = (
        "observations",
        "actions",
        "terminals",
        "timeouts",
    )

    for key in required_keys:
        if key not in clean_dataset:
            raise KeyError(
                f"missing required dataset key: {key}"
            )

    observations = np.asarray(
        clean_dataset["observations"]
    )

    actions = np.asarray(
        clean_dataset["actions"]
    )

    terminals = np.asarray(
        clean_dataset["terminals"]
    )

    timeouts = np.asarray(
        clean_dataset["timeouts"]
    )

    num_transitions = observations.shape[0]

    if actions.shape[0] != num_transitions:
        raise ValueError(
            "observations/actions transition counts differ"
        )

    if terminals.shape[0] != num_transitions:
        raise ValueError(
            "observations/terminals transition counts differ"
        )

    if timeouts.shape[0] != num_transitions:
        raise ValueError(
            "observations/timeouts transition counts differ"
        )

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive"
        )

    if eta < 0.0:
        raise ValueError(
            "eta cannot be negative"
        )

    if num_candidates <= 0:
        raise ValueError(
            "num_candidates must be positive"
        )

    decision_units = build_raw_decision_units(
        observations,
        actions,
    )

    (
        clustering_model,
        clustering,
    ) = fit_kmeans_decision_units(
        decision_units,
        num_clusters=num_clusters,
        seed=attack_seed,
    )

    trajectories, _ = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    windows = tuple(
        iter_sequence_windows(
            clustering.labels,
            trajectories,
            sequence_length=sequence_length,
        )
    )

    pattern_frequencies = Counter(
        window.pattern
        for window in windows
    )

    return PreparedCSDPC(
        attack_seed=int(
            attack_seed
        ),
        num_transitions=int(
            num_transitions
        ),
        sequence_length=int(
            sequence_length
        ),
        eta=float(
            eta
        ),
        num_candidates=int(
            num_candidates
        ),
        clustering_model=(
            clustering_model
        ),
        clustering=clustering,
        windows=windows,
        pattern_frequencies=dict(
            pattern_frequencies
        ),
    )

def apply_csdpc_attack(
    clean_dataset: Mapping[str, np.ndarray],
    prepared: PreparedCSDPC,
    *,
    rho: float,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> CSDPCAttackResult:
    """
    Apply CSDPC using an already-prepared clean clustering context.

    The input dataset is never mutated.
    """

    observations = np.asarray(
        clean_dataset["observations"]
    )

    actions = np.asarray(
        clean_dataset["actions"]
    )

    if (
        observations.shape[0]
        != prepared.num_transitions
    ):
        raise ValueError(
            "dataset transition count does not match prepared context"
        )

    if (
        actions.shape[0]
        != prepared.num_transitions
    ):
        raise ValueError(
            "dataset transition count does not match prepared context"
        )

    poisoned_dataset = {
        key: np.asarray(value).copy()
        for key, value
        in clean_dataset.items()
    }

    transition_budget = (
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
                transition_budget
            ),
        )
    )

    rng = np.random.default_rng(
        prepared.attack_seed
    )

    perturbed_windows = []

    modified_indices = []

    for selected_window in (
        selection.selected_windows
    ):
        perturbed = (
            perturb_selected_window(
                observations,
                actions,
                selected_window,
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
                action_low=action_low,
                action_high=action_high,
            )
        )

        start = (
            selected_window.global_start
        )

        end = (
            selected_window.global_end
        )

        poisoned_dataset[
            "observations"
        ][start:end] = (
            perturbed.observations
        )

        poisoned_dataset[
            "actions"
        ][start:end] = (
            perturbed.actions
        )

        perturbed_windows.append(
            perturbed
        )

        modified_indices.extend(
            selected_window.transition_indices
        )

    modified_indices = tuple(
        sorted(
            set(modified_indices)
        )
    )

    actual_budget = (
        selection.actual_transition_budget
    )

    if len(modified_indices) != actual_budget:
        raise RuntimeError(
            "modified-transition accounting mismatch"
        )

    actual_rho = (
        actual_budget
        / prepared.num_transitions
        if prepared.num_transitions
        else 0.0
    )

    return CSDPCAttackResult(
        poisoned_dataset=(
            poisoned_dataset
        ),
        requested_rho=float(rho),
        actual_rho=float(
            actual_rho
        ),
        requested_transition_budget=(
            transition_budget
        ),
        actual_transition_budget=(
            actual_budget
        ),
        selected_windows=(
            selection.selected_windows
        ),
        perturbed_windows=tuple(
            perturbed_windows
        ),
        modified_transition_indices=(
            modified_indices
        ),
    )