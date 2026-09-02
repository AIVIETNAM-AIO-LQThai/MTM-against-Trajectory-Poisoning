from __future__ import annotations

from typing import Mapping

import numpy as np

from .patterns import (
    deduplicate_consecutive,
)
from .types import (
    Pattern,
    PerturbedWindow,
    SelectedWindow,
)


def _relative_linf_scales(
    values: np.ndarray,
    eta: float,
) -> np.ndarray:
    """
    Return one L-infinity perturbation scale per transition.

    Shape:
        [L, 1]
    """

    return (
        eta
        * np.max(
            np.abs(values),
            axis=1,
            keepdims=True,
        )
    )


def _total_linf_perturbation(
    state_deltas: np.ndarray,
    action_deltas: np.ndarray,
) -> float:
    state_cost = np.max(
        np.abs(state_deltas),
        axis=1,
    )

    action_cost = np.max(
        np.abs(action_deltas),
        axis=1,
    )

    return float(
        np.sum(
            state_cost
            + action_cost
        )
    )


def perturb_selected_window(
    observations: np.ndarray,
    actions: np.ndarray,
    selected_window: SelectedWindow,
    *,
    kmeans_model,
    clean_pattern_frequencies: Mapping[
        Pattern,
        int,
    ],
    eta: float,
    num_candidates: int,
    rng: np.random.Generator,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> PerturbedWindow:
    """
    Generate bounded candidate perturbations and choose the
    candidate whose resulting deduplicated cluster pattern has
    the highest clean-data occurrence frequency.

    Tie-break:
        1. higher target pattern frequency
        2. lower total L-infinity perturbation
        3. lower candidate index
    """

    observations = np.asarray(
        observations
    )

    actions = np.asarray(
        actions
    )

    if observations.ndim != 2:
        raise ValueError(
            "observations must be 2D"
        )

    if actions.ndim != 2:
        raise ValueError(
            "actions must be 2D"
        )

    if (
        observations.shape[0]
        != actions.shape[0]
    ):
        raise ValueError(
            "observation/action transition counts differ"
        )

    if eta < 0.0:
        raise ValueError(
            "eta cannot be negative"
        )

    if num_candidates <= 0:
        raise ValueError(
            "num_candidates must be positive"
        )

    if action_low > action_high:
        raise ValueError(
            "action_low cannot exceed action_high"
        )

    start = selected_window.global_start
    end = selected_window.global_end

    if start < 0 or end > len(observations):
        raise ValueError(
            "selected window lies outside dataset"
        )

    if end <= start:
        raise ValueError(
            "selected window must be non-empty"
        )

    source_observations = (
        observations[
            start:end
        ].copy()
    )

    source_actions = (
        actions[
            start:end
        ].copy()
    )

    sequence_length = (
        end
        - start
    )

    state_scales = (
        _relative_linf_scales(
            source_observations,
            eta,
        )
    )

    action_scales = (
        _relative_linf_scales(
            source_actions,
            eta,
        )
    )

    state_noise = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(
            num_candidates,
            sequence_length,
            source_observations.shape[1],
        ),
    )

    action_noise = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(
            num_candidates,
            sequence_length,
            source_actions.shape[1],
        ),
    )

    candidate_observations = (
        source_observations[None, :, :]
        + state_noise
        * state_scales[None, :, :]
    )

    candidate_actions = (
        source_actions[None, :, :]
        + action_noise
        * action_scales[None, :, :]
    )

    candidate_actions = np.clip(
        candidate_actions,
        action_low,
        action_high,
    )

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

    decision_units = np.concatenate(
        [
            candidate_observations,
            candidate_actions,
        ],
        axis=2,
    )

    # sklearn KMeans expects prediction inputs to use the
    # same floating-point dtype as the fitted cluster centers.
    #
    # D4RL observations/actions are commonly float32, while
    # NumPy RNG perturbations are generated as float64.
    # Align only the KMeans prediction representation here.
    cluster_centers = getattr(
        kmeans_model,
        "cluster_centers_",
        None,
    )

    if cluster_centers is not None:
        prediction_dtype = np.asarray(
            cluster_centers
        ).dtype

        decision_units_for_prediction = (
            np.asarray(
                decision_units,
                dtype=prediction_dtype,
            )
        )
    else:
        # Supports deterministic lightweight stand-ins used
        # by unit tests that implement predict() but do not
        # expose sklearn's cluster_centers_ attribute.
        decision_units_for_prediction = (
            decision_units
        )

    flattened = (
        decision_units_for_prediction.reshape(
            num_candidates
            * sequence_length,
            decision_units_for_prediction.shape[2],
        )
    )

    predicted_labels = (
        np.asarray(
            kmeans_model.predict(
                flattened
            ),
            dtype=np.int64,
        )
        .reshape(
            num_candidates,
            sequence_length,
        )
    )

    best_candidate_index = None
    best_pattern = None
    best_frequency = None
    best_perturbation_cost = None

    for candidate_index in range(
        num_candidates
    ):
        candidate_pattern = (
            deduplicate_consecutive(
                predicted_labels[
                    candidate_index
                ]
            )
        )

        candidate_frequency = int(
            clean_pattern_frequencies.get(
                candidate_pattern,
                0,
            )
        )

        perturbation_cost = (
            _total_linf_perturbation(
                state_deltas[
                    candidate_index
                ],
                action_deltas[
                    candidate_index
                ],
            )
        )

        score = (
            -candidate_frequency,
            perturbation_cost,
            candidate_index,
        )

        if best_candidate_index is None:
            best_score = score
            best_candidate_index = (
                candidate_index
            )
            best_pattern = (
                candidate_pattern
            )
            best_frequency = (
                candidate_frequency
            )
            best_perturbation_cost = (
                perturbation_cost
            )
            continue

        if score < best_score:
            best_score = score
            best_candidate_index = (
                candidate_index
            )
            best_pattern = (
                candidate_pattern
            )
            best_frequency = (
                candidate_frequency
            )
            best_perturbation_cost = (
                perturbation_cost
            )

    assert (
        best_candidate_index
        is not None
    )

    source_frequency = int(
        clean_pattern_frequencies.get(
            selected_window.source_pattern,
            0,
        )
    )

    return PerturbedWindow(
        trajectory_id=(
            selected_window.trajectory_id
        ),
        global_start=start,
        global_end=end,
        source_pattern=(
            selected_window.source_pattern
        ),
        target_pattern=best_pattern,
        source_frequency=source_frequency,
        target_frequency=int(
            best_frequency
        ),
        candidate_index=int(
            best_candidate_index
        ),
        total_linf_perturbation=float(
            best_perturbation_cost
        ),
        observations=(
            candidate_observations[
                best_candidate_index
            ].copy()
        ),
        actions=(
            candidate_actions[
                best_candidate_index
            ].copy()
        ),
        state_deltas=(
            state_deltas[
                best_candidate_index
            ].copy()
        ),
        action_deltas=(
            action_deltas[
                best_candidate_index
            ].copy()
        ),
    )