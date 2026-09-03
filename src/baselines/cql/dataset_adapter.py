from __future__ import annotations

from typing import Mapping

import numpy as np


def build_cql_training_view(
    dataset: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Reproduce the default D4RL qlearning_dataset view used by
    Q-learning algorithms.

    Frozen convention:
        terminate_on_end = False

    The returned raw_indices field is project metadata used only
    for auditing which original transitions enter CQL training.
    """

    required = (
        "observations",
        "actions",
        "rewards",
        "terminals",
        "timeouts",
    )

    for key in required:
        if key not in dataset:
            raise KeyError(
                f"missing required dataset key: {key}"
            )

    observations = np.asarray(
        dataset["observations"]
    )

    actions = np.asarray(
        dataset["actions"]
    )

    rewards = np.asarray(
        dataset["rewards"]
    )

    terminals = np.asarray(
        dataset["terminals"],
        dtype=bool,
    )

    timeouts = np.asarray(
        dataset["timeouts"],
        dtype=bool,
    )

    n = len(observations)

    if n < 2:
        raise ValueError(
            "dataset must contain at least two transitions"
        )

    for name, array in (
        ("actions", actions),
        ("rewards", rewards),
        ("terminals", terminals),
        ("timeouts", timeouts),
    ):
        if len(array) != n:
            raise ValueError(
                f"{name} length differs from observations"
            )

    candidate_indices = np.arange(
        n - 1,
        dtype=np.int64,
    )

    # Default D4RL qlearning_dataset behavior:
    # timeout transitions are omitted when terminate_on_end=False.
    raw_indices = candidate_indices[
        ~timeouts[:-1]
    ]

    return {
        "observations": np.asarray(
            observations[raw_indices],
            dtype=np.float32,
        ),
        "actions": np.asarray(
            actions[raw_indices],
            dtype=np.float32,
        ),
        "next_observations": np.asarray(
            observations[
                raw_indices + 1
            ],
            dtype=np.float32,
        ),
        "rewards": np.asarray(
            rewards[raw_indices],
            dtype=np.float32,
        ),
        "terminals": np.asarray(
            terminals[raw_indices],
            dtype=bool,
        ),
        "raw_indices": raw_indices.copy(),
    }


def audit_poison_exposure(
    raw_indices: np.ndarray,
    modified_transition_indices: np.ndarray,
) -> dict[str, float | int]:
    """
    Measure how raw CSDPC-modified transitions appear in the
    CQL Q-learning training view.

    A modified raw index can affect:
      - current observation
      - current action
      - previous transition's next observation
    """

    raw_indices = np.asarray(
        raw_indices,
        dtype=np.int64,
    )

    modified = np.asarray(
        modified_transition_indices,
        dtype=np.int64,
    )

    modified = np.unique(
        modified
    )

    current_modified = np.isin(
        raw_indices,
        modified,
    )

    next_state_modified = np.isin(
        raw_indices + 1,
        modified,
    )

    any_modified_tuple = (
        current_modified
        | next_state_modified
    )

    num_training = len(
        raw_indices
    )

    num_raw_modified = len(
        modified
    )

    current_count = int(
        np.sum(current_modified)
    )

    next_count = int(
        np.sum(next_state_modified)
    )

    any_count = int(
        np.sum(any_modified_tuple)
    )

    return {
        "num_training_transitions": int(
            num_training
        ),
        "num_raw_modified_transitions": int(
            num_raw_modified
        ),

        # Action at raw index i exists only if i itself survives.
        "poisoned_action_training_count": (
            current_count
        ),

        "poisoned_current_state_count": (
            current_count
        ),

        # Observation i can also appear as next_obs for i-1.
        "poisoned_next_state_count": (
            next_count
        ),

        "poison_exposed_training_tuple_count": (
            any_count
        ),

        "action_exposure_rate": (
            current_count
            / num_training
            if num_training
            else 0.0
        ),

        "current_state_exposure_rate": (
            current_count
            / num_training
            if num_training
            else 0.0
        ),

        "next_state_exposure_rate": (
            next_count
            / num_training
            if num_training
            else 0.0
        ),

        "any_tuple_exposure_rate": (
            any_count
            / num_training
            if num_training
            else 0.0
        ),

        "raw_modified_current_retention": (
            current_count
            / num_raw_modified
            if num_raw_modified
            else 0.0
        ),
    }