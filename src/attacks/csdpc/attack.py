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
from .types import (
    PreparedCSDPC,
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