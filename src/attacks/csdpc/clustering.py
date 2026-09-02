from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans

from .types import ClusteringResult


def build_raw_decision_units(
    observations: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """
    Construct raw CSDPC state-action decision-unit features.

    Each transition t is represented as:

        u_t = concat(s_t, a_t)

    No normalization or standardization is performed here.
    """

    observations = np.asarray(observations)
    actions = np.asarray(actions)

    if observations.ndim != 2:
        raise ValueError(
            "observations must have shape [N, state_dim]"
        )

    if actions.ndim != 2:
        raise ValueError(
            "actions must have shape [N, action_dim]"
        )

    if observations.shape[0] != actions.shape[0]:
        raise ValueError(
            "observations and actions must contain the same "
            "number of transitions"
        )

    if observations.shape[1] == 0:
        raise ValueError(
            "observations must contain at least one feature"
        )

    if actions.shape[1] == 0:
        raise ValueError(
            "actions must contain at least one feature"
        )

    if not np.isfinite(observations).all():
        raise ValueError(
            "observations contain NaN or Inf"
        )

    if not np.isfinite(actions).all():
        raise ValueError(
            "actions contain NaN or Inf"
        )

    # np.concatenate already allocates a new array. copy() makes the
    # ownership requirement explicit: downstream attack code must not
    # share mutable storage with the clean dataset.
    return np.concatenate(
        [observations, actions],
        axis=1,
    ).copy()


def fit_kmeans_decision_units(
    features: np.ndarray,
    *,
    num_clusters: int,
    seed: int,
) -> Tuple[KMeans, ClusteringResult]:
    """
    Fit the frozen CSDPC KMeans reproduction configuration.

    Reproduction choices:
        init="k-means++"
        n_init=10
        max_iter=300
        tol=1e-4
        algorithm="lloyd"

    The attack seed is used directly as sklearn's random_state.
    """

    features = np.asarray(features)

    if features.ndim != 2:
        raise ValueError(
            "features must have shape [N, feature_dim]"
        )

    if features.shape[0] == 0:
        raise ValueError(
            "features must contain at least one sample"
        )

    if features.shape[1] == 0:
        raise ValueError(
            "features must contain at least one feature"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "features contain NaN or Inf"
        )

    if num_clusters <= 1:
        raise ValueError(
            "num_clusters must be greater than 1"
        )

    if features.shape[0] < num_clusters:
        raise ValueError(
            "number of samples must be at least num_clusters"
        )

    model = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1.0e-4,
        random_state=seed,
        algorithm="lloyd",
    )

    labels = model.fit_predict(features)

    result = ClusteringResult(
        labels=np.asarray(labels, dtype=np.int64).copy(),
        centers=np.asarray(
            model.cluster_centers_
        ).copy(),
        inertia=float(model.inertia_),
        n_iter=int(model.n_iter_),
    )

    return model, result