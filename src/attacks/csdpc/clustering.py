import numpy as np


def build_raw_decision_units(
    observations: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Construct raw CSDPC state-action decision-unit features."""

    if observations.ndim != 2:
        raise ValueError("observations must be [N, state_dim]")

    if actions.ndim != 2:
        raise ValueError("actions must be [N, action_dim]")

    if observations.shape[0] != actions.shape[0]:
        raise ValueError(
            "observations and actions must contain the same "
            "number of transitions"
        )

    if not np.isfinite(observations).all():
        raise ValueError("observations contain NaN or Inf")

    if not np.isfinite(actions).all():
        raise ValueError("actions contain NaN or Inf")

    return np.concatenate(
        [observations, actions],
        axis=-1,
    )