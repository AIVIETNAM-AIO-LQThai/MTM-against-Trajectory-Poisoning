from __future__ import annotations
import numpy as np

def compute_rtg(
    rewards: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Compute return-to-go for one trajectory.

    For Group 1 / Decision Transformer reproduction:

        R_t = r_t + gamma * R_{t+1}

    with gamma = 1.0.

    Parameters
    ----------
    rewards:
        1D reward array with shape (T,).

    gamma:
        Discount factor. The reference DT D4RL setup uses 1.0.

    Returns
    -------
    np.ndarray
        RTG array with the same shape and dtype as rewards.
    """
    rewards = np.asarray(rewards)
    if rewards.ndim != 1:
        raise ValueError(
            f"Expected rewards with shape (T,), "
            f"got {rewards.shape}"
        )

    if len(rewards) == 0:
        return rewards.copy()

    if not np.isfinite(rewards).all():
        raise ValueError(
            "Rewards contain NaN or Inf."
        )

    rtg = np.zeros_like(rewards)
    rtg[-1] = rewards[-1]
    for t in range(len(rewards) - 2, -1, -1):
        rtg[t] = rewards[t] + gamma * rtg[t + 1]
    return rtg