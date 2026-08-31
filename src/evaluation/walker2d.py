from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.evaluation.scoring import (
    walker2d_normalized_score,
)
from src.methods.dt.inference import (
    get_action,
)


@dataclass(frozen=True)
class EpisodeResult:
    raw_return: float
    normalized_return: float
    episode_length: int
    episode_seed: int
    target_return: float


def evaluate_dt_episode(
    env,
    model,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    *,
    target_return: float,
    episode_seed: int,
    context_length: int = 20,
    scale: float = 1000.0,
    max_ep_len: int = 1000,
    device: torch.device,
) -> EpisodeResult:

    model.eval()

    # Reference Gym 0.18.3 seeding API.
    env.seed(
        episode_seed
    )

    np.random.seed(
        episode_seed
    )

    torch.manual_seed(
        episode_seed
    )

    state = env.reset()

    # Defensive support if accidentally called under
    # a newer Gym-style API.
    if isinstance(
        state,
        tuple,
    ):
        state = state[0]

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    if state.shape != (17,):
        raise RuntimeError(
            "Unexpected Walker2d observation "
            f"shape: {state.shape}"
        )

    states = state.reshape(
        1,
        17,
    )

    actions = np.empty(
        (0, 6),
        dtype=np.float32,
    )

    # DT uses scaled return:
    #
    # target 5000 -> RTG 5.0
    returns_to_go = np.array(
        [
            target_return / scale
        ],
        dtype=np.float32,
    )

    timesteps = np.array(
        [0],
        dtype=np.int64,
    )

    episode_return = 0.0
    episode_length = 0

    for t in range(
        max_ep_len
    ):
        action = get_action(
            model=model,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            state_mean=state_mean,
            state_std=state_std,
            context_length=context_length,
            device=device,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if action.shape != (6,):
            raise RuntimeError(
                "Unexpected DT action shape: "
                f"{action.shape}"
            )

        if not np.isfinite(
            action
        ).all():
            raise FloatingPointError(
                "DT produced NaN or Inf action."
            )

        # The model uses tanh, so this should naturally hold.
        if (
            np.any(action < -1.00001)
            or np.any(action > 1.00001)
        ):
            raise RuntimeError(
                "DT produced action outside "
                f"Walker2d bounds: {action}"
            )

        step_result = env.step(
            action
        )

        if len(step_result) == 4:
            (
                next_state,
                reward,
                done,
                info,
            ) = step_result

        elif len(step_result) == 5:
            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = step_result

            done = bool(
                terminated
                or truncated
            )

        else:
            raise RuntimeError(
                "Unexpected env.step() "
                f"return length: {len(step_result)}"
            )

        reward = float(
            reward
        )

        episode_return += reward
        episode_length += 1

        # Record a_t.
        actions = np.concatenate(
            [
                actions,
                action.reshape(
                    1,
                    6,
                ),
            ],
            axis=0,
        )

        if done:
            break

        next_state = np.asarray(
            next_state,
            dtype=np.float32,
        )

        if next_state.shape != (17,):
            raise RuntimeError(
                "Unexpected next-state shape: "
                f"{next_state.shape}"
            )

        # Record s_(t+1).
        states = np.concatenate(
            [
                states,
                next_state.reshape(
                    1,
                    17,
                ),
            ],
            axis=0,
        )

        # Reference DT RTG update:
        #
        # R_(t+1) = R_t - r_t / scale
        next_rtg = (
            returns_to_go[-1]
            - reward / scale
        )

        returns_to_go = np.concatenate(
            [
                returns_to_go,
                np.array(
                    [next_rtg],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )

        # Walker2d max episode length is 1000,
        # so valid embedding indices are 0..999.
        next_timestep = min(
            t + 1,
            max_ep_len - 1,
        )

        timesteps = np.concatenate(
            [
                timesteps,
                np.array(
                    [next_timestep],
                    dtype=np.int64,
                ),
            ],
            axis=0,
        )

    normalized_return = (
        walker2d_normalized_score(
            episode_return
        )
    )

    return EpisodeResult(
        raw_return=float(
            episode_return
        ),
        normalized_return=float(
            normalized_return
        ),
        episode_length=int(
            episode_length
        ),
        episode_seed=int(
            episode_seed
        ),
        target_return=float(
            target_return
        ),
    )