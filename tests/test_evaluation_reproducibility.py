from pathlib import Path

import gym
import numpy as np
import torch

from src.evaluation.walker2d import (
    evaluate_dt_episode,
)
from src.methods.dt.checkpoint import (
    load_dt_checkpoint_compat,
)
from src.methods.dt.model import (
    DecisionTransformer,
)


CHECKPOINT_PATH = Path(
    "experiments/dt/"
    "walker2d_medium_clean/"
    "seed_0/smoke/"
    "checkpoint_final.pt"
)

NORMALIZATION_PATH = Path(
    "data/metadata/"
    "walker2d_medium_normalization.npz"
)


def make_model():
    model = DecisionTransformer(
        state_dim=17,
        action_dim=6,
        hidden_size=128,
        max_ep_len=1000,
        n_layer=3,
        n_head=1,
        n_inner=512,
        activation_function="relu",
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        embd_pdrop=0.1,
        action_tanh=True,
    )

    load_dt_checkpoint_compat(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu",
    )

    return model


def run_once(seed):
    model = make_model()

    with np.load(
        NORMALIZATION_PATH
    ) as f:
        state_mean = (
            f["state_mean"].copy()
        )

        state_std = (
            f["state_std"].copy()
        )

    env = gym.make(
        "Walker2d-v3"
    )

    try:
        result = evaluate_dt_episode(
            env=env,
            model=model,
            state_mean=state_mean,
            state_std=state_std,
            target_return=5000.0,
            episode_seed=seed,
            context_length=20,
            scale=1000.0,
            max_ep_len=1000,
            device=torch.device("cpu"),
        )

    finally:
        env.close()

    return result


def test_same_seed_reproduces_same_episode():
    first = run_once(
        12345
    )

    second = run_once(
        12345
    )

    assert (
        first.episode_length
        == second.episode_length
    )

    np.testing.assert_allclose(
        first.raw_return,
        second.raw_return,
        rtol=0.0,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        first.normalized_return,
        second.normalized_return,
        rtol=0.0,
        atol=1e-8,
    )