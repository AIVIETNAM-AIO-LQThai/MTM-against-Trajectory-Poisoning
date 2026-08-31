from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.methods.dt.checkpoint import (
    load_dt_checkpoint_compat,
)
from src.methods.dt.inference import (
    get_action,
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

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    model = make_model()

    load_dt_checkpoint_compat(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu",
    )

    with np.load(
        NORMALIZATION_PATH
    ) as f:
        state_mean = (
            f["state_mean"].copy()
        )

        state_std = (
            f["state_std"].copy()
        )

    # Fixed synthetic history.
    rng = np.random.default_rng(
        20260831
    )

    T = 7

    states = rng.normal(
        size=(T, 17)
    ).astype(np.float32)

    actions = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(T - 1, 6),
    ).astype(np.float32)

    returns_to_go = np.linspace(
        5.0,
        4.9,
        T,
        dtype=np.float32,
    )

    timesteps = np.arange(
        T,
        dtype=np.int64,
    )

    action = get_action(
        model=model,
        states=states,
        actions=actions,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        state_mean=state_mean,
        state_std=state_std,
        context_length=20,
        device=torch.device("cpu"),
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        args.output,
        action,
    )

    print(
        "Predicted action:",
        action,
    )

    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()