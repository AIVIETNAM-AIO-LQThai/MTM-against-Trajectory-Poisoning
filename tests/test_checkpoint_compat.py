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

    model.eval()

    return model


def test_smoke_checkpoint_loads_safely():
    assert CHECKPOINT_PATH.exists()

    model = make_model()

    checkpoint = load_dt_checkpoint_compat(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu",
    )

    assert checkpoint["update"] == 100
    assert checkpoint["seed"] == 0

    # Our DT deliberately disables GPT-2's ordinary
    # positional embedding.
    assert torch.equal(
        model.transformer.wpe.weight,
        torch.zeros_like(
            model.transformer.wpe.weight
        ),
    )

    assert (
        model.transformer
        .wpe.weight
        .requires_grad
        is False
    )


def test_smoke_checkpoint_can_run_inference():
    assert CHECKPOINT_PATH.exists()
    assert NORMALIZATION_PATH.exists()

    model = make_model()

    checkpoint = load_dt_checkpoint_compat(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu",
    )

    assert checkpoint["update"] == 100

    with np.load(
        NORMALIZATION_PATH
    ) as f:
        state_mean = (
            f["state_mean"].copy()
        )

        state_std = (
            f["state_std"].copy()
        )

    states = np.zeros(
        (1, 17),
        dtype=np.float32,
    )

    # At t=0 there are no previous actions.
    actions = np.empty(
        (0, 6),
        dtype=np.float32,
    )

    # target return 5000 / scale 1000
    returns_to_go = np.array(
        [5.0],
        dtype=np.float32,
    )

    timesteps = np.array(
        [0],
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

    assert action.shape == (6,)
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)