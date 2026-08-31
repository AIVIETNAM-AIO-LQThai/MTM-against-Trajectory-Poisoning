from random import Random

import numpy as np
import torch

from src.data.batching import sample_dt_batch
from src.data.trajectories import (
    find_completed_trajectories,
)
from src.methods.dt.model import (
    DecisionTransformer,
)
from src.methods.dt.optim import (
    create_dt_optimizer_and_scheduler,
)
from src.methods.dt.trainer import DTTrainer


def make_tiny_batch():
    rng = np.random.default_rng(123)
    n = 30

    observations = rng.normal(
        size=(n, 17)
    ).astype(np.float32)
    actions = np.tanh(
        rng.normal(size=(n, 6))
    ).astype(np.float32)
    rewards = rng.normal(
        size=n
    ).astype(np.float32)
    terminals = np.zeros(
        n,
        dtype=bool,
    )
    timeouts = np.zeros(
        n,
        dtype=bool,
    )

    # One completed trajectory.
    terminals[-1] = True

    trajectories, trailing = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    assert trailing == 0

    state_mean = np.mean(
        observations,
        axis=0,
    )
    state_std = (
        np.std(
            observations,
            axis=0,
        )
        + 1e-6
    )

    return sample_dt_batch(
        observations, actions, rewards,
        terminals, trajectories,
        state_mean, state_std,
        batch_size=4, context_length=20, max_ep_len=1000,
        rtg_scale=1000.0,
        np_rng=np.random.default_rng(456),
        py_rng=Random(456),
    )


def test_production_dt_train_step():
    torch.manual_seed(0)

    device = torch.device("cpu")

    model = DecisionTransformer(
        state_dim=17, action_dim=6, hidden_size=128,
        max_ep_len=1000,
        n_layer=3, n_head=1, n_inner=512,
        activation_function="relu",
        resid_pdrop=0.1, attn_pdrop=0.1, embd_pdrop=0.1,
        action_tanh=True,
    ).to(device)

    optimizer, scheduler = (
        create_dt_optimizer_and_scheduler(
            model,
            learning_rate=1e-4,
            weight_decay=1e-4,
            warmup_steps=10_000,
        )
    )

    trainer = DTTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        grad_clip_norm=0.25,
    )

    batch = make_tiny_batch()

    metrics = trainer.train_step(
        batch
    )

    assert np.isfinite(metrics.loss)
    assert metrics.loss >= 0

    assert np.isfinite(metrics.grad_norm_pre_clip)
    assert metrics.grad_norm_pre_clip >= 0

    assert metrics.learning_rate > 0
    assert metrics.learning_rate <= 1e-4

def test_reference_lr_warmup():
    torch.manual_seed(0)

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

    optimizer, scheduler = (
        create_dt_optimizer_and_scheduler(
            model,
            learning_rate=1e-4,
            weight_decay=1e-4,
            warmup_steps=10_000,
        )
    )

    initial_lr = (
        optimizer.param_groups[0]["lr"]
    )

    # It should initially be at the
    # warmup-scaled learning rate.
    assert initial_lr <= 1e-4

    for _ in range(100):
        optimizer.step()
        scheduler.step()

    lr_after_100 = (
        optimizer.param_groups[0]["lr"]
    )

    assert lr_after_100 > initial_lr
    assert lr_after_100 < 1e-4