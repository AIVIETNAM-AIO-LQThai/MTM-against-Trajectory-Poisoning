from pathlib import Path
from random import Random

import h5py
import numpy as np
import torch

from src.data.batching import sample_dt_batch
from src.data.trajectories import find_completed_trajectories
from src.methods.dt.losses import masked_action_mse
from src.methods.dt.model import DecisionTransformer


DATASET_PATH = Path(
    "data/raw/walker2d-medium-v2/"
    "walker2d_medium-v2.hdf5"
)

NORMALIZATION_PATH = Path(
    "data/metadata/"
    "walker2d_medium_normalization.npz"
)


def test_dt_can_overfit_tiny_fixed_batch():
    torch.manual_seed(0)
    np.random.seed(0)

    # --------------------------------------------------
    # Load frozen clean Walker2d dataset.
    # --------------------------------------------------

    with h5py.File(DATASET_PATH, "r") as f:
        observations = f["observations"][:]
        actions = f["actions"][:]
        rewards = f["rewards"][:]

        terminals = (
            f["terminals"][:]
            .astype(bool)
        )

        timeouts = (
            f["timeouts"][:]
            .astype(bool)
        )

    trajectories, trailing = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    assert trailing == 5

    # --------------------------------------------------
    # Load the normalization artifact we already froze.
    # --------------------------------------------------

    with np.load(NORMALIZATION_PATH) as f:
        state_mean = f["state_mean"]
        state_std = f["state_std"]

    # --------------------------------------------------
    # Create ONE fixed batch.
    #
    # We deliberately never resample this batch.
    # --------------------------------------------------

    batch = sample_dt_batch(
        observations,
        actions,
        rewards,
        terminals,
        trajectories,
        state_mean,
        state_std,
        batch_size=4,
        context_length=20,
        max_ep_len=1000,
        rtg_scale=1000.0,
        np_rng=np.random.default_rng(123),
        py_rng=Random(123),
    )

    # --------------------------------------------------
    # Device.
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------
    # Convert fixed batch to PyTorch.
    # --------------------------------------------------

    states = torch.tensor(
        batch.states,
        dtype=torch.float32,
        device=device,
    )

    action_targets = torch.tensor(
        batch.actions,
        dtype=torch.float32,
        device=device,
    )

    # Sampler has K+1 RTG entries.
    # DT consumes only the first K.
    returns_to_go = torch.tensor(
        batch.rtg[:, :-1],
        dtype=torch.float32,
        device=device,
    )

    timesteps = torch.tensor(
        batch.timesteps,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.tensor(
        batch.attention_mask,
        dtype=torch.long,
        device=device,
    )

    # --------------------------------------------------
    # Tiny-overfit model.
    #
    # IMPORTANT:
    # dropout=0 here is intentional.
    #
    # This is a diagnostic test, NOT our final baseline
    # configuration. The production DT will use dropout=0.1.
    # --------------------------------------------------

    model = DecisionTransformer(
        state_dim=17,
        action_dim=6,
        hidden_size=128,
        max_ep_len=1000,
        n_layer=3,
        n_head=1,
        n_inner=512,
        activation_function="relu",
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        action_tanh=True,
    ).to(device)

    model.train()

    # --------------------------------------------------
    # Diagnostic optimizer.
    #
    # Again, this is NOT the final baseline optimizer.
    # Higher LR is intentional for a fast memorization test.
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )

    # --------------------------------------------------
    # Initial loss BEFORE training.
    # --------------------------------------------------

    with torch.no_grad():
        _, initial_action_preds, _ = model(
            states,
            action_targets,
            returns_to_go,
            timesteps,
            attention_mask,
        )

        initial_loss = masked_action_mse(
            initial_action_preds,
            action_targets,
            attention_mask,
        ).item()

    print(
        f"\nInitial tiny-overfit loss: "
        f"{initial_loss:.8f}"
    )

    # --------------------------------------------------
    # Repeatedly optimize the SAME fixed batch.
    # --------------------------------------------------

    num_steps = 500

    for step in range(num_steps):
        optimizer.zero_grad(
            set_to_none=True
        )

        _, action_preds, _ = model(
            states,
            action_targets,
            returns_to_go,
            timesteps,
            attention_mask,
        )

        loss = masked_action_mse(
            action_preds,
            action_targets,
            attention_mask,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        if (
            step == 0
            or (step + 1) % 100 == 0
        ):
            print(
                f"step {step + 1:4d}: "
                f"loss={loss.item():.8f}"
            )

    # --------------------------------------------------
    # Final loss.
    # --------------------------------------------------

    model.eval()

    with torch.no_grad():
        _, final_action_preds, _ = model(
            states,
            action_targets,
            returns_to_go,
            timesteps,
            attention_mask,
        )

        final_loss = masked_action_mse(
            final_action_preds,
            action_targets,
            attention_mask,
        ).item()

    print(
        f"Final tiny-overfit loss: "
        f"{final_loss:.8f}"
    )

    reduction_ratio = (
        final_loss / initial_loss
    )

    print(
        f"Final / initial ratio: "
        f"{reduction_ratio:.8f}"
    )

    # --------------------------------------------------
    # PASS criteria.
    #
    # Don't require mathematically zero loss.
    # We simply require clear memorization.
    # --------------------------------------------------

    assert final_loss < initial_loss

    assert reduction_ratio < 0.05, (
        "DT failed to strongly overfit the tiny fixed batch: "
        f"initial={initial_loss:.6f}, "
        f"final={final_loss:.6f}, "
        f"ratio={reduction_ratio:.6f}"
    )