from __future__ import annotations

import math
import random

import numpy as np
import torch

from src.methods.mtm.losses import (
    reference_mtm_loss,
)

from src.methods.mtm.masking import (
    sample_reference_auto_mask,
)

from src.methods.mtm.model import (
    MTMConfig,
    ReferenceMTM,
)


SEED = 321

NUM_SAMPLES = 16
TRAJ_LENGTH = 4

STATE_DIM = 17
ACTION_DIM = 6

NUM_UPDATES = 2500
LEARNING_RATE = 1e-3

NUM_EVAL_MASKS = 64


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_structured_dataset(
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Build a tiny deterministic dataset with structure shared
    across modalities and timesteps.

    This is intentional.

    For random independent targets, masked positions would be
    fundamentally unpredictable. Here, visible trajectory context
    contains information about hidden positions, so the smoke test
    can verify that stochastic MTM reconstruction actually learns.
    """

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        SEED + 1
    )

    latent = torch.randn(
        NUM_SAMPLES,
        3,
        generator=generator,
    )

    times = torch.arange(
        TRAJ_LENGTH,
        dtype=torch.float32,
    )

    # --------------------------------------------------
    # States
    # --------------------------------------------------

    state_weights_0 = torch.linspace(
        0.5,
        1.5,
        STATE_DIM,
    )

    state_weights_1 = torch.linspace(
        -1.0,
        1.0,
        STATE_DIM,
    )

    state_weights_2 = torch.linspace(
        1.25,
        0.25,
        STATE_DIM,
    )

    states = []

    for t in range(
        TRAJ_LENGTH
    ):
        time_value = times[t]

        state_t = (
            latent[:, 0:1]
            * state_weights_0[None, :]
            + latent[:, 1:2]
            * state_weights_1[None, :]
            + latent[:, 2:3]
            * state_weights_2[None, :]
            + 0.10
            * time_value
            * state_weights_0[None, :]
        )

        states.append(
            state_t
        )

    states = torch.stack(
        states,
        dim=1,
    )

    # --------------------------------------------------
    # Actions
    #
    # They depend on the same latent variables as states.
    # --------------------------------------------------

    action_weights_0 = torch.linspace(
        -0.8,
        0.8,
        ACTION_DIM,
    )

    action_weights_1 = torch.linspace(
        1.0,
        0.2,
        ACTION_DIM,
    )

    actions = []

    for t in range(
        TRAJ_LENGTH
    ):
        time_value = times[t]

        action_t = (
            0.7
            * latent[:, 0:1]
            * action_weights_0[None, :]
            + 0.4
            * latent[:, 1:2]
            * action_weights_1[None, :]
            - 0.2
            * latent[:, 2:3]
            + 0.08
            * time_value
        )

        actions.append(
            action_t
        )

    actions = torch.stack(
        actions,
        dim=1,
    )

    # --------------------------------------------------
    # Returns
    #
    # Again generated from the same latent trajectory.
    # --------------------------------------------------

    returns = []

    for t in range(
        TRAJ_LENGTH
    ):
        time_value = times[t]

        return_t = (
            0.9
            * latent[:, 0]
            - 0.5
            * latent[:, 1]
            + 0.3
            * latent[:, 2]
            + 0.15
            * time_value
        )

        returns.append(
            return_t
        )

    returns = torch.stack(
        returns,
        dim=1,
    )[..., None]

    return {
        "states": states[
            :,
            :,
            None,
            :,
        ].to(device),

        "actions": actions[
            :,
            :,
            None,
            :,
        ].to(device),

        "returns": returns[
            :,
            :,
            None,
            :,
        ].to(device),
    }


def count_visible_tokens(
    masks: dict[
        str,
        torch.Tensor,
    ],
) -> int:
    return int(
        sum(
            mask.sum().item()
            for mask
            in masks.values()
        )
    )


def evaluate_mask_bank(
    model: ReferenceMTM,
    trajectories: dict[
        str,
        torch.Tensor,
    ],
    *,
    device: torch.device,
) -> tuple[
    float,
    float,
    int,
]:
    """
    Evaluate on a fixed deterministic bank of AUTO_MASK
    realizations.

    Returning an average over a bank is important because
    individual stochastic masks can have very different losses.
    """

    model.eval()

    full_losses = []
    masked_losses = []

    all_hidden_count = 0

    with torch.no_grad():

        for mask_index in range(
            NUM_EVAL_MASKS
        ):
            rng = np.random.RandomState(
                10_000
                + mask_index
            )

            sample = (
                sample_reference_auto_mask(
                    {
                        "states": (
                            1,
                            STATE_DIM,
                        ),
                        "actions": (
                            1,
                            ACTION_DIM,
                        ),
                        "returns": (
                            1,
                            1,
                        ),
                    },
                    traj_length=(
                        TRAJ_LENGTH
                    ),
                    device=device,
                    rng=rng,
                )
            )

            masks = sample.masks

            if (
                count_visible_tokens(
                    masks
                )
                == 0
            ):
                all_hidden_count += 1

            predictions = model(
                trajectories,
                masks,
            )

            loss_output = (
                reference_mtm_loss(
                    trajectories,
                    predictions,
                    masks,
                )
            )

            full_losses.append(
                float(
                    loss_output
                    .total_loss
                    .item()
                )
            )

            finite_masked = []

            for value in (
                loss_output
                .masked_losses
                .values()
            ):
                scalar = float(
                    value.item()
                )

                if math.isfinite(
                    scalar
                ):
                    finite_masked.append(
                        scalar
                    )

            if finite_masked:
                masked_losses.append(
                    float(
                        np.mean(
                            finite_masked
                        )
                    )
                )

    return (
        float(
            np.mean(
                full_losses
            )
        ),
        float(
            np.mean(
                masked_losses
            )
        ),
        all_hidden_count,
    )


def main() -> None:
    set_seed(
        SEED
    )

    device = torch.device(
        "cpu"
    )

    data_shapes = {
        "states": (
            1,
            STATE_DIM,
        ),
        "actions": (
            1,
            ACTION_DIM,
        ),
        "returns": (
            1,
            1,
        ),
    }

    trajectories = (
        make_structured_dataset(
            device=device
        )
    )

    model = ReferenceMTM(
        data_shapes,
        traj_length=(
            TRAJ_LENGTH
        ),
        config=MTMConfig(
            n_embd=64,
            n_head=4,
            n_enc_layer=2,
            n_dec_layer=1,
            dropout=0.0,
        ),
    ).to(
        device
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.0,
        )
    )

    print(
        "=" * 70
    )

    print(
        "RANDOM AUTO_MASK MTM LEARNING SMOKE"
    )

    print(
        "=" * 70
    )

    print(
        "device:",
        device,
    )

    print(
        "samples:",
        NUM_SAMPLES,
    )

    print(
        "updates:",
        NUM_UPDATES,
    )

    print(
        "parameters:",
        sum(
            parameter.numel()
            for parameter
            in model.parameters()
        ),
    )

    (
        initial_eval_full,
        initial_eval_masked,
        eval_all_hidden,
    ) = evaluate_mask_bank(
        model,
        trajectories,
        device=device,
    )

    print()
    print(
        "initial_eval_full:",
        f"{initial_eval_full:.8f}",
    )

    print(
        "initial_eval_masked:",
        f"{initial_eval_masked:.8f}",
    )

    print(
        "eval_all_hidden_masks:",
        eval_all_hidden,
        "/",
        NUM_EVAL_MASKS,
    )

    # One evolving legacy NumPy RNG.
    #
    # This follows the same RNG family as the reference MTM
    # masking implementation rather than resetting a seed
    # every update.
    train_rng = np.random.RandomState(
        SEED + 2
    )

    all_hidden_train_count = 0

    running_losses = []

    model.train()

    for update in range(
        1,
        NUM_UPDATES + 1,
    ):
        sample = (
            sample_reference_auto_mask(
                data_shapes,
                traj_length=(
                    TRAJ_LENGTH
                ),
                device=device,
                rng=train_rng,
            )
        )

        masks = sample.masks

        if (
            count_visible_tokens(
                masks
            )
            == 0
        ):
            all_hidden_train_count += 1

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(
            trajectories,
            masks,
        )

        loss_output = (
            reference_mtm_loss(
                trajectories,
                predictions,
                masks,
            )
        )

        loss = (
            loss_output.total_loss
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "non-finite training loss"
            )

        loss.backward()

        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                model.parameters(),
                max_norm=100.0,
            )
        )

        if not torch.isfinite(
            grad_norm
        ):
            raise RuntimeError(
                "non-finite gradient norm"
            )

        optimizer.step()

        running_losses.append(
            float(
                loss.item()
            )
        )

        if len(
            running_losses
        ) > 100:
            running_losses.pop(
                0
            )

        if (
            update == 1
            or update % 100 == 0
        ):
            print(
                f"update={update:4d} "
                f"loss={loss.item():.8f} "
                f"running100="
                f"{np.mean(running_losses):.8f} "
                f"mode="
                f"{sample.selected_mode} "
                f"position="
                f"{sample.selected_position}"
            )

    (
        final_eval_full,
        final_eval_masked,
        final_eval_all_hidden,
    ) = evaluate_mask_bank(
        model,
        trajectories,
        device=device,
    )

    full_ratio = (
        final_eval_full
        / initial_eval_full
    )

    masked_ratio = (
        final_eval_masked
        / initial_eval_masked
    )

    print()
    print(
        "initial_eval_full:",
        f"{initial_eval_full:.8f}",
    )

    print(
        "final_eval_full:  ",
        f"{final_eval_full:.8f}",
    )

    print(
        "full_ratio:       ",
        f"{full_ratio:.8f}",
    )

    print()

    print(
        "initial_eval_masked:",
        f"{initial_eval_masked:.8f}",
    )

    print(
        "final_eval_masked:  ",
        f"{final_eval_masked:.8f}",
    )

    print(
        "masked_ratio:       ",
        f"{masked_ratio:.8f}",
    )

    print()

    print(
        "train_all_hidden_masks:",
        all_hidden_train_count,
        "/",
        NUM_UPDATES,
    )

    print(
        "eval_all_hidden_masks:",
        final_eval_all_hidden,
        "/",
        NUM_EVAL_MASKS,
    )

    if not math.isfinite(
        final_eval_full
    ):
        raise RuntimeError(
            "RANDOM MASK SMOKE FAILED: "
            "final evaluation loss "
            "is non-finite"
        )

    # This is deliberately a weaker threshold than the
    # fixed-mask overfit gate.
    #
    # Masks vary every update, including occasional
    # all-hidden masks, so zero reconstruction error is
    # not the purpose of this test.
    if full_ratio >= 0.50:
        raise RuntimeError(
            "RANDOM MASK SMOKE FAILED: "
            "fixed-bank full loss did not fall "
            "below 50% of its initial value"
        )

    # We also want masked-position reconstruction to move
    # in the correct direction. It need not collapse to zero.
    if masked_ratio >= 0.90:
        raise RuntimeError(
            "RANDOM MASK SMOKE FAILED: "
            "masked reconstruction diagnostic "
            "did not improve by at least 10%"
        )

    print()
    print(
        "[PASS] random AUTO_MASK learning smoke"
    )


if __name__ == "__main__":
    main()