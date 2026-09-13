from __future__ import annotations

import random

import numpy as np
import torch

from src.methods.mtm.losses import (
    reference_mtm_loss,
)

from src.methods.mtm.model import (
    MTMConfig,
    ReferenceMTM,
)


SEED = 123
NUM_SAMPLES = 16
TRAJ_LENGTH = 4

STATE_DIM = 17
ACTION_DIM = 6

NUM_UPDATES = 2000
LEARNING_RATE = 1e-3


def set_seed(
    seed: int,
) -> None:
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
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

    # --------------------------------------------------
    # Fixed synthetic dataset
    # --------------------------------------------------

    torch.manual_seed(
        SEED + 1
    )

    trajectories = {
        "states": torch.randn(
            NUM_SAMPLES,
            TRAJ_LENGTH,
            1,
            STATE_DIM,
            device=device,
        ),

        "actions": torch.randn(
            NUM_SAMPLES,
            TRAJ_LENGTH,
            1,
            ACTION_DIM,
            device=device,
        ),

        "returns": torch.randn(
            NUM_SAMPLES,
            TRAJ_LENGTH,
            1,
            1,
            device=device,
        ),
    }

    # --------------------------------------------------
    # Fixed deterministic masks.
    #
    # No random masking in this first overfit test.
    # --------------------------------------------------

    masks = {
        "states": torch.tensor(
            [
                [1.0],
                [0.0],
                [1.0],
                [0.0],
            ],
            device=device,
        ),

        "actions": torch.tensor(
            [
                [1.0],
                [1.0],
                [0.0],
                [0.0],
            ],
            device=device,
        ),

        "returns": torch.tensor(
            [
                [1.0],
                [0.0],
                [1.0],
                [0.0],
            ],
            device=device,
        ),
    }

    # --------------------------------------------------
    # Deliberately small network.
    #
    # Tiny-overfit tests implementation correctness;
    # it is NOT a reference training run.
    # --------------------------------------------------

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    print(
        "=" * 70
    )

    print(
        "FIXED-MASK MTM TINY OVERFIT"
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

    initial_loss = None
    final_loss = None

    model.train()

    for update in range(
        1,
        NUM_UPDATES + 1,
    ):
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

        if initial_loss is None:
            initial_loss = float(
                loss.item()
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

        final_loss = float(
            loss.item()
        )

        if (
            update == 1
            or update % 100 == 0
        ):
            full = {
                key: float(
                    value.item()
                )
                for key, value
                in loss_output
                .full_losses
                .items()
            }

            masked = {
                key: float(
                    value.item()
                )
                for key, value
                in loss_output
                .masked_losses
                .items()
            }

            print(
                f"update={update:4d} "
                f"total={final_loss:.8f} "
                f"full={full} "
                f"masked={masked}"
            )

    assert (
        initial_loss
        is not None
    )

    assert (
        final_loss
        is not None
    )

    ratio = (
        final_loss
        / initial_loss
    )

    print()
    print(
        f"initial_loss: "
        f"{initial_loss:.8f}"
    )

    print(
        f"final_loss:   "
        f"{final_loss:.8f}"
    )

    print(
        f"final/initial:"
        f" {ratio:.8f}"
    )

    # Tiny-overfit is a debugging gate, not a benchmark.
    #
    # We require a dramatic reduction rather than merely
    # a statistically detectable improvement.
    if ratio >= 0.05:
        raise RuntimeError(
            "TINY OVERFIT FAILED: "
            "final loss did not fall below "
            "5% of initial loss"
        )

    print()
    print(
        "[PASS] fixed-mask tiny overfit"
    )


if __name__ == "__main__":
    main()