from __future__ import annotations

import math
import random
from pathlib import Path

import h5py
import numpy as np
import torch

from src.data.mtm_batching import (
    MTMBatch,
    build_split_window_ranges,
    sample_mtm_batch,
)

from src.data.mtm_dataset import (
    ReferenceMTMDataset,
)

from src.data.mtm_split import (
    reference_trajectory_split,
)

from src.data.mtm_statistics import (
    compute_reference_mtm_statistics,
)

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

from src.methods.mtm.tokenizers import (
    ContinuousTokenizer,
)


DATASET_PATH = Path(
    "data/raw/walker2d-medium-v2/"
    "walker2d_medium-v2.hdf5"
)

SEED = 404

TRAJ_LENGTH = 4
MAX_PATH_LENGTH = 1000

TRAIN_FRACTION = 0.95

BATCH_SIZE = 64
NUM_UPDATES = 1000

LEARNING_RATE = 1e-3

NUM_EVAL_BATCHES = 8

STATE_DIM = 17
ACTION_DIM = 6


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_batch(
    batch: MTMBatch,
    tokenizers: dict[
        str,
        ContinuousTokenizer,
    ],
    *,
    device: torch.device,
) -> dict[
    str,
    torch.Tensor,
]:
    raw = {
        "states": torch.from_numpy(
            batch.states
        ).to(
            device=device,
            dtype=torch.float32,
        ),

        "actions": torch.from_numpy(
            batch.actions
        ).to(
            device=device,
            dtype=torch.float32,
        ),

        "returns": torch.from_numpy(
            batch.returns
        ).to(
            device=device,
            dtype=torch.float32,
        ),
    }

    return {
        key: tokenizers[
            key
        ].encode(
            value
        )
        for key, value
        in raw.items()
    }


def finite_masked_mean(
    values: dict[
        str,
        torch.Tensor,
    ],
) -> float | None:
    finite_values = []

    for value in values.values():
        scalar = float(
            value.item()
        )

        if math.isfinite(
            scalar
        ):
            finite_values.append(
                scalar
            )

    if not finite_values:
        return None

    return float(
        np.mean(
            finite_values
        )
    )


def evaluate(
    model: ReferenceMTM,
    dataset: ReferenceMTMDataset,
    window_range,
    tokenizers,
    *,
    device: torch.device,
) -> tuple[
    float,
    float,
]:
    """
    Deterministic validation.

    Batch RNG and mask RNG are fixed every time this function
    is called, so initial and final evaluation use exactly the
    same validation windows and masks.
    """

    batch_rng = (
        np.random.RandomState(
            SEED + 10_000
        )
    )

    mask_rng = (
        np.random.RandomState(
            SEED + 20_000
        )
    )

    full_losses = []
    masked_losses = []

    model.eval()

    with torch.no_grad():

        for _ in range(
            NUM_EVAL_BATCHES
        ):
            batch = sample_mtm_batch(
                dataset,
                window_range,
                batch_size=BATCH_SIZE,
                rng=batch_rng,
            )

            encoded = encode_batch(
                batch,
                tokenizers,
                device=device,
            )

            mask_sample = (
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
                    rng=mask_rng,
                )
            )

            predictions = model(
                encoded,
                mask_sample.masks,
            )

            loss_output = (
                reference_mtm_loss(
                    encoded,
                    predictions,
                    mask_sample.masks,
                )
            )

            full_losses.append(
                float(
                    loss_output
                    .total_loss
                    .item()
                )
            )

            masked = (
                finite_masked_mean(
                    loss_output
                    .masked_losses
                )
            )

            if masked is not None:
                masked_losses.append(
                    masked
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
    )


def main() -> None:
    set_seed(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 72
    )

    print(
        "WALKER2D CLEAN MTM END-TO-END SMOKE"
    )

    print(
        "=" * 72
    )

    print(
        "device:",
        device,
    )

    print(
        "dataset:",
        DATASET_PATH,
    )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"missing frozen dataset: "
            f"{DATASET_PATH}"
        )

    # --------------------------------------------------
    # Load frozen raw dataset
    # --------------------------------------------------

    with h5py.File(
        DATASET_PATH,
        "r",
    ) as handle:

        observations = (
            handle[
                "observations"
            ][:]
        )

        actions = (
            handle[
                "actions"
            ][:]
        )

        rewards = (
            handle[
                "rewards"
            ][:]
        )

        terminals = (
            handle[
                "terminals"
            ][:]
            .astype(bool)
        )

        timeouts = (
            handle[
                "timeouts"
            ][:]
            .astype(bool)
        )

    # --------------------------------------------------
    # Reference-style project MTM dataset
    # --------------------------------------------------

    dataset = ReferenceMTMDataset(
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
        trajectory_length=(
            TRAJ_LENGTH
        ),
        max_path_length=(
            MAX_PATH_LENGTH
        ),
        discount=1.5,
    )

    assert (
        dataset.num_completed_trajectories
        == 1190
    )

    assert (
        dataset.num_used_transitions
        == 999_995
    )

    assert (
        dataset.trailing_transitions
        == 5
    )

    assert len(
        dataset
    ) == 996_425

    # --------------------------------------------------
    # Frozen reference split
    # --------------------------------------------------

    split = (
        reference_trajectory_split(
            dataset.trajectories,
            train_fraction=(
                TRAIN_FRACTION
            ),
        )
    )

    assert len(
        split.train_trajectories
    ) == 1130

    assert len(
        split.validation_trajectories
    ) == 60

    ranges = (
        build_split_window_ranges(
            dataset,
            split,
        )
    )

    print(
        "completed trajectories:",
        dataset.num_completed_trajectories,
    )

    print(
        "train trajectories:",
        len(
            split.train_trajectories
        ),
    )

    print(
        "validation trajectories:",
        len(
            split.validation_trajectories
        ),
    )

    print(
        "train windows:",
        ranges.train.count,
    )

    print(
        "validation windows:",
        ranges.validation.count,
    )

    assert (
        ranges.train.count
        + ranges.validation.count
        == len(dataset)
    )

    # --------------------------------------------------
    # Reference padded TRAIN statistics
    #
    # Validation trajectories are intentionally excluded.
    # --------------------------------------------------

    statistics = (
        compute_reference_mtm_statistics(
            observations,
            actions,
            dataset.returns,
            split.train_trajectories,
            max_path_length=(
                MAX_PATH_LENGTH
            ),
        )
    )

    tokenizers = {
        key: (
            ContinuousTokenizer
            .from_statistics(
                statistics[
                    key
                ]
            )
            .to(device)
        )
        for key in (
            "states",
            "actions",
            "returns",
        )
    }

    # --------------------------------------------------
    # Model
    #
    # This remains a smoke architecture, NOT the final
    # 512-dimensional reference run.
    # --------------------------------------------------

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
        "parameters:",
        sum(
            parameter.numel()
            for parameter
            in model.parameters()
        ),
    )

    print(
        "batch size:",
        BATCH_SIZE,
    )

    print(
        "updates:",
        NUM_UPDATES,
    )

    # --------------------------------------------------
    # Initial deterministic validation
    # --------------------------------------------------

    (
        initial_val_full,
        initial_val_masked,
    ) = evaluate(
        model,
        dataset,
        ranges.validation,
        tokenizers,
        device=device,
    )

    print()
    print(
        "initial_val_full:",
        f"{initial_val_full:.8f}",
    )

    print(
        "initial_val_masked:",
        f"{initial_val_masked:.8f}",
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    train_batch_rng = (
        np.random.RandomState(
            SEED + 1
        )
    )

    train_mask_rng = (
        np.random.RandomState(
            SEED + 2
        )
    )

    running_losses = []

    model.train()

    for update in range(
        1,
        NUM_UPDATES + 1,
    ):
        batch = sample_mtm_batch(
            dataset,
            ranges.train,
            batch_size=BATCH_SIZE,
            rng=train_batch_rng,
        )

        # Hard leakage assertion.
        if not bool(
            np.all(
                batch.trajectory_ids
                < len(
                    split.train_ids
                )
            )
        ):
            raise RuntimeError(
                "validation trajectory leaked "
                "into training batch"
            )

        encoded = encode_batch(
            batch,
            tokenizers,
            device=device,
        )

        mask_sample = (
            sample_reference_auto_mask(
                data_shapes,
                traj_length=(
                    TRAJ_LENGTH
                ),
                device=device,
                rng=train_mask_rng,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(
            encoded,
            mask_sample.masks,
        )

        loss_output = (
            reference_mtm_loss(
                encoded,
                predictions,
                mask_sample.masks,
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
                f"loss="
                f"{loss.item():.8f} "
                f"running100="
                f"{np.mean(running_losses):.8f} "
                f"mode="
                f"{mask_sample.selected_mode} "
                f"position="
                f"{mask_sample.selected_position}"
            )

    # --------------------------------------------------
    # Final deterministic validation
    # --------------------------------------------------

    (
        final_val_full,
        final_val_masked,
    ) = evaluate(
        model,
        dataset,
        ranges.validation,
        tokenizers,
        device=device,
    )

    full_ratio = (
        final_val_full
        / initial_val_full
    )

    masked_ratio = (
        final_val_masked
        / initial_val_masked
    )

    print()
    print(
        "initial_val_full:",
        f"{initial_val_full:.8f}",
    )

    print(
        "final_val_full:  ",
        f"{final_val_full:.8f}",
    )

    print(
        "full_ratio:      ",
        f"{full_ratio:.8f}",
    )

    print()

    print(
        "initial_val_masked:",
        f"{initial_val_masked:.8f}",
    )

    print(
        "final_val_masked:  ",
        f"{final_val_masked:.8f}",
    )

    print(
        "masked_ratio:      ",
        f"{masked_ratio:.8f}",
    )

    if not (
        math.isfinite(
            final_val_full
        )
        and math.isfinite(
            final_val_masked
        )
    ):
        raise RuntimeError(
            "WALKER2D SMOKE FAILED: "
            "non-finite validation metric"
        )

    # This is an integration smoke gate,
    # not a benchmark-performance threshold.
    #
    # The fixed validation bank must improve materially.
    if full_ratio >= 0.90:
        raise RuntimeError(
            "WALKER2D SMOKE FAILED: "
            "full validation reconstruction "
            "did not improve by at least 10%"
        )

    if masked_ratio >= 1.00:
        raise RuntimeError(
            "WALKER2D SMOKE FAILED: "
            "masked validation reconstruction "
            "did not improve"
        )

    print()
    print(
        "[PASS] Walker2d clean MTM "
        "end-to-end smoke"
    )


if __name__ == "__main__":
    main()