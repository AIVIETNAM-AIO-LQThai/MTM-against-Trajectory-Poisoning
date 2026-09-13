from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
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

from src.methods.mtm.optim import (
    create_reference_mtm_optimizer_and_scheduler,
)

from src.methods.mtm.tokenizers import (
    ContinuousTokenizer,
)


DATASET_PATH = Path(
    "data/raw/walker2d-medium-v2/"
    "walker2d_medium-v2.hdf5"
)

STATE_DIM = 17
ACTION_DIM = 6

TRAJ_LENGTH = 4
MAX_PATH_LENGTH = 1000
TRAIN_FRACTION = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    # --------------------------------------------------
    # Official/reference-scale defaults
    # --------------------------------------------------

    parser.add_argument(
        "--num-updates",
        type=int,
        default=140_010,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=40_000,
    )

    parser.add_argument(
        "--n-embd",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--n-head",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--n-enc-layer",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--n-dec-layer",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--num-eval-batches",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def resolve_device(
    requested: str,
) -> torch.device:

    if requested == "cpu":
        return torch.device(
            "cpu"
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested "
                "but CUDA is unavailable"
            )

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            )
            .strip()
        )

    except Exception:
        return None


def append_jsonl(
    path: Path,
    record: dict,
) -> None:
    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                sort_keys=True,
            )
            + "\n"
        )


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


def finite_mean(
    values: dict[
        str,
        torch.Tensor,
    ],
) -> float | None:

    result = []

    for value in values.values():
        scalar = float(
            value.item()
        )

        if math.isfinite(
            scalar
        ):
            result.append(
                scalar
            )

    if not result:
        return None

    return float(
        np.mean(
            result
        )
    )


def evaluate(
    model: ReferenceMTM,
    dataset: ReferenceMTMDataset,
    validation_range,
    tokenizers,
    *,
    seed: int,
    batch_size: int,
    num_batches: int,
    device: torch.device,
) -> dict:

    # Reset every evaluation.
    #
    # Therefore step 0, step 20000, etc. evaluate on
    # exactly the same windows and masks.

    batch_rng = (
        np.random.RandomState(
            seed + 100_000
        )
    )

    mask_rng = (
        np.random.RandomState(
            seed + 200_000
        )
    )

    total_losses = []

    modality_full = {
        "states": [],
        "actions": [],
        "returns": [],
    }

    modality_masked = {
        "states": [],
        "actions": [],
        "returns": [],
    }

    model.eval()

    with torch.no_grad():

        for _ in range(
            num_batches
        ):
            batch = sample_mtm_batch(
                dataset,
                validation_range,
                batch_size=batch_size,
                rng=batch_rng,
            )

            encoded = encode_batch(
                batch,
                tokenizers,
                device=device,
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
                    rng=mask_rng,
                )
            )

            predictions = model(
                encoded,
                sample.masks,
            )

            losses = (
                reference_mtm_loss(
                    encoded,
                    predictions,
                    sample.masks,
                )
            )

            total_losses.append(
                float(
                    losses
                    .total_loss
                    .item()
                )
            )

            for key in (
                "states",
                "actions",
                "returns",
            ):
                modality_full[
                    key
                ].append(
                    float(
                        losses
                        .full_losses[
                            key
                        ]
                        .item()
                    )
                )

                masked_value = float(
                    losses
                    .masked_losses[
                        key
                    ]
                    .item()
                )

                if math.isfinite(
                    masked_value
                ):
                    modality_masked[
                        key
                    ].append(
                        masked_value
                    )

    result = {
        "total_full": float(
            np.mean(
                total_losses
            )
        ),
    }

    for key in (
        "states",
        "actions",
        "returns",
    ):
        result[
            f"full_{key}"
        ] = float(
            np.mean(
                modality_full[
                    key
                ]
            )
        )

        if modality_masked[
            key
        ]:
            result[
                f"masked_{key}"
            ] = float(
                np.mean(
                    modality_masked[
                        key
                    ]
                )
            )

        else:
            result[
                f"masked_{key}"
            ] = None

    return result


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: ReferenceMTM,
    optimizer,
    scheduler,
    args: argparse.Namespace,
    git_commit: str | None,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = {
        "step": step,

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "scheduler_state_dict": (
            scheduler.state_dict()
        ),

        "args": vars(
            args
        ),

        "git_commit": git_commit,

        "python_random_state": (
            random.getstate()
        ),

        "numpy_random_state": (
            np.random.get_state()
        ),

        "torch_random_state": (
            torch.get_rng_state()
        ),
    }

    if torch.cuda.is_available():
        state[
            "cuda_random_states"
        ] = (
            torch.cuda
            .get_rng_state_all()
        )

    torch.save(
        state,
        path,
    )


def main() -> None:
    args = parse_args()

    if (
        args.num_updates
        <= args.warmup_steps
    ):
        raise ValueError(
            "num-updates must exceed "
            "warmup-steps"
        )

    set_seed(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    if args.output_dir is None:
        args.output_dir = Path(
            "experiments/mtm/"
            "walker2d_medium_reference/"
            f"seed_{args.seed}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoints_dir = (
        args.output_dir
        / "checkpoints"
    )

    checkpoints_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        args.output_dir
        / "training_metrics.jsonl"
    )

    summary_path = (
        args.output_dir
        / "evaluation_summary.json"
    )

    # Fresh run.
    metrics_path.write_text(
        "",
        encoding="utf-8",
    )

    git_commit = (
        get_git_commit()
    )

    print(
        "=" * 72
    )

    print(
        "REFERENCE-SCALE CLEAN MTM TRAINING"
    )

    print(
        "=" * 72
    )

    print(
        "device:",
        device,
    )

    print(
        "seed:",
        args.seed,
    )

    print(
        "updates:",
        args.num_updates,
    )

    print(
        "batch_size:",
        args.batch_size,
    )

    print(
        "git_commit:",
        git_commit,
    )

    # --------------------------------------------------
    # Dataset
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
        dataset
        .num_completed_trajectories
        == 1190
    )

    assert (
        dataset
        .num_used_transitions
        == 999_995
    )

    assert (
        dataset
        .trailing_transitions
        == 5
    )

    split = (
        reference_trajectory_split(
            dataset.trajectories,
            train_fraction=(
                TRAIN_FRACTION
            ),
        )
    )

    ranges = (
        build_split_window_ranges(
            dataset,
            split,
        )
    )

    assert len(
        split.train_trajectories
    ) == 1130

    assert len(
        split.validation_trajectories
    ) == 60

    # --------------------------------------------------
    # Train-only reference statistics
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
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_enc_layer=(
                args.n_enc_layer
            ),
            n_dec_layer=(
                args.n_dec_layer
            ),
            dropout=args.dropout,
        ),
    ).to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "parameters:",
        parameter_count,
    )

    # --------------------------------------------------
    # Optimizer / official schedule
    # --------------------------------------------------

    (
        optimizer,
        scheduler,
    ) = (
        create_reference_mtm_optimizer_and_scheduler(
            model.parameters(),
            learning_rate=(
                args.learning_rate
            ),
            weight_decay=(
                args.weight_decay
            ),
            warmup_steps=(
                args.warmup_steps
            ),
            num_train_steps=(
                args.num_updates
            ),
        )
    )

    # --------------------------------------------------
    # Record immutable run setup
    # --------------------------------------------------

    append_jsonl(
        metrics_path,
        {
            "type": "config",
            "git_commit": git_commit,
            "seed": args.seed,
            "device": str(
                device
            ),
            "num_updates": (
                args.num_updates
            ),
            "batch_size": (
                args.batch_size
            ),
            "learning_rate": (
                args.learning_rate
            ),
            "weight_decay": (
                args.weight_decay
            ),
            "warmup_steps": (
                args.warmup_steps
            ),
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_enc_layer": (
                args.n_enc_layer
            ),
            "n_dec_layer": (
                args.n_dec_layer
            ),
            "dropout": (
                args.dropout
            ),
            "parameters": (
                parameter_count
            ),
            "train_trajectories": (
                len(
                    split
                    .train_trajectories
                )
            ),
            "validation_trajectories": (
                len(
                    split
                    .validation_trajectories
                )
            ),
            "train_windows": (
                ranges.train.count
            ),
            "validation_windows": (
                ranges.validation.count
            ),
        },
    )

    # --------------------------------------------------
    # Initial fixed validation bank
    # --------------------------------------------------

    initial_eval = evaluate(
        model,
        dataset,
        ranges.validation,
        tokenizers,
        seed=args.seed,
        batch_size=min(
            args.batch_size,
            256,
        ),
        num_batches=(
            args.num_eval_batches
        ),
        device=device,
    )

    append_jsonl(
        metrics_path,
        {
            "type": "validation",
            "step": 0,
            **initial_eval,
        },
    )

    print(
        "initial validation:",
        initial_eval,
    )

    # --------------------------------------------------
    # Training RNG streams
    # --------------------------------------------------

    batch_rng = (
        np.random.RandomState(
            args.seed
            + 1_000
        )
    )

    mask_rng = (
        np.random.RandomState(
            args.seed
            + 2_000
        )
    )

    running = []

    start_time = time.time()

    for step in range(
        1,
        args.num_updates + 1,
    ):

        model.train()

        batch = sample_mtm_batch(
            dataset,
            ranges.train,
            batch_size=(
                args.batch_size
            ),
            rng=batch_rng,
        )

        if not bool(
            np.all(
                batch.trajectory_ids
                < len(
                    split.train_ids
                )
            )
        ):
            raise RuntimeError(
                "validation leakage into "
                "training batch"
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
                rng=mask_rng,
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
                f"non-finite loss "
                f"at step {step}"
            )

        # LR actually used for this update.
        lr_used = float(
            optimizer
            .param_groups[
                0
            ][
                "lr"
            ]
        )

        loss.backward()

        # Reference code:
        # optimizer first, scheduler second.
        optimizer.step()
        scheduler.step()

        loss_value = float(
            loss.item()
        )

        running.append(
            loss_value
        )

        if len(running) > 100:
            running.pop(
                0
            )

        if (
            step == 1
            or step
            % args.log_every
            == 0
        ):

            masked_mean = (
                finite_mean(
                    loss_output
                    .masked_losses
                )
            )

            elapsed = (
                time.time()
                - start_time
            )

            record = {
                "type": "train",
                "step": step,
                "loss": loss_value,
                "running100": float(
                    np.mean(
                        running
                    )
                ),
                "lr_used": lr_used,
                "masked_mean": (
                    masked_mean
                ),
                "mode": (
                    mask_sample
                    .selected_mode
                ),
                "position": (
                    mask_sample
                    .selected_position
                ),
                "elapsed_seconds": (
                    elapsed
                ),
            }

            for key, value in (
                loss_output
                .full_losses
                .items()
            ):
                record[
                    f"full_{key}"
                ] = float(
                    value.item()
                )

            append_jsonl(
                metrics_path,
                record,
            )

            print(
                f"step={step:6d} "
                f"loss="
                f"{loss_value:.6f} "
                f"running100="
                f"{np.mean(running):.6f} "
                f"lr="
                f"{lr_used:.8g}"
            )

        if (
            step
            % args.eval_every
            == 0
            or step
            == args.num_updates
        ):

            evaluation = evaluate(
                model,
                dataset,
                ranges.validation,
                tokenizers,
                seed=args.seed,
                batch_size=min(
                    args.batch_size,
                    256,
                ),
                num_batches=(
                    args.num_eval_batches
                ),
                device=device,
            )

            append_jsonl(
                metrics_path,
                {
                    "type": (
                        "validation"
                    ),
                    "step": step,
                    **evaluation,
                },
            )

            print(
                "validation:",
                step,
                evaluation,
            )

        if (
            step
            % args.checkpoint_every
            == 0
            or step
            == args.num_updates
        ):

            save_checkpoint(
                checkpoints_dir
                / (
                    f"step_"
                    f"{step:06d}.pt"
                ),
                step=step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                git_commit=(
                    git_commit
                ),
            )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    final_eval = evaluate(
        model,
        dataset,
        ranges.validation,
        tokenizers,
        seed=args.seed,
        batch_size=min(
            args.batch_size,
            256,
        ),
        num_batches=(
            args.num_eval_batches
        ),
        device=device,
    )

    summary = {
        "seed": args.seed,
        "git_commit": git_commit,
        "parameters": (
            parameter_count
        ),
        "num_updates": (
            args.num_updates
        ),
        "initial_validation": (
            initial_eval
        ),
        "final_validation": (
            final_eval
        ),
        "elapsed_seconds": (
            time.time()
            - start_time
        ),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 72
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()