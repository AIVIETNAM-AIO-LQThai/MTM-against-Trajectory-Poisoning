from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

from src.data.batching import sample_dt_batch
from src.data.trajectories import find_completed_trajectories
from src.methods.dt.model import DecisionTransformer
from src.methods.dt.optim import (
    create_dt_optimizer_and_scheduler,
)
from src.methods.dt.trainer import DTTrainer


DATASET_PATH = Path(
    "data/raw/walker2d-medium-v2/"
    "walker2d_medium-v2.hdf5"
)

NORMALIZATION_PATH = Path(
    "data/metadata/"
    "walker2d_medium_normalization.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-updates",
        type=int,
        default=100,
        help=(
            "Use 100 for smoke testing. "
            "Full reference run will use 100000."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/dt/"
            "walker2d_medium_clean/"
            "seed_0/smoke"
        ),
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("DECISION TRANSFORMER TRAINING")
    print("=" * 70)

    print(f"device:       {device}")
    print(f"seed:         {args.seed}")
    print(f"num_updates:  {args.num_updates}")
    print(f"batch_size:   {args.batch_size}")
    print(f"output_dir:   {args.output_dir}")

    # --------------------------------------------------
    # Load clean dataset
    # --------------------------------------------------

    with h5py.File(DATASET_PATH, "r") as f:
        observations = f[
            "observations"
        ][:]

        actions = f[
            "actions"
        ][:]

        rewards = f[
            "rewards"
        ][:]

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

    used_transitions = sum(
        trajectory.length
        for trajectory in trajectories
    )

    assert len(trajectories) == 1190
    assert used_transitions == 999_995
    assert trailing == 5

    print(
        f"trajectories: "
        f"{len(trajectories)}"
    )

    print(
        f"training transitions: "
        f"{used_transitions}"
    )

    # --------------------------------------------------
    # Frozen normalization
    # --------------------------------------------------

    with np.load(
        NORMALIZATION_PATH
    ) as f:
        state_mean = f[
            "state_mean"
        ].copy()

        state_std = f[
            "state_std"
        ].copy()

        norm_num_transitions = int(
            f[
                "num_training_transitions"
            ]
        )

        norm_num_trajectories = int(
            f["num_trajectories"]
        )

    assert (
        norm_num_transitions
        == used_transitions
    )

    assert (
        norm_num_trajectories
        == len(trajectories)
    )

    # --------------------------------------------------
    # Model
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

        resid_pdrop=0.1,
        attn_pdrop=0.1,
        embd_pdrop=0.1,

        action_tanh=True,
    ).to(device)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"trainable parameters: "
        f"{num_parameters:,}"
    )

    # --------------------------------------------------
    # Optimizer + reference warmup
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Independent RNGs used by batch sampler.
    # --------------------------------------------------

    np_rng = np.random.default_rng(
        args.seed
    )

    py_rng = random.Random(
        args.seed
    )

    # --------------------------------------------------
    # Save run manifest BEFORE training.
    # --------------------------------------------------

    manifest = {
        "dataset": "walker2d-medium-v2",
        "method": "dt",
        "attack": "none",
        "poison_rate": 0.0,

        "seed": args.seed,

        "num_updates": args.num_updates,
        "batch_size": args.batch_size,

        "context_length": 20,

        "state_dim": 17,
        "action_dim": 6,

        "hidden_size": 128,
        "n_layer": 3,
        "n_head": 1,
        "n_inner": 512,

        "activation": "relu",
        "dropout": 0.1,

        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 10_000,
        "grad_clip_norm": 0.25,

        "rtg_scale": 1000.0,
        "max_ep_len": 1000,

        "num_trajectories": len(
            trajectories
        ),

        "num_training_transitions": (
            used_transitions
        ),

        "trailing_transitions": trailing,

        "trainable_parameters": (
            num_parameters
        ),

        "device": str(device),

        "torch_version": torch.__version__,
        "numpy_version": np.__version__,

        "git_commit": get_git_commit(),

        "started_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
    }

    manifest_path = (
        args.output_dir
        / "run_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    # --------------------------------------------------
    # Training metrics JSONL
    # --------------------------------------------------

    metrics_path = (
        args.output_dir
        / "training_metrics.jsonl"
    )

    # Start clean for a new run.
    if metrics_path.exists():
        metrics_path.unlink()

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    print("\nTraining starts...\n")

    for update in range(
        1,
        args.num_updates + 1,
    ):
        batch = sample_dt_batch(
            observations,
            actions,
            rewards,
            terminals,
            trajectories,
            state_mean,
            state_std,

            batch_size=args.batch_size,
            context_length=20,
            max_ep_len=1000,
            rtg_scale=1000.0,

            np_rng=np_rng,
            py_rng=py_rng,
        )

        metrics = trainer.train_step(
            batch
        )

        record = {
            "update": update,
            "loss": metrics.loss,
            "grad_norm_pre_clip": metrics.grad_norm_pre_clip,
            "learning_rate": (
                metrics.learning_rate
            ),
        }

        with metrics_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(record)
                + "\n"
            )

        if (
            update == 1
            or update
            % args.log_every == 0
        ):
            print(
                f"update "
                f"{update:6d}/"
                f"{args.num_updates:6d} | "
                f"loss="
                f"{metrics.loss:.6f} | "
                f"grad="
                f"{metrics.grad_norm_pre_clip:.4f} | "
                f"lr="
                f"{metrics.learning_rate:.8e}"
            )

        if (
            args.checkpoint_every > 0
            and update
            % args.checkpoint_every == 0
        ):
            checkpoint_path = (
                args.output_dir
                / f"checkpoint_{update}.pt"
            )

            torch.save(
                {
                    "update": update,
                    "seed": args.seed,

                    "model_state_dict": (
                        model.state_dict()
                    ),

                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),

                    "scheduler_state_dict": (
                        scheduler.state_dict()
                    ),
                },
                checkpoint_path,
            )

    # --------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------

    final_checkpoint_path = (
        args.output_dir
        / "checkpoint_final.pt"
    )

    torch.save(
        {
            "update": args.num_updates,
            "seed": args.seed,

            "model_state_dict": (
                model.state_dict()
            ),

            "optimizer_state_dict": (
                optimizer.state_dict()
            ),

            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
        },
        final_checkpoint_path,
    )

    manifest["finished_at"] = (
        datetime.now()
        .astimezone()
        .isoformat()
    )

    manifest["final_checkpoint"] = (
        final_checkpoint_path.name
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(
        f"checkpoint -> "
        f"{final_checkpoint_path}"
    )
    print(
        f"metrics    -> "
        f"{metrics_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()