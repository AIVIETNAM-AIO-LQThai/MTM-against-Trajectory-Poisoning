from __future__ import annotations

import argparse
import json
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


NORMALIZATION_PATH = Path(
    "data/metadata/"
    "walker2d_medium_normalization.npz"
)


def make_model():
    return DecisionTransformer(
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--target-return",
        type=float,
        default=5000.0,
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--eval-seed-base",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.num_episodes <= 0:
        raise ValueError(
            "--num-episodes must be positive."
        )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            args.checkpoint
        )

    if not NORMALIZATION_PATH.exists():
        raise FileNotFoundError(
            NORMALIZATION_PATH
        )

    device = torch.device(
        "cpu"
    )

    model = make_model().to(
        device
    )

    checkpoint = (
        load_dt_checkpoint_compat(
            model=model,
            checkpoint_path=args.checkpoint,
            device=device,
        )
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

    if state_mean.shape != (17,):
        raise RuntimeError(
            "Unexpected state_mean shape."
        )

    if state_std.shape != (17,):
        raise RuntimeError(
            "Unexpected state_std shape."
        )

    env = gym.make(
        "Walker2d-v3"
    )

    if (
        env.observation_space.shape
        != (17,)
    ):
        raise RuntimeError(
            "Unexpected Walker2d "
            "observation space."
        )

    if (
        env.action_space.shape
        != (6,)
    ):
        raise RuntimeError(
            "Unexpected Walker2d "
            "action space."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    episodes_path = (
        args.output_dir
        / "episodes.jsonl"
    )

    summary_path = (
        args.output_dir
        / "summary.json"
    )

    # Never accidentally append new evaluation episodes
    # to an old run.
    if episodes_path.exists():
        raise FileExistsError(
            "Evaluation output already exists: "
            f"{episodes_path}"
        )

    if summary_path.exists():
        raise FileExistsError(
            "Evaluation output already exists: "
            f"{summary_path}"
        )

    records = []

    try:
        for episode_index in range(
            args.num_episodes
        ):
            episode_seed = (
                args.eval_seed_base
                + episode_index
            )

            result = evaluate_dt_episode(
                env=env,
                model=model,
                state_mean=state_mean,
                state_std=state_std,
                target_return=args.target_return,
                episode_seed=episode_seed,
                context_length=20,
                scale=1000.0,
                max_ep_len=1000,
                device=device,
            )

            record = {
                "dataset": (
                    "walker2d-medium-v2"
                ),
                "environment": (
                    "Walker2d-v3"
                ),
                "method": "dt",
                "attack": "none",
                "poison_rate": 0.0,
                "training_seed": (
                    int(checkpoint["seed"])
                ),
                "training_update": (
                    int(checkpoint["update"])
                ),
                "checkpoint": str(
                    args.checkpoint
                ),
                "target_return": float(
                    args.target_return
                ),
                "evaluation_episode": int(
                    episode_index
                ),
                "evaluation_seed": int(
                    episode_seed
                ),
                "episode_length": int(
                    result.episode_length
                ),
                "raw_return": float(
                    result.raw_return
                ),
                "normalized_return": float(
                    result.normalized_return
                ),
            }

            records.append(
                record
            )

            with episodes_path.open(
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                    )
                    + "\n"
                )

            print(
                "episode "
                f"{episode_index:03d} | "
                f"seed={episode_seed} | "
                f"len={result.episode_length} | "
                f"raw={result.raw_return:.3f} | "
                "normalized="
                f"{result.normalized_return:.3f}"
            )

    finally:
        env.close()

    raw_returns = np.asarray(
        [
            record["raw_return"]
            for record in records
        ],
        dtype=np.float64,
    )

    normalized_returns = np.asarray(
        [
            record["normalized_return"]
            for record in records
        ],
        dtype=np.float64,
    )

    episode_lengths = np.asarray(
        [
            record["episode_length"]
            for record in records
        ],
        dtype=np.float64,
    )

    summary = {
        "dataset": (
            "walker2d-medium-v2"
        ),
        "environment": (
            "Walker2d-v3"
        ),
        "method": "dt",
        "attack": "none",
        "poison_rate": 0.0,
        "training_seed": int(
            checkpoint["seed"]
        ),
        "training_update": int(
            checkpoint["update"]
        ),
        "checkpoint": str(
            args.checkpoint
        ),
        "target_return": float(
            args.target_return
        ),
        "num_episodes": int(
            args.num_episodes
        ),
        "eval_seed_base": int(
            args.eval_seed_base
        ),
        "raw_return_mean": float(
            raw_returns.mean()
        ),
        "raw_return_std": float(
            raw_returns.std()
        ),
        "normalized_return_mean": float(
            normalized_returns.mean()
        ),
        "normalized_return_std": float(
            normalized_returns.std()
        ),
        "episode_length_mean": float(
            episode_lengths.mean()
        ),
        "episode_length_std": float(
            episode_lengths.std()
        ),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            sort_keys=True,
        )

    print()
    print("Evaluation summary:")
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()