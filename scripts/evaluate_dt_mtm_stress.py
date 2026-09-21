from __future__ import annotations

import argparse
import json
from pathlib import Path

import gym
import numpy as np
import torch

from src.evaluation.walker2d import evaluate_dt_episode
from src.methods.dt.checkpoint import load_dt_checkpoint_compat
from src.methods.dt.model import DecisionTransformer


NORMALIZATION_PATH = Path(
    "data/metadata/walker2d_medium_normalization.npz"
)


def build_policy_model() -> DecisionTransformer:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=("canonical", "s2_overlap_r0"),
        required=True,
    )
    parser.add_argument("--rho", type=float, choices=(0.01, 0.05), required=True)
    parser.add_argument("--attack-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--target-return", type=float, default=5000.0)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--eval-seed-base", type=int, default=30000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not NORMALIZATION_PATH.exists():
        raise FileNotFoundError(NORMALIZATION_PATH)

    device = torch.device("cpu")
    model = build_policy_model().to(device)
    checkpoint = load_dt_checkpoint_compat(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    with np.load(NORMALIZATION_PATH) as handle:
        state_mean = handle["state_mean"].copy()
        state_std = handle["state_std"].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = args.output_dir / "episodes.jsonl"
    summary_path = args.output_dir / "summary.json"
    if episodes_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"Evaluation output already exists under {args.output_dir}"
        )

    env = gym.make("Walker2d-v3")
    records = []
    try:
        for episode_index in range(args.num_episodes):
            episode_seed = args.eval_seed_base + episode_index
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
                "dataset": "walker2d-medium-v2",
                "environment": "Walker2d-v3",
                "method": "dt_mtm_joint",
                "condition": args.condition,
                "rho": float(args.rho),
                "attack_seed": int(args.attack_seed),
                "training_seed": int(checkpoint["seed"]),
                "training_update": int(checkpoint["update"]),
                "lambda_mtm": float(checkpoint["lambda_mtm"]),
                "checkpoint": str(args.checkpoint),
                "target_return": float(args.target_return),
                "evaluation_episode": episode_index,
                "evaluation_seed": episode_seed,
                "episode_length": int(result.episode_length),
                "raw_return": float(result.raw_return),
                "normalized_return": float(result.normalized_return),
            }
            records.append(record)
            with episodes_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"episode {episode_index:03d} | seed={episode_seed} | "
                f"len={result.episode_length} | raw={result.raw_return:.3f} | "
                f"normalized={result.normalized_return:.3f}"
            )
    finally:
        env.close()

    raw_returns = np.asarray([x["raw_return"] for x in records], dtype=np.float64)
    normalized_returns = np.asarray(
        [x["normalized_return"] for x in records], dtype=np.float64
    )
    lengths = np.asarray([x["episode_length"] for x in records], dtype=np.float64)

    summary = {
        "dataset": "walker2d-medium-v2",
        "environment": "Walker2d-v3",
        "method": "dt_mtm_joint",
        "condition": args.condition,
        "rho": float(args.rho),
        "attack_seed": int(args.attack_seed),
        "training_seed": int(checkpoint["seed"]),
        "training_update": int(checkpoint["update"]),
        "lambda_mtm": float(checkpoint["lambda_mtm"]),
        "checkpoint": str(args.checkpoint),
        "target_return": float(args.target_return),
        "num_episodes": args.num_episodes,
        "eval_seed_base": args.eval_seed_base,
        "raw_return_mean": float(raw_returns.mean()),
        "raw_return_std": float(raw_returns.std()),
        "normalized_return_mean": float(normalized_returns.mean()),
        "normalized_return_std": float(normalized_returns.std()),
        "episode_length_mean": float(lengths.mean()),
        "episode_length_std": float(lengths.std()),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
