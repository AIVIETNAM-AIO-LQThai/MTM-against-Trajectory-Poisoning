from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
from random import Random

import h5py
import numpy as np
import torch

from src.data.batching import sample_dt_batch
from src.data.trajectories import find_completed_trajectories
from src.methods.dt.model import DecisionTransformer
from src.methods.dt.optim import create_dt_optimizer_and_scheduler
from src.methods.dt.trainer import DTTrainer


CLEAN_DATASET_PATH = Path(
    "data/raw/walker2d-medium-v2/walker2d_medium-v2.hdf5"
)
NORMALIZATION_PATH = Path(
    "data/metadata/walker2d_medium_normalization.npz"
)

RESUME_LOCKED_FIELDS = (
    "dataset",
    "condition",
    "rho",
    "attack_seed",
    "seed",
    "num_updates",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "grad_clip_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--attack-seed", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-updates", type=int, default=100_000)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--grad-clip-norm", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def make_model() -> DecisionTransformer:
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


def load_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        observations = handle["observations"][:]
        actions = handle["actions"][:]
        rewards = handle["rewards"][:]
        terminals = handle["terminals"][:].astype(bool)
        timeouts = handle["timeouts"][:].astype(bool)

    trajectories, trailing = find_completed_trajectories(terminals, timeouts)
    used = sum(t.length for t in trajectories)
    if len(trajectories) != 1190 or used != 999_995 or trailing != 5:
        raise RuntimeError(
            "dataset trajectory contract changed: "
            f"trajectories={len(trajectories)} used={used} trailing={trailing}"
        )
    return observations, actions, rewards, terminals, trajectories, used, trailing


def load_normalization():
    with np.load(NORMALIZATION_PATH) as handle:
        state_mean = handle["state_mean"].copy()
        state_std = handle["state_std"].copy()
        ntrans = int(handle["num_training_transitions"])
        ntraj = int(handle["num_trajectories"])
    if ntrans != 999_995 or ntraj != 1190:
        raise RuntimeError("frozen clean normalization metadata mismatch")
    return state_mean, state_std


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model,
    optimizer,
    scheduler,
    args: argparse.Namespace,
    np_rng: np.random.Generator,
    py_rng: Random,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "update": step,
        "seed": args.seed,
        "condition": args.condition,
        "rho": args.rho,
        "attack_seed": args.attack_seed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
        "elapsed_seconds": float(elapsed_seconds),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "dt_np_rng_state": np_rng.bit_generator.state,
        "dt_py_rng_state": py_rng.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda_random_states"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def validate_resume(args: argparse.Namespace, saved: dict) -> None:
    mismatches = []
    for field in RESUME_LOCKED_FIELDS:
        if field not in saved:
            raise ValueError(f"checkpoint missing locked field: {field}")
        current = getattr(args, field)
        previous = saved[field]
        if field == "dataset":
            current = Path(current)
            previous = Path(previous)
        if current != previous:
            mismatches.append((field, previous, current))
    if mismatches:
        text = "\n".join(
            f"  {field}: checkpoint={old!r}, current={new!r}"
            for field, old, new in mismatches
        )
        raise ValueError("resume configuration mismatch:\n" + text)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.num_updates <= 0:
        raise ValueError("--num-updates must be positive")
    if args.stop_after is not None and not (0 < args.stop_after <= args.num_updates):
        raise ValueError("--stop-after must lie in [1, num-updates]")
    if args.batch_size != 64:
        raise ValueError("Group 4C freezes --batch-size at 64")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        observations,
        actions,
        rewards,
        terminals,
        trajectories,
        used_transitions,
        trailing,
    ) = load_dataset(args.dataset)
    state_mean, state_std = load_normalization()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.jsonl"
    manifest_path = args.output_dir / "run_manifest.json"
    summary_path = args.output_dir / "summary.json"
    checkpoint_dir = args.output_dir / "checkpoints"

    model = make_model().to(device)
    optimizer, scheduler = create_dt_optimizer_and_scheduler(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
    )
    trainer = DTTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        grad_clip_norm=args.grad_clip_norm,
    )

    np_rng = np.random.default_rng(args.seed)
    py_rng = Random(args.seed)
    start_step = 0
    elapsed_before = 0.0

    if args.resume is None:
        for path in (metrics_path, manifest_path, summary_path):
            if path.exists():
                raise FileExistsError(f"fresh run would overwrite {path}")
        metrics_path.write_text("", encoding="utf-8")
        manifest = {
            "dataset": str(args.dataset),
            "dataset_sha256": sha256_file(args.dataset),
            "clean_normalization": str(NORMALIZATION_PATH),
            "condition": args.condition,
            "rho": args.rho,
            "attack_seed": args.attack_seed,
            "seed": args.seed,
            "num_updates": args.num_updates,
            "batch_size": args.batch_size,
            "context_length": 20,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "grad_clip_norm": args.grad_clip_norm,
            "rtg_scale": 1000.0,
            "max_ep_len": 1000,
            "num_trajectories": len(trajectories),
            "num_training_transitions": used_transitions,
            "trailing_transitions": trailing,
            "device": str(device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "git_commit": get_git_commit(),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    else:
        if not args.resume.exists():
            raise FileNotFoundError(args.resume)
        if not metrics_path.exists() or not manifest_path.exists():
            raise FileNotFoundError("resume requested but run metadata is missing")
        checkpoint = load_checkpoint(args.resume, device)
        validate_resume(args, checkpoint["args"])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        if device.type == "cuda" and "cuda_random_states" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_random_states"]]
            )
        np_rng.bit_generator.state = checkpoint["dt_np_rng_state"]
        py_rng.setstate(checkpoint["dt_py_rng_state"])
        start_step = int(checkpoint["step"])
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        append_jsonl(
            metrics_path,
            {"type": "resume", "step": start_step, "checkpoint": str(args.resume)},
        )
        print(f"resumed from step: {start_step}")

    target_step = args.stop_after or args.num_updates
    if target_step <= start_step:
        raise ValueError(
            f"target step {target_step} must be greater than resume step {start_step}"
        )

    print("=" * 72)
    print("GROUP 4C VANILLA DT TRAINING")
    print("=" * 72)
    print("device:", device)
    print("dataset:", args.dataset)
    print("condition:", args.condition)
    print("rho:", args.rho)
    print("attack_seed:", args.attack_seed)
    print("training seed:", args.seed)
    print("start step:", start_step)
    print("target step:", target_step)

    start_time = time.time() - elapsed_before
    for update in range(start_step + 1, target_step + 1):
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
        metrics = trainer.train_step(batch)
        if update == 1 or update % args.log_every == 0:
            payload = {
                "type": "train",
                "update": update,
                "loss": metrics.loss,
                "grad_norm_pre_clip": metrics.grad_norm_pre_clip,
                "learning_rate": metrics.learning_rate,
                "elapsed_seconds": time.time() - start_time,
            }
            append_jsonl(metrics_path, payload)
            print(
                f"update={update:6d}/{args.num_updates:6d} "
                f"loss={metrics.loss:.6f} "
                f"grad={metrics.grad_norm_pre_clip:.4f} "
                f"lr={metrics.learning_rate:.8e}"
            )

        if update % args.checkpoint_every == 0 or update == target_step:
            save_checkpoint(
                checkpoint_dir / f"checkpoint_step_{update:06d}.pt",
                step=update,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                np_rng=np_rng,
                py_rng=py_rng,
                elapsed_seconds=time.time() - start_time,
            )

    final_checkpoint = checkpoint_dir / f"checkpoint_step_{target_step:06d}.pt"
    summary = {
        "status": "complete" if target_step == args.num_updates else "partial",
        "final_step": target_step,
        "condition": args.condition,
        "rho": args.rho,
        "attack_seed": args.attack_seed,
        "training_seed": args.seed,
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "checkpoint": str(final_checkpoint),
        "elapsed_seconds": time.time() - start_time,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
