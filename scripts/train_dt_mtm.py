from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path
from random import Random

import numpy as np
import torch

from src.data.mtm_batching import MTMShuffledWindowSampler
from src.methods.dt.optim import create_dt_optimizer_and_scheduler
from src.methods.dt_mtm.clean_pipeline import (
    build_fixed_clean_probe,
    build_joint_model,
    build_mtm_tokenizers,
    evaluate_clean_probe,
    load_clean_walker2d,
    sample_clean_joint_train_batch,
)
from src.methods.dt_mtm.trainer import DTMTMTrainer
from src.methods.dt_mtm.walker2d import trainable_parameter_count


RESUME_LOCKED_FIELDS = (
    "seed",
    "num_updates",
    "dt_batch_size",
    "mtm_batch_size",
    "lambda_mtm",
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "grad_clip_norm",
    "auxiliary_grad_clip_norm",
    "diagnostics_every",
    "probe_every",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--lambda-mtm", type=float, required=True)

    # Full Group-1 training horizon. Use --stop-after for smoke/partial runs so
    # a later resume keeps the same scientific schedule.
    parser.add_argument("--num-updates", type=int, default=100_000)
    parser.add_argument("--stop-after", type=int, default=None)

    parser.add_argument("--dt-batch-size", type=int, default=64)
    parser.add_argument("--mtm-batch-size", type=int, default=512)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--grad-clip-norm", type=float, default=0.25)
    parser.add_argument("--auxiliary-grad-clip-norm", type=float, default=0.25)

    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--diagnostics-every", type=int, default=100)
    parser.add_argument("--probe-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)

    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def lambda_slug(value: float) -> str:
    return format(value, ".8g").replace("-", "m").replace(".", "p")


def save_checkpoint(
    path: Path,
    policy_path: Path,
    *,
    step: int,
    model,
    optimizer,
    scheduler,
    args: argparse.Namespace,
    git_commit: str | None,
    dt_np_rng: np.random.Generator,
    dt_py_rng: Random,
    mtm_sampler: MTMShuffledWindowSampler,
    mask_rng: np.random.RandomState,
    initial_probe: dict,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
        "git_commit": git_commit,
        "initial_probe": initial_probe,
        "elapsed_seconds": float(elapsed_seconds),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "dt_np_rng_state": dt_np_rng.bit_generator.state,
        "dt_py_rng_state": dt_py_rng.getstate(),
        "mtm_sampler_state": mtm_sampler.state_dict(),
        "mask_rng_state": mask_rng.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda_random_states"] = torch.cuda.get_rng_state_all()

    torch.save(state, path)

    # DT-only policy checkpoint: this keeps evaluation independent of the MTM
    # branch and compatible with the existing Decision Transformer loader.
    torch.save(
        {
            "update": step,
            "seed": args.seed,
            "lambda_mtm": args.lambda_mtm,
            "joint_checkpoint": str(path),
            "model_state_dict": model.dt.state_dict(),
        },
        policy_path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def validate_resume(args: argparse.Namespace, saved_args: dict) -> None:
    mismatches = []
    for field in RESUME_LOCKED_FIELDS:
        if field not in saved_args:
            raise ValueError(f"checkpoint missing locked field: {field}")
        if getattr(args, field) != saved_args[field]:
            mismatches.append((field, saved_args[field], getattr(args, field)))
    if mismatches:
        lines = ["resume configuration mismatch:"]
        lines.extend(
            f"  {field}: checkpoint={saved!r}, current={current!r}"
            for field, saved, current in mismatches
        )
        raise ValueError("\n".join(lines))


def main() -> None:
    args = parse_args()

    if args.lambda_mtm <= 0.0:
        raise ValueError("Primary Group-4 joint run requires --lambda-mtm > 0")
    if args.num_updates <= 0:
        raise ValueError("--num-updates must be positive")
    if args.num_updates < args.warmup_steps:
        raise ValueError("--num-updates must be >= --warmup-steps")
    if args.stop_after is not None and not (0 < args.stop_after <= args.num_updates):
        raise ValueError("--stop-after must lie in [1, num-updates]")
    for name in (
        "dt_batch_size",
        "mtm_batch_size",
        "log_every",
        "diagnostics_every",
        "probe_every",
        "checkpoint_every",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(args.resume)
        if args.output_dir is None:
            args.output_dir = args.resume.parent.parent
    elif args.output_dir is None:
        args.output_dir = Path(
            "experiments/dt_mtm/walker2d_medium_clean/"
            f"seed_{args.seed}/lambda_{lambda_slug(args.lambda_mtm)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    policies_dir = args.output_dir / "policies"
    metrics_path = args.output_dir / "training_metrics.jsonl"
    probe_path = args.output_dir / "clean_probe.jsonl"
    summary_path = args.output_dir / "summary.json"

    if args.resume is None:
        for path in (metrics_path, probe_path, summary_path):
            if path.exists():
                raise FileExistsError(
                    f"Fresh run would overwrite existing artifact: {path}"
                )
        metrics_path.write_text("", encoding="utf-8")
        probe_path.write_text("", encoding="utf-8")
    else:
        if not metrics_path.exists() or not probe_path.exists():
            raise FileNotFoundError(
                "resume requested but training_metrics.jsonl/clean_probe.jsonl "
                "is missing from the run directory"
            )

    git_commit = get_git_commit()

    print("=" * 72)
    print("GROUP 4B CLEAN DT+MTM TRAINING")
    print("=" * 72)
    print("device:", device)
    print("seed:", args.seed)
    print("lambda_mtm:", args.lambda_mtm)
    print("full horizon:", args.num_updates)
    print("requested stop:", args.stop_after or args.num_updates)
    print("DT batch:", args.dt_batch_size)
    print("MTM batch:", args.mtm_batch_size)
    print("git commit:", git_commit)

    data = load_clean_walker2d()
    tokenizers = build_mtm_tokenizers(data, device=device)
    model = build_joint_model(seed=args.seed, device=device)

    print("trainable parameters:", trainable_parameter_count(model))

    optimizer, scheduler = create_dt_optimizer_and_scheduler(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
    )
    trainer = DTMTMTrainer(
        model,
        optimizer,
        scheduler,
        device,
        lambda_mtm=args.lambda_mtm,
        grad_clip_norm=args.grad_clip_norm,
        auxiliary_grad_clip_norm=args.auxiliary_grad_clip_norm,
        diagnose_shared_gradients=True,
    )

    dt_np_rng = np.random.default_rng(args.seed)
    dt_py_rng = Random(args.seed)
    mtm_sampler = MTMShuffledWindowSampler(
        data.mtm_ranges.train,
        seed=args.seed + 1_000,
    )
    mask_rng = np.random.RandomState(args.seed + 2_000)

    fixed_probe = build_fixed_clean_probe(
        data,
        tokenizers,
        seed=args.seed,
        device=device,
        dt_batch_size=256,
        mtm_batch_size=256,
    )

    start_step = 0
    elapsed_before = 0.0

    if args.resume is None:
        initial_probe_metrics = evaluate_clean_probe(
            model,
            fixed_probe,
            device=device,
        )
        initial_probe = initial_probe_metrics.__dict__
        append_jsonl(
            probe_path,
            {"step": 0, **initial_probe},
        )
        append_jsonl(
            metrics_path,
            {
                "type": "config",
                "git_commit": git_commit,
                "seed": args.seed,
                "lambda_mtm": args.lambda_mtm,
                "num_updates": args.num_updates,
                "dt_batch_size": args.dt_batch_size,
                "mtm_batch_size": args.mtm_batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "grad_clip_norm": args.grad_clip_norm,
                "auxiliary_grad_clip_norm": args.auxiliary_grad_clip_norm,
                "diagnostics_every": args.diagnostics_every,
                "probe_every": args.probe_every,
                "trainable_parameters": trainable_parameter_count(model),
                "train_trajectories_mtm": len(data.mtm_split.train_ids),
                "validation_trajectories_mtm": len(data.mtm_split.validation_ids),
            },
        )
    else:
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

        dt_np_rng.bit_generator.state = checkpoint["dt_np_rng_state"]
        dt_py_rng.setstate(checkpoint["dt_py_rng_state"])
        mtm_sampler.load_state_dict(checkpoint["mtm_sampler_state"])
        mask_rng.set_state(checkpoint["mask_rng_state"])

        start_step = int(checkpoint["step"])
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        initial_probe = dict(checkpoint["initial_probe"])

        append_jsonl(
            metrics_path,
            {
                "type": "resume",
                "step": start_step,
                "checkpoint": str(args.resume),
                "git_commit": git_commit,
            },
        )
        print("resumed from step:", start_step)

    target_step = args.stop_after or args.num_updates
    if target_step <= start_step:
        raise ValueError(
            f"target step {target_step} must be greater than resume step {start_step}"
        )

    start_time = time.time() - elapsed_before

    for step in range(start_step + 1, target_step + 1):
        batch, audit = sample_clean_joint_train_batch(
            data,
            tokenizers,
            dt_np_rng=dt_np_rng,
            dt_py_rng=dt_py_rng,
            mtm_sampler=mtm_sampler,
            mask_rng=mask_rng,
            device=device,
            dt_batch_size=args.dt_batch_size,
            mtm_batch_size=args.mtm_batch_size,
        )

        trainer.diagnose_shared_gradients = (
            step == 1 or step % args.diagnostics_every == 0
        )
        metrics = trainer.train_step(batch)

        if step == 1 or step % args.log_every == 0:
            record = {
                "type": "train",
                "step": step,
                "total_loss": metrics.total_loss,
                "dt_loss": metrics.dt_loss,
                "mtm_loss": metrics.mtm_loss,
                "mtm_scaled_loss": metrics.mtm_scaled_loss,
                "mtm_state_loss": metrics.mtm_state_loss,
                "mtm_action_loss": metrics.mtm_action_loss,
                "mtm_return_loss": metrics.mtm_return_loss,
                "grad_norm_pre_clip": metrics.grad_norm_pre_clip,
                "dt_grad_norm_pre_clip": metrics.dt_grad_norm_pre_clip,
                "auxiliary_grad_norm_pre_clip": metrics.auxiliary_grad_norm_pre_clip,
                "shared_dt_grad_norm": finite_or_none(metrics.shared_dt_grad_norm),
                "shared_mtm_grad_norm": finite_or_none(metrics.shared_mtm_grad_norm),
                "shared_mtm_scaled_grad_norm": finite_or_none(
                    metrics.shared_mtm_scaled_grad_norm
                ),
                "shared_grad_cosine": finite_or_none(metrics.shared_grad_cosine),
                "learning_rate": metrics.learning_rate,
                "mask_mode": audit.mask_mode,
                "mask_position": audit.mask_position,
                "elapsed_seconds": time.time() - start_time,
            }
            append_jsonl(metrics_path, record)
            cosine_text = (
                "n/a"
                if record["shared_grad_cosine"] is None
                else f"{record['shared_grad_cosine']:+.4f}"
            )
            print(
                f"step={step:6d}/{args.num_updates:6d} "
                f"total={metrics.total_loss:.6f} "
                f"dt={metrics.dt_loss:.6f} "
                f"mtm={metrics.mtm_loss:.6f} "
                f"cos={cosine_text} "
                f"lr={metrics.learning_rate:.8e}"
            )

        if step % args.probe_every == 0 or step == target_step:
            probe_metrics = evaluate_clean_probe(
                model,
                fixed_probe,
                device=device,
            )
            append_jsonl(
                probe_path,
                {"step": step, **probe_metrics.__dict__},
            )
            print(
                "clean probe "
                f"step={step} "
                f"dt_mse={probe_metrics.dt_action_mse:.6f} "
                f"mtm={probe_metrics.mtm_total_loss:.6f}"
            )

        if step % args.checkpoint_every == 0 or step == target_step:
            save_checkpoint(
                checkpoints_dir / f"joint_step_{step:06d}.pt",
                policies_dir / f"dt_step_{step:06d}.pt",
                step=step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                git_commit=git_commit,
                dt_np_rng=dt_np_rng,
                dt_py_rng=dt_py_rng,
                mtm_sampler=mtm_sampler,
                mask_rng=mask_rng,
                initial_probe=initial_probe,
                elapsed_seconds=time.time() - start_time,
            )

    final_probe_metrics = evaluate_clean_probe(
        model,
        fixed_probe,
        device=device,
    )

    summary = {
        "seed": args.seed,
        "git_commit": git_commit,
        "lambda_mtm": args.lambda_mtm,
        "num_updates": args.num_updates,
        "final_step": target_step,
        "status": "complete" if target_step == args.num_updates else "partial",
        "dt_batch_size": args.dt_batch_size,
        "mtm_batch_size": args.mtm_batch_size,
        "initial_probe": initial_probe,
        "final_probe": final_probe_metrics.__dict__,
        "elapsed_seconds": time.time() - start_time,
        "joint_checkpoint": str(
            checkpoints_dir / f"joint_step_{target_step:06d}.pt"
        ),
        "dt_policy_checkpoint": str(
            policies_dir / f"dt_step_{target_step:06d}.pt"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("GROUP 4B CLEAN TRAINING STOPPED AT REQUESTED STEP")
    print("=" * 72)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
