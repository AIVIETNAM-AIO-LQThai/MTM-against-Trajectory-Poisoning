from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from src.methods.rdt_source_dt.data import (
    SourceSequenceDataset,
    load_source_trajectories,
)
from src.methods.rdt_source_dt.losses import (
    source_masked_action_mse,
)
from src.methods.rdt_source_dt.model import (
    make_source_compatible_dt,
)


ROOT = Path(
    __file__
).resolve().parents[
    1
]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "attack_qualification"
    / "rdt_source_dt_victim.json"
)

DEFAULT_DATASET = (
    ROOT
    / "data"
    / "derived"
    / "rdt_source_random_state_corruption"
    / "walker2d-medium-v2"
    / "clean_ratio_0p02.hdf5"
)

EXPECTED_DATASET_SHA256 = (
    "a2b0eedb6083b2d28f4ea0b5ce220dd"
    "6763df8c8152ce8209e98292a359ad401"
)

RESUME_LOCKED_FIELDS = (
    "dataset",
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

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--num-updates",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def sha256_file(
    path: Path,
) -> str:
    digest = (
        hashlib.sha256()
    )

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024
                * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


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


def set_source_seed(
    seed: int,
) -> None:
    # Match public RDT repository set_seed semantics.
    import os

    os.environ[
        "PYTHONHASHSEED"
    ] = str(
        seed
    )

    np.random.seed(
        seed
    )

    random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = (
        True
    )

    torch.backends.cudnn.benchmark = (
        False
    )


def make_optimizer_and_scheduler(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
):
    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter:
            parameter.requires_grad,
            model.parameters(),
        ),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(
            0.9,
            0.999,
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=(
                lambda step:
                min(
                    (
                        step
                        + 1
                    )
                    / warmup_steps,
                    1.0,
                )
            ),
        )
    )

    return (
        optimizer,
        scheduler,
    )


def save_checkpoint(
    path: Path,
    *,
    update: int,
    model,
    optimizer,
    scheduler,
    args: argparse.Namespace,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = {
        "step": update,
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
        "args": vars(
            args
        ),
        "elapsed_seconds": (
            float(
                elapsed_seconds
            )
        ),
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
            torch.cuda.get_rng_state_all()
        )

    torch.save(
        state,
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict:
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def validate_resume(
    args: argparse.Namespace,
    saved_args: dict,
) -> None:
    mismatches = []

    for field in (
        RESUME_LOCKED_FIELDS
    ):
        if field not in (
            saved_args
        ):
            raise ValueError(
                "checkpoint missing locked field: "
                f"{field}"
            )

        current = getattr(
            args,
            field,
        )

        previous = (
            saved_args[
                field
            ]
        )

        if field == "dataset":
            current = Path(
                current
            )

            previous = Path(
                previous
            )

        if (
            current
            != previous
        ):
            mismatches.append(
                (
                    field,
                    previous,
                    current,
                )
            )

    if mismatches:
        details = "\n".join(
            (
                f"  {field}: "
                f"checkpoint={old!r}, "
                f"current={new!r}"
            )
            for (
                field,
                old,
                new,
            )
            in mismatches
        )

        raise ValueError(
            "resume configuration mismatch:\n"
            + details
        )


def append_jsonl(
    path: Path,
    payload: dict,
) -> None:
    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                payload,
                sort_keys=True,
            )
            + "\n"
        )


def model_is_finite(
    model,
) -> bool:
    return all(
        bool(
            torch.isfinite(
                parameter.detach()
            ).all()
        )
        for parameter
        in model.parameters()
    )


def gradient_elements_are_finite(
    model,
) -> bool:
    """
    Source-faithful viability check.

    The public RDT trainer does not treat the scalar value
    returned by clip_grad_norm_ as a control-flow signal.

    We therefore distinguish:
    - actual non-finite gradient tensor elements: failure;
    - float32 overflow of the reported aggregate norm:
      telemetry only.
    """
    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        if not bool(
            torch.isfinite(
                parameter.grad.detach()
            ).all()
        ):
            return False

    return True


def optimizer_is_finite(
    optimizer,
) -> bool:
    for state in (
        optimizer.state.values()
    ):
        for key in (
            "exp_avg",
            "exp_avg_sq",
        ):
            value = state.get(
                key
            )

            if (
                value
                is not None
                and not bool(
                    torch.isfinite(
                        value
                    ).all()
                )
            ):
                return False

    return True


def write_failure(
    path: Path,
    *,
    update: int,
    reason: str,
    loss: float | None = None,
    grad_norm: float | None = None,
) -> None:
    payload = {
        "status": (
            "failed"
        ),
        "update": int(
            update
        ),
        "reason": (
            reason
        ),
        "loss": loss,
        "grad_norm_pre_clip": (
            grad_norm
        ),
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if (
        args.batch_size
        != 64
    ):
        raise ValueError(
            "source-compatible protocol freezes "
            "batch size at 64"
        )

    if (
        args.num_updates
        != 100_000
    ):
        raise ValueError(
            "source-compatible protocol freezes "
            "total updates at 100000"
        )

    if (
        args.learning_rate
        != 1e-4
        or args.weight_decay
        != 1e-4
        or args.warmup_steps
        != 10_000
        or args.grad_clip_norm
        != 0.25
    ):
        raise ValueError(
            "source-compatible optimizer protocol changed"
        )

    target_update = (
        args.stop_after
        or args.num_updates
    )

    if not (
        1
        <= target_update
        <= args.num_updates
    ):
        raise ValueError(
            "invalid --stop-after"
        )

    args.dataset = (
        args.dataset.resolve()
    )

    if (
        sha256_file(
            args.dataset
        )
        != EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "clean 2% dataset SHA256 mismatch"
        )

    set_source_seed(
        args.seed
    )

    trajectories = (
        load_source_trajectories(
            args.dataset,
            expected_num_trajectories=23,
            expected_num_transitions=20_147,
            expected_trailing_transitions=0,
        )
    )

    dataset = SourceSequenceDataset(
        trajectories,
        seq_len=20,
        episode_len=1000,
        reward_scale=0.001,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = (
        make_source_compatible_dt()
        .to(
            device
        )
    )

    (
        optimizer,
        scheduler,
    ) = (
        make_optimizer_and_scheduler(
            model,
            learning_rate=(
                args.learning_rate
            ),
            weight_decay=(
                args.weight_decay
            ),
            warmup_steps=(
                args.warmup_steps
            ),
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        args.output_dir
        / "training_metrics.jsonl"
    )

    manifest_path = (
        args.output_dir
        / "run_manifest.json"
    )

    summary_path = (
        args.output_dir
        / "summary.json"
    )

    failure_path = (
        args.output_dir
        / "failure.json"
    )

    checkpoint_dir = (
        args.output_dir
        / "checkpoints"
    )

    start_update = 0
    elapsed_before = 0.0

    if args.resume is None:
        for path in (
            metrics_path,
            manifest_path,
            summary_path,
            failure_path,
        ):
            if path.exists():
                raise FileExistsError(
                    f"fresh run would overwrite {path}"
                )

        metrics_path.write_text(
            "",
            encoding="utf-8",
        )

        manifest = {
            "experiment": (
                "rdt_source_dt_clean_viability"
            ),
            "source_repository_commit": (
                "865fb60632153ed7d8a49e7941b675f026480006"
            ),
            "dataset": str(
                args.dataset
            ),
            "dataset_sha256": (
                EXPECTED_DATASET_SHA256
            ),
            "seed": args.seed,
            "state_normalization": (
                False
            ),
            "padding_side": (
                "right"
            ),
            "action_padding": (
                0.0
            ),
            "sequence_length": (
                20
            ),
            "reward_scale": (
                0.001
            ),
            "batch_size": (
                args.batch_size
            ),
            "num_updates": (
                args.num_updates
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
            "grad_clip_norm": (
                args.grad_clip_norm
            ),
            "grad_clip_return_is_failure_condition": (
                False
            ),
            "gradient_element_finiteness_required": (
                True
            ),
            "device": str(
                device
            ),
            "torch_version": (
                torch.__version__
            ),
            "numpy_version": (
                np.__version__
            ),
            "git_commit": (
                get_git_commit()
            ),
        }

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    else:
        if failure_path.exists():
            raise RuntimeError(
                "run has an existing failure.json; "
                "do not resume a failed viability cell"
            )

        if not (
            metrics_path.exists()
            and manifest_path.exists()
        ):
            raise FileNotFoundError(
                "resume metadata missing"
            )

        checkpoint = (
            load_checkpoint(
                args.resume,
                device,
            )
        )

        validate_resume(
            args,
            checkpoint[
                "args"
            ],
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

        random.setstate(
            checkpoint[
                "python_random_state"
            ]
        )

        np.random.set_state(
            checkpoint[
                "numpy_random_state"
            ]
        )

        torch.set_rng_state(
            checkpoint[
                "torch_random_state"
            ].cpu()
        )

        if (
            device.type
            == "cuda"
            and "cuda_random_states"
            in checkpoint
        ):
            torch.cuda.set_rng_state_all(
                [
                    state.cpu()
                    for state
                    in checkpoint[
                        "cuda_random_states"
                    ]
                ]
            )

        start_update = int(
            checkpoint[
                "update"
            ]
        )

        elapsed_before = float(
            checkpoint.get(
                "elapsed_seconds",
                0.0,
            )
        )

        append_jsonl(
            metrics_path,
            {
                "type": (
                    "resume"
                ),
                "update": (
                    start_update
                ),
                "checkpoint": str(
                    args.resume
                ),
            },
        )

    if (
        target_update
        <= start_update
    ):
        raise ValueError(
            "target update must exceed resume update"
        )

    print(
        "=" * 80
    )

    print(
        "SOURCE-COMPATIBLE RDT VANILLA DT — CLEAN VIABILITY"
    )

    print(
        "=" * 80
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
        "dataset:",
        args.dataset,
    )

    print(
        "start update:",
        start_update,
    )

    print(
        "target update:",
        target_update,
    )

    start_time = (
        time.time()
        - elapsed_before
    )

    for update in range(
        start_update + 1,
        target_update + 1,
    ):
        (
            states,
            actions,
            returns,
            time_steps,
            mask,
            trajectory_ids,
            start_indices,
        ) = dataset.get_batch(
            args.batch_size
        )

        states = states.to(
            device
        )

        actions = actions.to(
            device
        )

        returns = returns.to(
            device
        )

        time_steps = (
            time_steps.to(
                device
            )
        )

        mask = mask.to(
            device
        )

        padding_mask = ~mask.to(
            torch.bool
        )

        model.train()

        predicted_actions = (
            model(
                states=states,
                actions=actions,
                returns_to_go=(
                    returns
                ),
                time_steps=(
                    time_steps
                ),
                padding_mask=(
                    padding_mask
                ),
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss = (
            source_masked_action_mse(
                predicted_actions,
                actions,
                mask,
            )
        )

        if not bool(
            torch.isfinite(
                loss
            )
        ):
            loss_value = float(
                loss.detach()
                .cpu()
            )

            write_failure(
                failure_path,
                update=update,
                reason=(
                    "non_finite_loss"
                ),
                loss=loss_value,
            )

            raise FloatingPointError(
                "non-finite source-compatible DT loss: "
                f"{loss_value}"
            )

        loss.backward()

        loss_value = float(
            loss.detach()
            .cpu()
        )

        gradient_elements_finite = (
            gradient_elements_are_finite(
                model
            )
        )

        if not gradient_elements_finite:
            write_failure(
                failure_path,
                update=update,
                reason=(
                    "non_finite_gradient_elements"
                ),
                loss=loss_value,
            )

            raise FloatingPointError(
                "non-finite source-compatible DT "
                "gradient element"
            )

        # Source-faithful clipping semantics:
        #
        # The public RDT trainer invokes clip_grad_norm_
        # but does not branch on its returned total norm.
        #
        # On float32 the aggregate norm can overflow to inf
        # even when every individual gradient element is
        # finite. We log that scalar but do not treat it as
        # a failure condition.
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    args.grad_clip_norm
                ),
            )
        )

        grad_norm_value = float(
            grad_norm.detach()
            .cpu()
        )

        grad_norm_reported_finite = bool(
            torch.isfinite(
                grad_norm
            )
        )

        optimizer.step()
        scheduler.step()

        if not model_is_finite(
            model
        ):
            write_failure(
                failure_path,
                update=update,
                reason=(
                    "non_finite_model_parameters"
                ),
                loss=loss_value,
                grad_norm=(
                    grad_norm_value
                ),
            )

            raise FloatingPointError(
                "non-finite model parameter"
            )

        if not optimizer_is_finite(
            optimizer
        ):
            write_failure(
                failure_path,
                update=update,
                reason=(
                    "non_finite_optimizer_state"
                ),
                loss=loss_value,
                grad_norm=(
                    grad_norm_value
                ),
            )

            raise FloatingPointError(
                "non-finite Adam state"
            )

        if (
            update == 1
            or update
            % args.log_every
            == 0
        ):
            payload = {
                "type": (
                    "train"
                ),
                "update": (
                    update
                ),
                "loss": (
                    loss_value
                ),
                "grad_norm_pre_clip": (
                    grad_norm_value
                ),
                "gradient_elements_finite": (
                    gradient_elements_finite
                ),
                "grad_norm_reported_finite": (
                    grad_norm_reported_finite
                ),
                "learning_rate": float(
                    optimizer.param_groups[
                        0
                    ][
                        "lr"
                    ]
                ),
                "elapsed_seconds": (
                    time.time()
                    - start_time
                ),
                "trajectory_ids": [
                    int(
                        x
                    )
                    for x in (
                        trajectory_ids.cpu()
                        .numpy()
                    )
                ],
                "start_indices": [
                    int(
                        x
                    )
                    for x in (
                        start_indices.cpu()
                        .numpy()
                    )
                ],
            }

            append_jsonl(
                metrics_path,
                payload,
            )

            print(
                f"update={update:6d}/"
                f"{args.num_updates:6d} "
                f"loss={loss_value:.6f} "
                f"grad={grad_norm_value:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8e}"
            )

        if (
            update
            % args.checkpoint_every
            == 0
            or update
            == target_update
        ):
            save_checkpoint(
                checkpoint_dir
                / (
                    "checkpoint_step_"
                    f"{update:06d}.pt"
                ),
                update=update,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                elapsed_seconds=(
                    time.time()
                    - start_time
                ),
            )

    final_checkpoint = (
        checkpoint_dir
        / (
            "checkpoint_step_"
            f"{target_update:06d}.pt"
        )
    )

    summary = {
        "status": (
            "complete"
            if target_update
            == args.num_updates
            else "partial"
        ),
        "seed": args.seed,
        "final_update": (
            target_update
        ),
        "checkpoint": str(
            final_checkpoint
        ),
        "dataset_sha256": (
            EXPECTED_DATASET_SHA256
        ),
        "loss_finite": True,
        "gradient_elements_finite": (
            True
        ),
        "reported_gradient_norm_finite_required": (
            False
        ),
        "model_parameters_finite": (
            model_is_finite(
                model
            )
        ),
        "optimizer_state_finite": (
            optimizer_is_finite(
                optimizer
            )
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
        )
        + "\n",
        encoding="utf-8",
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
