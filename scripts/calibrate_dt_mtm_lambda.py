from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from random import Random

import numpy as np
import torch

from src.methods.dt.losses import masked_action_mse
from src.methods.dt_mtm.calibration import (
    DEFAULT_LAMBDA_CANDIDATES,
    choose_lambda_from_gradient_ratios,
)
from src.methods.dt_mtm.clean_pipeline import (
    build_joint_model,
    build_mtm_tokenizers,
    load_clean_walker2d,
    sample_clean_joint_train_batch,
)
from src.methods.dt_mtm.diagnostics import shared_gradient_diagnostics
from src.data.mtm_batching import MTMShuffledWindowSampler
from src.methods.mtm.losses import reference_mtm_loss


DEFAULT_OUTPUT = Path(
    "experiments/dt_mtm/walker2d_medium_clean/"
    "lambda_calibration_seed_0.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--dt-batch-size", type=int, default=64)
    parser.add_argument("--mtm-batch-size", type=int, default=512)
    parser.add_argument(
        "--candidate-lambdas",
        nargs="+",
        type=float,
        default=list(DEFAULT_LAMBDA_CANDIDATES),
    )
    parser.add_argument("--target-max-scaled-ratio", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
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


def main() -> None:
    args = parse_args()

    if args.num_batches <= 0:
        raise ValueError("--num-batches must be positive")
    if args.dt_batch_size <= 0 or args.mtm_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Calibration output already exists: {args.output}. "
            "Use --overwrite only if you intentionally want to replace it."
        )

    set_seed(args.seed)
    device = resolve_device(args.device)

    print("=" * 72)
    print("GROUP 4B CLEAN LAMBDA CALIBRATION")
    print("=" * 72)
    print("device:", device)
    print("seed:", args.seed)
    print("num_batches:", args.num_batches)
    print("DT batch:", args.dt_batch_size)
    print("MTM batch:", args.mtm_batch_size)
    print("candidates:", args.candidate_lambdas)
    print("scaled-gradient cap:", args.target_max_scaled_ratio)

    data = load_clean_walker2d()
    tokenizers = build_mtm_tokenizers(data, device=device)
    model = build_joint_model(seed=args.seed, device=device)
    model.train()

    dt_np_rng = np.random.default_rng(args.seed)
    dt_py_rng = Random(args.seed)
    mtm_sampler = MTMShuffledWindowSampler(
        data.mtm_ranges.train,
        seed=args.seed + 1_000,
    )
    mask_rng = np.random.RandomState(args.seed + 2_000)

    ratios: list[float] = []
    cosines: list[float] = []
    rows: list[dict] = []
    bridge_audit = None

    for batch_index in range(args.num_batches):
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

        if bridge_audit is None:
            bridge_audit = model.audit_initial_bridge_equivalence(
                batch.mtm_trajectories
            )
            if (
                bridge_audit.state_max_abs_error > 1e-5
                or bridge_audit.action_max_abs_error > 1e-5
            ):
                raise RuntimeError(
                    "Initial bridge equivalence error is unexpectedly large: "
                    f"{bridge_audit}"
                )

        states = torch.as_tensor(batch.states, dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=device)
        returns_to_go = torch.as_tensor(
            batch.returns_to_go, dtype=torch.float32, device=device
        )
        timesteps = torch.as_tensor(batch.timesteps, dtype=torch.long, device=device)
        attention_mask = torch.as_tensor(
            batch.attention_mask, dtype=torch.long, device=device
        )

        _, action_predictions, _ = model.forward_dt(
            states,
            actions,
            returns_to_go,
            timesteps,
            attention_mask,
        )
        dt_loss = masked_action_mse(
            action_predictions,
            actions,
            attention_mask,
        )

        mtm_predictions = model.forward_mtm(
            batch.mtm_trajectories,
            batch.mtm_masks,
        )
        mtm_output = reference_mtm_loss(
            batch.mtm_trajectories,
            mtm_predictions,
            batch.mtm_masks,
        )

        diagnostic = shared_gradient_diagnostics(
            model,
            dt_loss,
            mtm_output.total_loss,
            lambda_mtm=1.0,
        )

        if not (
            math.isfinite(diagnostic.dt_norm)
            and diagnostic.dt_norm > 0.0
        ):
            raise RuntimeError(
                "Non-finite/zero DT shared gradient during calibration"
            )

        if not (
            math.isfinite(diagnostic.mtm_norm_unscaled)
            and diagnostic.mtm_norm_unscaled >= 0.0
        ):
            raise RuntimeError(
                "Non-finite/negative MTM shared gradient during calibration"
            )

        # Zero MTM shared gradient is a valid reference-mask outcome.
        # In particular, AUTO_MASK can hide all state/action inputs, so the
        # MTM objective has no path to the DT state/action embeddings on that
        # batch. Keep the sample as ratio 0 instead of resampling or aborting.
        ratio = diagnostic.mtm_norm_unscaled / diagnostic.dt_norm
        ratios.append(float(ratio))

        cosine_is_finite = math.isfinite(
            diagnostic.cosine_similarity
        )
        if cosine_is_finite:
            cosines.append(float(diagnostic.cosine_similarity))

        zero_shared_mtm_gradient = (
            diagnostic.mtm_norm_unscaled == 0.0
        )

        row = {
            "batch": batch_index + 1,
            "dt_loss": float(dt_loss.detach().cpu()),
            "mtm_loss": float(mtm_output.total_loss.detach().cpu()),
            "shared_dt_grad_norm": diagnostic.dt_norm,
            "shared_mtm_grad_norm": diagnostic.mtm_norm_unscaled,
            "raw_gradient_ratio": float(ratio),
            "shared_grad_cosine": (
                float(diagnostic.cosine_similarity)
                if cosine_is_finite
                else None
            ),
            "zero_shared_mtm_gradient": zero_shared_mtm_gradient,
            "mask_mode": audit.mask_mode,
            "mask_position": audit.mask_position,
        }
        rows.append(row)

        cosine_text = (
            f"{diagnostic.cosine_similarity:+.4f}"
            if cosine_is_finite
            else "undefined"
        )

        print(
            f"batch={batch_index + 1:02d} "
            f"dt={row['dt_loss']:.6f} "
            f"mtm={row['mtm_loss']:.6f} "
            f"ratio={ratio:.4f} "
            f"cos={cosine_text}"
        )

        # Explicitly release the graphs before the next large MTM batch.
        del dt_loss, mtm_output, mtm_predictions, action_predictions

    result = choose_lambda_from_gradient_ratios(
        ratios,
        cosines,
        candidates=args.candidate_lambdas,
        target_max_scaled_ratio=args.target_max_scaled_ratio,
    )

    assert bridge_audit is not None

    payload = {
        "protocol": "group4b_clean_gradient_balance_v2",
        "seed": args.seed,
        "device": str(device),
        "num_batches": args.num_batches,
        "dt_batch_size": args.dt_batch_size,
        "mtm_batch_size": args.mtm_batch_size,
        "candidate_lambdas": [float(x) for x in args.candidate_lambdas],
        "target_max_scaled_ratio": args.target_max_scaled_ratio,
        "zero_shared_mtm_gradient_policy": (
            "include_as_ratio_zero_and_omit_undefined_cosine"
        ),
        "num_zero_shared_mtm_gradient_batches": sum(
            int(row["zero_shared_mtm_gradient"]) for row in rows
        ),
        "recommended_lambda": result.recommended_lambda,
        "median_raw_gradient_ratio": result.median_raw_gradient_ratio,
        "median_cosine_similarity": result.median_cosine_similarity,
        "candidate_scaled_ratios": {
            str(key): value
            for key, value in result.candidate_scaled_ratios.items()
        },
        "bridge_audit": {
            "state_rank": bridge_audit.state_rank,
            "state_input_dim": bridge_audit.state_input_dim,
            "state_max_abs_error": bridge_audit.state_max_abs_error,
            "action_rank": bridge_audit.action_rank,
            "action_input_dim": bridge_audit.action_input_dim,
            "action_max_abs_error": bridge_audit.action_max_abs_error,
        },
        "batches": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print("Calibration result")
    print("------------------")
    print("median raw MTM/DT shared-grad ratio:", result.median_raw_gradient_ratio)
    print("median shared-gradient cosine:", result.median_cosine_similarity)
    print("recommended lambda:", result.recommended_lambda)
    print("saved:", args.output)


if __name__ == "__main__":
    main()
