from __future__ import annotations

import json
import math
import random
import types
from pathlib import Path

import numpy as np
import torch

from scripts.train_rdt_source_dt_clean import (
    DEFAULT_DATASET,
    EXPECTED_DATASET_SHA256,
    load_checkpoint,
    make_optimizer_and_scheduler,
    set_source_seed,
    sha256_file,
)

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


ROOT = Path.cwd()

CHECKPOINT = (
    ROOT
    / "experiments"
    / "attack_qualification"
    / "rdt_source_dt_source_clip"
    / "clean"
    / "model_seed_0"
    / "checkpoints"
    / "checkpoint_step_010000.pt"
)

OUTPUT = (
    ROOT
    / "experiments"
    / "attack_qualification"
    / "rdt_source_dt_source_clip"
    / "diagnostics"
    / "attention_path_comparison_from_10000.json"
)

START_UPDATE = 10_001
END_UPDATE = 15_000

LOG_EVERY = 100


def all_grad_elements_finite(model):
    bad = []

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        mask = ~torch.isfinite(
            parameter.grad.detach()
        )

        count = int(
            mask.sum().cpu()
        )

        if count:
            bad.append(
                {
                    "parameter": name,
                    "nonfinite_count": count,
                    "numel": parameter.grad.numel(),
                }
            )

    return len(bad) == 0, bad


def global_norm64(model):
    total_sq = 0.0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach()

        if not bool(
            torch.isfinite(
                gradient
            ).all()
        ):
            return float("nan")

        norm = float(
            torch.linalg.vector_norm(
                gradient.double(),
                ord=2,
            ).cpu()
        )

        total_sq += norm * norm

    return math.sqrt(
        total_sq
    )


def model_finite(model):
    return all(
        bool(
            torch.isfinite(
                parameter.detach()
            ).all()
        )
        for parameter
        in model.parameters()
    )


def optimizer_finite(optimizer):
    for state in optimizer.state.values():
        for key in (
            "exp_avg",
            "exp_avg_sq",
        ):
            value = state.get(
                key
            )

            if (
                value is not None
                and not bool(
                    torch.isfinite(
                        value
                    ).all()
                )
            ):
                return False

    return True


def legacy_attention_forward(
    self,
    x,
    *,
    padding_mask=None,
):
    """
    Same TransformerBlock algebra, but request attention weights.

    In modern PyTorch this avoids the need_weights=False optimized
    scaled-dot-product-attention execution path and uses the older
    explicit MHA path, which is closer to PyTorch 1.8.1.
    """

    causal_mask = self.causal_mask[
        : x.shape[1],
        : x.shape[1],
    ]

    norm_x = self.norm1(
        x
    )

    # Match the public RDT source:
    # batch-first -> sequence-first.
    norm_x = norm_x.transpose(
        1,
        0,
    )

    attention_out = self.attention(
        query=norm_x,
        key=norm_x,
        value=norm_x,
        attn_mask=causal_mask,
        key_padding_mask=padding_mask,
        need_weights=True,
    )[0]

    attention_out = (
        attention_out.transpose(
            1,
            0,
        )
    )

    x = (
        x
        + self.attention_residual_dropout(
            attention_out
        )
    )

    x = (
        x
        + self.mlp(
            self.norm2(
                x
            )
        )
    )

    return x


def install_legacy_attention_path(model):
    for block in model.blocks:
        block.forward = types.MethodType(
            legacy_attention_forward,
            block,
        )


def restore_checkpoint(
    checkpoint,
    *,
    model,
    optimizer,
    scheduler,
    device,
):
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
        device.type == "cuda"
        and "cuda_random_states"
        in checkpoint
    ):
        torch.cuda.set_rng_state_all(
            [
                state.cpu()
                for state in checkpoint[
                    "cuda_random_states"
                ]
            ]
        )


def build_run(
    *,
    device,
    checkpoint,
    trajectories,
    legacy_attention,
):
    # Recreate model under the same initialization sequence.
    set_source_seed(
        0
    )

    dataset = SourceSequenceDataset(
        trajectories,
        seq_len=20,
        episode_len=1000,
        reward_scale=0.001,
    )

    model = (
        make_source_compatible_dt()
        .to(
            device
        )
    )

    if legacy_attention:
        install_legacy_attention_path(
            model
        )

    optimizer, scheduler = (
        make_optimizer_and_scheduler(
            model,
            learning_rate=1e-4,
            weight_decay=1e-4,
            warmup_steps=10_000,
        )
    )

    restore_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    return (
        dataset,
        model,
        optimizer,
        scheduler,
    )


def run_path(
    *,
    name,
    device,
    checkpoint,
    trajectories,
    legacy_attention,
):
    (
        dataset,
        model,
        optimizer,
        scheduler,
    ) = build_run(
        device=device,
        checkpoint=checkpoint,
        trajectories=trajectories,
        legacy_attention=legacy_attention,
    )

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    result = {
        "name": name,
        "legacy_attention": legacy_attention,
        "start_update": START_UPDATE,
        "target_update": END_UPDATE,
        "status": None,
        "failure_update": None,
        "failure_reason": None,
        "events": [],
    }

    for update in range(
        START_UPDATE,
        END_UPDATE + 1,
    ):
        (
            states,
            actions,
            returns,
            time_steps,
            mask,
            _trajectory_ids,
            _start_indices,
        ) = dataset.get_batch(
            64
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

        time_steps = time_steps.to(
            device
        )

        mask = mask.to(
            device
        )

        padding_mask = ~mask.to(
            torch.bool
        )

        model.train()

        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss = source_masked_action_mse(
            predicted_actions,
            actions,
            mask,
        )

        loss_value = float(
            loss.detach().cpu()
        )

        if not bool(
            torch.isfinite(
                loss
            )
        ):
            result[
                "status"
            ] = "FAIL"

            result[
                "failure_update"
            ] = update

            result[
                "failure_reason"
            ] = "non_finite_loss"

            print(
                f"update={update} "
                "NON-FINITE LOSS"
            )

            break

        loss.backward()

        (
            gradients_finite,
            bad_gradients,
        ) = all_grad_elements_finite(
            model
        )

        norm64 = global_norm64(
            model
        )

        if not gradients_finite:
            result[
                "status"
            ] = "FAIL"

            result[
                "failure_update"
            ] = update

            result[
                "failure_reason"
            ] = (
                "non_finite_gradient_elements"
            )

            result[
                "bad_gradients"
            ] = bad_gradients

            print()
            print(
                f"update={update}"
            )

            print(
                "ACTUAL NON-FINITE "
                "GRADIENT ELEMENT"
            )

            for row in (
                bad_gradients[
                    :10
                ]
            ):
                print(
                    " ",
                    row,
                )

            break

        # Source-faithful clipping:
        # the returned scalar is telemetry only.
        reported_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=0.25,
            )
        )

        reported_norm_value = float(
            reported_norm.detach().cpu()
        )

        optimizer.step()
        scheduler.step()

        if not model_finite(
            model
        ):
            result[
                "status"
            ] = "FAIL"

            result[
                "failure_update"
            ] = update

            result[
                "failure_reason"
            ] = (
                "non_finite_model_parameters"
            )

            print(
                f"update={update} "
                "NON-FINITE MODEL PARAMETER"
            )

            break

        if not optimizer_finite(
            optimizer
        ):
            result[
                "status"
            ] = "FAIL"

            result[
                "failure_update"
            ] = update

            result[
                "failure_reason"
            ] = (
                "non_finite_optimizer_state"
            )

            print(
                f"update={update} "
                "NON-FINITE OPTIMIZER STATE"
            )

            break

        important = (
            update
            % LOG_EVERY
            == 0
            or not math.isfinite(
                reported_norm_value
            )
            or (
                math.isfinite(
                    norm64
                )
                and norm64
                >= 1e6
            )
        )

        if important:
            event = {
                "update": update,
                "loss": loss_value,
                "norm64": norm64,
                "reported_norm": (
                    reported_norm_value
                ),
            }

            result[
                "events"
            ].append(
                event
            )

            print(
                f"update={update:6d} "
                f"loss={loss_value:.8f} "
                f"norm64={norm64:.6g} "
                f"clip_return="
                f"{reported_norm_value:.6g}"
            )

    else:
        result[
            "status"
        ] = "PASS_TO_15000"

    print()
    print(
        "RESULT:",
        result[
            "status"
        ],
    )

    if (
        result[
            "failure_update"
        ]
        is not None
    ):
        print(
            "failure update:",
            result[
                "failure_update"
            ],
        )

        print(
            "reason:",
            result[
                "failure_reason"
            ],
        )

    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this diagnostic"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            CHECKPOINT
        )

    dataset_path = Path(
        DEFAULT_DATASET
    ).resolve()

    if (
        sha256_file(
            dataset_path
        )
        != EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "dataset SHA mismatch"
        )

    device = torch.device(
        "cuda"
    )

    trajectories = (
        load_source_trajectories(
            dataset_path,
            expected_num_trajectories=23,
            expected_num_transitions=20_147,
            expected_trailing_transitions=0,
        )
    )

    checkpoint = load_checkpoint(
        CHECKPOINT,
        device,
    )

    if int(
        checkpoint[
            "update"
        ]
    ) != 10_000:
        raise RuntimeError(
            "expected update-10000 checkpoint"
        )

    print(
        "torch:",
        torch.__version__,
    )

    print(
        "CUDA:",
        torch.version.cuda,
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )

    print(
        "checkpoint:",
        CHECKPOINT,
    )

    current_result = run_path(
        name=(
            "CURRENT PYTORCH 2.7 "
            "ATTENTION PATH "
            "(need_weights=False)"
        ),
        device=device,
        checkpoint=checkpoint,
        trajectories=trajectories,
        legacy_attention=False,
    )

    legacy_result = run_path(
        name=(
            "LEGACY-LIKE MHA PATH "
            "(need_weights=True)"
        ),
        device=device,
        checkpoint=checkpoint,
        trajectories=trajectories,
        legacy_attention=True,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            {
                "torch_version": (
                    torch.__version__
                ),
                "cuda_version": (
                    torch.version.cuda
                ),
                "gpu": (
                    torch.cuda.get_device_name(
                        0
                    )
                ),
                "checkpoint": str(
                    CHECKPOINT
                ),
                "current_attention": (
                    current_result
                ),
                "legacy_like_attention": (
                    legacy_result
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(
        "ATTENTION PATH COMPARISON"
    )
    print("=" * 100)

    print(
        "current path:",
        current_result[
            "status"
        ],
        current_result[
            "failure_update"
        ],
        current_result[
            "failure_reason"
        ],
    )

    print(
        "legacy-like path:",
        legacy_result[
            "status"
        ],
        legacy_result[
            "failure_update"
        ],
        legacy_result[
            "failure_reason"
        ],
    )

    print()
    print(
        "output ->",
        OUTPUT,
    )


if __name__ == "__main__":
    main()