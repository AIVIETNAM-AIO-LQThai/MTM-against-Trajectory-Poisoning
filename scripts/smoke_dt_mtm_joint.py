from __future__ import annotations

import argparse

import torch

from src.methods.dt.model import DecisionTransformer
from src.methods.dt_mtm.model import DTMTMModel
from src.methods.dt_mtm.trainer import DTMTMTrainBatch, DTMTMTrainer
from src.methods.mtm.model import MTMConfig, ReferenceMTM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--lambda-mtm", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def build_model() -> DTMTMModel:
    dt = DecisionTransformer(
        state_dim=17,
        action_dim=6,
        hidden_size=32,
        max_ep_len=100,
        n_layer=1,
        n_head=4,
        n_inner=64,
        activation_function="relu",
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        action_tanh=True,
    )
    mtm = ReferenceMTM(
        {"states": (1, 17), "actions": (1, 6), "returns": (1, 1)},
        traj_length=5,
        config=MTMConfig(
            n_embd=32,
            n_head=4,
            n_enc_layer=1,
            n_dec_layer=1,
            dropout=0.0,
        ),
    )
    return DTMTMModel(dt, mtm)


def make_batch() -> DTMTMTrainBatch:
    torch.manual_seed(2026)
    batch_size = 2
    length = 5
    states = torch.randn(batch_size, length, 17)
    actions = torch.randn(batch_size, length, 6)
    rtg = torch.randn(batch_size, length, 1)
    timesteps = torch.arange(length).unsqueeze(0).repeat(batch_size, 1)
    attention_mask = torch.ones(batch_size, length, dtype=torch.long)

    trajectories = {
        "states": states.unsqueeze(2).clone(),
        "actions": actions.unsqueeze(2).clone(),
        "returns": torch.randn(batch_size, length, 1, 1),
    }
    masks = {
        "states": torch.tensor([[0.0], [1.0], [1.0], [0.0], [1.0]]),
        "actions": torch.tensor([[1.0], [0.0], [1.0], [1.0], [1.0]]),
        "returns": torch.tensor([[1.0], [1.0], [0.0], [1.0], [1.0]]),
    }

    return DTMTMTrainBatch(
        states=states,
        actions=actions,
        returns_to_go=rtg,
        timesteps=timesteps,
        attention_mask=attention_mask,
        mtm_trajectories=trajectories,
        mtm_masks=masks,
    )


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(101)
    model = build_model().to(device)
    audit = model.audit_initial_bridge_equivalence(
        {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in make_batch().mtm_trajectories.items()
        }
    )

    print("DT+MTM synthetic integration smoke")
    print(f"device={device}")
    print(f"lambda_mtm={args.lambda_mtm}")
    print(
        "bridge_error "
        f"state={audit.state_max_abs_error:.3e} "
        f"action={audit.action_max_abs_error:.3e}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    trainer = DTMTMTrainer(
        model,
        optimizer,
        scheduler=None,
        device=device,
        lambda_mtm=args.lambda_mtm,
        grad_clip_norm=5.0,
        diagnose_shared_gradients=True,
    )
    batch = make_batch()

    first = None
    last = None
    for step in range(1, args.steps + 1):
        metrics = trainer.train_step(batch)
        if first is None:
            first = metrics.total_loss
        last = metrics.total_loss
        if step == 1 or step == args.steps or step % 5 == 0:
            print(
                f"step={step:03d} total={metrics.total_loss:.6f} "
                f"dt={metrics.dt_loss:.6f} mtm={metrics.mtm_loss:.6f} "
                f"shared_cos={metrics.shared_grad_cosine:+.4f}"
            )

    assert first is not None and last is not None
    print(f"loss_ratio={last / first:.6f}")


if __name__ == "__main__":
    main()
