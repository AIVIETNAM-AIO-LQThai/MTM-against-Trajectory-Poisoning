from __future__ import annotations

import copy
import math

import torch

from src.methods.dt.model import DecisionTransformer
from src.methods.dt.trainer import DTTrainer
from src.methods.dt_mtm.model import DTMTMModel
from src.methods.dt_mtm.trainer import DTMTMTrainBatch, DTMTMTrainer
from src.methods.mtm.model import MTMConfig, ReferenceMTM


DATA_SHAPES = {
    "states": (1, 17),
    "actions": (1, 6),
    "returns": (1, 1),
}


def make_dt() -> DecisionTransformer:
    return DecisionTransformer(
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


def make_mtm(length: int = 5) -> ReferenceMTM:
    return ReferenceMTM(
        DATA_SHAPES,
        traj_length=length,
        config=MTMConfig(
            n_embd=32,
            n_head=4,
            n_enc_layer=1,
            n_dec_layer=1,
            dropout=0.0,
        ),
    )


def make_batch(batch_size: int = 3, length: int = 5) -> DTMTMTrainBatch:
    torch.manual_seed(9001)
    states = torch.randn(batch_size, length, 17)
    actions = torch.randn(batch_size, length, 6)
    rtg = torch.randn(batch_size, length, 1)
    timesteps = torch.arange(length).unsqueeze(0).repeat(batch_size, 1)
    attention_mask = torch.ones(batch_size, length, dtype=torch.long)

    mtm_trajectories = {
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
        mtm_trajectories=mtm_trajectories,
        mtm_masks=masks,
    )


def test_joint_train_step_reports_finite_losses_and_gradient_diagnostics():
    torch.manual_seed(123)
    model = DTMTMModel(make_dt(), make_mtm())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    trainer = DTMTMTrainer(
        model,
        optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        lambda_mtm=0.1,
        grad_clip_norm=1.0,
        diagnose_shared_gradients=True,
    )

    metrics = trainer.train_step(make_batch())

    for value in (
        metrics.total_loss,
        metrics.dt_loss,
        metrics.mtm_loss,
        metrics.mtm_scaled_loss,
        metrics.mtm_state_loss,
        metrics.mtm_action_loss,
        metrics.mtm_return_loss,
        metrics.grad_norm_pre_clip,
        metrics.shared_dt_grad_norm,
        metrics.shared_mtm_grad_norm,
        metrics.shared_mtm_scaled_grad_norm,
    ):
        assert math.isfinite(value)

    assert metrics.shared_dt_grad_norm > 0.0
    assert metrics.shared_mtm_grad_norm > 0.0
    assert metrics.shared_mtm_scaled_grad_norm > 0.0
    assert math.isfinite(metrics.shared_grad_cosine)
    assert -1.000001 <= metrics.shared_grad_cosine <= 1.000001


def test_lambda_zero_one_step_matches_vanilla_dt_parameter_update():
    torch.manual_seed(456)
    baseline = make_dt()
    joint_dt = copy.deepcopy(baseline)
    joint = DTMTMModel(joint_dt, make_mtm())

    baseline_optimizer = torch.optim.AdamW(
        baseline.parameters(), lr=1e-3, weight_decay=1e-4
    )
    joint_optimizer = torch.optim.AdamW(
        joint.parameters(), lr=1e-3, weight_decay=1e-4
    )

    baseline_trainer = DTTrainer(
        baseline,
        baseline_optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        grad_clip_norm=0.25,
    )
    joint_trainer = DTMTMTrainer(
        joint,
        joint_optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        lambda_mtm=0.0,
        grad_clip_norm=0.25,
        diagnose_shared_gradients=True,
    )

    batch = make_batch()

    class BaselineBatch:
        pass

    dt_batch = BaselineBatch()
    dt_batch.states = batch.states.numpy()
    dt_batch.actions = batch.actions.numpy()
    # DTTrainer expects RTG with one extra terminal position and slices :-1.
    extra = torch.zeros(batch.returns_to_go.shape[0], 1, 1)
    dt_batch.rtg = torch.cat([batch.returns_to_go, extra], dim=1).numpy()
    dt_batch.timesteps = batch.timesteps.numpy()
    dt_batch.attention_mask = batch.attention_mask.numpy()

    baseline_metrics = baseline_trainer.train_step(dt_batch)
    joint_metrics = joint_trainer.train_step(batch)

    assert abs(baseline_metrics.loss - joint_metrics.dt_loss) < 1e-8

    baseline_state = baseline.state_dict()
    joint_state = joint.dt.state_dict()
    assert baseline_state.keys() == joint_state.keys()
    for key in baseline_state:
        torch.testing.assert_close(
            joint_state[key], baseline_state[key], rtol=0.0, atol=1e-7
        )


def test_fixed_batch_joint_training_reduces_total_loss():
    torch.manual_seed(777)
    model = DTMTMModel(make_dt(), make_mtm())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    trainer = DTMTMTrainer(
        model,
        optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        lambda_mtm=0.1,
        grad_clip_norm=5.0,
        diagnose_shared_gradients=False,
    )

    batch = make_batch(batch_size=2)
    losses = []
    for _ in range(40):
        losses.append(trainer.train_step(batch).total_loss)

    assert math.isfinite(losses[0])
    assert math.isfinite(losses[-1])
    assert losses[-1] < losses[0] * 0.8, (losses[0], losses[-1])
