from __future__ import annotations

import copy

import torch

from src.methods.dt.losses import masked_action_mse
from src.methods.dt.model import DecisionTransformer
from src.methods.dt_mtm.losses import compose_dt_mtm_loss
from src.methods.dt_mtm.model import DTMTMModel
from src.methods.mtm.losses import reference_mtm_loss
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


def make_mtm(traj_length: int = 5) -> ReferenceMTM:
    return ReferenceMTM(
        DATA_SHAPES,
        traj_length=traj_length,
        config=MTMConfig(
            n_embd=32,
            n_head=4,
            n_enc_layer=1,
            n_dec_layer=1,
            dropout=0.0,
        ),
    )


def make_joint(traj_length: int = 5) -> DTMTMModel:
    return DTMTMModel(make_dt(), make_mtm(traj_length=traj_length))


def make_dt_inputs(batch_size: int = 2, length: int = 5):
    states = torch.randn(batch_size, length, 17)
    actions = torch.randn(batch_size, length, 6)
    rtg = torch.randn(batch_size, length, 1)
    timesteps = torch.arange(length).unsqueeze(0).repeat(batch_size, 1)
    attention_mask = torch.ones(batch_size, length, dtype=torch.long)
    return states, actions, rtg, timesteps, attention_mask


def make_mtm_inputs(batch_size: int = 2, length: int = 5):
    trajectories = {
        "states": torch.randn(batch_size, length, 1, 17),
        "actions": torch.randn(batch_size, length, 1, 6),
        "returns": torch.randn(batch_size, length, 1, 1),
    }
    masks = {
        "states": torch.tensor([[0.0], [1.0], [1.0], [1.0], [1.0]])[:length],
        "actions": torch.ones(length, 1),
        "returns": torch.ones(length, 1),
    }
    return trajectories, masks


def test_joint_model_exposes_explicit_separate_passes():
    model = make_joint()

    assert callable(model.forward_dt)
    assert callable(model.forward_mtm)

    shared = dict(model.shared_named_parameters())
    assert set(shared.keys()) == {
        "dt.embed_state.weight",
        "dt.embed_state.bias",
        "dt.embed_action.weight",
        "dt.embed_action.bias",
    }


def test_lambda_zero_dt_forward_is_exactly_vanilla_dt():
    torch.manual_seed(123)

    baseline = make_dt()
    joint_dt = copy.deepcopy(baseline)
    joint = DTMTMModel(joint_dt, make_mtm())

    baseline.eval()
    joint.eval()

    inputs = make_dt_inputs()

    with torch.no_grad():
        baseline_outputs = baseline(*inputs)
        joint_outputs = joint.forward_dt(*inputs)

    for expected, actual in zip(baseline_outputs, joint_outputs):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_lambda_zero_dt_gradients_are_equivalent():
    torch.manual_seed(456)

    baseline = make_dt()
    joint_dt = copy.deepcopy(baseline)
    joint = DTMTMModel(joint_dt, make_mtm())

    baseline.train()
    joint.train()

    states, actions, rtg, timesteps, attention_mask = make_dt_inputs()
    trajectories, masks = make_mtm_inputs()

    _, baseline_action_preds, _ = baseline(
        states,
        actions,
        rtg,
        timesteps,
        attention_mask,
    )
    baseline_loss = masked_action_mse(
        baseline_action_preds,
        actions,
        attention_mask,
    )
    baseline_loss.backward()

    _, joint_action_preds, _ = joint.forward_dt(
        states,
        actions,
        rtg,
        timesteps,
        attention_mask,
    )
    dt_loss = masked_action_mse(
        joint_action_preds,
        actions,
        attention_mask,
    )

    mtm_predictions = joint.forward_mtm(trajectories, masks)
    mtm_loss = reference_mtm_loss(
        trajectories,
        mtm_predictions,
        masks,
    )
    total = compose_dt_mtm_loss(
        dt_loss,
        mtm_loss,
        lambda_mtm=0.0,
    )
    total.total_loss.backward()

    baseline_params = dict(baseline.named_parameters())
    joint_params = dict(joint.dt.named_parameters())

    assert baseline_params.keys() == joint_params.keys()

    for name in baseline_params.keys():
        expected = baseline_params[name].grad
        actual = joint_params[name].grad

        if expected is None:
            assert actual is None, name
        else:
            assert actual is not None, name
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0.0,
                atol=1e-7,
                msg=lambda message, n=name: f"{n}: {message}",
            )


def test_bridge_reproduces_original_mtm_state_action_input_embeddings():
    torch.manual_seed(789)

    model = make_joint()
    trajectories, _ = make_mtm_inputs()

    audit = model.audit_initial_bridge_equivalence(trajectories)

    assert audit.state_rank == audit.state_input_dim
    assert audit.action_rank == audit.action_input_dim
    assert audit.state_max_abs_error < 2e-5
    assert audit.action_max_abs_error < 2e-5


def test_joint_dt_path_remains_strictly_causal():
    torch.manual_seed(321)

    model = make_joint(traj_length=8)
    model.eval()

    states_a, actions_a, rtg_a, timesteps, attention_mask = make_dt_inputs(
        batch_size=1,
        length=8,
    )

    states_b = states_a.clone()
    actions_b = actions_a.clone()
    rtg_b = rtg_a.clone()

    t = 3
    actions_b[:, t, :] = 999.0
    states_b[:, t + 1 :, :] = -777.0
    actions_b[:, t + 1 :, :] = 888.0
    rtg_b[:, t + 1 :, :] = 555.0

    with torch.no_grad():
        _, preds_a, _ = model.forward_dt(
            states_a,
            actions_a,
            rtg_a,
            timesteps,
            attention_mask,
        )
        _, preds_b, _ = model.forward_dt(
            states_b,
            actions_b,
            rtg_b,
            timesteps,
            attention_mask,
        )

    torch.testing.assert_close(
        preds_a[:, : t + 1],
        preds_b[:, : t + 1],
        rtol=0.0,
        atol=1e-6,
    )


def test_joint_mtm_path_remains_bidirectional():
    torch.manual_seed(654)

    model = make_joint(traj_length=4)
    model.eval()

    trajectories_a = {
        "states": torch.randn(1, 4, 1, 17),
        "actions": torch.randn(1, 4, 1, 6),
        "returns": torch.randn(1, 4, 1, 1),
    }
    trajectories_b = {
        key: value.clone()
        for key, value in trajectories_a.items()
    }

    trajectories_b["states"][0, 3, 0, :] += 10.0

    masks = {
        "states": torch.tensor([[0.0], [1.0], [1.0], [1.0]]),
        "actions": torch.ones(4, 1),
        "returns": torch.ones(4, 1),
    }

    with torch.no_grad():
        output_a = model.forward_mtm(trajectories_a, masks)
        output_b = model.forward_mtm(trajectories_b, masks)

    difference = (
        output_a["states"][0, 0, 0, :]
        - output_b["states"][0, 0, 0, :]
    ).abs().max()

    assert difference.item() > 1e-6


def test_joint_loss_decomposition_and_lambda_zero():
    dt_loss = torch.tensor(2.0, requires_grad=True)

    mtm_loss = reference_mtm_loss(
        {
            "states": torch.zeros(1, 2, 1, 2),
            "actions": torch.zeros(1, 2, 1, 1),
            "returns": torch.zeros(1, 2, 1, 1),
        },
        {
            "states": torch.ones(1, 2, 1, 2),
            "actions": 2.0 * torch.ones(1, 2, 1, 1),
            "returns": 3.0 * torch.ones(1, 2, 1, 1),
        },
        {
            "states": torch.ones(2, 1),
            "actions": torch.ones(2, 1),
            "returns": torch.ones(2, 1),
        },
    )

    zero = compose_dt_mtm_loss(dt_loss, mtm_loss, lambda_mtm=0.0)
    torch.testing.assert_close(zero.total_loss, dt_loss)
    torch.testing.assert_close(zero.mtm_scaled_loss, torch.tensor(0.0))
    torch.testing.assert_close(zero.mtm_weighted_loss, mtm_loss.total_loss)

    weighted = compose_dt_mtm_loss(dt_loss, mtm_loss, lambda_mtm=0.5)
    torch.testing.assert_close(
        weighted.total_loss,
        dt_loss + 0.5 * mtm_loss.total_loss,
    )


def test_mtm_objective_reaches_only_declared_shared_dt_embeddings():
    torch.manual_seed(987)

    model = make_joint()
    model.train()

    trajectories, masks = make_mtm_inputs()
    predictions = model.forward_mtm(trajectories, masks)
    mtm_loss = reference_mtm_loss(trajectories, predictions, masks)
    mtm_loss.total_loss.backward()

    state_weight_grad = model.dt.embed_state.weight.grad
    action_weight_grad = model.dt.embed_action.weight.grad

    assert state_weight_grad is not None
    assert action_weight_grad is not None
    assert torch.linalg.vector_norm(state_weight_grad).item() > 0.0
    assert torch.linalg.vector_norm(action_weight_grad).item() > 0.0

    # Return semantics are deliberately separate between DT and MTM.
    assert model.dt.embed_return.weight.grad is None
    assert model.dt.embed_return.bias.grad is None


def test_dt_and_mtm_prediction_heads_remain_parameter_disjoint():
    model = make_joint()

    dt_head_ids = {
        id(parameter)
        for parameter in model.dt.predict_action.parameters()
    }
    mtm_head_ids = {
        id(parameter)
        for parameter in model.mtm.output_heads["actions"].parameters()
    }

    assert dt_head_ids.isdisjoint(mtm_head_ids)
