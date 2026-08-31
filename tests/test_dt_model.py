import torch

from src.methods.dt.model import (
    DecisionTransformer,
)


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


def test_dt_forward_shapes():
    torch.manual_seed(0)

    model = make_model()

    B = 4
    K = 20

    states = torch.randn(
        B,
        K,
        17,
    )

    actions = torch.randn(
        B,
        K,
        6,
    )

    returns_to_go = torch.randn(
        B,
        K,
        1,
    )

    timesteps = (
        torch.arange(K)
        .unsqueeze(0)
        .repeat(B, 1)
    )

    attention_mask = torch.ones(
        B,
        K,
        dtype=torch.long,
    )

    (
        state_preds,
        action_preds,
        return_preds,
    ) = model(
        states,
        actions,
        returns_to_go,
        timesteps,
        attention_mask,
    )

    assert state_preds.shape == (
        B,
        K,
        17,
    )

    assert action_preds.shape == (
        B,
        K,
        6,
    )

    assert return_preds.shape == (
        B,
        K,
        1,
    )


def test_action_predictions_respect_action_bounds():
    torch.manual_seed(0)

    model = make_model()
    model.eval()

    B = 2
    K = 20

    states = torch.randn(
        B,
        K,
        17,
    )

    actions = torch.randn(
        B,
        K,
        6,
    )

    returns_to_go = torch.randn(
        B,
        K,
        1,
    )

    timesteps = (
        torch.arange(K)
        .unsqueeze(0)
        .repeat(B, 1)
    )

    attention_mask = torch.ones(
        B,
        K,
        dtype=torch.long,
    )

    with torch.no_grad():
        _, action_preds, _ = model(
            states,
            actions,
            returns_to_go,
            timesteps,
            attention_mask,
        )

    assert torch.all(
        action_preds <= 1.0
    )

    assert torch.all(
        action_preds >= -1.0
    )


def test_gpt_position_embeddings_are_disabled():
    model = make_model()

    position_embeddings = (
        model.transformer.wpe.weight
    )

    assert not position_embeddings.requires_grad

    torch.testing.assert_close(
        position_embeddings,
        torch.zeros_like(
            position_embeddings
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_model_rejects_k_plus_one_rtg():
    model = make_model()

    B = 2
    K = 20

    states = torch.randn(
        B,
        K,
        17,
    )

    actions = torch.randn(
        B,
        K,
        6,
    )

    # Deliberately wrong:
    # sampler provides K+1 but model needs K.
    returns_to_go = torch.randn(
        B,
        K + 1,
        1,
    )

    timesteps = (
        torch.arange(K)
        .unsqueeze(0)
        .repeat(B, 1)
    )

    try:
        model(
            states,
            actions,
            returns_to_go,
            timesteps,
        )

    except ValueError as error:
        assert "batch.rtg[:, :-1]" in str(
            error
        )

    else:
        raise AssertionError(
            "Model accepted K+1 RTG values."
        )