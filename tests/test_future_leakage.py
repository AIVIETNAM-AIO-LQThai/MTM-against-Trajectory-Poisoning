import torch

from src.methods.dt.model import (
    DecisionTransformer,
)


def make_model():
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


def test_future_information_cannot_change_current_action():
    torch.manual_seed(123)

    model = make_model()

    # CRITICAL:
    # Disable dropout so two forwards are deterministic.
    model.eval()

    B = 1
    K = 20

    states_a = torch.randn(
        B, K, 17
    )

    actions_a = torch.randn(
        B, K, 6
    )

    rtg_a = torch.randn(
        B, K, 1
    )

    timesteps = (
        torch.arange(K)
        .unsqueeze(0)
    )

    attention_mask = torch.ones(
        B,
        K,
        dtype=torch.long,
    )

    # Clone into experiment B.
    states_b = states_a.clone()
    actions_b = actions_a.clone()
    rtg_b = rtg_a.clone()

    # We inspect prediction of a_t.
    t = 7

    # --------------------------------------------------
    # Change CURRENT action a_t itself.
    #
    # Since action prediction comes from state token s_t,
    # the true a_t token occurs AFTER that prediction
    # position and therefore must not affect a_hat_t.
    # --------------------------------------------------

    actions_b[:, t, :] = 999.0

    # --------------------------------------------------
    # Completely destroy everything strictly in future.
    # --------------------------------------------------

    states_b[:, t + 1 :, :] = -777.0
    actions_b[:, t + 1 :, :] = 888.0
    rtg_b[:, t + 1 :, :] = 555.0

    with torch.no_grad():
        _, preds_a, _ = model(
            states_a,
            actions_a,
            rtg_a,
            timesteps,
            attention_mask,
        )

        _, preds_b, _ = model(
            states_b,
            actions_b,
            rtg_b,
            timesteps,
            attention_mask,
        )

    # All predictions through timestep t must be identical.
    torch.testing.assert_close(
        preds_a[:, : t + 1],
        preds_b[:, : t + 1],
        rtol=0.0,
        atol=1e-6,
    )