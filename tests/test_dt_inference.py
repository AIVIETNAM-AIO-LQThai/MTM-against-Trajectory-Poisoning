import numpy as np
import torch

from src.methods.dt.inference import get_action
from src.methods.dt.model import DecisionTransformer


def make_model():
    torch.manual_seed(0)

    model = DecisionTransformer(
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

    model.eval()

    return model

def test_get_action_shape_and_bounds():
    model = make_model()

    device = torch.device("cpu")
    model.to(device)

    states = np.random.default_rng(0).normal(
        size=(1, 17)
    ).astype(np.float32)

    # No previous actions at timestep 0.
    actions = np.empty(
        (0, 6),
        dtype=np.float32,
    )
    returns_to_go = np.array(
        [5.0],
        dtype=np.float32,
    )
    timesteps = np.array(
        [0],
        dtype=np.int64,
    )
    state_mean = np.zeros(
        17,
        dtype=np.float32,
    )
    state_std = np.ones(
        17,
        dtype=np.float32,
    )

    action = get_action(
        model, states, actions,
        returns_to_go, timesteps,
        state_mean, state_std,
        context_length=20, device=device,
    )

    assert action.shape == (6,)
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

def test_get_action_handles_history_longer_than_context():
    model = make_model()

    device = torch.device("cpu")
    model.to(device)

    rng = np.random.default_rng(123)

    T = 35

    states = rng.normal(
        size=(T, 17)
    ).astype(np.float32)

    actions = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(T - 1, 6),
    ).astype(np.float32)

    returns_to_go = np.linspace(
        5.0,
        4.0,
        T,
        dtype=np.float32,
    )

    timesteps = np.arange(
        T,
        dtype=np.int64,
    )

    state_mean = np.zeros(
        17,
        dtype=np.float32,
    )

    state_std = np.ones(
        17,
        dtype=np.float32,
    )

    action = get_action(
        model,
        states,
        actions,
        returns_to_go,
        timesteps,
        state_mean,
        state_std,
        context_length=20,
        device=device,
    )

    assert action.shape == (6,)
    assert np.isfinite(action).all()

def test_get_action_uses_only_last_k_timesteps():
    model = make_model()

    device = torch.device("cpu")
    model.to(device)

    rng = np.random.default_rng(456)

    K = 20
    T = 30

    states_a = rng.normal(
        size=(T, 17)
    ).astype(np.float32)
    actions_a = rng.uniform(
        -1.0,
        1.0,
        size=(T - 1, 6),
    ).astype(np.float32)
    rtg_a = rng.normal(
        size=T
    ).astype(np.float32)

    timesteps = np.arange(
        T,
        dtype=np.int64,
    )

    # Clone identical history.
    states_b = states_a.copy()
    actions_b = actions_a.copy()
    rtg_b = rtg_a.copy()

    # Destroy ONLY history older than the last K.
    old_length = T - K
    states_b[:old_length] = 9999.0

    # Actions correspond to transitions.
    actions_b[:old_length] = -9999.0
    rtg_b[:old_length] = 7777.0
    state_mean = np.zeros(
        17,
        dtype=np.float32,
    )
    state_std = np.ones(
        17,
        dtype=np.float32,
    )

    action_a = get_action(
        model, states_a, actions_a,
        rtg_a, timesteps,
        state_mean, state_std,
        context_length=K, device=device,
    )

    action_b = get_action(
        model, states_b, actions_b,
        rtg_b, timesteps,
        state_mean, state_std,
        context_length=K, device=device,
    )

    np.testing.assert_allclose(
        action_a, action_b,
        rtol=0.0, atol=1e-6,
    )

def test_get_action_is_deterministic_in_eval_mode():
    model = make_model()

    device = torch.device("cpu")
    model.to(device)

    rng = np.random.default_rng(999)

    states = rng.normal(
        size=(8, 17)
    ).astype(np.float32)

    actions = rng.uniform(
        -1,
        1,
        size=(7, 6),
    ).astype(np.float32)

    rtg = np.linspace(
        5.0,
        4.5,
        8,
        dtype=np.float32,
    )

    timesteps = np.arange(
        8,
        dtype=np.int64,
    )

    state_mean = np.zeros(
        17,
        dtype=np.float32,
    )

    state_std = np.ones(
        17,
        dtype=np.float32,
    )

    action_a = get_action(
        model,
        states,
        actions,
        rtg,
        timesteps,
        state_mean,
        state_std,
        context_length=20,
        device=device,
    )

    action_b = get_action(
        model,
        states,
        actions,
        rtg,
        timesteps,
        state_mean,
        state_std,
        context_length=20,
        device=device,
    )

    np.testing.assert_array_equal(
        action_a,
        action_b,
    )

def test_get_action_uses_reference_evaluation_preprocessing():
    model = make_model()

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    rng = np.random.default_rng(1234)

    T = 3
    K = 20

    states = rng.normal(
        size=(T, 17)
    ).astype(np.float32)

    actions = rng.uniform(
        -1.0,
        1.0,
        size=(T - 1, 6),
    ).astype(np.float32)

    returns_to_go = np.array(
        [5.0, 4.99, 4.98],
        dtype=np.float32,
    )

    timesteps = np.arange(
        T,
        dtype=np.int64,
    )

    # Nontrivial stats are intentional.
    # This lets the test distinguish:
    #
    #   normalize -> zero pad
    #
    # from:
    #
    #   zero pad -> normalize
    state_mean = np.linspace(
        -1.0,
        1.0,
        17,
        dtype=np.float32,
    )

    state_std = np.linspace(
        0.5,
        1.5,
        17,
        dtype=np.float32,
    )

    captured = {}

    def capture_inputs(
        module,
        args,
    ):
        (
            states_tensor,
            actions_tensor,
            rtg_tensor,
            timesteps_tensor,
            mask_tensor,
        ) = args

        captured["states"] = (
            states_tensor.detach()
            .cpu()
            .numpy()
        )

        captured["actions"] = (
            actions_tensor.detach()
            .cpu()
            .numpy()
        )

        captured["rtg"] = (
            rtg_tensor.detach()
            .cpu()
            .numpy()
        )

        captured["timesteps"] = (
            timesteps_tensor.detach()
            .cpu()
            .numpy()
        )

        captured["mask"] = (
            mask_tensor.detach()
            .cpu()
            .numpy()
        )

    hook = model.register_forward_pre_hook(
        capture_inputs
    )

    try:
        get_action(
            model=model,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            state_mean=state_mean,
            state_std=state_std,
            context_length=K,
            device=device,
        )

    finally:
        hook.remove()

    pad_len = K - T

    model_states = (
        captured["states"][0]
    )

    model_actions = (
        captured["actions"][0]
    )

    model_rtg = (
        captured["rtg"][0, :, 0]
    )

    model_timesteps = (
        captured["timesteps"][0]
    )

    model_mask = (
        captured["mask"][0]
    )

    # --------------------------------------------------
    # 1. Evaluation state padding must be EXACT zero
    #    in normalized state space.
    # --------------------------------------------------

    np.testing.assert_array_equal(
        model_states[:pad_len],
        np.zeros(
            (pad_len, 17),
            dtype=np.float32,
        ),
    )

    # --------------------------------------------------
    # 2. Real states must be normalized.
    # --------------------------------------------------

    expected_real_states = (
        states - state_mean
    ) / state_std

    np.testing.assert_allclose(
        model_states[pad_len:],
        expected_real_states,
        rtol=0.0,
        atol=1e-6,
    )

    # --------------------------------------------------
    # 3. Evaluation action padding must be ZERO,
    #    NOT training sentinel -10.
    # --------------------------------------------------

    np.testing.assert_array_equal(
        model_actions[:pad_len],
        np.zeros(
            (pad_len, 6),
            dtype=np.float32,
        ),
    )

    # Previous real actions must remain unchanged.
    np.testing.assert_array_equal(
        model_actions[
            pad_len:pad_len + T - 1
        ],
        actions,
    )

    # Current unknown action is dummy zero.
    np.testing.assert_array_equal(
        model_actions[-1],
        np.zeros(
            6,
            dtype=np.float32,
        ),
    )

    # --------------------------------------------------
    # 4. RTG padding.
    # --------------------------------------------------

    np.testing.assert_array_equal(
        model_rtg[:pad_len],
        np.zeros(
            pad_len,
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        model_rtg[pad_len:],
        returns_to_go,
    )

    # --------------------------------------------------
    # 5. Timestep padding.
    # --------------------------------------------------

    np.testing.assert_array_equal(
        model_timesteps[:pad_len],
        np.zeros(
            pad_len,
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        model_timesteps[pad_len:],
        timesteps,
    )

    # --------------------------------------------------
    # 6. Attention mask.
    # --------------------------------------------------

    np.testing.assert_array_equal(
        model_mask[:pad_len],
        np.zeros(
            pad_len,
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        model_mask[pad_len:],
        np.ones(
            T,
            dtype=np.int64,
        ),
    )