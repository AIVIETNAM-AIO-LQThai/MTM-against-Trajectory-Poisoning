from random import Random
import torch
import numpy as np
from src.data.batching import sample_dt_batch
from src.data.trajectories import (
    find_completed_trajectories,
)
from src.methods.dt.losses import masked_action_mse

def make_short_synthetic_dataset():
    """
    One completed trajectory of length 3.

    With K=20 this guarantees left padding,
    regardless of which start index is sampled.
    """
    observations = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=np.float32,
    )
    actions = np.array(
        [
            [0.1],
            [0.2],
            [0.3],
        ],
        dtype=np.float32,
    )

    rewards = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )
    terminals = np.array(
        [False, False, True],
        dtype=bool,
    )
    timeouts = np.array(
        [False, False, False],
        dtype=bool,
    )
    trajectories, trailing = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    assert trailing == 0
    assert len(trajectories) == 1

    # Deliberately simple frozen statistics.
    state_mean = np.array(
        [2.0, 20.0],
        dtype=np.float32,
    )
    state_std = np.array(
        [1.0, 10.0],
        dtype=np.float32,
    )

    return (
        observations, actions, rewards,
        terminals, trajectories,
        state_mean, state_std,
    )

def test_left_padding_values_and_mask():
    (
        observations, actions, rewards,
        terminals, trajectories,
        state_mean, state_std,
    ) = make_short_synthetic_dataset()

    K = 20

    batch = sample_dt_batch(
        observations, actions, rewards,
        terminals, trajectories,
        state_mean, state_std,
        batch_size=1, context_length=K, max_ep_len=1000,
        rtg_scale=1000.0,
        np_rng=np.random.default_rng(0),
        py_rng=Random(0),
    )
    local_start = int(
        batch.start_indices[0]
    )
    trajectory = trajectories[0]
    remaining_length = (
        trajectory.length - local_start
    )
    tlen = min(
        K,
        remaining_length,
    )
    pad_len = K - tlen

    assert pad_len > 0
    assert int(
        batch.attention_mask[0].sum()
    ) == tlen
    assert (
        np.count_nonzero(
            batch.attention_mask[0] == 0
        )
        == pad_len
    )

    # --------------------------------------------------
    # Attention mask
    # --------------------------------------------------
    np.testing.assert_array_equal(
        batch.attention_mask[
            0,
            :pad_len,
        ],
        np.zeros(
            pad_len,
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        batch.attention_mask[
            0,
            pad_len:,
        ],
        np.ones(
            tlen,
            dtype=np.float32,
        ),
    )

    # --------------------------------------------------
    # Action padding = -10
    # --------------------------------------------------
    np.testing.assert_array_equal(
        batch.actions[
            0,
            :pad_len,
        ],
        np.full(
            (pad_len, 1),
            -10.0,
            dtype=np.float32,
        ),
    )

    # --------------------------------------------------
    # Reward padding = 0
    # --------------------------------------------------
    np.testing.assert_array_equal(
        batch.rewards[
            0,
            :pad_len,
            0,
        ],
        np.zeros(
            pad_len,
            dtype=np.float32,
        ),
    )

    # --------------------------------------------------
    # Done padding = 2
    # --------------------------------------------------
    np.testing.assert_array_equal(
        batch.dones[
            0,
            :pad_len,
        ],
        np.full(
            pad_len,
            2,
            dtype=np.int64,
        ),
    )

    # --------------------------------------------------
    # Timestep padding = 0
    # --------------------------------------------------
    np.testing.assert_array_equal(
        batch.timesteps[
            0,
            :pad_len,
        ],
        np.zeros(
            pad_len,
            dtype=np.int64,
        ),
    )

    # --------------------------------------------------
    # RTG left padding = 0
    #
    # RTG has K+1 entries, but still exactly pad_len
    # leading padding entries.
    # --------------------------------------------------
    np.testing.assert_array_equal(
        batch.rtg[
            0,
            :pad_len,
            0,
        ],
        np.zeros(
            pad_len,
            dtype=np.float32,
        ),
    )

    expected_padded_state = (
        np.zeros(
            state_mean.shape,
            dtype=np.float32,
        )
        - state_mean
    ) / state_std

    expected_padded_states = np.repeat(
        expected_padded_state[None, :],
        pad_len,
        axis=0,
    )

    np.testing.assert_allclose(
        batch.states[
            0,
            :pad_len,
        ],
        expected_padded_states,
        rtol=0.0,
        atol=1e-6,
    )

def masked_action_mse_numpy(
    predictions: np.ndarray,
    targets: np.ndarray,
    attention_mask: np.ndarray,
) -> float:
    """
    Reference semantics for DT action loss.

    Only positions whose attention mask is > 0
    contribute to MSE.
    """
    valid = attention_mask > 0
    valid_predictions = predictions[valid]
    valid_targets = targets[valid]

    if valid_predictions.size == 0:
        raise ValueError(
            "No valid action positions."
        )

    return float(
        np.mean(
            (valid_predictions - valid_targets) ** 2
        )
    )

def test_padded_actions_do_not_affect_action_loss():
    # B=1, K=5, action_dim=1
    targets = np.array(
        [[
            [-10.0],
            [-10.0],
            [0.1],
            [0.2],
            [0.3],
        ]],
        dtype=np.float32,
    )

    attention_mask = np.array(
        [[0, 0, 1, 1, 1]],
        dtype=np.float32,
    )

    predictions_a = np.array(
        [
            [
                [999999.0],
                [-999999.0],
                [0.0],
                [0.0],
                [0.0],
            ]
        ],
        dtype=np.float32,
    )

    predictions_b = predictions_a.copy()

    # Change ONLY padded predictions dramatically.
    predictions_b[0, 0, 0] = -123456789.0
    predictions_b[0, 1, 0] = 987654321.0

    loss_a = masked_action_mse_numpy(
        predictions_a,
        targets,
        attention_mask,
    )

    loss_b = masked_action_mse_numpy(
        predictions_b,
        targets,
        attention_mask,
    )

    # If padding is correctly masked,
    # changing padded predictions cannot affect loss.
    assert np.isclose(
        loss_a,
        loss_b,
        rtol=0.0,
        atol=0.0,
    )

    expected_loss = np.mean(
        np.array(
            [
                (0.0 - 0.1) ** 2,
                (0.0 - 0.2) ** 2,
                (0.0 - 0.3) ** 2,
            ]
        )
    )

    assert np.isclose(
        loss_a,
        expected_loss,
        atol=1e-7,
    )

def test_sampler_padding_matches_loss_mask():
    (
        observations,
        actions,
        rewards,
        terminals,
        trajectories,
        state_mean,
        state_std,
    ) = make_short_synthetic_dataset()

    batch = sample_dt_batch(
        observations,
        actions,
        rewards,
        terminals,
        trajectories,
        state_mean,
        state_std,
        batch_size=1,
        context_length=20,
        max_ep_len=1000,
        rtg_scale=1000.0,
        np_rng=np.random.default_rng(7),
        py_rng=Random(7),
    )

    mask = batch.attention_mask[0]

    padded = mask == 0
    valid = mask == 1

    assert padded.any()
    assert valid.any()

    # Every padded action should contain the -10 sentinel.
    np.testing.assert_array_equal(
        batch.actions[0][padded],
        np.full_like(
            batch.actions[0][padded],
            -10.0,
        ),
    )

    # No valid action should contain the padding sentinel.
    assert not np.any(
        batch.actions[0][valid]
        == -10.0
    )

def test_torch_masked_loss_ignores_padding():
    targets = torch.tensor(
        [
            [
                [-10.0],
                [-10.0],
                [0.1],
                [0.2],
                [0.3],
            ]
        ],
        dtype=torch.float32,
    )

    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1]
        ],
        dtype=torch.long,
    )

    predictions_a = torch.tensor(
        [
            [
                [999999.0],
                [-999999.0],
                [0.0],
                [0.0],
                [0.0],
            ]
        ],
        dtype=torch.float32,
    )

    predictions_b = (
        predictions_a.clone()
    )

    # Change only padded positions.
    predictions_b[0, 0, 0] = -123456.0
    predictions_b[0, 1, 0] = 987654.0

    loss_a = masked_action_mse(
        predictions_a,
        targets,
        attention_mask,
    )

    loss_b = masked_action_mse(
        predictions_b,
        targets,
        attention_mask,
    )

    torch.testing.assert_close(
        loss_a,
        loss_b,
        rtol=0.0,
        atol=0.0,
    )

    expected = torch.mean(
        torch.tensor(
            [
                0.1**2,
                0.2**2,
                0.3**2,
            ],
            dtype=torch.float32,
        )
    )

    torch.testing.assert_close(
        loss_a,
        expected,
        rtol=1e-6,
        atol=1e-7,
    )

def test_masked_positions_receive_zero_gradient():
    predictions = torch.tensor(
        [
            [
                [5.0],
                [-7.0],
                [0.5],
                [0.6],
                [0.7],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    targets = torch.tensor(
        [
            [
                [-10.0],
                [-10.0],
                [0.1],
                [0.2],
                [0.3],
            ]
        ],
        dtype=torch.float32,
    )

    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1]
        ],
        dtype=torch.long,
    )

    loss = masked_action_mse(
        predictions,
        targets,
        attention_mask,
    )

    loss.backward()

    assert predictions.grad is not None

    # Padded positions receive exactly no gradient.
    torch.testing.assert_close(
        predictions.grad[0, :2],
        torch.zeros_like(
            predictions.grad[0, :2]
        ),
        rtol=0.0,
        atol=0.0,
    )

    # Valid positions should contribute gradients.
    assert torch.any(
        predictions.grad[0, 2:] != 0
    )