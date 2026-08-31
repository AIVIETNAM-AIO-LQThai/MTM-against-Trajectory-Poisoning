from __future__ import annotations

import numpy as np
import torch

from src.methods.dt.model import DecisionTransformer


@torch.no_grad()
def get_action(
    model: DecisionTransformer, states: np.ndarray, actions: np.ndarray,
    returns_to_go: np.ndarray, timesteps: np.ndarray,
    state_mean: np.ndarray, state_std: np.ndarray,
    *, context_length: int = 20, device: torch.device,
) -> np.ndarray:
    """
    Predict the action for the most recent state.

    Parameters
    ----------
    states:
        Raw environment states up through the current timestep.

        Shape:
            (T, state_dim)

    actions:
        Previously executed actions only.

        Shape:
            (T - 1, action_dim)

        A zero dummy action is appended for the current
        timestep before the model forward pass.

    returns_to_go:
        SCALED RTG values through the current timestep.

        Shape:
            (T,)

        Example for target return 5000 with scale 1000:
            initial RTG = 5.0

    timesteps:
        Episode timestep indices through current timestep.

        Shape:
            (T,)

    state_mean/state_std:
        Frozen clean-dataset normalization statistics.

    Returns
    -------
    np.ndarray
        Predicted action with shape (action_dim,).
    """
    model.eval()

    states = np.asarray(
        states,
        dtype=np.float32,
    )
    actions = np.asarray(
        actions,
        dtype=np.float32,
    )
    returns_to_go = np.asarray(
        returns_to_go,
        dtype=np.float32,
    )
    timesteps = np.asarray(
        timesteps,
        dtype=np.int64,
    )
    state_mean = np.asarray(
        state_mean,
        dtype=np.float32,
    )
    state_std = np.asarray(
        state_std,
        dtype=np.float32,
    )

    # --------------------------------------------------
    # Validate history lengths.
    # --------------------------------------------------
    T = len(states)

    if T == 0:
        raise ValueError(
            "At least one state is required."
        )
    if len(actions) != T - 1:
        raise ValueError(
            "actions must contain only previously executed "
            f"actions: expected {T - 1}, got {len(actions)}"
        )
    if len(returns_to_go) != T:
        raise ValueError(
            "returns_to_go must have one value per state."
        )
    if len(timesteps) != T:
        raise ValueError(
            "timesteps must have one value per state."
        )

    state_dim = model.state_dim
    action_dim = model.action_dim

    if states.shape != (
        T,
        state_dim,
    ):
        raise ValueError(
            f"Expected states shape ({T}, {state_dim}), "
            f"got {states.shape}"
        )

    if actions.shape != (
        T - 1,
        action_dim,
    ):
        raise ValueError(
            f"Expected actions shape "
            f"({T - 1}, {action_dim}), "
            f"got {actions.shape}"
        )

        # --------------------------------------------------
    # Reference DT evaluation preprocessing
    # --------------------------------------------------

    # Normalize REAL observed states first.
    #
    # Important:
    # Reference DT evaluation normalizes the real history
    # first, then pads the normalized sequence with zeros.
    states = (
        states - state_mean
    ) / state_std

    # --------------------------------------------------
    # Append placeholder action for CURRENT timestep.
    #
    # At timestep t:
    #
    # states  = s_0 ... s_t
    # actions = a_0 ... a_(t-1)
    #
    # a_t is what we are predicting.
    # --------------------------------------------------

    current_dummy_action = np.zeros(
        (1, action_dim),
        dtype=np.float32,
    )

    actions_with_dummy = np.concatenate(
        [
            actions,
            current_dummy_action,
        ],
        axis=0,
    )

    # --------------------------------------------------
    # Keep only the most recent K timesteps.
    # --------------------------------------------------
    states = states[-context_length:]
    actions_with_dummy = actions_with_dummy[-context_length:]
    returns_to_go = returns_to_go[-context_length:]
    timesteps = timesteps[-context_length:]

    tlen = len(states)
    pad_len = context_length - tlen

    # --------------------------------------------------
    # Reference DT EVALUATION padding.
    #
    # NOTE:
    # This intentionally differs from training:
    #
    # training:
    #   state raw-zero -> normalize
    #   action padding = -10
    #
    # evaluation:
    #   normalize real states first
    #   normalized-state padding = 0
    #   action padding = 0
    # --------------------------------------------------
    states = np.concatenate(
        [
            np.zeros((pad_len, state_dim), dtype=np.float32,
            ),
            states,
        ],
        axis=0,
    )

    actions_with_dummy = np.concatenate(
        [
            np.zeros((pad_len, action_dim), dtype=np.float32,
            ),
            actions_with_dummy,
        ],
        axis=0,
    )

    returns_to_go = np.concatenate(
        [
            np.zeros(pad_len, dtype=np.float32),
            returns_to_go,
        ],
        axis=0,
    )

    timesteps = np.concatenate(
        [
            np.zeros(pad_len, dtype=np.int64),
            timesteps,
        ],
        axis=0,
    )

    attention_mask = np.concatenate(
        [
            np.zeros(pad_len, dtype=np.int64),
            np.ones(tlen, dtype=np.int64),
        ],
        axis=0,
    )

    # --------------------------------------------------
    # Convert to tensors.
    # --------------------------------------------------

    states_tensor = torch.as_tensor(
        states,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    actions_tensor = torch.as_tensor(
        actions_with_dummy,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    rtg_tensor = torch.as_tensor(
        returns_to_go,
        dtype=torch.float32,
        device=device,
    ).reshape(
        1,
        context_length,
        1,
    )

    timesteps_tensor = torch.as_tensor(
        timesteps,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    mask_tensor = torch.as_tensor(
        attention_mask,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    # --------------------------------------------------
    # DT prediction.
    # --------------------------------------------------

    _, action_preds, _ = model(
        states_tensor,
        actions_tensor,
        rtg_tensor,
        timesteps_tensor,
        mask_tensor,
    )

    # Because of LEFT padding, the newest state always
    # occupies the final sequence position.
    action = action_preds[
        0,
        -1,
    ]

    return action.detach().cpu().numpy()