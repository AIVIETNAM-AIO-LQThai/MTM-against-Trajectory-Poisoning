from __future__ import annotations

import torch

from src.methods.dt.model import DecisionTransformer
from src.methods.dt_mtm.model import DTMTMModel
from src.methods.mtm.model import MTMConfig, ReferenceMTM


STATE_DIM = 17
ACTION_DIM = 6
DT_CONTEXT_LENGTH = 20
MTM_TRAJECTORY_LENGTH = 4
MAX_EP_LEN = 1000
MTM_TRAIN_FRACTION = 0.95
MTM_DISCOUNT = 1.5


def build_walker2d_dt() -> DecisionTransformer:
    """Build the frozen Group-1 Walker2d Decision Transformer."""

    return DecisionTransformer(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_size=128,
        max_ep_len=MAX_EP_LEN,
        n_layer=3,
        n_head=1,
        n_inner=512,
        activation_function="relu",
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        embd_pdrop=0.1,
        action_tanh=True,
    )


def build_walker2d_mtm() -> ReferenceMTM:
    """Build the validated Group-3 MTM architecture from scratch."""

    return ReferenceMTM(
        {
            "states": (1, STATE_DIM),
            "actions": (1, ACTION_DIM),
            "returns": (1, 1),
        },
        traj_length=MTM_TRAJECTORY_LENGTH,
        config=MTMConfig(
            n_embd=512,
            n_head=4,
            n_enc_layer=2,
            n_dec_layer=1,
            dropout=0.1,
        ),
    )


def build_walker2d_dt_mtm() -> DTMTMModel:
    """
    Build the Group-4 primary joint model.

    DT is constructed first so its initialization for a given torch seed is
    identical to building the frozen Group-1 DT alone. The Group-3 checkpoint
    is deliberately NOT loaded: the primary comparison studies the auxiliary
    masked objective/shared representation rather than MTM pretraining.
    """

    dt = build_walker2d_dt()
    mtm = build_walker2d_mtm()
    return DTMTMModel(dt, mtm)


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
