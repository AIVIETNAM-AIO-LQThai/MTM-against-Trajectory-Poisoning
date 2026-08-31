from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class DecisionTransformer(nn.Module):
    """
    Vanilla Decision Transformer for continuous-control trajectories.

    Token order for each timestep t:

        (R_t, s_t, a_t)

    giving the flattened Transformer sequence:

        R_0, s_0, a_0,
        R_1, s_1, a_1,
        ...

    Action a_t is predicted from the hidden representation
    of state token s_t.

    The Transformer itself is causal.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        max_ep_len: int = 1000,
        n_layer: int = 3,
        n_head: int = 1,
        n_inner: int = 512,
        activation_function: str = "relu",
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        action_tanh: bool = True,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.max_ep_len = max_ep_len

        # --------------------------------------------------
        # GPT-2 backbone
        # --------------------------------------------------

        config = GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            embd_pdrop=embd_pdrop,
            use_cache=False,
        )

        self.transformer = GPT2Model(config)

        # --------------------------------------------------
        # IMPORTANT:
        #
        # Reference Decision Transformer removes GPT-2's
        # ordinary positional embeddings because timestep
        # embeddings are supplied explicitly below.
        #
        # Modern Hugging Face GPT2Model still contains wpe,
        # so force it permanently to zero.
        # --------------------------------------------------

        with torch.no_grad():
            self.transformer.wpe.weight.zero_()

        self.transformer.wpe.weight.requires_grad_(False)

        # --------------------------------------------------
        # Modality embeddings
        # --------------------------------------------------

        self.embed_timestep = nn.Embedding(
            max_ep_len,
            hidden_size,
        )

        self.embed_return = nn.Linear(
            1,
            hidden_size,
        )

        self.embed_state = nn.Linear(
            state_dim,
            hidden_size,
        )

        self.embed_action = nn.Linear(
            action_dim,
            hidden_size,
        )

        self.embed_ln = nn.LayerNorm(
            hidden_size,
        )

        # --------------------------------------------------
        # Prediction heads
        #
        # Reference DT contains all three heads, although
        # the paper's continuous-control training objective
        # uses only action prediction.
        # --------------------------------------------------

        self.predict_state = nn.Linear(
            hidden_size,
            state_dim,
        )

        if action_tanh:
            self.predict_action = nn.Sequential(
                nn.Linear(
                    hidden_size,
                    action_dim,
                ),
                nn.Tanh(),
            )
        else:
            self.predict_action = nn.Linear(
                hidden_size,
                action_dim,
            )

        self.predict_return = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Parameters
        ----------
        states:
            (B, K, state_dim)

        actions:
            (B, K, action_dim)

        returns_to_go:
            (B, K, 1)

            IMPORTANT:
            The batch sampler stores K+1 RTG entries.
            Pass batch.rtg[:, :-1] into this model.

        timesteps:
            (B, K)

        attention_mask:
            (B, K)

            1 = valid trajectory position
            0 = left padding

        Returns
        -------
        state_preds:
            (B, K, state_dim)

        action_preds:
            (B, K, action_dim)

        return_preds:
            (B, K, 1)
        """

        batch_size, seq_length, state_dim = (
            states.shape
        )

        if state_dim != self.state_dim:
            raise ValueError(
                "State dimension mismatch: "
                f"expected {self.state_dim}, "
                f"got {state_dim}"
            )
        if actions.shape != (
            batch_size,
            seq_length,
            self.action_dim,
        ):
            raise ValueError(
                "Unexpected action shape: "
                f"{actions.shape}"
            )

        if returns_to_go.shape != (
            batch_size,
            seq_length,
            1,
        ):
            raise ValueError(
                "returns_to_go must have shape "
                f"({batch_size}, {seq_length}, 1), "
                f"got {returns_to_go.shape}. "
                "If using DTBatch, pass "
                "batch.rtg[:, :-1]."
            )

        if timesteps.shape != (
            batch_size,
            seq_length,
        ):
            raise ValueError(
                "Unexpected timestep shape: "
                f"{timesteps.shape}"
            )
        if torch.any(timesteps < 0):
            raise ValueError(
                "Timesteps cannot be negative."
            )
        if torch.any(
            timesteps >= self.max_ep_len
        ):
            raise ValueError(
                "Timestep exceeds max_ep_len."
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                (
                    batch_size,
                    seq_length,
                ),
                dtype=torch.long,
                device=states.device,
            )

        if attention_mask.shape != (
            batch_size,
            seq_length,
        ):
            raise ValueError(
                "Unexpected attention-mask shape: "
                f"{attention_mask.shape}"
            )

        # --------------------------------------------------
        # Embed each trajectory modality.
        # --------------------------------------------------
        time_embeddings = self.embed_timestep(
            timesteps
        )
        return_embeddings = (
            self.embed_return(
                returns_to_go
            )
            + time_embeddings
        )
        state_embeddings = (
            self.embed_state(
                states
            )
            + time_embeddings
        )
        action_embeddings = (
            self.embed_action(
                actions
            )
            + time_embeddings
        )

        # --------------------------------------------------
        # Stack:
        #
        # (R_0, s_0, a_0,
        #  R_1, s_1, a_1, ...)
        # --------------------------------------------------
        stacked_inputs = torch.stack(
            (
                return_embeddings,
                state_embeddings,
                action_embeddings,
            ),
            dim=2,
        )
        stacked_inputs = stacked_inputs.reshape(
            batch_size,
            3 * seq_length,
            self.hidden_size,
        )
        stacked_inputs = self.embed_ln(
            stacked_inputs
        )

        # Same validity mask applies to each of:
        #
        # R_t, s_t, a_t
        stacked_attention_mask = (
            attention_mask.unsqueeze(-1)
            .expand(
                batch_size,
                seq_length,
                3,
            )
            .reshape(
                batch_size,
                3 * seq_length,
            )
        )

        # --------------------------------------------------
        # Strictly causal GPT-2 forward pass.
        # --------------------------------------------------
        transformer_output = (
            self.transformer(
                inputs_embeds=stacked_inputs,
                attention_mask=(
                    stacked_attention_mask
                ),
                use_cache=False,
            )
        )

        x = (
            transformer_output
            .last_hidden_state
        )

        # Restore modality dimension:
        #
        # x[:, :, 0] = return tokens
        # x[:, :, 1] = state tokens
        # x[:, :, 2] = action tokens
        x = x.reshape(
            batch_size,
            seq_length,
            3,
            self.hidden_size,
        )

        return_hidden = x[:, :, 0]
        state_hidden = x[:, :, 1]
        action_hidden = x[:, :, 2]

        # --------------------------------------------------
        # Reference DT prediction alignment.
        #
        # state token predicts CURRENT action:
        #
        #       s_t -> a_t
        #
        # Action token is used for next-state / next-return
        # prediction heads.
        # --------------------------------------------------
        action_preds = self.predict_action(
            state_hidden
        )
        state_preds = self.predict_state(
            action_hidden
        )
        return_preds = self.predict_return(
            action_hidden
        )

        return (
            state_preds,
            action_preds,
            return_preds,
        )