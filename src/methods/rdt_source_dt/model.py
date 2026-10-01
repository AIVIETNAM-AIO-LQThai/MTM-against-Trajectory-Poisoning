from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """Small prediction head matching the public RDT DT implementation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        num_layers: int = 1,
        use_tanh: bool = False,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        for _ in range(num_layers - 1):
            layers.append(
                nn.Linear(
                    input_dim,
                    input_dim,
                )
            )
            layers.append(
                nn.GELU()
            )

        layers.append(
            nn.Linear(
                input_dim,
                output_dim,
            )
        )

        if use_tanh:
            layers.append(
                nn.Tanh()
            )

        self.model = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            x
        )


class TransformerBlock(nn.Module):
    """
    Pre-norm causal Transformer block matching the public RDT DT source.

    The public source uses nn.MultiheadAttention in sequence-first mode,
    then manually transposes batch-first tensors.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

        self.attention_residual_dropout = (
            nn.Dropout(
                residual_dropout
            )
        )

        self.attention = (
            nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                dropout=attention_dropout,
            )
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                embedding_dim,
                4 * embedding_dim,
            ),
            nn.GELU(),
            nn.Linear(
                4 * embedding_dim,
                embedding_dim,
            ),
            nn.Dropout(
                residual_dropout
            ),
        )

        causal_mask = ~torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
            )
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        padding_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[
            : x.shape[1],
            : x.shape[1],
        ]

        norm_x = self.norm1(
            x
        )

        # Public source manually converts
        # (batch, sequence, embedding)
        # to nn.MultiheadAttention's
        # (sequence, batch, embedding).
        norm_x_seq_first = (
            norm_x.transpose(
                0,
                1,
            )
        )

        attention_out = self.attention(
            query=norm_x_seq_first,
            key=norm_x_seq_first,
            value=norm_x_seq_first,
            attn_mask=causal_mask,
            key_padding_mask=(
                padding_mask
            ),
            need_weights=False,
        )[0]

        attention_out = (
            attention_out.transpose(
                0,
                1,
            )
        )

        x = (
            x
            + self.attention_residual_dropout(
                attention_out
            )
        )

        x = (
            x
            + self.mlp(
                self.norm2(
                    x
                )
            )
        )

        return x


class SourceCompatibleDecisionTransformer(
    nn.Module
):
    """
    Vanilla DT victim matching the public RDT repository's DT semantics.

    This is intentionally separate from this repository's older
    Hugging-Face-GPT2 DecisionTransformer.
    """

    def __init__(
        self,
        *,
        state_dim: int = 17,
        action_dim: int = 6,
        seq_len: int = 20,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 1,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.1,
        embedding_dropout: float | None = None,
        predict_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.state_dim = (
            state_dim
        )

        self.action_dim = (
            action_dim
        )

        self.seq_len = seq_len
        self.episode_len = (
            episode_len
        )

        self.embedding_dim = (
            embedding_dim
        )

        self.embedding_norm = (
            nn.LayerNorm(
                embedding_dim
            )
        )

        self.output_norm = (
            nn.LayerNorm(
                embedding_dim
            )
        )

        if (
            embedding_dropout
            is not None
        ):
            self.embedding_dropout = (
                nn.Dropout(
                    embedding_dropout
                )
            )

        # Source allocates extra positions for
        # right-padded timestep indices.
        self.timestep_embedding = (
            nn.Embedding(
                episode_len + seq_len,
                embedding_dim,
            )
        )

        self.state_embedding = (
            nn.Linear(
                state_dim,
                embedding_dim,
            )
        )

        self.action_embedding = (
            nn.Linear(
                action_dim,
                embedding_dim,
            )
        )

        self.return_embedding = (
            nn.Linear(
                1,
                embedding_dim,
            )
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=(
                        3
                        * seq_len
                    ),
                    embedding_dim=(
                        embedding_dim
                    ),
                    num_heads=(
                        num_heads
                    ),
                    attention_dropout=(
                        attention_dropout
                    ),
                    residual_dropout=(
                        residual_dropout
                    ),
                )
                for _ in range(
                    num_layers
                )
            ]
        )

        self.predict_dropout = (
            nn.Dropout(
                predict_dropout
            )
        )

        self.action_head = MLPBlock(
            embedding_dim,
            action_dim,
            num_layers=1,
            use_tanh=True,
        )

        self.apply(
            self._initialize_module
        )

    @staticmethod
    def _initialize_module(
        module: nn.Module,
    ) -> None:
        # Matches the public source's apply()-based initialization.
        # Note that MultiheadAttention.in_proj_weight is a raw Parameter
        # and therefore retains PyTorch's MultiheadAttention initialization,
        # exactly as in the public implementation.
        if isinstance(
            module,
            (
                nn.Linear,
                nn.Embedding,
            ),
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if (
                isinstance(
                    module,
                    nn.Linear,
                )
                and module.bias
                is not None
            ):
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.LayerNorm,
        ):
            nn.init.zeros_(
                module.bias
            )

            nn.init.ones_(
                module.weight
            )

    def forward(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        time_steps: torch.Tensor,
        padding_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:
        batch_size = states.shape[
            0
        ]

        sequence_length = (
            states.shape[
                1
            ]
        )

        if (
            sequence_length
            > self.seq_len
        ):
            raise ValueError(
                "sequence exceeds configured "
                f"length {self.seq_len}"
            )

        if actions.shape != (
            batch_size,
            sequence_length,
            self.action_dim,
        ):
            raise ValueError(
                "unexpected action shape: "
                f"{actions.shape}"
            )

        if returns_to_go.shape != (
            batch_size,
            sequence_length,
            1,
        ):
            raise ValueError(
                "unexpected RTG shape: "
                f"{returns_to_go.shape}"
            )

        if time_steps.shape != (
            batch_size,
            sequence_length,
        ):
            raise ValueError(
                "unexpected timestep shape: "
                f"{time_steps.shape}"
            )

        if torch.any(
            time_steps < 0
        ):
            raise ValueError(
                "negative timestep"
            )

        if torch.any(
            time_steps
            >= (
                self.episode_len
                + self.seq_len
            )
        ):
            raise ValueError(
                "timestep exceeds source-compatible "
                "embedding table"
            )

        time_embedding = (
            self.timestep_embedding(
                time_steps
            )
        )

        state_embedding = (
            self.state_embedding(
                states
            )
        )

        action_embedding = (
            self.action_embedding(
                actions
            )
        )

        return_embedding = (
            self.return_embedding(
                returns_to_go
            )
        )

        # Source token order:
        # R_t, s_t, a_t.
        sequence = torch.stack(
            [
                return_embedding,
                state_embedding,
                action_embedding,
            ],
            dim=1,
        )

        sequence = (
            sequence.permute(
                0,
                2,
                1,
                3,
            )
            .reshape(
                batch_size,
                3
                * sequence_length,
                self.embedding_dim,
            )
        )

        sequence = (
            sequence
            + time_embedding.repeat_interleave(
                3,
                dim=1,
            )
        )

        if (
            padding_mask
            is not None
        ):
            if padding_mask.shape != (
                batch_size,
                sequence_length,
            ):
                raise ValueError(
                    "unexpected padding-mask shape"
                )

            token_padding_mask = (
                torch.stack(
                    [
                        padding_mask,
                        padding_mask,
                        padding_mask,
                    ],
                    dim=1,
                )
                .permute(
                    0,
                    2,
                    1,
                )
                .reshape(
                    batch_size,
                    3
                    * sequence_length,
                )
            )

        else:
            token_padding_mask = (
                None
            )

        out = self.embedding_norm(
            sequence
        )

        if hasattr(
            self,
            "embedding_dropout",
        ):
            out = (
                self.embedding_dropout(
                    out
                )
            )

        for block in self.blocks:
            out = block(
                out,
                padding_mask=(
                    token_padding_mask
                ),
            )

        out = self.output_norm(
            out
        )

        out = self.predict_dropout(
            out
        )

        state_token_output = (
            out[
                :,
                1::3,
            ]
        )

        return self.action_head(
            state_token_output
        )


def make_source_compatible_dt(
) -> SourceCompatibleDecisionTransformer:
    return (
        SourceCompatibleDecisionTransformer(
            state_dim=17,
            action_dim=6,
            seq_len=20,
            episode_len=1000,
            embedding_dim=128,
            num_layers=3,
            num_heads=1,
            attention_dropout=0.0,
            residual_dropout=0.1,
            embedding_dropout=None,
            predict_dropout=0.1,
        )
    )
