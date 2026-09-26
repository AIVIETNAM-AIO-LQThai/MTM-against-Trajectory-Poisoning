from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class MTMConfig:
    """
    Reference continuous-D4RL MTM configuration.

    Defaults match the official d4rl_cont experiment.
    """

    n_embd: int = 512
    n_head: int = 4
    n_enc_layer: int = 2
    n_dec_layer: int = 1
    dropout: float = 0.1


def make_1d_sincos_position_embedding(
    embed_dim: int,
    length: int,
) -> torch.Tensor:
    """
    Reproduce the fixed 1D sin/cos positional embedding used
    by the reference MTM.

    Output shape:
        [1, T, 1, embed_dim]

    The official implementation divides the embedding by 2.
    """

    if embed_dim <= 0:
        raise ValueError(
            "embed_dim must be positive"
        )

    if embed_dim % 2 != 0:
        raise ValueError(
            "embed_dim must be even"
        )

    if length <= 0:
        raise ValueError(
            "length must be positive"
        )

    half_dim = embed_dim // 2

    omega = np.arange(
        half_dim,
        dtype=np.float32,
    )

    omega /= float(
        half_dim
    )

    omega = (
        1.0
        / (
            10000.0
            ** omega
        )
    )

    positions = np.arange(
        length,
        dtype=np.float32,
    )

    angles = np.einsum(
        "t,d->td",
        positions,
        omega,
    )

    embedding = np.concatenate(
        [
            np.sin(angles),
            np.cos(angles),
        ],
        axis=1,
    )

    embedding = (
        torch.from_numpy(
            embedding
        )
        .to(torch.float32)
        .unsqueeze(0)
        .unsqueeze(2)
        / 2.0
    )

    return embedding


class ReferenceMTM(nn.Module):
    """
    Standalone reference-style Masked Trajectory Model.

    Expected encoded continuous modality shape:

        [B, T, P, D]

    For our current continuous tokenizers:

        P = 1

    Current modeled modalities:

        states
        actions
        returns

    This class contains NO:

        - Decision Transformer logic
        - causal attention
        - poisoning logic
        - reconstruction loss
        - project-specific masks
    """

    def __init__(
        self,
        data_shapes: Mapping[
            str,
            tuple[int, int],
        ],
        *,
        traj_length: int,
        config: MTMConfig | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = MTMConfig()

        if traj_length <= 0:
            raise ValueError(
                "traj_length must be positive"
            )

        if config.n_embd <= 0:
            raise ValueError(
                "n_embd must be positive"
            )

        if config.n_head <= 0:
            raise ValueError(
                "n_head must be positive"
            )

        if (
            config.n_embd
            % config.n_head
            != 0
        ):
            raise ValueError(
                "n_embd must be divisible "
                "by n_head"
            )

        if config.n_embd % 2 != 0:
            raise ValueError(
                "n_embd must be even"
            )

        if config.n_enc_layer <= 0:
            raise ValueError(
                "n_enc_layer must be positive"
            )

        if config.n_dec_layer <= 0:
            raise ValueError(
                "n_dec_layer must be positive"
            )

        if len(data_shapes) == 0:
            raise ValueError(
                "data_shapes must not be empty"
            )

        self.config = config
        self.traj_length = int(
            traj_length
        )

        # Preserve insertion order because the reference
        # model concatenates modalities in dict order.
        self.data_shapes: dict[
            str,
            tuple[int, int],
        ] = {}

        for key, shape in (
            data_shapes.items()
        ):
            if len(shape) != 2:
                raise ValueError(
                    f"{key}: data shape must be "
                    "(tokens_per_timestep, feature_dim)"
                )

            tokens_per_timestep = int(
                shape[0]
            )

            feature_dim = int(
                shape[1]
            )

            if tokens_per_timestep <= 0:
                raise ValueError(
                    f"{key}: token count must be positive"
                )

            if feature_dim <= 0:
                raise ValueError(
                    f"{key}: feature dimension "
                    "must be positive"
                )

            self.data_shapes[key] = (
                tokens_per_timestep,
                feature_dim,
            )

        self.modality_order = tuple(
            self.data_shapes.keys()
        )

        hidden_dim = config.n_embd

        # --------------------------------------------------
        # Encoder-side modality-specific components
        # --------------------------------------------------

        self.encoder_embed = (
            nn.ModuleDict()
        )

        self.encoder_per_dim_encoding = (
            nn.ParameterDict()
        )

        # --------------------------------------------------
        # Decoder-side components
        # --------------------------------------------------

        self.decoder_embed = (
            nn.ModuleDict()
        )

        self.decoder_per_dim_encoding = (
            nn.ParameterDict()
        )

        self.mask_tokens = (
            nn.ParameterDict()
        )

        self.output_heads = (
            nn.ModuleDict()
        )

        for key, (
            tokens_per_timestep,
            feature_dim,
        ) in self.data_shapes.items():

            self.encoder_embed[
                key
            ] = nn.Linear(
                feature_dim,
                hidden_dim,
            )

            self.encoder_per_dim_encoding[
                key
            ] = nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    tokens_per_timestep,
                    hidden_dim,
                )
            )

            self.decoder_embed[
                key
            ] = nn.Linear(
                hidden_dim,
                hidden_dim,
            )

            self.decoder_per_dim_encoding[
                key
            ] = nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    tokens_per_timestep,
                    hidden_dim,
                )
            )

            self.mask_tokens[
                key
            ] = nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    hidden_dim,
                )
            )

            self.output_heads[
                key
            ] = nn.Sequential(
                nn.LayerNorm(
                    hidden_dim
                ),
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(
                    hidden_dim,
                    feature_dim,
                ),
            )

        # --------------------------------------------------
        # Bidirectional encoder
        #
        # IMPORTANT:
        # no causal attention mask is supplied.
        # --------------------------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=config.n_head,
                dim_feedforward=(
                    hidden_dim * 4
                ),
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=(
                    config.n_enc_layer
                ),
                norm=nn.LayerNorm(
                    hidden_dim
                ),
            )
        )

        # --------------------------------------------------
        # Bidirectional decoder
        #
        # The reference implementation also uses
        # TransformerEncoder blocks for the decoder.
        # --------------------------------------------------

        decoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=config.n_head,
                dim_feedforward=(
                    hidden_dim * 4
                ),
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.decoder = (
            nn.TransformerEncoder(
                decoder_layer,
                num_layers=(
                    config.n_dec_layer
                ),
                norm=nn.LayerNorm(
                    hidden_dim
                ),
            )
        )

        position_embedding = (
            make_1d_sincos_position_embedding(
                hidden_dim,
                self.traj_length,
            )
        )

        self.register_buffer(
            "position_embedding",
            position_embedding,
        )

    def _validate_trajectories(
        self,
        trajectories: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> tuple[int, int]:
        """
        Validate trajectory modality keys and tensor shapes.

        Returns:
            batch_size,
            trajectory_length
        """

        if set(
            trajectories.keys()
        ) != set(
            self.modality_order
        ):
            raise ValueError(
                "trajectory modalities do not "
                "match model modalities"
            )

        batch_size = None
        trajectory_length = None

        for key in self.modality_order:
            tensor = trajectories[
                key
            ]

            if tensor.ndim != 4:
                raise ValueError(
                    f"{key}: expected [B,T,P,D], "
                    f"got {tuple(tensor.shape)}"
                )

            (
                current_batch,
                current_length,
                token_count,
                feature_dim,
            ) = tensor.shape

            (
                expected_tokens,
                expected_features,
            ) = self.data_shapes[
                key
            ]

            if token_count != (
                expected_tokens
            ):
                raise ValueError(
                    f"{key}: expected "
                    f"{expected_tokens} tokens "
                    "per timestep"
                )

            if feature_dim != (
                expected_features
            ):
                raise ValueError(
                    f"{key}: expected feature "
                    f"dimension "
                    f"{expected_features}"
                )

            if current_length > (
                self.traj_length
            ):
                raise ValueError(
                    f"{key}: trajectory length "
                    "exceeds model maximum"
                )

            if batch_size is None:
                batch_size = int(
                    current_batch
                )

                trajectory_length = int(
                    current_length
                )

            else:
                if current_batch != (
                    batch_size
                ):
                    raise ValueError(
                        "all modalities must "
                        "share batch size"
                    )

                if current_length != (
                    trajectory_length
                ):
                    raise ValueError(
                        "all modalities must "
                        "share trajectory length"
                    )

        assert batch_size is not None
        assert trajectory_length is not None

        return (
            batch_size,
            trajectory_length,
        )

    @staticmethod
    def _expand_mask(
        mask: torch.Tensor,
        *,
        trajectory_length: int,
        tokens_per_timestep: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert a reference MTM mask to flattened [T * P].

        Accepted input:

            [T]
            [T, P]

        Convention:

            1 = visible
            0 = hidden
        """

        mask = torch.as_tensor(
            mask,
            device=device,
        )

        if mask.ndim == 1:
            if mask.shape[0] != (
                trajectory_length
            ):
                raise ValueError(
                    "1D mask has wrong "
                    "trajectory length"
                )

            mask = (
                mask[:, None]
                .repeat(
                    1,
                    tokens_per_timestep,
                )
            )

        elif mask.ndim == 2:
            expected_shape = (
                trajectory_length,
                tokens_per_timestep,
            )

            if tuple(
                mask.shape
            ) != expected_shape:
                raise ValueError(
                    "2D mask has wrong shape"
                )

        else:
            raise ValueError(
                "mask must have shape "
                "[T] or [T,P]"
            )

        valid = (
            (mask == 0)
            | (mask == 1)
        )

        if not bool(
            valid.all()
        ):
            raise ValueError(
                "mask must contain only "
                "0 or 1"
            )

        return mask.reshape(
            -1
        )

    def process_masks(
        self,
        trajectories: Mapping[
            str,
            torch.Tensor,
        ],
        masks: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        """
        Validate and flatten all modality masks.
        """

        _, trajectory_length = (
            self._validate_trajectories(
                trajectories
            )
        )

        if set(
            masks.keys()
        ) != set(
            self.modality_order
        ):
            raise ValueError(
                "mask modalities do not "
                "match model modalities"
            )

        processed = {}

        for key in self.modality_order:
            tokens_per_timestep = (
                self.data_shapes[
                    key
                ][0]
            )

            processed[key] = (
                self._expand_mask(
                    masks[key],
                    trajectory_length=(
                        trajectory_length
                    ),
                    tokens_per_timestep=(
                        tokens_per_timestep
                    ),
                    device=(
                        trajectories[
                            key
                        ].device
                    ),
                )
            )

        return processed

    def trajectory_encoding(
        self,
        trajectories: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        """
        Encoder-side modality embedding.

        [B,T,P,D]
            ->
        [B,T*P,C]
        """

        self._validate_trajectories(
            trajectories
        )

        encoded = {}

        for key in self.modality_order:
            trajectory = (
                trajectories[
                    key
                ].to(
                    torch.float32
                )
            )

            (
                batch_size,
                trajectory_length,
                token_count,
                _,
            ) = trajectory.shape

            x = (
                self.encoder_embed[
                    key
                ](
                    trajectory
                )
                + self.encoder_per_dim_encoding[
                    key
                ]
                + self.position_embedding[
                    :,
                    :trajectory_length,
                    :,
                    :,
                ]
            )

            encoded[key] = (
                x.reshape(
                    batch_size,
                    trajectory_length
                    * token_count,
                    self.config.n_embd,
                )
            )

        return encoded

    @staticmethod
    def _select_visible_tokens(
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """
        Remove hidden tokens before the encoder.

        Returns:

            visible_tokens
            restore_indices
            visible_count
        """

        if mask.ndim != 1:
            raise ValueError(
                "flattened mask must be 1D"
            )

        if x.shape[1] != (
            mask.numel()
        ):
            raise ValueError(
                "mask/token length mismatch"
            )

        visible_ids = (
            (mask == 1)
            .nonzero(
                as_tuple=True
            )[0]
        )

        hidden_ids = (
            (mask == 0)
            .nonzero(
                as_tuple=True
            )[0]
        )

        reordered_ids = (
            torch.cat(
                [
                    visible_ids,
                    hidden_ids,
                ],
                dim=0,
            )
        )

        restore_indices = (
            torch.argsort(
                reordered_ids
            )
        )

        visible_tokens = x[
            :,
            visible_ids,
            :,
        ]

        return (
            visible_tokens,
            restore_indices,
            int(
                visible_ids.numel()
            ),
        )

    def forward_encoder(
        self,
        encoded_trajectories: Mapping[
            str,
            torch.Tensor,
        ],
        masks: Mapping[
            str,
            torch.Tensor,
        ],
    ):
        """
        Concatenate visible tokens across modalities and run the
        bidirectional encoder.
        """

        visible_parts = []

        restore_indices = {}
        visible_counts = {}

        for key in self.modality_order:
            (
                visible,
                restore,
                visible_count,
            ) = self._select_visible_tokens(
                encoded_trajectories[
                    key
                ],
                masks[
                    key
                ],
            )

            visible_parts.append(
                visible
            )

            restore_indices[
                key
            ] = restore

            visible_counts[
                key
            ] = visible_count

        all_visible = torch.cat(
            visible_parts,
            dim=1,
        )

        # AUTO_MASK can legitimately produce an all-hidden
        # realization, for example selected mode=states at t=0.
        #
        # In that case there is nothing for the encoder to process.
        # The decoder will operate only from learned mask tokens.
        if all_visible.shape[1] == 0:
            encoded_all = all_visible
        else:
            encoded_all = self.encoder(
                all_visible
            )

        encoded_by_modality = {}

        offset = 0

        for key in self.modality_order:
            count = visible_counts[
                key
            ]

            encoded_by_modality[
                key
            ] = encoded_all[
                :,
                offset:
                offset + count,
                :,
            ]

            offset += count

        return (
            encoded_by_modality,
            restore_indices,
            visible_counts,
        )

    def restore_decoder_tokens(
        self,
        encoded_trajectories: Mapping[
            str,
            torch.Tensor,
        ],
        restore_indices: Mapping[
            str,
            torch.Tensor,
        ],
        visible_counts: Mapping[
            str,
            int,
        ],
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        """
        Reinsert learned mask tokens and restore original token
        positions for each modality.
        """

        restored = {}

        for key in self.modality_order:
            visible = (
                encoded_trajectories[
                    key
                ]
            )

            indices = (
                restore_indices[
                    key
                ]
            )

            batch_size = (
                visible.shape[0]
            )

            total_tokens = int(
                indices.numel()
            )

            visible_count = int(
                visible_counts[
                    key
                ]
            )

            hidden_count = (
                total_tokens
                - visible_count
            )

            mask_tokens = (
                self.mask_tokens[
                    key
                ].repeat(
                    batch_size,
                    hidden_count,
                    1,
                )
            )

            visible_then_hidden = (
                torch.cat(
                    [
                        visible,
                        mask_tokens,
                    ],
                    dim=1,
                )
            )

            gather_index = (
                indices[
                    None,
                    :,
                    None,
                ]
                .repeat(
                    batch_size,
                    1,
                    self.config.n_embd,
                )
            )

            restored[key] = (
                torch.gather(
                    visible_then_hidden,
                    dim=1,
                    index=gather_index,
                )
            )

        return restored

    def decoder_trajectory_encoding(
        self,
        restored: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        """
        Decoder-side modality embedding.
        """

        encoded = {}

        for key in self.modality_order:
            (
                tokens_per_timestep,
                _,
            ) = self.data_shapes[
                key
            ]

            latent = restored[
                key
            ]

            batch_size = (
                latent.shape[0]
            )

            total_tokens = (
                latent.shape[1]
            )

            if (
                total_tokens
                % tokens_per_timestep
                != 0
            ):
                raise RuntimeError(
                    "token count cannot be "
                    "reshaped into timesteps"
                )

            trajectory_length = (
                total_tokens
                // tokens_per_timestep
            )

            latent = latent.reshape(
                batch_size,
                trajectory_length,
                tokens_per_timestep,
                self.config.n_embd,
            )

            x = (
                self.decoder_embed[
                    key
                ](
                    latent
                )
                + self.decoder_per_dim_encoding[
                    key
                ]
                + self.position_embedding[
                    :,
                    :trajectory_length,
                    :,
                    :,
                ]
            )

            encoded[key] = (
                x.reshape(
                    batch_size,
                    total_tokens,
                    self.config.n_embd,
                )
            )

        return encoded

    def forward_decoder(
        self,
        encoded_trajectories: Mapping[
            str,
            torch.Tensor,
        ],
        restore_indices: Mapping[
            str,
            torch.Tensor,
        ],
        visible_counts: Mapping[
            str,
            int,
        ],
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        """
        Restore hidden positions, run the bidirectional decoder,
        then reconstruct each modality.
        """

        restored = (
            self.restore_decoder_tokens(
                encoded_trajectories,
                restore_indices,
                visible_counts,
            )
        )

        decoder_inputs = (
            self.decoder_trajectory_encoding(
                restored
            )
        )

        concatenated = torch.cat(
            [
                decoder_inputs[
                    key
                ]
                for key
                in self.modality_order
            ],
            dim=1,
        )

        decoded = self.decoder(
            concatenated
        )

        predictions = {}

        offset = 0

        for key in self.modality_order:
            token_count = (
                decoder_inputs[
                    key
                ].shape[1]
            )

            segment = decoded[
                :,
                offset:
                offset + token_count,
                :,
            ]

            offset += token_count

            (
                tokens_per_timestep,
                _,
            ) = self.data_shapes[
                key
            ]

            batch_size = (
                segment.shape[0]
            )

            trajectory_length = (
                token_count
                // tokens_per_timestep
            )

            segment = segment.reshape(
                batch_size,
                trajectory_length,
                tokens_per_timestep,
                self.config.n_embd,
            )

            predictions[key] = (
                self.output_heads[
                    key
                ](
                    segment
                )
            )

        return predictions

    def encode(
        self,
        trajectories: Mapping[
            str,
            torch.Tensor,
        ],
        masks: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        """
        Return encoder outputs for visible tokens only.
        """

        processed_masks = (
            self.process_masks(
                trajectories,
                masks,
            )
        )

        trajectory_embeddings = (
            self.trajectory_encoding(
                trajectories
            )
        )

        (
            encoded,
            _,
            _,
        ) = self.forward_encoder(
            trajectory_embeddings,
            processed_masks,
        )

        return encoded

    def forward(
        self,
        trajectories: Mapping[
            str,
            torch.Tensor,
        ],
        masks: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        processed_masks = (
            self.process_masks(
                trajectories,
                masks,
            )
        )

        trajectory_embeddings = (
            self.trajectory_encoding(
                trajectories
            )
        )

        (
            encoded,
            restore_indices,
            visible_counts,
        ) = self.forward_encoder(
            trajectory_embeddings,
            processed_masks,
        )

        return self.forward_decoder(
            encoded,
            restore_indices,
            visible_counts,
        )