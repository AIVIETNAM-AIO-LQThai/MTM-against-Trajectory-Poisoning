from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

import torch
from torch import nn

from src.methods.dt.model import DecisionTransformer
from src.methods.mtm.model import ReferenceMTM


@dataclass(frozen=True)
class MTMBridgeAudit:
    """Diagnostics for the DT->MTM shared input bridges."""

    state_rank: int
    state_input_dim: int
    state_max_abs_error: float
    action_rank: int
    action_input_dim: int
    action_max_abs_error: float


class DTMTMModel(nn.Module):
    """
    Joint Decision Transformer + Masked Trajectory Model.

    The two objectives intentionally keep separate Transformer passes:

    * ``forward_dt`` is exactly the existing causal Decision Transformer.
    * ``forward_mtm`` is exactly the existing bidirectional MTM encoder/
      decoder after a shared state/action input representation.

    Shared parameters are limited to ``dt.embed_state`` and
    ``dt.embed_action``. MTM returns are intentionally NOT shared with DT
    return-to-go because the two repositories use different target
    semantics.

    To preserve the validated standalone MTM initialization, the joint MTM
    state/action path is initialized as

        bridge(DT_embedding(x)) ~= original_MTM_embedding(x)

    using a pseudoinverse construction. When the DT input projection has
    full column rank, this equality holds up to floating-point error for all
    inputs. The original MTM state/action input linears are retained as
    frozen audit references and are not used by ``forward_mtm``.
    """

    REQUIRED_MODALITIES = ("states", "actions", "returns")

    def __init__(
        self,
        dt: DecisionTransformer,
        mtm: ReferenceMTM,
    ) -> None:
        super().__init__()

        self.dt = dt
        self.mtm = mtm

        if tuple(self.mtm.modality_order) != self.REQUIRED_MODALITIES:
            raise ValueError(
                "Joint DT+MTM currently requires MTM modalities in exact "
                "order ('states', 'actions', 'returns'). Got "
                f"{self.mtm.modality_order}."
            )

        state_tokens, state_dim = self.mtm.data_shapes["states"]
        action_tokens, action_dim = self.mtm.data_shapes["actions"]
        return_tokens, _ = self.mtm.data_shapes["returns"]

        if state_tokens != 1 or action_tokens != 1 or return_tokens != 1:
            raise ValueError(
                "Joint DT+MTM currently supports exactly one continuous "
                "token per timestep for states/actions/returns."
            )

        if state_dim != self.dt.state_dim:
            raise ValueError(
                "DT and MTM state dimensions differ: "
                f"{self.dt.state_dim} vs {state_dim}."
            )

        if action_dim != self.dt.action_dim:
            raise ValueError(
                "DT and MTM action dimensions differ: "
                f"{self.dt.action_dim} vs {action_dim}."
            )

        self.state_bridge = nn.Linear(
            self.dt.hidden_size,
            self.mtm.config.n_embd,
        )
        self.action_bridge = nn.Linear(
            self.dt.hidden_size,
            self.mtm.config.n_embd,
        )

        self._state_rank = self._fit_bridge(
            source=self.dt.embed_state,
            target=self.mtm.encoder_embed["states"],
            bridge=self.state_bridge,
            name="states",
        )
        self._action_rank = self._fit_bridge(
            source=self.dt.embed_action,
            target=self.mtm.encoder_embed["actions"],
            bridge=self.action_bridge,
            name="actions",
        )

        # These modules remain in the object solely as exact audit references
        # to the validated standalone MTM initialization. The joint path uses
        # DT embedding -> bridge instead.
        for key in ("states", "actions"):
            for parameter in self.mtm.encoder_embed[key].parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _fit_bridge(
        *,
        source: nn.Linear,
        target: nn.Linear,
        bridge: nn.Linear,
        name: str,
    ) -> int:
        """Initialize ``bridge(source(x))`` to reproduce ``target(x)``."""

        if source.in_features != target.in_features:
            raise ValueError(
                f"{name}: source/target input dimensions differ: "
                f"{source.in_features} vs {target.in_features}."
            )

        if bridge.in_features != source.out_features:
            raise ValueError(
                f"{name}: bridge input dimension does not match DT "
                "embedding output."
            )

        if bridge.out_features != target.out_features:
            raise ValueError(
                f"{name}: bridge output dimension does not match MTM "
                "embedding output."
            )

        # source.weight: [H_dt, D]
        # pinv(source.weight): [D, H_dt]
        # target.weight @ pinv(source.weight): [H_mtm, H_dt]
        source_weight_f64 = source.weight.detach().to(torch.float64)
        target_weight_f64 = target.weight.detach().to(torch.float64)

        rank = int(torch.linalg.matrix_rank(source_weight_f64).item())
        required_rank = source.in_features

        if rank != required_rank:
            raise RuntimeError(
                f"{name}: DT embedding matrix rank {rank} is below input "
                f"dimension {required_rank}; exact bridge initialization "
                "cannot be guaranteed."
            )

        bridge_weight = target_weight_f64 @ torch.linalg.pinv(source_weight_f64)

        if source.bias is None:
            source_bias_f64 = torch.zeros(
                source.out_features,
                dtype=torch.float64,
                device=source_weight_f64.device,
            )
        else:
            source_bias_f64 = source.bias.detach().to(torch.float64)

        if target.bias is None:
            target_bias_f64 = torch.zeros(
                target.out_features,
                dtype=torch.float64,
                device=target_weight_f64.device,
            )
        else:
            target_bias_f64 = target.bias.detach().to(torch.float64)

        bridge_bias = target_bias_f64 - bridge_weight @ source_bias_f64

        with torch.no_grad():
            bridge.weight.copy_(
                bridge_weight.to(
                    dtype=bridge.weight.dtype,
                    device=bridge.weight.device,
                )
            )
            bridge.bias.copy_(
                bridge_bias.to(
                    dtype=bridge.bias.dtype,
                    device=bridge.bias.device,
                )
            )

        return rank

    def forward_dt(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the unchanged strictly-causal Decision Transformer path."""

        return self.dt(
            states,
            actions,
            returns_to_go,
            timesteps,
            attention_mask,
        )

    def _joint_mtm_trajectory_encoding(
        self,
        trajectories: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Build MTM encoder inputs while sharing DT state/action embeddings.

        Input modality tensors keep the existing MTM shape ``[B,T,P,D]``.
        """

        self.mtm._validate_trajectories(trajectories)

        encoded: dict[str, torch.Tensor] = {}

        for key in self.mtm.modality_order:
            trajectory = trajectories[key].to(torch.float32)
            batch_size, trajectory_length, token_count, _ = trajectory.shape

            if key == "states":
                x = self.state_bridge(self.dt.embed_state(trajectory))
            elif key == "actions":
                x = self.action_bridge(self.dt.embed_action(trajectory))
            else:
                # MTM return target is distinct from DT return-to-go, so this
                # path deliberately stays MTM-specific.
                x = self.mtm.encoder_embed[key](trajectory)

            x = (
                x
                + self.mtm.encoder_per_dim_encoding[key]
                + self.mtm.position_embedding[
                    :,
                    :trajectory_length,
                    :,
                    :,
                ]
            )

            encoded[key] = x.reshape(
                batch_size,
                trajectory_length * token_count,
                self.mtm.config.n_embd,
            )

        return encoded

    def forward_mtm(
        self,
        trajectories: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run the bidirectional MTM reconstruction path."""

        processed_masks = self.mtm.process_masks(trajectories, masks)
        trajectory_embeddings = self._joint_mtm_trajectory_encoding(
            trajectories
        )

        encoded, restore_indices, visible_counts = self.mtm.forward_encoder(
            trajectory_embeddings,
            processed_masks,
        )

        return self.mtm.forward_decoder(
            encoded,
            restore_indices,
            visible_counts,
        )

    def audit_initial_bridge_equivalence(
        self,
        trajectories: Mapping[str, torch.Tensor],
    ) -> MTMBridgeAudit:
        """Compare bridge projections with original standalone MTM linears."""

        self.mtm._validate_trajectories(trajectories)

        with torch.no_grad():
            state_input = trajectories["states"].to(torch.float32)
            action_input = trajectories["actions"].to(torch.float32)

            state_reference = self.mtm.encoder_embed["states"](state_input)
            state_joint = self.state_bridge(self.dt.embed_state(state_input))

            action_reference = self.mtm.encoder_embed["actions"](action_input)
            action_joint = self.action_bridge(self.dt.embed_action(action_input))

            state_error = float(
                (state_reference - state_joint).abs().max().detach().cpu()
            )
            action_error = float(
                (action_reference - action_joint).abs().max().detach().cpu()
            )

        return MTMBridgeAudit(
            state_rank=self._state_rank,
            state_input_dim=self.dt.embed_state.in_features,
            state_max_abs_error=state_error,
            action_rank=self._action_rank,
            action_input_dim=self.dt.embed_action.in_features,
            action_max_abs_error=action_error,
        )

    def shared_named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield exactly the parameters optimized by both DT and MTM losses."""

        for name, parameter in self.dt.embed_state.named_parameters():
            yield f"dt.embed_state.{name}", parameter

        for name, parameter in self.dt.embed_action.named_parameters():
            yield f"dt.embed_action.{name}", parameter
