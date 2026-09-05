from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.data.mtm_statistics import (
    DataStatistics,
)


class ContinuousTokenizer(nn.Module):
    """
    Reference-style MTM continuous tokenizer.

    Input:
        [B, T, D]

    Encoded:
        [B, T, 1, D]

    Official normalization:
        (x - mean) / std

    and any feature with std < 0.1 uses std = 1.
    """

    def __init__(
        self,
        data_mean: np.ndarray,
        data_std: np.ndarray,
        *,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        data_mean = np.asarray(
            data_mean,
            dtype=np.float32,
        )

        data_std = np.asarray(
            data_std,
            dtype=np.float32,
        ).copy()

        if data_mean.ndim != 1:
            raise ValueError(
                "data_mean must be 1D"
            )

        if data_std.ndim != 1:
            raise ValueError(
                "data_std must be 1D"
            )

        if data_mean.shape != data_std.shape:
            raise ValueError(
                "data_mean and data_std must "
                "have identical shape"
            )

        if not np.isfinite(
            data_mean
        ).all():
            raise ValueError(
                "data_mean contains non-finite values"
            )

        if not np.isfinite(
            data_std
        ).all():
            raise ValueError(
                "data_std contains non-finite values"
            )

        # Exact official threshold:
        #
        # std < 0.1 -> 1
        #
        # Note this is strictly "<", not "<=".
        data_std[
            data_std < 0.1
        ] = 1.0

        self._data_mean = nn.Parameter(
            torch.tensor(
                data_mean,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        self._data_std = nn.Parameter(
            torch.tensor(
                data_std,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        self.normalize = bool(
            normalize
        )

    @classmethod
    def from_statistics(
        cls,
        stats: DataStatistics,
        *,
        normalize: bool = True,
    ) -> "ContinuousTokenizer":
        return cls(
            stats.mean,
            stats.std,
            normalize=normalize,
        )

    @property
    def feature_dim(self) -> int:
        return int(
            self._data_mean.numel()
        )

    @property
    def data_mean(self) -> torch.Tensor:
        return self._data_mean

    @property
    def data_std(self) -> torch.Tensor:
        return self._data_std

    def encode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        if trajectory.ndim != 3:
            raise ValueError(
                "trajectory must have shape "
                "[B, T, D]"
            )

        if trajectory.shape[-1] != (
            self.feature_dim
        ):
            raise ValueError(
                "trajectory feature dimension "
                "does not match tokenizer"
            )

        trajectory = trajectory.to(
            dtype=torch.float32
        )

        if self.normalize:
            mean = self._data_mean.to(
                trajectory.device
            )

            std = self._data_std.to(
                trajectory.device
            )

            trajectory = (
                trajectory - mean
            ) / std

        return trajectory.unsqueeze(
            2
        )

    def decode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        if trajectory.ndim != 4:
            raise ValueError(
                "encoded trajectory must have "
                "shape [B, T, 1, D]"
            )

        if trajectory.shape[2] != 1:
            raise ValueError(
                "continuous tokenizer expects "
                "exactly one token per modality timestep"
            )

        if trajectory.shape[-1] != (
            self.feature_dim
        ):
            raise ValueError(
                "encoded feature dimension does "
                "not match tokenizer"
            )

        trajectory = trajectory.to(
            dtype=torch.float32
        )

        trajectory = trajectory.squeeze(
            2
        )

        if self.normalize:
            mean = self._data_mean.to(
                trajectory.device
            )

            std = self._data_std.to(
                trajectory.device
            )

            trajectory = (
                trajectory * std
                + mean
            )

        return trajectory