from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from src.methods.dt_mtm.model import DTMTMModel


@dataclass(frozen=True)
class SharedGradientDiagnostics:
    """Gradient interaction on the parameters shared by DT and MTM."""

    dt_norm: float
    mtm_norm_unscaled: float
    mtm_norm_scaled: float
    cosine_similarity: float


def _gradient_vector(
    loss: torch.Tensor,
    parameters: Iterable[nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    params = tuple(parameters)

    if not params:
        raise ValueError("No shared parameters were supplied")

    gradients = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )

    flattened: list[torch.Tensor] = []
    for parameter, gradient in zip(params, gradients):
        if gradient is None:
            flattened.append(torch.zeros_like(parameter).reshape(-1))
        else:
            flattened.append(gradient.detach().reshape(-1))

    return torch.cat(flattened)


def shared_gradient_diagnostics(
    model: DTMTMModel,
    dt_loss: torch.Tensor,
    mtm_loss: torch.Tensor,
    *,
    lambda_mtm: float,
) -> SharedGradientDiagnostics:
    """
    Measure DT/MTM interaction without mutating ``parameter.grad``.

    Cosine similarity is measured between the unscaled DT and MTM objective
    gradients. The scaled MTM norm is also reported because that is the
    magnitude that enters the actual joint update.
    """

    if lambda_mtm < 0.0:
        raise ValueError("lambda_mtm must be non-negative")

    params = tuple(
        parameter
        for _, parameter in model.shared_named_parameters()
        if parameter.requires_grad
    )

    dt_vector = _gradient_vector(
        dt_loss,
        params,
        retain_graph=True,
    )
    mtm_vector = _gradient_vector(
        mtm_loss,
        params,
        retain_graph=True,
    )

    dt_norm_tensor = torch.linalg.vector_norm(dt_vector)
    mtm_norm_tensor = torch.linalg.vector_norm(mtm_vector)

    denominator = dt_norm_tensor * mtm_norm_tensor
    if float(denominator.detach().cpu()) == 0.0:
        cosine = float("nan")
    else:
        cosine = float(
            torch.dot(dt_vector, mtm_vector).div(denominator).detach().cpu()
        )

    mtm_norm = float(mtm_norm_tensor.detach().cpu())

    return SharedGradientDiagnostics(
        dt_norm=float(dt_norm_tensor.detach().cpu()),
        mtm_norm_unscaled=mtm_norm,
        mtm_norm_scaled=mtm_norm * float(lambda_mtm),
        cosine_similarity=cosine,
    )
