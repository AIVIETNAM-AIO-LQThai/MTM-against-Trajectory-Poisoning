from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_LAMBDA_CANDIDATES = (
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
)


@dataclass(frozen=True)
class LambdaCalibrationResult:
    recommended_lambda: float
    median_raw_gradient_ratio: float
    median_cosine_similarity: float
    target_max_scaled_ratio: float
    candidate_scaled_ratios: dict[float, float]


def choose_lambda_from_gradient_ratios(
    raw_gradient_ratios: Iterable[float],
    cosine_similarities: Iterable[float],
    *,
    candidates: Iterable[float] = DEFAULT_LAMBDA_CANDIDATES,
    target_max_scaled_ratio: float = 0.25,
) -> LambdaCalibrationResult:
    """
    Freeze lambda from clean shared-gradient scale only.

    We choose the largest predeclared candidate satisfying

        lambda * median(||g_MTM|| / ||g_DT||) <= target_max_scaled_ratio.

    This deliberately makes the auxiliary gradient non-dominant at the
    shared state/action embeddings. Poisoned performance is never consulted.
    """

    if not 0.0 < target_max_scaled_ratio:
        raise ValueError("target_max_scaled_ratio must be positive")

    ratios = np.asarray(list(raw_gradient_ratios), dtype=np.float64)
    cosines = np.asarray(list(cosine_similarities), dtype=np.float64)

    ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    cosines = cosines[np.isfinite(cosines)]

    if ratios.size == 0:
        raise ValueError("no finite positive gradient ratios were supplied")

    candidate_values = sorted({float(value) for value in candidates})
    if not candidate_values or candidate_values[0] <= 0.0:
        raise ValueError("all lambda candidates must be positive")

    median_ratio = float(np.median(ratios))
    median_cosine = (
        float(np.median(cosines))
        if cosines.size
        else float("nan")
    )

    scaled = {
        value: float(value * median_ratio)
        for value in candidate_values
    }

    admissible = [
        value
        for value in candidate_values
        if scaled[value] <= target_max_scaled_ratio
    ]

    if not admissible:
        raise RuntimeError(
            "No predeclared lambda candidate satisfies the clean gradient "
            "cap. Do not extend the grid after observing poisoned results; "
            "inspect the clean integration first."
        )

    recommended = max(admissible)

    return LambdaCalibrationResult(
        recommended_lambda=float(recommended),
        median_raw_gradient_ratio=median_ratio,
        median_cosine_similarity=median_cosine,
        target_max_scaled_ratio=float(target_max_scaled_ratio),
        candidate_scaled_ratios=scaled,
    )
