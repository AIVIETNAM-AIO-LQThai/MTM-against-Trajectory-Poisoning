import numpy as np

from src.evaluation.scoring import (
    WALKER2D_EXPERT_SCORE,
    WALKER2D_RANDOM_SCORE,
    walker2d_normalized_score,
)


def test_random_score_maps_to_zero():
    score = walker2d_normalized_score(
        WALKER2D_RANDOM_SCORE
    )

    np.testing.assert_allclose(
        score,
        0.0,
        atol=1e-10,
    )


def test_expert_score_maps_to_100():
    score = walker2d_normalized_score(
        WALKER2D_EXPERT_SCORE
    )

    np.testing.assert_allclose(
        score,
        100.0,
        atol=1e-10,
    )


def test_normalization_is_not_clamped():
    score = walker2d_normalized_score(
        WALKER2D_EXPERT_SCORE + 1000.0
    )

    assert score > 100.0