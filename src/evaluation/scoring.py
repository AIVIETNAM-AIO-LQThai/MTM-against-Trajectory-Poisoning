WALKER2D_RANDOM_SCORE = 1.629008
WALKER2D_EXPERT_SCORE = 4592.3


def walker2d_normalized_score(
    raw_return: float,
) -> float:
    """
    Convert raw Walker2d return to D4RL normalized score.
    """

    return 100.0 * (
        raw_return
        - WALKER2D_RANDOM_SCORE
    ) / (
        WALKER2D_EXPERT_SCORE
        - WALKER2D_RANDOM_SCORE
    )