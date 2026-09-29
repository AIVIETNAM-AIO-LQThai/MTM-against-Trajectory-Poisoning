import numpy as np

from scripts.preflight_state_matched_action_substitution_qualification import (
    transition_to_trajectory,
)
from src.data.trajectories import TrajectorySlice


def test_transition_to_trajectory():
    trajectories = [
        TrajectorySlice(
            0,
            2,
        ),
        TrajectorySlice(
            2,
            5,
        ),
    ]

    mapping = (
        transition_to_trajectory(
            trajectories,
            5,
        )
    )

    np.testing.assert_array_equal(
        mapping,
        np.asarray(
            [0, 0, 1, 1, 1],
            dtype=np.int32,
        ),
    )
