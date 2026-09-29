from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from scripts.generate_state_matched_action_substitution import (
    build_transition_to_trajectory,
    choose_donors_for_batch,
    pool_transition_indices,
    select_targets,
    trajectory_returns,
)
from src.data.trajectories import TrajectorySlice


def test_target_selection_is_exact_unique_and_deterministic():
    pool = np.arange(
        100,
        dtype=np.int64,
    )

    first = select_targets(
        pool,
        requested_budget=25,
        attack_seed=30,
    )

    second = select_targets(
        pool,
        requested_budget=25,
        attack_seed=30,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )

    assert len(first) == 25
    assert len(
        np.unique(
            first
        )
    ) == 25


def test_transition_pool_uses_trajectory_return():
    trajectories = [
        TrajectorySlice(
            0,
            2,
        ),
        TrajectorySlice(
            2,
            5,
        ),
        TrajectorySlice(
            5,
            9,
        ),
    ]

    rewards = np.asarray(
        [
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ],
        dtype=np.float32,
    )

    returns = trajectory_returns(
        rewards,
        trajectories,
    )

    high = pool_transition_indices(
        trajectories,
        returns,
        threshold=10.0,
        mode="ge",
    )

    low = pool_transition_indices(
        trajectories,
        returns,
        threshold=3.0,
        mode="le",
    )

    np.testing.assert_array_equal(
        high,
        np.asarray(
            [5, 6, 7, 8],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        low,
        np.asarray(
            [0, 1],
            dtype=np.int64,
        ),
    )


def test_transition_to_trajectory_mapping():
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
        build_transition_to_trajectory(
            trajectories,
            5,
        )
    )

    np.testing.assert_array_equal(
        mapping,
        np.asarray(
            [
                0,
                0,
                1,
                1,
                1,
            ],
            dtype=np.int32,
        ),
    )


def test_donor_selection_prefers_max_action_distance_within_knn():
    normalized_states = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [0.4, 0.0],
        ],
        dtype=np.float64,
    )

    actions = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.9, 0.0],
            [0.4, 0.0],
        ],
        dtype=np.float32,
    )

    target_indices = np.asarray(
        [0],
        dtype=np.int64,
    )

    donor_indices = np.asarray(
        [
            1,
            2,
            3,
            4,
        ],
        dtype=np.int64,
    )

    tree = cKDTree(
        normalized_states[
            donor_indices
        ]
    )

    (
        donors,
        state_dist,
        action_disp,
        rank,
    ) = choose_donors_for_batch(
        target_indices=target_indices,
        normalized_states=(
            normalized_states
        ),
        clean_actions=actions,
        donor_indices=(
            donor_indices
        ),
        donor_tree=tree,
        k=3,
        p=2,
        eps=0.0,
        workers=1,
    )

    # K=3 means donor indices 1,2,3 are eligible.
    # Index 3 has the largest action displacement.
    assert int(
        donors[
            0
        ]
    ) == 3

    assert np.isclose(
        state_dist[
            0
        ],
        0.3,
    )

    assert np.isclose(
        action_disp[
            0
        ],
        0.9,
    )

    assert int(
        rank[
            0
        ]
    ) == 3


def test_donor_selection_tie_breaks_by_state_distance_then_index():
    normalized_states = np.asarray(
        [
            [0.0],
            [0.2],
            [0.1],
            [0.1],
        ],
        dtype=np.float64,
    )

    actions = np.asarray(
        [
            [0.0],
            [1.0],
            [1.0],
            [1.0],
        ],
        dtype=np.float32,
    )

    target_indices = np.asarray(
        [0],
        dtype=np.int64,
    )

    donor_indices = np.asarray(
        [1, 2, 3],
        dtype=np.int64,
    )

    tree = cKDTree(
        normalized_states[
            donor_indices
        ]
    )

    (
        donors,
        _,
        _,
        rank,
    ) = choose_donors_for_batch(
        target_indices=target_indices,
        normalized_states=(
            normalized_states
        ),
        clean_actions=actions,
        donor_indices=(
            donor_indices
        ),
        donor_tree=tree,
        k=3,
        p=2,
        eps=0.0,
        workers=1,
    )

    # All have equal action displacement.
    # Global 2 and 3 are closer than global 1;
    # global 2 then wins the exact-distance tie.
    assert int(
        donors[
            0
        ]
    ) == 2

    assert int(
        rank[
            0
        ]
    ) == 1
