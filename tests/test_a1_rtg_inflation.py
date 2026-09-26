import numpy as np

from scripts.generate_a1_rtg_inflation import (
    apply_reward_inflation,
    select_trajectories,
    trajectory_returns,
    validate_artifact,
)
from src.data.trajectories import (
    TrajectorySlice,
)


def synthetic_dataset():
    lengths = [4, 5, 6, 7, 8, 9]

    starts = np.cumsum(
        [0] + lengths[:-1]
    )

    trajectories = [
        TrajectorySlice(
            start=int(start),
            end=int(start + length),
        )
        for start, length
        in zip(
            starts,
            lengths,
        )
    ]

    n = sum(lengths)

    rewards = np.concatenate(
        [
            np.full(
                length,
                float(index + 1),
                dtype=np.float32,
            )
            for index, length
            in enumerate(lengths)
        ]
    )

    clean = {
        "observations": np.arange(
            n * 2,
            dtype=np.float32,
        ).reshape(n, 2),
        "actions": np.zeros(
            (n, 1),
            dtype=np.float32,
        ),
        "rewards": rewards,
        "terminals": np.zeros(
            n,
            dtype=bool,
        ),
        "timeouts": np.zeros(
            n,
            dtype=bool,
        ),
    }

    return (
        clean,
        trajectories,
    )


def test_trajectory_returns():
    clean, trajectories = (
        synthetic_dataset()
    )

    values = trajectory_returns(
        clean["rewards"],
        trajectories,
    )

    expected = np.asarray(
        [
            4.0,
            10.0,
            18.0,
            28.0,
            40.0,
            54.0,
        ]
    )

    np.testing.assert_allclose(
        values,
        expected,
    )


def test_selection_is_deterministic():
    clean, trajectories = (
        synthetic_dataset()
    )

    returns = trajectory_returns(
        clean["rewards"],
        trajectories,
    )

    first = select_trajectories(
        trajectories,
        returns,
        candidate_threshold=28.0,
        requested_budget=15,
        attack_seed=10,
    )

    second = select_trajectories(
        trajectories,
        returns,
        candidate_threshold=28.0,
        requested_budget=15,
        attack_seed=10,
    )

    assert first == second


def test_selection_never_exceeds_budget():
    clean, trajectories = (
        synthetic_dataset()
    )

    returns = trajectory_returns(
        clean["rewards"],
        trajectories,
    )

    selected, used, _ = (
        select_trajectories(
            trajectories,
            returns,
            candidate_threshold=28.0,
            requested_budget=15,
            attack_seed=11,
        )
    )

    assert used <= 15

    assert all(
        returns[index] <= 28.0
        for index in selected
    )


def test_reward_only_transform():
    clean, trajectories = (
        synthetic_dataset()
    )

    poisoned, _ = (
        apply_reward_inflation(
            clean,
            trajectories,
            [0, 1],
            target_return=30.0,
        )
    )

    for key in (
        "observations",
        "actions",
        "terminals",
        "timeouts",
    ):
        np.testing.assert_array_equal(
            poisoned[key],
            clean[key],
        )

    assert not np.array_equal(
        poisoned["rewards"],
        clean["rewards"],
    )


def test_selected_returns_hit_target():
    clean, trajectories = (
        synthetic_dataset()
    )

    poisoned, _ = (
        apply_reward_inflation(
            clean,
            trajectories,
            [0, 1],
            target_return=30.0,
        )
    )

    returns = trajectory_returns(
        poisoned["rewards"],
        trajectories,
    )

    np.testing.assert_allclose(
        returns[[0, 1]],
        [30.0, 30.0],
        atol=1e-5,
        rtol=0.0,
    )


def test_validator_accepts_valid_artifact():
    clean, trajectories = (
        synthetic_dataset()
    )

    selected = [0, 1]

    poisoned, _ = (
        apply_reward_inflation(
            clean,
            trajectories,
            selected,
            target_return=30.0,
        )
    )

    result = validate_artifact(
        clean,
        poisoned,
        trajectories,
        selected,
        used_transitions=sum(
            t.length
            for t in trajectories
        ),
        requested_budget=10,
        actual_budget=9,
        minimum_budget_utilization=0.8,
        candidate_threshold=10.0,
        target_return=30.0,
    )

    assert (
        result[
            "non_reward_arrays_identical"
        ]
        is True
    )

    assert (
        result[
            "actual_transition_budget"
        ]
        == 9
    )
