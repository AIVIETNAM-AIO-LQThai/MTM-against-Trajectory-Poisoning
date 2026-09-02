import numpy as np

from src.attacks.csdpc.attack import (
    prepare_csdpc_attack,
)


def _dataset():
    observations = np.array(
        [
            [-3.0, 0.0],
            [-2.5, 0.1],
            [-2.0, 0.2],
            [-1.5, 0.3],
            [1.5, 0.4],
            [2.0, 0.5],
            [2.5, 0.6],
            [3.0, 0.7],
        ],
        dtype=np.float64,
    )

    actions = np.array(
        [
            [-0.2],
            [-0.2],
            [-0.1],
            [-0.1],
            [0.1],
            [0.1],
            [0.2],
            [0.2],
        ],
        dtype=np.float64,
    )

    terminals = np.array(
        [
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            True,
        ]
    )

    timeouts = np.zeros(
        8,
        dtype=bool,
    )

    rewards = np.arange(
        8,
        dtype=np.float64,
    )

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
    }


def test_prepare_does_not_mutate_dataset():
    dataset = _dataset()

    before = {
        key: value.copy()
        for key, value
        in dataset.items()
    }

    prepare_csdpc_attack(
        dataset,
        attack_seed=0,
        num_clusters=2,
        sequence_length=3,
        eta=0.05,
        num_candidates=10,
    )

    for key in dataset:
        np.testing.assert_array_equal(
            dataset[key],
            before[key],
        )


def test_prepare_has_correct_transition_count():
    prepared = prepare_csdpc_attack(
        _dataset(),
        attack_seed=0,
        num_clusters=2,
        sequence_length=3,
        eta=0.05,
        num_candidates=10,
    )

    assert (
        prepared.num_transitions
        == 8
    )


def test_windows_do_not_cross_trajectories():
    prepared = prepare_csdpc_attack(
        _dataset(),
        attack_seed=0,
        num_clusters=2,
        sequence_length=3,
        eta=0.05,
        num_candidates=10,
    )

    for window in prepared.windows:
        assert (
            window.global_start < 4
            and window.global_end <= 4
        ) or (
            window.global_start >= 4
            and window.global_end <= 8
        )


def test_same_seed_has_same_discrete_preparation():
    first = prepare_csdpc_attack(
        _dataset(),
        attack_seed=7,
        num_clusters=2,
        sequence_length=3,
        eta=0.05,
        num_candidates=10,
    )

    second = prepare_csdpc_attack(
        _dataset(),
        attack_seed=7,
        num_clusters=2,
        sequence_length=3,
        eta=0.05,
        num_candidates=10,
    )

    np.testing.assert_array_equal(
        first.clustering.labels,
        second.clustering.labels,
    )

    assert (
        first.pattern_frequencies
        == second.pattern_frequencies
    )

    assert (
        first.windows
        == second.windows
    )