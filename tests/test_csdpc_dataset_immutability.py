import numpy as np

from src.attacks.csdpc.attack import (
    apply_csdpc_attack,
    prepare_csdpc_attack,
)


def _dataset():
    rng = np.random.default_rng(123)

    n = 100

    observations = rng.normal(
        size=(n, 4)
    )

    actions = rng.uniform(
        -0.8,
        0.8,
        size=(n, 2),
    )

    rewards = rng.normal(
        size=n
    )

    terminals = np.zeros(
        n,
        dtype=bool,
    )

    timeouts = np.zeros(
        n,
        dtype=bool,
    )

    timeouts[49] = True
    timeouts[99] = True

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
    }


def _prepare(dataset):
    return prepare_csdpc_attack(
        dataset,
        attack_seed=0,
        num_clusters=4,
        sequence_length=5,
        eta=0.05,
        num_candidates=20,
    )


def test_attack_does_not_mutate_clean_dataset():
    dataset = _dataset()

    before = {
        key: value.copy()
        for key, value
        in dataset.items()
    }

    prepared = _prepare(
        dataset
    )

    apply_csdpc_attack(
        dataset,
        prepared,
        rho=0.10,
    )

    for key in dataset:
        np.testing.assert_array_equal(
            dataset[key],
            before[key],
        )


def test_nonattack_fields_are_identical():
    dataset = _dataset()

    prepared = _prepare(
        dataset
    )

    result = apply_csdpc_attack(
        dataset,
        prepared,
        rho=0.10,
    )

    for key in (
        "rewards",
        "terminals",
        "timeouts",
    ):
        np.testing.assert_array_equal(
            result.poisoned_dataset[key],
            dataset[key],
        )


def test_zero_rho_is_array_identical():
    dataset = _dataset()

    prepared = _prepare(
        dataset
    )

    result = apply_csdpc_attack(
        dataset,
        prepared,
        rho=0.0,
    )

    for key in dataset:
        np.testing.assert_array_equal(
            result.poisoned_dataset[key],
            dataset[key],
        )

    assert (
        result.modified_transition_indices
        == ()
    )

    assert (
        result.actual_transition_budget
        == 0
    )


def test_modified_indices_are_unique():
    dataset = _dataset()

    prepared = _prepare(
        dataset
    )

    result = apply_csdpc_attack(
        dataset,
        prepared,
        rho=0.10,
    )

    indices = (
        result.modified_transition_indices
    )

    assert len(indices) == len(
        set(indices)
    )


def test_budget_is_not_exceeded():
    dataset = _dataset()

    prepared = _prepare(
        dataset
    )

    result = apply_csdpc_attack(
        dataset,
        prepared,
        rho=0.10,
    )

    assert (
        result.actual_transition_budget
        <= result.requested_transition_budget
    )

    assert (
        len(
            result.modified_transition_indices
        )
        == result.actual_transition_budget
    )


def test_actions_remain_in_bounds():
    dataset = _dataset()

    prepared = _prepare(
        dataset
    )

    result = apply_csdpc_attack(
        dataset,
        prepared,
        rho=0.10,
    )

    assert np.all(
        result.poisoned_dataset[
            "actions"
        ] <= 1.0
    )

    assert np.all(
        result.poisoned_dataset[
            "actions"
        ] >= -1.0
    )


def test_same_seed_and_rho_are_deterministic():
    dataset = _dataset()

    prepared = _prepare(
        dataset
    )

    first = apply_csdpc_attack(
        dataset,
        prepared,
        rho=0.10,
    )

    second = apply_csdpc_attack(
        dataset,
        prepared,
        rho=0.10,
    )

    np.testing.assert_array_equal(
        first.poisoned_dataset[
            "observations"
        ],
        second.poisoned_dataset[
            "observations"
        ],
    )

    np.testing.assert_array_equal(
        first.poisoned_dataset[
            "actions"
        ],
        second.poisoned_dataset[
            "actions"
        ],
    )

    assert (
        first.modified_transition_indices
        == second.modified_transition_indices
    )


def test_low_budget_is_prefix_of_high_budget():
    dataset = _dataset()

    prepared = _prepare(
        dataset
    )

    low = apply_csdpc_attack(
        dataset,
        prepared,
        rho=0.10,
    )

    high = apply_csdpc_attack(
        dataset,
        prepared,
        rho=0.20,
    )

    low_indices = set(
        low.modified_transition_indices
    )

    high_indices = set(
        high.modified_transition_indices
    )

    assert low_indices.issubset(
        high_indices
    )

    np.testing.assert_array_equal(
        low.poisoned_dataset[
            "observations"
        ][
            list(
                low.modified_transition_indices
            )
        ],
        high.poisoned_dataset[
            "observations"
        ][
            list(
                low.modified_transition_indices
            )
        ],
    )

    np.testing.assert_array_equal(
        low.poisoned_dataset[
            "actions"
        ][
            list(
                low.modified_transition_indices
            )
        ],
        high.poisoned_dataset[
            "actions"
        ][
            list(
                low.modified_transition_indices
            )
        ],
    )