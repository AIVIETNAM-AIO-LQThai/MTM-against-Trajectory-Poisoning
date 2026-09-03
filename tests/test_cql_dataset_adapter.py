import numpy as np

from src.baselines.cql.dataset_adapter import (
    audit_poison_exposure,
    build_cql_training_view,
)


def _dataset():
    n = 8

    observations = np.arange(
        n * 2,
        dtype=np.float64,
    ).reshape(n, 2)

    actions = np.arange(
        n,
        dtype=np.float64,
    ).reshape(n, 1)

    rewards = np.arange(
        n,
        dtype=np.float64,
    )

    terminals = np.zeros(
        n,
        dtype=bool,
    )

    timeouts = np.zeros(
        n,
        dtype=bool,
    )

    # Timeout transition must be dropped.
    timeouts[2] = True

    # True terminal is retained.
    terminals[5] = True

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
    }


def test_default_view_drops_timeout_transition():
    dataset = _dataset()

    view = build_cql_training_view(
        dataset
    )

    np.testing.assert_array_equal(
        view["raw_indices"],
        np.array(
            [0, 1, 3, 4, 5, 6],
            dtype=np.int64,
        ),
    )


def test_last_raw_transition_is_not_used_as_current():
    view = build_cql_training_view(
        _dataset()
    )

    assert 7 not in set(
        view["raw_indices"]
    )


def test_true_terminal_is_retained():
    view = build_cql_training_view(
        _dataset()
    )

    position = np.where(
        view["raw_indices"] == 5
    )[0][0]

    assert bool(
        view["terminals"][
            position
        ]
    )


def test_next_observation_uses_next_raw_index():
    dataset = _dataset()

    view = build_cql_training_view(
        dataset
    )

    for position, raw_index in enumerate(
        view["raw_indices"]
    ):
        np.testing.assert_array_equal(
            view["next_observations"][
                position
            ],
            dataset["observations"][
                raw_index + 1
            ].astype(
                np.float32
            ),
        )


def test_cql_numeric_arrays_are_float32():
    view = build_cql_training_view(
        _dataset()
    )

    assert (
        view["observations"].dtype
        == np.float32
    )

    assert (
        view["actions"].dtype
        == np.float32
    )

    assert (
        view["next_observations"].dtype
        == np.float32
    )

    assert (
        view["rewards"].dtype
        == np.float32
    )


def test_poison_exposure_distinguishes_current_and_next():
    view = build_cql_training_view(
        _dataset()
    )

    # Raw index 2 is dropped as a current CQL transition,
    # but observation[2] is still next_obs for raw index 1.
    audit = audit_poison_exposure(
        view["raw_indices"],
        np.array(
            [2],
            dtype=np.int64,
        ),
    )

    assert (
        audit[
            "poisoned_action_training_count"
        ]
        == 0
    )

    assert (
        audit[
            "poisoned_current_state_count"
        ]
        == 0
    )

    assert (
        audit[
            "poisoned_next_state_count"
        ]
        == 1
    )

    assert (
        audit[
            "poison_exposed_training_tuple_count"
        ]
        == 1
    )