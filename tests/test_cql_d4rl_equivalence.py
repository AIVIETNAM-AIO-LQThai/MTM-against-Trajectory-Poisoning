from __future__ import annotations

import numpy as np
import d4rl

from src.baselines.cql.dataset_adapter import (
    build_cql_training_view,
)


class FakeEnv:
    _max_episode_steps = 1000

    def __init__(self, dataset):
        self._dataset = dataset

    def get_dataset(self, **kwargs):
        return self._dataset


def test_project_adapter_matches_d4rl_qlearning_dataset():
    n = 8

    observations = np.arange(
        n * 3,
        dtype=np.float64,
    ).reshape(n, 3)

    actions = np.arange(
        n * 2,
        dtype=np.float64,
    ).reshape(n, 2)

    rewards = np.arange(
        n,
        dtype=np.float64,
    )

    terminals = np.zeros(
        n,
        dtype=bool,
    )
    terminals[5] = True

    timeouts = np.zeros(
        n,
        dtype=bool,
    )
    timeouts[2] = True

    raw = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
    }

    env = FakeEnv(raw)

    expected = d4rl.qlearning_dataset(
        env,
        dataset=raw,
        terminate_on_end=False,
    )

    actual = build_cql_training_view(raw)

    for key in (
        "observations",
        "actions",
        "next_observations",
        "rewards",
        "terminals",
    ):
        np.testing.assert_array_equal(
            actual[key],
            expected[key],
        )

        assert (
            actual[key].dtype
            == expected[key].dtype
        )

    np.testing.assert_array_equal(
        actual["raw_indices"],
        np.array(
            [0, 1, 3, 4, 5, 6],
            dtype=np.int64,
        ),
    )