import numpy as np
import pytest

from src.attacks.csdpc.clustering import (
    build_raw_decision_units,
    fit_kmeans_decision_units,
)


def test_build_raw_decision_units_shape_and_values():
    observations = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )

    actions = np.array(
        [
            [10.0],
            [20.0],
            [30.0],
        ],
        dtype=np.float32,
    )

    features = build_raw_decision_units(
        observations,
        actions,
    )

    expected = np.array(
        [
            [1.0, 2.0, 10.0],
            [3.0, 4.0, 20.0],
            [5.0, 6.0, 30.0],
        ],
        dtype=np.float32,
    )

    assert features.shape == (3, 3)
    np.testing.assert_array_equal(
        features,
        expected,
    )


def test_build_raw_decision_units_does_not_mutate_inputs():
    observations = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    actions = np.array(
        [
            [5.0],
            [6.0],
        ],
        dtype=np.float32,
    )

    observations_before = observations.copy()
    actions_before = actions.copy()

    features = build_raw_decision_units(
        observations,
        actions,
    )

    np.testing.assert_array_equal(
        observations,
        observations_before,
    )

    np.testing.assert_array_equal(
        actions,
        actions_before,
    )

    assert not np.shares_memory(
        features,
        observations,
    )

    assert not np.shares_memory(
        features,
        actions,
    )


def test_build_raw_decision_units_rejects_mismatched_lengths():
    observations = np.zeros(
        (3, 2),
        dtype=np.float32,
    )

    actions = np.zeros(
        (2, 1),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="same number of transitions",
    ):
        build_raw_decision_units(
            observations,
            actions,
        )


def test_build_raw_decision_units_rejects_nonfinite_values():
    observations = np.array(
        [
            [1.0, np.nan],
            [2.0, 3.0],
        ],
        dtype=np.float32,
    )

    actions = np.zeros(
        (2, 1),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="NaN or Inf",
    ):
        build_raw_decision_units(
            observations,
            actions,
        )


def test_kmeans_is_deterministic_for_same_seed():
    features = np.array(
        [
            [-5.0, -5.0],
            [-5.1, -4.9],
            [-4.9, -5.1],
            [5.0, 5.0],
            [5.1, 4.9],
            [4.9, 5.1],
        ],
        dtype=np.float64,
    )

    _, result_a = fit_kmeans_decision_units(
        features,
        num_clusters=2,
        seed=17,
    )

    _, result_b = fit_kmeans_decision_units(
        features,
        num_clusters=2,
        seed=17,
    )

    np.testing.assert_array_equal(
        result_a.labels,
        result_b.labels,
    )

    np.testing.assert_allclose(
        result_a.centers,
        result_b.centers,
    )

    assert result_a.inertia == pytest.approx(
        result_b.inertia
    )

    assert result_a.n_iter == result_b.n_iter


def test_kmeans_uses_requested_number_of_clusters():
    features = np.array(
        [
            [-10.0, -10.0],
            [-10.1, -9.9],
            [0.0, 0.0],
            [0.1, -0.1],
            [10.0, 10.0],
            [9.9, 10.1],
        ],
        dtype=np.float64,
    )

    model, result = fit_kmeans_decision_units(
        features,
        num_clusters=3,
        seed=0,
    )

    assert model.n_clusters == 3
    assert result.centers.shape == (3, 2)
    assert len(np.unique(result.labels)) == 3
    assert result.labels.shape == (6,)