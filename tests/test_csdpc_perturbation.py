import numpy as np

from src.attacks.csdpc.perturbation import perturb_selected_window
from src.attacks.csdpc.types import SelectedWindow
from src.attacks.csdpc.clustering import build_raw_decision_units, fit_kmeans_decision_units


class SignKMeans:
    """
    Tiny deterministic stand-in for sklearn KMeans.

    Cluster 0: first feature < 0
    Cluster 1: first feature >= 0
    """

    def predict(self, features):
        features = np.asarray(features)

        return (
            features[:, 0] >= 0.0
        ).astype(np.int64)


def _selected_window():
    return SelectedWindow(
        trajectory_id=0,
        global_start=0,
        global_end=3,
        source_pattern=(0,),
    )


def test_perturbation_does_not_mutate_clean_inputs():
    observations = np.array(
        [
            [-2.0, 1.0],
            [-2.0, 1.0],
            [-2.0, 1.0],
        ],
        dtype=np.float64,
    )

    actions = np.array(
        [
            [0.2],
            [0.2],
            [0.2],
        ],
        dtype=np.float64,
    )

    observations_before = (
        observations.copy()
    )

    actions_before = (
        actions.copy()
    )

    perturb_selected_window(
        observations,
        actions,
        _selected_window(),
        kmeans_model=SignKMeans(),
        clean_pattern_frequencies={
            (0,): 10,
            (1,): 100,
        },
        eta=0.05,
        num_candidates=10,
        rng=np.random.default_rng(0),
    )

    np.testing.assert_array_equal(
        observations,
        observations_before,
    )

    np.testing.assert_array_equal(
        actions,
        actions_before,
    )


def test_state_perturbation_respects_relative_linf_bound():
    observations = np.array(
        [
            [-2.0, 1.0],
            [-4.0, 2.0],
            [-6.0, 3.0],
        ],
        dtype=np.float64,
    )

    actions = np.array(
        [
            [0.2],
            [0.3],
            [0.4],
        ],
        dtype=np.float64,
    )

    eta = 0.05

    result = perturb_selected_window(
        observations,
        actions,
        _selected_window(),
        kmeans_model=SignKMeans(),
        clean_pattern_frequencies={
            (0,): 10,
            (1,): 100,
        },
        eta=eta,
        num_candidates=20,
        rng=np.random.default_rng(1),
    )

    allowed = (
        eta
        * np.max(
            np.abs(observations),
            axis=1,
        )
    )

    actual = np.max(
        np.abs(
            result.state_deltas
        ),
        axis=1,
    )

    assert np.all(
        actual
        <= allowed + 1e-12
    )


def test_action_perturbation_respects_relative_linf_bound():
    observations = np.array(
        [
            [-2.0, 1.0],
            [-2.0, 1.0],
            [-2.0, 1.0],
        ],
        dtype=np.float64,
    )

    actions = np.array(
        [
            [0.2],
            [0.4],
            [0.8],
        ],
        dtype=np.float64,
    )

    eta = 0.05

    result = perturb_selected_window(
        observations,
        actions,
        _selected_window(),
        kmeans_model=SignKMeans(),
        clean_pattern_frequencies={
            (0,): 10,
            (1,): 100,
        },
        eta=eta,
        num_candidates=20,
        rng=np.random.default_rng(2),
    )

    allowed = (
        eta
        * np.max(
            np.abs(actions),
            axis=1,
        )
    )

    actual = np.max(
        np.abs(
            result.action_deltas
        ),
        axis=1,
    )

    assert np.all(
        actual
        <= allowed + 1e-12
    )


def test_actions_are_clipped_to_environment_bounds():
    observations = np.array(
        [
            [-2.0, 1.0],
            [-2.0, 1.0],
            [-2.0, 1.0],
        ],
        dtype=np.float64,
    )

    actions = np.array(
        [
            [0.999],
            [0.999],
            [0.999],
        ],
        dtype=np.float64,
    )

    result = perturb_selected_window(
        observations,
        actions,
        _selected_window(),
        kmeans_model=SignKMeans(),
        clean_pattern_frequencies={
            (0,): 10,
            (1,): 100,
        },
        eta=0.05,
        num_candidates=100,
        rng=np.random.default_rng(3),
    )

    assert np.all(
        result.actions <= 1.0
    )

    assert np.all(
        result.actions >= -1.0
    )


def test_same_seed_is_deterministic():
    observations = np.array(
        [
            [-2.0, 1.0],
            [-2.0, 1.0],
            [-2.0, 1.0],
        ],
        dtype=np.float64,
    )

    actions = np.array(
        [
            [0.2],
            [0.2],
            [0.2],
        ],
        dtype=np.float64,
    )

    kwargs = dict(
        kmeans_model=SignKMeans(),
        clean_pattern_frequencies={
            (0,): 10,
            (1,): 100,
        },
        eta=0.05,
        num_candidates=25,
    )

    first = perturb_selected_window(
        observations,
        actions,
        _selected_window(),
        rng=np.random.default_rng(42),
        **kwargs,
    )

    second = perturb_selected_window(
        observations,
        actions,
        _selected_window(),
        rng=np.random.default_rng(42),
        **kwargs,
    )

    np.testing.assert_array_equal(
        first.observations,
        second.observations,
    )

    np.testing.assert_array_equal(
        first.actions,
        second.actions,
    )

    assert (
        first.target_pattern
        == second.target_pattern
    )

    assert (
        first.candidate_index
        == second.candidate_index
    )

def test_float32_kmeans_accepts_rng_generated_candidates():
    observations = np.array(
        [
            [-3.0, 0.0],
            [-2.5, 0.1],
            [-2.0, 0.2],
            [2.0, 0.3],
            [2.5, 0.4],
            [3.0, 0.5],
        ],
        dtype=np.float32,
    )

    actions = np.array(
        [
            [-0.2],
            [-0.2],
            [-0.1],
            [0.1],
            [0.2],
            [0.2],
        ],
        dtype=np.float32,
    )

    features = build_raw_decision_units(
        observations,
        actions,
    )

    model, clustering = (
        fit_kmeans_decision_units(
            features,
            num_clusters=2,
            seed=0,
        )
    )

    assert (
        model.cluster_centers_.dtype
        == np.float32
    )

    selected = SelectedWindow(
        trajectory_id=0,
        global_start=0,
        global_end=3,
        source_pattern=(
            int(
                clustering.labels[0]
            ),
        ),
    )

    result = perturb_selected_window(
        observations,
        actions,
        selected,
        kmeans_model=model,
        clean_pattern_frequencies={
            selected.source_pattern: 10,
        },
        eta=0.05,
        num_candidates=10,
        rng=np.random.default_rng(0),
    )

    assert (
        result.observations.shape
        == (3, 2)
    )

    assert (
        result.actions.shape
        == (3, 1)
    )