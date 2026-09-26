import numpy as np

from scripts.audit_e3_csdpc_candidate_objective_attainment import (
    _aggregate_rows,
    _candidate_pool_diagnostics,
)
from src.attacks.csdpc.perturbation import perturb_selected_window
from src.attacks.csdpc.types import SelectedWindow


class TinyKMeans:
    def __init__(self):
        self.cluster_centers_ = np.asarray(
            [[-1.0, 0.0], [1.0, 0.0]],
            dtype=np.float64,
        )

    def predict(self, x):
        x = np.asarray(x)
        return (x[:, 0] >= 0.0).astype(np.int64)


def _fixture():
    observations = np.asarray(
        [[-0.10], [-0.10], [-0.10]],
        dtype=np.float64,
    )
    actions = np.asarray(
        [[0.2], [0.2], [0.2]],
        dtype=np.float64,
    )
    selected = SelectedWindow(
        trajectory_id=0,
        global_start=0,
        global_end=3,
        source_pattern=(0,),
    )
    frequencies = {
        (0,): 2,
        (1,): 20,
        (0, 1): 5,
        (1, 0): 6,
        (0, 1, 0): 7,
        (1, 0, 1): 8,
    }
    return observations, actions, selected, frequencies


def test_candidate_audit_matches_canonical_best_candidate():
    observations, actions, selected, frequencies = _fixture()
    seed = 1234

    diagnostic = _candidate_pool_diagnostics(
        observations,
        actions,
        selected,
        kmeans_model=TinyKMeans(),
        clean_pattern_frequencies=frequencies,
        eta=2.0,
        num_candidates=100,
        rng=np.random.default_rng(seed),
        global_max_frequency=20,
        action_low=-1.0,
        action_high=1.0,
    )

    canonical = perturb_selected_window(
        observations,
        actions,
        selected,
        kmeans_model=TinyKMeans(),
        clean_pattern_frequencies=frequencies,
        eta=2.0,
        num_candidates=100,
        rng=np.random.default_rng(seed),
        action_low=-1.0,
        action_high=1.0,
    )

    assert diagnostic["best_candidate_index"] == canonical.candidate_index
    assert tuple(diagnostic["best_target_pattern"]) == tuple(canonical.target_pattern)
    assert diagnostic["best_target_frequency"] == canonical.target_frequency


def test_candidate_pool_metrics_are_bounded():
    observations, actions, selected, frequencies = _fixture()

    row = _candidate_pool_diagnostics(
        observations,
        actions,
        selected,
        kmeans_model=TinyKMeans(),
        clean_pattern_frequencies=frequencies,
        eta=2.0,
        num_candidates=50,
        rng=np.random.default_rng(9),
        global_max_frequency=20,
        action_low=-1.0,
        action_high=1.0,
    )

    assert 0.0 <= row["fraction_candidate_patterns_changed"] <= 1.0
    assert 0.0 <= row["fraction_candidates_frequency_improved"] <= 1.0
    assert 0.0 <= row["best_to_global_max_ratio"] <= 1.0


def test_aggregate_rows():
    rows = [
        {
            "chosen_pattern_changed": True,
            "frequency_improved": True,
            "source_frequency": 1,
            "best_target_frequency": 5,
            "best_to_source_ratio": 5.0,
            "best_to_global_max_ratio": 0.5,
            "normalized_frequency_progress": 4.0 / 9.0,
            "distinct_candidate_pattern_count": 3,
            "candidate_pool_has_any_pattern_change": True,
            "candidate_pool_has_any_frequency_improvement": True,
            "candidate_pool_global_max_hit_count": 0,
        },
        {
            "chosen_pattern_changed": False,
            "frequency_improved": False,
            "source_frequency": 2,
            "best_target_frequency": 2,
            "best_to_source_ratio": 1.0,
            "best_to_global_max_ratio": 0.2,
            "normalized_frequency_progress": 0.0,
            "distinct_candidate_pattern_count": 1,
            "candidate_pool_has_any_pattern_change": False,
            "candidate_pool_has_any_frequency_improvement": False,
            "candidate_pool_global_max_hit_count": 0,
        },
    ]

    result = _aggregate_rows(rows)

    assert result["num_selected_windows"] == 2
    assert np.isclose(result["fraction_chosen_patterns_changed"], 0.5)
    assert np.isclose(result["fraction_chosen_frequencies_improved"], 0.5)
    assert np.isclose(result["mean_distinct_candidate_pattern_count"], 2.0)


def test_global_max_hit_fraction():
    rows = [
        {
            "chosen_pattern_changed": True,
            "frequency_improved": True,
            "source_frequency": 1,
            "best_target_frequency": 10,
            "best_to_source_ratio": 10.0,
            "best_to_global_max_ratio": 1.0,
            "normalized_frequency_progress": 1.0,
            "distinct_candidate_pattern_count": 3,
            "candidate_pool_has_any_pattern_change": True,
            "candidate_pool_has_any_frequency_improvement": True,
            "candidate_pool_global_max_hit_count": 1,
        },
        {
            "chosen_pattern_changed": True,
            "frequency_improved": True,
            "source_frequency": 1,
            "best_target_frequency": 5,
            "best_to_source_ratio": 5.0,
            "best_to_global_max_ratio": 0.5,
            "normalized_frequency_progress": 4.0 / 9.0,
            "distinct_candidate_pattern_count": 2,
            "candidate_pool_has_any_pattern_change": True,
            "candidate_pool_has_any_frequency_improvement": True,
            "candidate_pool_global_max_hit_count": 0,
        },
    ]

    result = _aggregate_rows(rows)
    assert np.isclose(result["fraction_pools_with_global_max_hit"], 0.5)
