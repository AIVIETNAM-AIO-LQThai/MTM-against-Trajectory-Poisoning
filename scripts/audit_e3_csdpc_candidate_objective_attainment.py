from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.attacks.csdpc.attack import prepare_csdpc_attack
from src.attacks.csdpc.metadata import sha256_file, write_metadata_json
from src.attacks.csdpc.patterns import deduplicate_consecutive
from src.attacks.csdpc.selection import (
    compute_transition_budget,
    select_rare_nonoverlapping_windows,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "raw" / "walker2d-medium-v2" / "walker2d_medium-v2.hdf5"
DEFAULT_CONFIG = ROOT / "configs" / "postfinal_controls" / "e3_csdpc_candidate_objective_attainment.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "postfinal_controls" / "e3_csdpc_candidate_objective_attainment.json"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))

    if config.get("experiment") != "E3_CSDPC_CANDIDATE_OBJECTIVE_ATTAINMENT":
        raise ValueError("unexpected E3 experiment identifier")
    if config.get("status") != "POSTFINAL_DIAGNOSTIC_ONLY":
        raise ValueError("E3 diagnostic status changed")
    if not config.get("canonical_attack_is_unchanged", False):
        raise ValueError("canonical attack boundary changed")

    expected = {
        "attack_seeds": [0, 1, 2],
        "num_clusters": 8,
        "sequence_length": 5,
        "eta": 0.05,
        "rho": 0.01,
        "num_candidates": 100,
        "action_low": -1.0,
        "action_high": 1.0,
    }

    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(
                "E3 frozen setting changed: "
                f"{key}={config[key]!r}, expected={value!r}"
            )

    return config


def _load_dataset(path):
    with h5py.File(path, "r") as handle:
        return {key: np.asarray(handle[key]) for key in handle.keys()}


def _relative_linf_scales(values, eta):
    return eta * np.max(np.abs(values), axis=1, keepdims=True)


def _total_linf_cost(state_deltas, action_deltas):
    state_cost = np.max(np.abs(state_deltas), axis=2)
    action_cost = np.max(np.abs(action_deltas), axis=2)
    return np.sum(state_cost + action_cost, axis=1)


def _candidate_pool_diagnostics(
    observations,
    actions,
    selected_window,
    *,
    kmeans_model,
    clean_pattern_frequencies,
    eta,
    num_candidates,
    rng,
    global_max_frequency,
    action_low,
    action_high,
):
    start = int(selected_window.global_start)
    end = int(selected_window.global_end)

    source_observations = np.asarray(observations[start:end]).copy()
    source_actions = np.asarray(actions[start:end]).copy()
    sequence_length = end - start

    state_scales = _relative_linf_scales(source_observations, eta)
    action_scales = _relative_linf_scales(source_actions, eta)

    state_noise = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(num_candidates, sequence_length, source_observations.shape[1]),
    )
    action_noise = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(num_candidates, sequence_length, source_actions.shape[1]),
    )

    candidate_observations = (
        source_observations[None, :, :]
        + state_noise * state_scales[None, :, :]
    )
    candidate_actions = (
        source_actions[None, :, :]
        + action_noise * action_scales[None, :, :]
    )
    candidate_actions = np.clip(candidate_actions, action_low, action_high)

    state_deltas = candidate_observations - source_observations[None, :, :]
    action_deltas = candidate_actions - source_actions[None, :, :]

    decision_units = np.concatenate(
        [candidate_observations, candidate_actions],
        axis=2,
    )

    centers = np.asarray(kmeans_model.cluster_centers_)
    prediction_input = np.asarray(decision_units, dtype=centers.dtype)

    labels = np.asarray(
        kmeans_model.predict(
            prediction_input.reshape(
                num_candidates * sequence_length,
                prediction_input.shape[2],
            )
        ),
        dtype=np.int64,
    ).reshape(num_candidates, sequence_length)

    patterns = [
        tuple(deduplicate_consecutive(labels[i]))
        for i in range(num_candidates)
    ]
    frequencies = np.asarray(
        [
            int(clean_pattern_frequencies.get(pattern, 0))
            for pattern in patterns
        ],
        dtype=np.int64,
    )
    costs = _total_linf_cost(state_deltas, action_deltas)

    best_index = min(
        range(num_candidates),
        key=lambda i: (-int(frequencies[i]), float(costs[i]), int(i)),
    )

    source_pattern = tuple(selected_window.source_pattern)
    source_frequency = int(clean_pattern_frequencies.get(source_pattern, 0))
    best_pattern = patterns[best_index]
    best_frequency = int(frequencies[best_index])

    changed = np.asarray(
        [pattern != source_pattern for pattern in patterns],
        dtype=bool,
    )
    improved = frequencies > source_frequency

    if global_max_frequency > source_frequency:
        normalized_progress = (
            (best_frequency - source_frequency)
            / (global_max_frequency - source_frequency)
        )
    else:
        normalized_progress = 0.0

    return {
        "trajectory_id": int(selected_window.trajectory_id),
        "global_start": start,
        "global_end": end,
        "source_pattern": [int(x) for x in source_pattern],
        "source_frequency": source_frequency,
        "best_candidate_index": int(best_index),
        "best_target_pattern": [int(x) for x in best_pattern],
        "best_target_frequency": best_frequency,
        "chosen_pattern_changed": bool(best_pattern != source_pattern),
        "frequency_improved": bool(best_frequency > source_frequency),
        "best_to_source_ratio": float(best_frequency / max(source_frequency, 1)),
        "best_to_global_max_ratio": float(best_frequency / max(global_max_frequency, 1)),
        "normalized_frequency_progress": float(normalized_progress),
        "distinct_candidate_pattern_count": int(len(set(patterns))),
        "fraction_candidate_patterns_changed": float(changed.mean()),
        "fraction_candidates_frequency_improved": float(improved.mean()),
        "candidate_pool_has_any_pattern_change": bool(np.any(changed)),
        "candidate_pool_has_any_frequency_improvement": bool(np.any(improved)),
        "candidate_pool_global_max_hit_count": int(
            np.sum(frequencies == global_max_frequency)
        ),
        "best_total_linf_perturbation": float(costs[best_index]),
    }


def _mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _aggregate_rows(rows):
    if not rows:
        raise ValueError("cannot aggregate empty E3 rows")

    return {
        "num_selected_windows": int(len(rows)),
        "fraction_chosen_patterns_changed": _mean(
            [row["chosen_pattern_changed"] for row in rows]
        ),
        "fraction_chosen_frequencies_improved": _mean(
            [row["frequency_improved"] for row in rows]
        ),
        "mean_source_frequency": _mean([row["source_frequency"] for row in rows]),
        "median_source_frequency": _median([row["source_frequency"] for row in rows]),
        "mean_best_target_frequency": _mean(
            [row["best_target_frequency"] for row in rows]
        ),
        "median_best_target_frequency": _median(
            [row["best_target_frequency"] for row in rows]
        ),
        "mean_best_to_source_ratio": _mean(
            [row["best_to_source_ratio"] for row in rows]
        ),
        "median_best_to_source_ratio": _median(
            [row["best_to_source_ratio"] for row in rows]
        ),
        "mean_best_to_global_max_ratio": _mean(
            [row["best_to_global_max_ratio"] for row in rows]
        ),
        "median_best_to_global_max_ratio": _median(
            [row["best_to_global_max_ratio"] for row in rows]
        ),
        "mean_normalized_frequency_progress": _mean(
            [row["normalized_frequency_progress"] for row in rows]
        ),
        "median_normalized_frequency_progress": _median(
            [row["normalized_frequency_progress"] for row in rows]
        ),
        "mean_distinct_candidate_pattern_count": _mean(
            [row["distinct_candidate_pattern_count"] for row in rows]
        ),
        "fraction_pools_with_any_pattern_change": _mean(
            [row["candidate_pool_has_any_pattern_change"] for row in rows]
        ),
        "fraction_pools_with_any_frequency_improvement": _mean(
            [row["candidate_pool_has_any_frequency_improvement"] for row in rows]
        ),
        "fraction_pools_with_global_max_hit": _mean(
            [row["candidate_pool_global_max_hit_count"] > 0 for row in rows]
        ),
    }


def main():
    args = _parse_args()
    config = _load_config(args.config)

    dataset_path = args.dataset.resolve()
    output_path = args.output.resolve()

    actual_sha = sha256_file(dataset_path)
    if actual_sha != config["dataset_sha256"]:
        raise RuntimeError("frozen clean dataset SHA256 mismatch")

    print("Frozen clean dataset SHA256: PASS")

    clean = _load_dataset(dataset_path)

    all_rows = []
    seed_summaries = {}

    print()
    print("=" * 132)
    print("E3 - CSDPC CANONICAL CANDIDATE-OBJECTIVE ATTAINMENT")
    print("=" * 132)
    print(
        "seed windows budget global_max  changed improved  "
        "best/src best/max progress diversity any_change any_improve"
    )

    for attack_seed in config["attack_seeds"]:
        attack_seed = int(attack_seed)

        prepared = prepare_csdpc_attack(
            clean,
            attack_seed=attack_seed,
            num_clusters=int(config["num_clusters"]),
            sequence_length=int(config["sequence_length"]),
            eta=float(config["eta"]),
            num_candidates=int(config["num_candidates"]),
        )

        requested_budget = compute_transition_budget(
            num_transitions=prepared.num_transitions,
            rho=float(config["rho"]),
        )

        selection = select_rare_nonoverlapping_windows(
            prepared.windows,
            prepared.pattern_frequencies,
            transition_budget=requested_budget,
        )

        global_max_frequency = int(max(prepared.pattern_frequencies.values()))
        rng = np.random.default_rng(attack_seed)
        rows = []

        for selected_window in selection.selected_windows:
            row = _candidate_pool_diagnostics(
                clean["observations"],
                clean["actions"],
                selected_window,
                kmeans_model=prepared.clustering_model,
                clean_pattern_frequencies=prepared.pattern_frequencies,
                eta=float(config["eta"]),
                num_candidates=int(config["num_candidates"]),
                rng=rng,
                global_max_frequency=global_max_frequency,
                action_low=float(config["action_low"]),
                action_high=float(config["action_high"]),
            )
            row["attack_seed"] = attack_seed
            rows.append(row)
            all_rows.append(row)

        aggregate = _aggregate_rows(rows)
        aggregate.update(
            {
                "attack_seed": attack_seed,
                "requested_transition_budget": int(requested_budget),
                "actual_transition_budget": int(selection.actual_transition_budget),
                "skipped_overlap_windows": int(selection.skipped_overlap_windows),
                "global_max_pattern_frequency": global_max_frequency,
            }
        )

        seed_summaries[str(attack_seed)] = aggregate

        print(
            f"{attack_seed:4d} "
            f"{aggregate['num_selected_windows']:7d} "
            f"{aggregate['actual_transition_budget']:6d} "
            f"{global_max_frequency:10d} "
            f"{aggregate['fraction_chosen_patterns_changed']:8.4f} "
            f"{aggregate['fraction_chosen_frequencies_improved']:8.4f} "
            f"{aggregate['median_best_to_source_ratio']:8.3f} "
            f"{aggregate['median_best_to_global_max_ratio']:8.5f} "
            f"{aggregate['median_normalized_frequency_progress']:8.5f} "
            f"{aggregate['mean_distinct_candidate_pattern_count']:9.3f} "
            f"{aggregate['fraction_pools_with_any_pattern_change']:10.4f} "
            f"{aggregate['fraction_pools_with_any_frequency_improvement']:11.4f}"
        )

    overall = _aggregate_rows(all_rows)

    result = {
        "schema_version": "e3-csdpc-candidate-objective-attainment-v1",
        "experiment": config["experiment"],
        "status": config["status"],
        "canonical_attack_is_unchanged": True,
        "dataset": {
            "name": config["dataset"],
            "sha256": actual_sha,
        },
        "frozen_settings": {
            "attack_seeds": config["attack_seeds"],
            "num_clusters": config["num_clusters"],
            "sequence_length": config["sequence_length"],
            "eta": config["eta"],
            "rho": config["rho"],
            "num_candidates": config["num_candidates"],
            "pipeline": config["pipeline"],
        },
        "seed_summaries": seed_summaries,
        "overall": overall,
        "rows": all_rows,
        "claim_boundary": [
            "No alternative candidate generator was tested.",
            "No poisoned HDF5 artifact was written.",
            "No learner was trained or evaluated.",
            "Global-max-frequency comparisons are objective references, not reachability proofs.",
            "Canonical Gate B and Group-2 closure remain unchanged."
        ],
    }

    write_metadata_json(output_path, result)

    print()
    print("=" * 132)
    print("E3 OVERALL SUMMARY")
    print("=" * 132)
    print(json.dumps(overall, indent=2))
    print()
    print("output ->", output_path)


if __name__ == "__main__":
    main()
