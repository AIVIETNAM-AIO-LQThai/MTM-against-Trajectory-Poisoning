from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_group4e_clean_policy_sensitivity import (
    CLEAN_DATA,
    NORMALIZATION,
    ROOTS,
    RHOS,
    SEEDS,
    load_clean_dt,
    load_clean_joint_policy,
    load_dataset,
    build_metadata,
    classify_endpoints,
    deterministic_sample,
)

from scripts.analyze_group4e_layerwise_history import (
    frozen_sample as layerwise_sample,
)

from scripts.analyze_group4e_attention_routing import (
    frozen_sample as attention_sample,
    state_changed_mask,
)

from scripts.analyze_e1_matched_random_control import (
    action_sensitivity,
    layerwise_metrics,
    attention_metrics,
)


E1_RESULT = Path(
    "experiments/postfinal_controls/"
    "e1_matched_random_control.json"
)

OUTPUT = Path(
    "experiments/postfinal_controls/"
    "e1b_circular_shift_control.json"
)

PARTIAL = Path(
    "experiments/postfinal_controls/"
    "e1b_circular_shift_control.partial.json"
)

REPLICATES = tuple(range(5))

TIE_SEED_BASE = 5_100_000
SENSITIVITY_SEED_BASE = 5_200_000
LAYERWISE_SEED_BASE = 5_300_000
ATTENTION_SEED_BASE = 5_400_000

ATOL = 1e-5


def trajectory_ranges(clean, used_n):
    done = (
        np.asarray(clean["terminals"][:used_n], dtype=bool)
        | np.asarray(clean["timeouts"][:used_n], dtype=bool)
    )

    ranges = []
    start = 0

    for i, flag in enumerate(done):
        if not flag:
            continue
        end = i + 1
        ranges.append((start, end))
        start = end

    if start != used_n:
        raise RuntimeError(
            f"Completed trajectory range ends at {start}, expected {used_n}."
        )

    return ranges


def contiguous_runs(indices):
    values = np.asarray(indices, dtype=np.int64)

    if len(values) == 0:
        return []

    split_at = np.flatnonzero(np.diff(values) != 1) + 1

    return [
        np.asarray(run, dtype=np.int64)
        for run in np.split(values, split_at)
    ]


def run_lengths(mask):
    indices = np.flatnonzero(mask)
    return sorted(len(run) for run in contiguous_runs(indices))


def shifted_local_indices(source_local, shift, length):
    return (source_local + shift) % length


def enumerate_valid_shifts(source_mask_local, *, tie_seed):
    source_mask_local = np.asarray(source_mask_local, dtype=bool)
    length = len(source_mask_local)

    source_local = np.flatnonzero(source_mask_local)
    source_lengths = run_lengths(source_mask_local)

    if len(source_local) == 0:
        return []

    candidates = []

    for shift in range(1, length):
        target_local = shifted_local_indices(
            source_local,
            shift,
            length,
        )

        target_mask = np.zeros(length, dtype=bool)
        target_mask[target_local] = True

        if run_lengths(target_mask) != source_lengths:
            continue

        overlap_count = int(
            np.sum(source_mask_local & target_mask)
        )

        candidates.append(
            {
                "shift": int(shift),
                "overlap_count": overlap_count,
            }
        )

    if not candidates:
        return []

    rng = np.random.default_rng(tie_seed)

    for row in candidates:
        row["_tie"] = float(rng.random())

    candidates.sort(
        key=lambda row: (
            row["overlap_count"],
            row["_tie"],
            row["shift"],
        )
    )

    for rank, row in enumerate(candidates):
        row["rank"] = int(rank)
        del row["_tie"]

    return candidates


def build_circular_shift_control(
    clean,
    poison,
    used_n,
    *,
    replicate,
    artifact_seed,
):
    clean_obs = np.asarray(clean["observations"])
    poison_obs = np.asarray(poison["observations"])

    source_delta = (
        poison_obs[:used_n]
        - clean_obs[:used_n]
    )

    source_mask = np.any(
        source_delta != 0.0,
        axis=1,
    )

    control_obs = clean_obs.copy()
    records = []

    ranges = trajectory_ranges(clean, used_n)

    for trajectory_id, (traj_start, traj_end) in enumerate(ranges):
        local_source_mask = source_mask[traj_start:traj_end]

        source_count = int(local_source_mask.sum())
        if source_count == 0:
            continue

        length = traj_end - traj_start

        tie_seed = (
            TIE_SEED_BASE
            + artifact_seed * 10_000
            + trajectory_id
        )

        shifts = enumerate_valid_shifts(
            local_source_mask,
            tie_seed=tie_seed,
        )

        if len(shifts) < len(REPLICATES):
            raise RuntimeError(
                "Fewer than five valid E1B circular shifts: "
                f"trajectory={trajectory_id}, "
                f"range=[{traj_start},{traj_end}), "
                f"length={length}, "
                f"source_modified_count={source_count}, "
                f"source_run_lengths={run_lengths(local_source_mask)}, "
                f"valid_shift_count={len(shifts)}"
            )

        selected = shifts[replicate]
        shift = int(selected["shift"])

        source_local = np.flatnonzero(local_source_mask)
        target_local = shifted_local_indices(
            source_local,
            shift,
            length,
        )

        source_global = traj_start + source_local
        target_global = traj_start + target_local

        control_obs[target_global] = (
            clean_obs[target_global]
            + source_delta[source_global]
        )

        overlap_count = int(selected["overlap_count"])

        records.append(
            {
                "trajectory_id": int(trajectory_id),
                "trajectory_start": int(traj_start),
                "trajectory_end": int(traj_end),
                "trajectory_length": int(length),
                "source_modified_count": int(source_count),
                "source_run_lengths": run_lengths(local_source_mask),
                "shift": shift,
                "rank": int(selected["rank"]),
                "valid_shift_count": int(len(shifts)),
                "overlap_count": overlap_count,
                "overlap_fraction": float(
                    overlap_count / source_count
                ),
            }
        )

    control = {
        "observations": control_obs,
        "actions": clean["actions"],
        "rewards": clean["rewards"],
        "terminals": clean["terminals"],
        "timeouts": clean["timeouts"],
    }

    return control, records


def validate_control(
    clean,
    poison,
    control,
    records,
    used_n,
):
    clean_obs = np.asarray(clean["observations"])
    poison_obs = np.asarray(poison["observations"])
    control_obs = np.asarray(control["observations"])

    source_delta = (
        poison_obs[:used_n]
        - clean_obs[:used_n]
    )

    source_mask = np.any(
        source_delta != 0.0,
        axis=1,
    )

    target_delta = (
        control_obs[:used_n]
        - clean_obs[:used_n]
    )

    target_mask = np.any(
        target_delta != 0.0,
        axis=1,
    )

    ranges = trajectory_ranges(clean, used_n)

    max_recovered_delta_error = 0.0

    total_source = int(source_mask.sum())
    total_target = int(target_mask.sum())

    if total_source != total_target:
        raise RuntimeError(
            "Global modified-count mismatch: "
            f"source={total_source}, target={total_target}"
        )

    record_by_traj = {
        int(row["trajectory_id"]): row
        for row in records
    }

    overlap_total = 0

    for trajectory_id, (traj_start, traj_end) in enumerate(ranges):
        source_local_mask = source_mask[traj_start:traj_end]
        target_local_mask = target_mask[traj_start:traj_end]

        source_count = int(source_local_mask.sum())
        target_count = int(target_local_mask.sum())

        if source_count == 0:
            if target_count != 0:
                raise RuntimeError(
                    "Unmodified source trajectory became modified."
                )
            continue

        if source_count != target_count:
            raise RuntimeError(
                "Per-trajectory modified-count mismatch: "
                f"trajectory={trajectory_id}, "
                f"source={source_count}, target={target_count}"
            )

        source_lengths = run_lengths(source_local_mask)
        target_lengths = run_lengths(target_local_mask)

        if source_lengths != target_lengths:
            raise RuntimeError(
                "Run-length mismatch: "
                f"trajectory={trajectory_id}, "
                f"source={source_lengths}, target={target_lengths}"
            )

        row = record_by_traj[trajectory_id]
        shift = int(row["shift"])
        length = traj_end - traj_start

        source_local = np.flatnonzero(source_local_mask)
        target_local = shifted_local_indices(
            source_local,
            shift,
            length,
        )

        source_global = traj_start + source_local
        target_global = traj_start + target_local

        expected_observation = (
            clean_obs[target_global]
            + source_delta[source_global]
        )

        if not np.array_equal(
            control_obs[target_global],
            expected_observation,
        ):
            raise RuntimeError(
                "Stored circular-shift observation does not equal "
                "clean target + source delta."
            )

        recovered = (
            control_obs[target_global]
            - clean_obs[target_global]
        )

        error = float(
            np.max(
                np.abs(
                    recovered
                    - source_delta[source_global]
                )
            )
        )

        max_recovered_delta_error = max(
            max_recovered_delta_error,
            error,
        )

        overlap = int(
            np.sum(
                source_local_mask
                & target_local_mask
            )
        )

        if overlap != int(row["overlap_count"]):
            raise RuntimeError("Recorded overlap mismatch.")

        overlap_total += overlap

    if max_recovered_delta_error > ATOL:
        raise RuntimeError(
            "Recovered float32 delta error exceeds tolerance: "
            f"{max_recovered_delta_error}"
        )

    return {
        "source_modified_count": total_source,
        "shifted_modified_count": total_target,
        "source_target_overlap_count": int(overlap_total),
        "source_target_overlap_fraction": float(
            overlap_total / max(total_source, 1)
        ),
        "moved_away_fraction": float(
            1.0 - overlap_total / max(total_source, 1)
        ),
        "max_recovered_delta_error": float(
            max_recovered_delta_error
        ),
        "n_modified_trajectories": int(len(records)),
        "mean_trajectory_overlap_fraction": float(
            np.mean(
                [
                    row["overlap_fraction"]
                    for row in records
                ]
            )
        ),
        "median_trajectory_overlap_fraction": float(
            np.median(
                [
                    row["overlap_fraction"]
                    for row in records
                ]
            )
        ),
    }


def metrics_for_control(
    *,
    dt_model,
    joint_model,
    clean,
    control,
    changed,
    condition_index,
    rho,
    seed,
    replicate,
    meta,
    state_mean,
    state_std,
    device,
):
    direct, history = classify_endpoints(
        changed,
        meta,
    )

    sample_key = (
        condition_index * 100_000
        + int(rho * 10_000) * 100
        + seed * 10
        + replicate
    )

    direct_sample = deterministic_sample(
        direct,
        seed=SENSITIVITY_SEED_BASE + sample_key + 1,
    )

    history_sample = deterministic_sample(
        history,
        seed=SENSITIVITY_SEED_BASE + sample_key + 2,
    )

    layer_endpoints = layerwise_sample(
        history,
        LAYERWISE_SEED_BASE + sample_key,
    )

    attention_endpoints = attention_sample(
        history,
        ATTENTION_SEED_BASE + sample_key,
    )

    sensitivity = action_sensitivity(
        dt_model=dt_model,
        joint_model=joint_model,
        clean=clean,
        perturbation=control,
        direct_endpoints=direct_sample,
        history_endpoints=history_sample,
        meta=meta,
        state_mean=state_mean,
        state_std=state_std,
        device=device,
    )

    layers = layerwise_metrics(
        dt_model=dt_model,
        joint_model=joint_model,
        clean=clean,
        perturbation=control,
        endpoints=layer_endpoints,
        meta=meta,
        state_mean=state_mean,
        state_std=state_std,
        device=device,
    )

    attention = attention_metrics(
        dt_model=dt_model,
        joint_model=joint_model,
        clean=clean,
        perturbation=control,
        endpoints=attention_endpoints,
        changed=changed,
        meta=meta,
        state_mean=state_mean,
        state_std=state_std,
        device=device,
    )

    return {
        "direct_population": int(len(direct)),
        "history_only_population": int(len(history)),
        "action_sensitivity": sensitivity,
        "layerwise": layers,
        "attention": attention,
    }


def e1_reference_lookup(e1):
    result = {}

    for row in e1["rows"]:
        key = (
            row["condition"],
            float(row["rho"]),
            int(row["seed"]),
        )

        source = row["sources"]["csdpc_state"]

        result[key] = {
            "A_direct": source[
                "action_sensitivity"
            ]["direct"]["A_joint_minus_dt"],
            "A_history": source[
                "action_sensitivity"
            ]["history_only"]["A_joint_minus_dt"],
            "C_block2": source[
                "layerwise"
            ]["block2"]["C_joint_minus_dt"],
            "D_block2": source[
                "attention"
            ]["block2"]["D_joint_minus_dt"],
        }

    return result


def summarize(rows, e1_reference):
    groups = {}

    for row in rows:
        key = (
            row["condition"],
            float(row["rho"]),
            int(row["seed"]),
        )

        groups.setdefault(key, []).append(row)

    artifact_summary = []

    for key, group in sorted(groups.items()):
        condition, rho, seed = key

        c2 = np.asarray(
            [
                row["metrics"]["layerwise"]["block2"][
                    "C_joint_minus_dt"
                ]
                for row in group
            ],
            dtype=np.float64,
        )

        ah = np.asarray(
            [
                row["metrics"]["action_sensitivity"][
                    "history_only"
                ]["A_joint_minus_dt"]
                for row in group
            ],
            dtype=np.float64,
        )

        ad = np.asarray(
            [
                row["metrics"]["action_sensitivity"][
                    "direct"
                ]["A_joint_minus_dt"]
                for row in group
            ],
            dtype=np.float64,
        )

        d2 = np.asarray(
            [
                row["metrics"]["attention"]["block2"][
                    "D_joint_minus_dt"
                ]
                for row in group
            ],
            dtype=np.float64,
        )

        overlap = np.asarray(
            [
                row["invariants"][
                    "source_target_overlap_fraction"
                ]
                for row in group
            ],
            dtype=np.float64,
        )

        ref = e1_reference[key]

        artifact_summary.append(
            {
                "condition": condition,
                "rho": rho,
                "seed": seed,
                "n_replicates": int(len(group)),
                "e1_csdpc_reference": ref,
                "shifted": {
                    "C_block2_mean": float(c2.mean()),
                    "C_block2_std": float(c2.std(ddof=0)),
                    "C_block2_positive": int(np.sum(c2 > 0)),
                    "A_history_mean": float(ah.mean()),
                    "A_history_std": float(ah.std(ddof=0)),
                    "A_history_positive": int(np.sum(ah > 0)),
                    "A_direct_mean": float(ad.mean()),
                    "D_block2_mean": float(d2.mean()),
                    "D_block2_std": float(d2.std(ddof=0)),
                    "D_block2_positive": int(np.sum(d2 > 0)),
                    "overlap_fraction_mean": float(
                        overlap.mean()
                    ),
                    "overlap_fraction_max": float(
                        overlap.max()
                    ),
                },
            }
        )

    seed_summary = {}

    for seed in SEEDS:
        selected = [
            row
            for row in artifact_summary
            if row["seed"] == seed
        ]

        shifted_c2 = np.asarray(
            [
                row["shifted"]["C_block2_mean"]
                for row in selected
            ],
            dtype=np.float64,
        )

        ref_c2 = np.asarray(
            [
                row["e1_csdpc_reference"]["C_block2"]
                for row in selected
            ],
            dtype=np.float64,
        )

        shifted_ah = np.asarray(
            [
                row["shifted"]["A_history_mean"]
                for row in selected
            ],
            dtype=np.float64,
        )

        seed_summary[str(seed)] = {
            "n_artifacts": int(len(selected)),
            "shifted_mean_C_block2": float(
                shifted_c2.mean()
            ),
            "e1_csdpc_mean_C_block2": float(
                ref_c2.mean()
            ),
            "C_block2_ratio_shifted_over_e1": (
                float(
                    shifted_c2.mean()
                    / ref_c2.mean()
                )
                if abs(ref_c2.mean()) > 1e-12
                else None
            ),
            "shifted_mean_A_history": float(
                shifted_ah.mean()
            ),
        }

    all_c2 = np.asarray(
        [
            row["metrics"]["layerwise"]["block2"][
                "C_joint_minus_dt"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    all_ah = np.asarray(
        [
            row["metrics"]["action_sensitivity"][
                "history_only"
            ]["A_joint_minus_dt"]
            for row in rows
        ],
        dtype=np.float64,
    )

    all_overlap = np.asarray(
        [
            row["invariants"][
                "source_target_overlap_fraction"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    e1_c2_mean = float(
        np.mean(
            [
                value["C_block2"]
                for value in e1_reference.values()
            ]
        )
    )

    overall = {
        "n_control_realizations": int(len(rows)),
        "shifted_mean_C_block2": float(
            all_c2.mean()
        ),
        "shifted_C_block2_positive": int(
            np.sum(all_c2 > 0)
        ),
        "e1_csdpc_mean_C_block2": e1_c2_mean,
        "C_block2_ratio_shifted_over_e1": (
            float(
                all_c2.mean() / e1_c2_mean
            )
            if abs(e1_c2_mean) > 1e-12
            else None
        ),
        "shifted_mean_A_history": float(
            all_ah.mean()
        ),
        "shifted_A_history_positive": int(
            np.sum(all_ah > 0)
        ),
        "mean_source_target_overlap_fraction": float(
            all_overlap.mean()
        ),
        "max_source_target_overlap_fraction": float(
            all_overlap.max()
        ),
        "mean_moved_away_fraction": float(
            1.0 - all_overlap.mean()
        ),
    }

    return {
        "artifact_summary": artifact_summary,
        "seed_summary": seed_summary,
        "overall": overall,
    }


def save_partial(rows):
    PARTIAL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PARTIAL.write_text(
        json.dumps(
            {
                "analysis": "e1b_circular_shift_control",
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_partial():
    if not PARTIAL.exists():
        return []

    data = json.loads(
        PARTIAL.read_text(
            encoding="utf-8"
        )
    )

    return list(data.get("rows", []))


def row_key(row):
    return (
        row["condition"],
        float(row["rho"]),
        int(row["seed"]),
        int(row["replicate"]),
    )


def main():
    if not E1_RESULT.exists():
        raise FileNotFoundError(
            "E1 result is required: "
            f"{E1_RESULT}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)

    clean = load_dataset(CLEAN_DATA)
    meta = build_metadata(clean)

    with np.load(NORMALIZATION) as f:
        state_mean = f["state_mean"].astype(
            np.float32
        )
        state_std = f["state_std"].astype(
            np.float32
        )

    e1 = json.loads(
        E1_RESULT.read_text(
            encoding="utf-8"
        )
    )

    e1_reference = e1_reference_lookup(e1)

    rows = load_partial()
    completed = {
        row_key(row)
        for row in rows
    }

    print()
    print("=" * 150)
    print(
        "E1B - CIRCULAR-SHIFT "
        "LOCATION CONTROL"
    )
    print("=" * 150)
    print(
        "condition       rho seed rep "
        "overlap moved   A_direct A_history "
        "C_block2 D_block2"
    )

    for seed in SEEDS:
        dt_model, _ = load_clean_dt(
            seed,
            device,
        )

        joint_model, _ = load_clean_joint_policy(
            seed,
            device,
        )

        for condition_index, (condition, root) in enumerate(
            ROOTS.items()
        ):
            for rho_slug, rho_key in RHOS:
                rho = float(rho_key)

                poison = load_dataset(
                    root
                    / (
                        f"rho_{rho_slug}_"
                        f"seed_{seed}.hdf5"
                    )
                )

                artifact_seed = (
                    condition_index * 100
                    + int(rho * 1000)
                    + seed
                )

                for replicate in REPLICATES:
                    key = (
                        condition,
                        rho,
                        seed,
                        replicate,
                    )

                    if key in completed:
                        print(
                            f"{condition:15s} "
                            f"{rho:.2f} "
                            f"{seed:4d} "
                            f"{replicate:3d} "
                            "SKIP (checkpointed)"
                        )
                        continue

                    control, records = (
                        build_circular_shift_control(
                            clean,
                            poison,
                            meta["used_n"],
                            replicate=replicate,
                            artifact_seed=artifact_seed,
                        )
                    )

                    invariants = validate_control(
                        clean,
                        poison,
                        control,
                        records,
                        meta["used_n"],
                    )

                    changed = state_changed_mask(
                        clean,
                        control,
                        meta["used_n"],
                    )

                    metrics = metrics_for_control(
                        dt_model=dt_model,
                        joint_model=joint_model,
                        clean=clean,
                        control=control,
                        changed=changed,
                        condition_index=condition_index,
                        rho=rho,
                        seed=seed,
                        replicate=replicate,
                        meta=meta,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )

                    row = {
                        "condition": condition,
                        "rho": rho,
                        "seed": int(seed),
                        "replicate": int(replicate),
                        "invariants": invariants,
                        "trajectory_shifts": records,
                        "metrics": metrics,
                    }

                    rows.append(row)
                    completed.add(key)
                    save_partial(rows)

                    print(
                        f"{condition:15s} "
                        f"{rho:.2f} "
                        f"{seed:4d} "
                        f"{replicate:3d} "
                        f"{invariants['source_target_overlap_fraction']:7.4f} "
                        f"{invariants['moved_away_fraction']:7.4f} "
                        f"{metrics['action_sensitivity']['direct']['A_joint_minus_dt']:+8.5f} "
                        f"{metrics['action_sensitivity']['history_only']['A_joint_minus_dt']:+9.5f} "
                        f"{metrics['layerwise']['block2']['C_joint_minus_dt']:+8.5f} "
                        f"{metrics['attention']['block2']['D_joint_minus_dt']:+9.6f}"
                    )

        del dt_model
        del joint_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(rows) != 60:
        raise RuntimeError(
            f"Expected 60 completed controls, got {len(rows)}."
        )

    summary = summarize(
        rows,
        e1_reference,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            {
                "analysis": "e1b_circular_shift_control",
                "replicates": list(REPLICATES),
                "rows": rows,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if PARTIAL.exists():
        PARTIAL.unlink()

    print()
    print("=" * 150)
    print("E1B SUMMARY")
    print("=" * 150)
    print(
        json.dumps(
            summary["overall"],
            indent=2,
        )
    )

    print()
    print("SEED SUMMARY")
    print(
        json.dumps(
            summary["seed_summary"],
            indent=2,
        )
    )

    print()
    print("output ->", OUTPUT)


if __name__ == "__main__":
    main()
