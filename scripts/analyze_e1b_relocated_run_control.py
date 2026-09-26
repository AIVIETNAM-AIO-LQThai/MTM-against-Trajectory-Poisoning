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
    "e1b_relocated_run_control.json"
)

REPLICATES = tuple(range(5))

RELOCATION_SEED_BASE = 4_100_000
SENSITIVITY_SEED_BASE = 4_200_000
LAYERWISE_SEED_BASE = 4_300_000
ATTENTION_SEED_BASE = 4_400_000

ATOL = 1e-5
MAX_RELOCATION_ATTEMPTS = 10_000

PARTIAL_OUTPUT = Path(
    "experiments/postfinal_controls/"
    "e1b_relocated_run_control.partial.json"
)


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
            f"Completed trajectory range ends at {start}, "
            f"expected {used_n}."
        )

    return ranges


def contiguous_runs(indices):
    indices = np.asarray(indices, dtype=np.int64)

    if len(indices) == 0:
        return []

    split_at = np.flatnonzero(
        np.diff(indices) != 1
    ) + 1

    return [
        np.asarray(x, dtype=np.int64)
        for x in np.split(indices, split_at)
    ]


def run_lengths_from_mask(mask, start, end):
    local = np.flatnonzero(
        mask[start:end]
    )

    return sorted(
        len(run)
        for run in contiguous_runs(local)
    )


def candidate_starts(
    *,
    traj_start,
    traj_end,
    length,
    source_mask,
    target_mask,
):
    out = []

    last_start = traj_end - length

    for target_start in range(
        traj_start,
        last_start + 1,
    ):
        target_end = (
            target_start + length
        )

        if np.any(
            source_mask[
                target_start:target_end
            ]
        ):
            continue

        if np.any(
            target_mask[
                target_start:target_end
            ]
        ):
            continue

        # Keep relocated runs separated so the
        # original run-length multiset is preserved.
        if (
            target_start > traj_start
            and target_mask[
                target_start - 1
            ]
        ):
            continue

        if (
            target_end < traj_end
            and target_mask[
                target_end
            ]
        ):
            continue

        out.append(target_start)

    return np.asarray(
        out,
        dtype=np.int64,
    )


def relocate_runs_within_trajectory(
    clean,
    poison,
    used_n,
    seed,
):
    rng = np.random.default_rng(seed)

    clean_obs = np.asarray(
        clean["observations"]
    )

    poison_obs = np.asarray(
        poison["observations"]
    )

    source_delta = (
        poison_obs[:used_n]
        - clean_obs[:used_n]
    )

    source_mask = np.any(
        source_delta != 0.0,
        axis=1,
    )

    target_mask = np.zeros(
        used_n,
        dtype=bool,
    )

    control_obs = clean_obs.copy()

    ranges = trajectory_ranges(
        clean,
        used_n,
    )

    records = []

    for trajectory_id, (
        traj_start,
        traj_end,
    ) in enumerate(ranges):

        source_indices = (
            np.flatnonzero(
                source_mask[
                    traj_start:traj_end
                ]
            )
            + traj_start
        )

        runs = contiguous_runs(
            source_indices
        )

        if not runs:
            continue

        # Keep the predeclared longest-first construction,
        # but restart the entire trajectory if an unlucky
        # sequence of uniform candidate choices creates a
        # later dead end. No scientific constraint changes.
        ordered_runs = sorted(
            runs,
            key=lambda run: (
                -len(run),
                int(run[0]),
            ),
        )

        placed = False
        successful_plan = None
        successful_attempt = None

        for attempt in range(
            1,
            MAX_RELOCATION_ATTEMPTS + 1,
        ):
            # Current trajectory has not been committed yet,
            # so it is safe to clear only this slice. Marks
            # from previously completed trajectories lie
            # outside this range.
            target_mask[
                traj_start:traj_end
            ] = False

            trial_plan = []
            failed = False

            for run in ordered_runs:
                source_start = int(
                    run[0]
                )

                source_end = int(
                    run[-1]
                ) + 1

                length = (
                    source_end
                    - source_start
                )

                candidates = candidate_starts(
                    traj_start=traj_start,
                    traj_end=traj_end,
                    length=length,
                    source_mask=source_mask,
                    target_mask=target_mask,
                )

                if len(candidates) == 0:
                    failed = True
                    break

                # Predeclared rule: choose uniformly from the
                # currently valid locations using the frozen
                # deterministic RNG stream.
                target_start = int(
                    rng.choice(candidates)
                )

                target_end = (
                    target_start + length
                )

                target_mask[
                    target_start:target_end
                ] = True

                trial_plan.append(
                    {
                        "source_start": source_start,
                        "source_end": source_end,
                        "target_start": target_start,
                        "target_end": target_end,
                        "length": int(length),
                    }
                )

            if not failed:
                placed = True
                successful_plan = trial_plan
                successful_attempt = attempt
                break

        if not placed:
            # Leave this trajectory uncommitted on failure.
            target_mask[
                traj_start:traj_end
            ] = False

            source_run_lengths = [
                int(len(run))
                for run in ordered_runs
            ]

            source_count = int(
                source_mask[
                    traj_start:traj_end
                ].sum()
            )

            raise RuntimeError(
                "No legal E1B relocation plan after "
                f"{MAX_RELOCATION_ATTEMPTS} attempts: "
                f"trajectory={trajectory_id}, "
                f"range=[{traj_start},{traj_end}), "
                f"length={traj_end - traj_start}, "
                f"source_modified_count={source_count}, "
                f"source_run_lengths={source_run_lengths}"
            )

        for plan in successful_plan:
            source_start = int(
                plan["source_start"]
            )
            source_end = int(
                plan["source_end"]
            )
            target_start = int(
                plan["target_start"]
            )
            target_end = int(
                plan["target_end"]
            )

            delta_sequence = (
                source_delta[
                    source_start:source_end
                ]
            )

            control_obs[
                target_start:target_end
            ] = (
                clean_obs[
                    target_start:target_end
                ]
                + delta_sequence
            )

            records.append(
                {
                    "trajectory_id": int(
                        trajectory_id
                    ),
                    "trajectory_start": int(
                        traj_start
                    ),
                    "trajectory_end": int(
                        traj_end
                    ),
                    "source_start": (
                        source_start
                    ),
                    "source_end": (
                        source_end
                    ),
                    "target_start": int(
                        target_start
                    ),
                    "target_end": int(
                        target_end
                    ),
                    "length": int(
                        plan["length"]
                    ),
                    "placement_attempt": int(
                        successful_attempt
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


def validate_relocation(
    clean,
    poison,
    control,
    records,
    used_n,
):
    clean_obs = np.asarray(
        clean["observations"]
    )

    poison_obs = np.asarray(
        poison["observations"]
    )

    control_obs = np.asarray(
        control["observations"]
    )

    source_delta = (
        poison_obs[:used_n]
        - clean_obs[:used_n]
    )

    target_delta = (
        control_obs[:used_n]
        - clean_obs[:used_n]
    )

    source_mask = np.any(
        source_delta != 0.0,
        axis=1,
    )

    target_mask = np.any(
        target_delta != 0.0,
        axis=1,
    )

    overlap = int(
        np.sum(
            source_mask
            & target_mask
        )
    )

    if overlap != 0:
        raise RuntimeError(
            "E1B target locations overlap original "
            f"CSDPC state locations: {overlap}"
        )

    source_count = int(
        source_mask.sum()
    )

    target_count = int(
        target_mask.sum()
    )

    if source_count != target_count:
        raise RuntimeError(
            "E1B modified-state count mismatch: "
            f"source={source_count}, "
            f"target={target_count}"
        )

    ranges = trajectory_ranges(
        clean,
        used_n,
    )

    max_delta_transfer_error = 0.0
    per_trajectory_count_match = True
    per_trajectory_run_length_match = True

    trajectory_summaries = []

    for trajectory_id, (
        traj_start,
        traj_end,
    ) in enumerate(ranges):

        source_n = int(
            source_mask[
                traj_start:traj_end
            ].sum()
        )

        target_n = int(
            target_mask[
                traj_start:traj_end
            ].sum()
        )

        if source_n != target_n:
            per_trajectory_count_match = (
                False
            )

        source_lengths = (
            run_lengths_from_mask(
                source_mask,
                traj_start,
                traj_end,
            )
        )

        target_lengths = (
            run_lengths_from_mask(
                target_mask,
                traj_start,
                traj_end,
            )
        )

        if source_lengths != target_lengths:
            per_trajectory_run_length_match = (
                False
            )

        if source_n or target_n:
            trajectory_summaries.append(
                {
                    "trajectory_id": int(
                        trajectory_id
                    ),
                    "source_count": (
                        source_n
                    ),
                    "target_count": (
                        target_n
                    ),
                    "source_run_lengths": (
                        source_lengths
                    ),
                    "target_run_lengths": (
                        target_lengths
                    ),
                }
            )

    if not per_trajectory_count_match:
        raise RuntimeError(
            "E1B failed per-trajectory "
            "modified-count matching."
        )

    if not per_trajectory_run_length_match:
        raise RuntimeError(
            "E1B failed per-trajectory "
            "run-length matching."
        )

    for record in records:
        s0 = int(
            record["source_start"]
        )
        s1 = int(
            record["source_end"]
        )
        t0 = int(
            record["target_start"]
        )
        t1 = int(
            record["target_end"]
        )

        # Primary construction invariant:
        # the stored relocated observation must equal the
        # target clean observation plus the source delta,
        # evaluated at the dataset's storage precision.
        expected_observation = (
            clean_obs[t0:t1]
            + source_delta[s0:s1]
        )

        if not np.array_equal(
            control_obs[t0:t1],
            expected_observation,
        ):
            raise RuntimeError(
                "E1B relocated observation does not match "
                "clean target + source delta."
            )

        # Diagnostic only: recovering a delta by subtracting
        # two float32 observations introduces a second
        # rounding step, so allow normal float32 error.
        error = float(
            np.max(
                np.abs(
                    source_delta[s0:s1]
                    - target_delta[t0:t1]
                )
            )
        )

        max_delta_transfer_error = max(
            max_delta_transfer_error,
            error,
        )

    if (
        max_delta_transfer_error
        > ATOL
    ):
        raise RuntimeError(
            "E1B delta transfer mismatch: "
            f"{max_delta_transfer_error}"
        )

    displacements = np.asarray(
        [
            abs(
                int(record["target_start"])
                - int(record["source_start"])
            )
            for record in records
        ],
        dtype=np.float64,
    )

    return {
        "source_modified_count": (
            source_count
        ),
        "relocated_modified_count": (
            target_count
        ),
        "source_target_overlap_count": (
            overlap
        ),
        "per_trajectory_count_match": (
            per_trajectory_count_match
        ),
        "per_trajectory_run_length_match": (
            per_trajectory_run_length_match
        ),
        "max_delta_transfer_error": (
            float(
                max_delta_transfer_error
            )
        ),
        "n_relocated_runs": int(
            len(records)
        ),
        "mean_run_displacement": (
            float(displacements.mean())
            if len(displacements)
            else 0.0
        ),
        "median_run_displacement": (
            float(np.median(displacements))
            if len(displacements)
            else 0.0
        ),
        "trajectory_summaries": (
            trajectory_summaries
        ),
    }


def compact_metrics(
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
    direct, history = (
        classify_endpoints(
            changed,
            meta,
        )
    )

    sample_key = (
        condition_index * 100_000
        + int(rho * 10_000) * 100
        + seed * 10
        + replicate
    )

    sensitivity_seed = (
        SENSITIVITY_SEED_BASE
        + sample_key
    )

    direct_sample = (
        deterministic_sample(
            direct,
            seed=(
                sensitivity_seed + 1
            ),
        )
    )

    history_sample = (
        deterministic_sample(
            history,
            seed=(
                sensitivity_seed + 2
            ),
        )
    )

    layer_endpoints = (
        layerwise_sample(
            history,
            LAYERWISE_SEED_BASE
            + sample_key,
        )
    )

    attention_endpoints = (
        attention_sample(
            history,
            ATTENTION_SEED_BASE
            + sample_key,
        )
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
        "direct_population": int(
            len(direct)
        ),
        "history_only_population": int(
            len(history)
        ),
        "direct_sample_size": int(
            len(direct_sample)
        ),
        "history_sample_size": int(
            len(history_sample)
        ),
        "layerwise_sample_size": int(
            len(layer_endpoints)
        ),
        "attention_sample_size": int(
            len(attention_endpoints)
        ),
        "action_sensitivity": (
            sensitivity
        ),
        "layerwise": layers,
        "attention": attention,
    }


def e1_reference_lookup(e1):
    lookup = {}

    for row in e1["rows"]:
        key = (
            row["condition"],
            float(row["rho"]),
            int(row["seed"]),
        )

        source = row["sources"][
            "csdpc_state"
        ]

        lookup[key] = {
            "A_direct": source[
                "action_sensitivity"
            ]["direct"][
                "A_joint_minus_dt"
            ],
            "A_history": source[
                "action_sensitivity"
            ]["history_only"][
                "A_joint_minus_dt"
            ],
            "C_block2": source[
                "layerwise"
            ]["block2"][
                "C_joint_minus_dt"
            ],
            "D_block2": source[
                "attention"
            ]["block2"][
                "D_joint_minus_dt"
            ],
        }

    return lookup


def summarize(rows, e1_reference):
    artifact_groups = {}

    for row in rows:
        key = (
            row["condition"],
            float(row["rho"]),
            int(row["seed"]),
        )

        artifact_groups.setdefault(
            key,
            [],
        ).append(row)

    artifact_summary = []

    for key, group in sorted(
        artifact_groups.items()
    ):
        condition, rho, seed = key

        block2 = np.asarray(
            [
                r["metrics"][
                    "layerwise"
                ]["block2"][
                    "C_joint_minus_dt"
                ]
                for r in group
            ],
            dtype=np.float64,
        )

        ah = np.asarray(
            [
                r["metrics"][
                    "action_sensitivity"
                ]["history_only"][
                    "A_joint_minus_dt"
                ]
                for r in group
            ],
            dtype=np.float64,
        )

        ad = np.asarray(
            [
                r["metrics"][
                    "action_sensitivity"
                ]["direct"][
                    "A_joint_minus_dt"
                ]
                for r in group
            ],
            dtype=np.float64,
        )

        d2 = np.asarray(
            [
                r["metrics"][
                    "attention"
                ]["block2"][
                    "D_joint_minus_dt"
                ]
                for r in group
            ],
            dtype=np.float64,
        )

        ref = e1_reference[key]

        artifact_summary.append(
            {
                "condition": condition,
                "rho": rho,
                "seed": seed,
                "n_replicates": int(
                    len(group)
                ),
                "e1_csdpc_reference": ref,
                "relocated": {
                    "C_block2_mean": float(
                        block2.mean()
                    ),
                    "C_block2_std": float(
                        block2.std(ddof=0)
                    ),
                    "C_block2_positive": int(
                        np.sum(block2 > 0)
                    ),
                    "A_history_mean": float(
                        ah.mean()
                    ),
                    "A_history_std": float(
                        ah.std(ddof=0)
                    ),
                    "A_history_positive": int(
                        np.sum(ah > 0)
                    ),
                    "A_direct_mean": float(
                        ad.mean()
                    ),
                    "D_block2_mean": float(
                        d2.mean()
                    ),
                    "D_block2_std": float(
                        d2.std(ddof=0)
                    ),
                    "D_block2_positive": int(
                        np.sum(d2 > 0)
                    ),
                },
                "relocated_minus_e1": {
                    "C_block2": float(
                        block2.mean()
                        - ref["C_block2"]
                    ),
                    "A_history": float(
                        ah.mean()
                        - ref["A_history"]
                    ),
                    "D_block2": float(
                        d2.mean()
                        - ref["D_block2"]
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

        relocated_c2 = np.asarray(
            [
                row["relocated"][
                    "C_block2_mean"
                ]
                for row in selected
            ],
            dtype=np.float64,
        )

        e1_c2 = np.asarray(
            [
                row[
                    "e1_csdpc_reference"
                ]["C_block2"]
                for row in selected
            ],
            dtype=np.float64,
        )

        relocated_ah = np.asarray(
            [
                row["relocated"][
                    "A_history_mean"
                ]
                for row in selected
            ],
            dtype=np.float64,
        )

        seed_summary[str(seed)] = {
            "n_artifacts": int(
                len(selected)
            ),
            "relocated_mean_C_block2": float(
                relocated_c2.mean()
            ),
            "e1_csdpc_mean_C_block2": float(
                e1_c2.mean()
            ),
            "C_block2_ratio_relocated_over_e1": (
                float(
                    relocated_c2.mean()
                    / e1_c2.mean()
                )
                if abs(e1_c2.mean()) > 1e-12
                else None
            ),
            "relocated_mean_A_history": float(
                relocated_ah.mean()
            ),
        }

    relocated_all = np.asarray(
        [
            row["metrics"][
                "layerwise"
            ]["block2"][
                "C_joint_minus_dt"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    relocated_history_all = np.asarray(
        [
            row["metrics"][
                "action_sensitivity"
            ]["history_only"][
                "A_joint_minus_dt"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    e1_csdpc_mean = float(
        np.mean(
            [
                ref["C_block2"]
                for ref in (
                    e1_reference.values()
                )
            ]
        )
    )

    e1_history_mean = float(
        np.mean(
            [
                ref["A_history"]
                for ref in (
                    e1_reference.values()
                )
            ]
        )
    )

    overall = {
        "n_control_realizations": int(
            len(rows)
        ),
        "relocated_mean_C_block2": float(
            relocated_all.mean()
        ),
        "relocated_C_block2_positive": int(
            np.sum(
                relocated_all > 0
            )
        ),
        "e1_csdpc_mean_C_block2": (
            e1_csdpc_mean
        ),
        "C_block2_ratio_relocated_over_e1": (
            float(
                relocated_all.mean()
                / e1_csdpc_mean
            )
            if abs(e1_csdpc_mean) > 1e-12
            else None
        ),
        "relocated_mean_A_history": float(
            relocated_history_all.mean()
        ),
        "relocated_A_history_positive": int(
            np.sum(
                relocated_history_all > 0
            )
        ),
        "e1_csdpc_mean_A_history": (
            e1_history_mean
        ),
    }

    return {
        "artifact_summary": (
            artifact_summary
        ),
        "seed_summary": seed_summary,
        "overall": overall,
    }


def main():
    if not E1_RESULT.exists():
        raise FileNotFoundError(
            "E1 result is required before E1B: "
            f"{E1_RESULT}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)

    clean = load_dataset(
        CLEAN_DATA
    )

    meta = build_metadata(
        clean
    )

    with np.load(
        NORMALIZATION
    ) as f:
        state_mean = (
            f["state_mean"]
            .astype(np.float32)
        )

        state_std = (
            f["state_std"]
            .astype(np.float32)
        )

    e1 = json.loads(
        E1_RESULT.read_text(
            encoding="utf-8"
        )
    )

    e1_reference = (
        e1_reference_lookup(e1)
    )

    rows = []

    if PARTIAL_OUTPUT.exists():
        partial = json.loads(
            PARTIAL_OUTPUT.read_text(
                encoding="utf-8"
            )
        )
        rows = list(
            partial.get("rows", [])
        )
        print(
            "resume rows:",
            len(rows),
            "from",
            PARTIAL_OUTPUT,
        )

    completed = {
        (
            row["condition"],
            float(row["rho"]),
            int(row["seed"]),
            int(row["replicate"]),
        )
        for row in rows
    }

    print()
    print("=" * 150)
    print(
        "E1B - WITHIN-TRAJECTORY "
        "RELOCATED-RUN CONTROL"
    )
    print("=" * 150)

    print(
        "condition       rho seed rep "
        "runs overlap  A_direct A_history "
        "C_block2 D_block2"
    )

    for seed in SEEDS:
        dt_model, _ = (
            load_clean_dt(
                seed,
                device,
            )
        )

        joint_model, _ = (
            load_clean_joint_policy(
                seed,
                device,
            )
        )

        for condition_index, (
            condition,
            root,
        ) in enumerate(
            ROOTS.items()
        ):
            for (
                rho_slug,
                rho_key,
            ) in RHOS:
                rho = float(
                    rho_key
                )

                poison = load_dataset(
                    root
                    / (
                        f"rho_{rho_slug}_"
                        f"seed_{seed}.hdf5"
                    )
                )

                for replicate in REPLICATES:
                    row_key = (
                        condition,
                        rho,
                        seed,
                        int(replicate),
                    )

                    if row_key in completed:
                        print(
                            f"{condition:15s} "
                            f"{rho:.2f} "
                            f"{seed:4d} "
                            f"{replicate:3d} "
                            "SKIP (checkpointed)"
                        )
                        continue

                    relocation_seed = (
                        RELOCATION_SEED_BASE
                        + condition_index
                        * 100_000
                        + int(
                            rho * 10_000
                        ) * 100
                        + seed * 10
                        + replicate
                    )

                    control, records = (
                        relocate_runs_within_trajectory(
                            clean,
                            poison,
                            meta["used_n"],
                            relocation_seed,
                        )
                    )

                    invariants = (
                        validate_relocation(
                            clean,
                            poison,
                            control,
                            records,
                            meta["used_n"],
                        )
                    )

                    changed = (
                        state_changed_mask(
                            clean,
                            control,
                            meta["used_n"],
                        )
                    )

                    metrics = compact_metrics(
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

                    rows.append(
                        {
                            "condition": condition,
                            "rho": rho,
                            "seed": seed,
                            "replicate": int(
                                replicate
                            ),
                            "relocation_seed": int(
                                relocation_seed
                            ),
                            "invariants": (
                                invariants
                            ),
                            "relocation_records": (
                                records
                            ),
                            "metrics": metrics,
                        }
                    )

                    completed.add(row_key)

                    PARTIAL_OUTPUT.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    PARTIAL_OUTPUT.write_text(
                        json.dumps(
                            {
                                "analysis": (
                                    "e1b_relocated_run_control_partial"
                                ),
                                "rows": rows,
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )

                    print(
                        f"{condition:15s} "
                        f"{rho:.2f} "
                        f"{seed:4d} "
                        f"{replicate:3d} "
                        f"{invariants['n_relocated_runs']:4d} "
                        f"{invariants['source_target_overlap_count']:7d} "
                        f"{metrics['action_sensitivity']['direct']['A_joint_minus_dt']:+8.5f} "
                        f"{metrics['action_sensitivity']['history_only']['A_joint_minus_dt']:+9.5f} "
                        f"{metrics['layerwise']['block2']['C_joint_minus_dt']:+8.5f} "
                        f"{metrics['attention']['block2']['D_joint_minus_dt']:+9.6f}"
                    )

        del dt_model
        del joint_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
                "analysis": (
                    "e1b_relocated_run_control"
                ),
                "replicates": list(
                    REPLICATES
                ),
                "rows": rows,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if PARTIAL_OUTPUT.exists():
        PARTIAL_OUTPUT.unlink()

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
