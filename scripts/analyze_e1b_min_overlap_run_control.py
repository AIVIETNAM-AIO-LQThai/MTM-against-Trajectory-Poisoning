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

from scripts.e1b_min_overlap_core import (
    REPLICATES,
    build_min_overlap_control,
    validate_min_overlap_control,
)


E1_RESULT = Path(
    "experiments/postfinal_controls/"
    "e1_matched_random_control.json"
)

PREFLIGHT = Path(
    "experiments/postfinal_controls/"
    "e1b_min_overlap_preflight.json"
)

OUTPUT = Path(
    "experiments/postfinal_controls/"
    "e1b_min_overlap_run_control.json"
)

PARTIAL = Path(
    "experiments/postfinal_controls/"
    "e1b_min_overlap_run_control.partial.json"
)

SENSITIVITY_SEED_BASE = 6_200_000
LAYERWISE_SEED_BASE = 6_300_000
ATTENTION_SEED_BASE = 6_400_000


def e1_reference_lookup(e1):
    result = {}

    for row in e1["rows"]:
        key = (
            row["condition"],
            float(row["rho"]),
            int(row["seed"]),
        )

        source = row[
            "sources"
        ]["csdpc_state"]

        result[key] = {
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

    return result


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

    direct_sample = (
        deterministic_sample(
            direct,
            seed=(
                SENSITIVITY_SEED_BASE
                + sample_key
                + 1
            ),
        )
    )

    history_sample = (
        deterministic_sample(
            history,
            seed=(
                SENSITIVITY_SEED_BASE
                + sample_key
                + 2
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

    sensitivity = (
        action_sensitivity(
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

    attention = (
        attention_metrics(
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
    )

    return {
        "direct_population": int(
            len(direct)
        ),
        "history_only_population": int(
            len(history)
        ),
        "action_sensitivity": (
            sensitivity
        ),
        "layerwise": layers,
        "attention": attention,
    }


def summarize(rows, reference):
    groups = {}

    for row in rows:
        key = (
            row["condition"],
            float(row["rho"]),
            int(row["seed"]),
        )
        groups.setdefault(
            key,
            [],
        ).append(row)

    artifact_summary = []

    for key, group in sorted(
        groups.items()
    ):
        condition, rho, seed = key

        c2 = np.asarray(
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

        overlap = np.asarray(
            [
                r["invariants"][
                    "source_target_overlap_fraction"
                ]
                for r in group
            ],
            dtype=np.float64,
        )

        ref = reference[key]

        artifact_summary.append(
            {
                "condition": condition,
                "rho": rho,
                "seed": seed,
                "n_replicates": int(
                    len(group)
                ),
                "e1_csdpc_reference": ref,
                "control": {
                    "C_block2_mean": float(
                        c2.mean()
                    ),
                    "C_block2_std": float(
                        c2.std(ddof=0)
                    ),
                    "C_block2_positive": int(
                        np.sum(c2 > 0)
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
            x
            for x in artifact_summary
            if x["seed"] == seed
        ]

        control_c2 = np.asarray(
            [
                x["control"][
                    "C_block2_mean"
                ]
                for x in selected
            ],
            dtype=np.float64,
        )

        ref_c2 = np.asarray(
            [
                x[
                    "e1_csdpc_reference"
                ]["C_block2"]
                for x in selected
            ],
            dtype=np.float64,
        )

        control_ah = np.asarray(
            [
                x["control"][
                    "A_history_mean"
                ]
                for x in selected
            ],
            dtype=np.float64,
        )

        seed_summary[str(seed)] = {
            "n_artifacts": int(
                len(selected)
            ),
            "control_mean_C_block2": float(
                control_c2.mean()
            ),
            "e1_csdpc_mean_C_block2": float(
                ref_c2.mean()
            ),
            "C_block2_ratio_control_over_e1": (
                float(
                    control_c2.mean()
                    / ref_c2.mean()
                )
                if abs(
                    ref_c2.mean()
                ) > 1e-12
                else None
            ),
            "control_mean_A_history": float(
                control_ah.mean()
            ),
        }

    all_c2 = np.asarray(
        [
            r["metrics"][
                "layerwise"
            ]["block2"][
                "C_joint_minus_dt"
            ]
            for r in rows
        ],
        dtype=np.float64,
    )

    all_ah = np.asarray(
        [
            r["metrics"][
                "action_sensitivity"
            ]["history_only"][
                "A_joint_minus_dt"
            ]
            for r in rows
        ],
        dtype=np.float64,
    )

    all_overlap = np.asarray(
        [
            r["invariants"][
                "source_target_overlap_fraction"
            ]
            for r in rows
        ],
        dtype=np.float64,
    )

    e1_c2_mean = float(
        np.mean(
            [
                x["C_block2"]
                for x in (
                    reference.values()
                )
            ]
        )
    )

    overall = {
        "n_control_realizations": int(
            len(rows)
        ),
        "control_mean_C_block2": float(
            all_c2.mean()
        ),
        "control_C_block2_positive": int(
            np.sum(all_c2 > 0)
        ),
        "e1_csdpc_mean_C_block2": (
            e1_c2_mean
        ),
        "C_block2_ratio_control_over_e1": (
            float(
                all_c2.mean()
                / e1_c2_mean
            )
            if abs(
                e1_c2_mean
            ) > 1e-12
            else None
        ),
        "control_mean_A_history": float(
            all_ah.mean()
        ),
        "control_A_history_positive": int(
            np.sum(all_ah > 0)
        ),
        "mean_source_target_overlap_fraction": float(
            all_overlap.mean()
        ),
        "max_source_target_overlap_fraction": float(
            all_overlap.max()
        ),
        "mean_moved_away_fraction": float(
            1.0
            - all_overlap.mean()
        ),
    }

    return {
        "artifact_summary": (
            artifact_summary
        ),
        "seed_summary": (
            seed_summary
        ),
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
                "analysis": (
                    "e1b_min_overlap_run_control"
                ),
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

    return list(
        data.get(
            "rows",
            [],
        )
    )


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
            f"Missing E1 result: {E1_RESULT}"
        )

    if not PREFLIGHT.exists():
        raise FileNotFoundError(
            "Run E1B preflight before model inference: "
            f"{PREFLIGHT}"
        )

    preflight = json.loads(
        PREFLIGHT.read_text(
            encoding="utf-8"
        )
    )

    if (
        preflight.get("status")
        != "PASS"
    ):
        raise RuntimeError(
            "E1B preflight did not PASS."
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

    reference = (
        e1_reference_lookup(e1)
    )

    rows = load_partial()

    completed = {
        row_key(row)
        for row in rows
    }

    print()
    print("=" * 150)
    print(
        "E1B - MINIMUM-OVERLAP "
        "RUN LOCATION CONTROL"
    )
    print("=" * 150)

    print(
        "condition       rho seed rep "
        "overlap moved   A_direct A_history "
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

                artifact_seed = (
                    condition_index
                    * 100
                    + int(
                        rho * 1000
                    )
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
                        build_min_overlap_control(
                            clean,
                            poison,
                            meta["used_n"],
                            replicate=replicate,
                            artifact_seed=artifact_seed,
                        )
                    )

                    invariants = (
                        validate_min_overlap_control(
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

                    metrics = (
                        metrics_for_control(
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
                    )

                    row = {
                        "condition": (
                            condition
                        ),
                        "rho": rho,
                        "seed": int(
                            seed
                        ),
                        "replicate": int(
                            replicate
                        ),
                        "invariants": (
                            invariants
                        ),
                        "trajectory_layouts": (
                            records
                        ),
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
            f"Expected 60 controls, got {len(rows)}."
        )

    summary = summarize(
        rows,
        reference,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            {
                "analysis": (
                    "e1b_min_overlap_run_control"
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
