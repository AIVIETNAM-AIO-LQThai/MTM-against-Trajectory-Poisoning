from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_group4e_clean_policy_sensitivity import (
    CLEAN_DATA,
    NORMALIZATION,
    GROUP4D,
    ROOTS,
    RHOS,
    SEEDS,
    K,
    load_clean_dt,
    load_clean_joint_policy,
    load_dataset,
    build_metadata,
    changed_mask,
    sensitivity_for_endpoints,
)


MAX_PER_BIN = 5000

DISTANCE_BINS = (
    ("1-3", 1, 3),
    ("4-7", 4, 7),
    ("8-12", 8, 12),
    ("13-19", 13, 19),
)


def collect_history_by_distance(mask, meta):
    bins = {
        name: []
        for name, _, _ in DISTANCE_BINS
    }

    for endpoint in range(meta["used_n"]):
        if mask[endpoint]:
            continue

        traj_start = int(
            meta["traj_start"][endpoint]
        )

        left = max(
            traj_start,
            endpoint - K + 1,
        )

        prior = np.flatnonzero(
            mask[left:endpoint]
        )

        if len(prior) == 0:
            continue

        nearest_global = (
            left + int(prior[-1])
        )

        distance = (
            endpoint - nearest_global
        )

        for name, lo, hi in DISTANCE_BINS:
            if lo <= distance <= hi:
                bins[name].append(endpoint)
                break

    return {
        name: np.asarray(
            values,
            dtype=np.int64,
        )
        for name, values in bins.items()
    }


def frozen_sample(values, seed):
    if len(values) <= MAX_PER_BIN:
        return values

    rng = np.random.default_rng(seed)

    selected = rng.choice(
        values,
        size=MAX_PER_BIN,
        replace=False,
    )

    return np.sort(selected)


def summarize(dt, joint):
    if len(dt) == 0:
        return None

    return {
        "n": int(len(dt)),
        "dt_mean": float(np.mean(dt)),
        "joint_mean": float(np.mean(joint)),
        "A": float(
            np.mean(joint) - np.mean(dt)
        ),
        "fraction_joint_gt_dt": float(
            np.mean(joint > dt)
        ),
    }


def corr(x, y):
    return float(
        np.corrcoef(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
        )[0, 1]
    )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)

    clean = load_dataset(CLEAN_DATA)
    meta = build_metadata(clean)

    with np.load(NORMALIZATION) as f:
        state_mean = (
            f["state_mean"]
            .astype(np.float32)
        )

        state_std = (
            f["state_std"]
            .astype(np.float32)
        )

    group4d = json.loads(
        GROUP4D.read_text(
            encoding="utf-8"
        )
    )

    G = {}

    for condition in ROOTS:
        for _, rho_key in RHOS:
            for row in group4d[
                "conditions"
            ][condition][rho_key]["rows"]:

                G[
                    (
                        condition,
                        float(rho_key),
                        int(row["seed"]),
                    )
                ] = float(
                    row["stress_response_gap"]
                )

    rows = []

    print()
    print("=" * 125)
    print(
        "GROUP 4E — HISTORY-DISTANCE SENSITIVITY"
    )
    print("=" * 125)

    print(
        "condition       rho seed       G "
        "distance      N      DT     Joint        A    J>DT"
    )

    for seed in SEEDS:

        dt_model, _ = load_clean_dt(
            seed,
            device,
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
        ) in enumerate(ROOTS.items()):

            for rho_slug, rho_key in RHOS:

                rho = float(rho_key)

                poison = load_dataset(
                    root
                    / f"rho_{rho_slug}_seed_{seed}.hdf5"
                )

                mask = changed_mask(
                    clean,
                    poison,
                    meta["used_n"],
                )

                distance_groups = (
                    collect_history_by_distance(
                        mask,
                        meta,
                    )
                )

                for bin_index, (
                    name,
                    _,
                    _,
                ) in enumerate(DISTANCE_BINS):

                    population = (
                        distance_groups[name]
                    )

                    analysis_seed = (
                        1_400_000
                        + seed * 100_000
                        + condition_index * 10_000
                        + int(rho * 1000) * 10
                        + bin_index
                    )

                    endpoints = frozen_sample(
                        population,
                        analysis_seed,
                    )

                    dt_shift, joint_shift = (
                        sensitivity_for_endpoints(
                            dt_model=dt_model,
                            joint_model=joint_model,
                            clean=clean,
                            poison=poison,
                            endpoints=endpoints,
                            meta=meta,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    )

                    summary = summarize(
                        dt_shift,
                        joint_shift,
                    )

                    row = {
                        "condition": condition,
                        "rho": rho,
                        "seed": seed,
                        "G": G[
                            (
                                condition,
                                rho,
                                seed,
                            )
                        ],
                        "distance_bin": name,
                        "population": int(
                            len(population)
                        ),
                        "summary": summary,
                    }

                    rows.append(row)

                    print(
                        f"{condition:15s} "
                        f"{rho:.2f} "
                        f"{seed:4d} "
                        f"{row['G']:+8.4f} "
                        f"{name:8s} "
                        f"{summary['n']:6d} "
                        f"{summary['dt_mean']:7.4f} "
                        f"{summary['joint_mean']:8.4f} "
                        f"{summary['A']:+8.5f} "
                        f"{summary['fraction_joint_gt_dt']:6.3f}"
                    )

        del dt_model
        del joint_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=" * 125)
    print("DISTANCE-BIN SUMMARY")
    print("=" * 125)

    for name, _, _ in DISTANCE_BINS:

        selected = [
            r for r in rows
            if r["distance_bin"] == name
        ]

        A = np.asarray(
            [
                r["summary"]["A"]
                for r in selected
            ],
            dtype=np.float64,
        )

        gs = np.asarray(
            [
                r["G"]
                for r in selected
            ],
            dtype=np.float64,
        )

        frac = np.asarray(
            [
                r["summary"][
                    "fraction_joint_gt_dt"
                ]
                for r in selected
            ],
            dtype=np.float64,
        )

        print(
            f"{name:8s} "
            f"mean_A={A.mean():+.6f} "
            f"median_A={np.median(A):+.6f} "
            f"A>0={np.sum(A > 0)}/12 "
            f"corr(G,A)={corr(gs,A):+.4f} "
            f"mean_frac_joint_gt={frac.mean():.4f}"
        )

    out = Path(
        "experiments/dt_mtm_stress/"
        "group4e_history_distance.json"
    )

    out.write_text(
        json.dumps(
            {
                "analysis": (
                    "group4e_history_distance"
                ),
                "distance_bins": [
                    {
                        "name": name,
                        "min": lo,
                        "max": hi,
                    }
                    for name, lo, hi
                    in DISTANCE_BINS
                ],
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("output ->", out)


if __name__ == "__main__":
    main()
