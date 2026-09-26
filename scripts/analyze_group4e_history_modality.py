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
    BATCH_SIZE,
    load_clean_dt,
    load_clean_joint_policy,
    load_dataset,
    build_metadata,
    changed_mask,
    classify_endpoints,
    deterministic_sample,
    build_context_batch,
    predict,
)


def hybrid(clean, poison, mode):
    if mode not in (
        "state_only",
        "action_only",
        "both",
    ):
        raise ValueError(mode)

    return {
        "observations": (
            poison["observations"]
            if mode in ("state_only", "both")
            else clean["observations"]
        ),
        "actions": (
            poison["actions"]
            if mode in ("action_only", "both")
            else clean["actions"]
        ),
        "rewards": clean["rewards"],
        "terminals": clean["terminals"],
        "timeouts": clean["timeouts"],
    }


def shifts(
    *,
    model,
    clean,
    variants,
    endpoints,
    meta,
    state_mean,
    state_std,
    device,
):
    output = {
        key: []
        for key in variants
    }

    for start in range(
        0,
        len(endpoints),
        BATCH_SIZE,
    ):
        e = endpoints[
            start:start + BATCH_SIZE
        ]

        clean_batch = build_context_batch(
            clean,
            e,
            meta,
            state_mean,
            state_std,
        )

        clean_pred = predict(
            model,
            clean_batch,
            device,
        )

        for name, dataset in variants.items():
            counter_batch = build_context_batch(
                dataset,
                e,
                meta,
                state_mean,
                state_std,
            )

            counter_pred = predict(
                model,
                counter_batch,
                device,
            )

            magnitude = np.linalg.norm(
                counter_pred - clean_pred,
                axis=1,
            )

            output[name].append(
                magnitude
            )

    return {
        name: np.concatenate(values)
        for name, values in output.items()
    }


def pearson(x, y):
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

    clean = load_dataset(
        CLEAN_DATA
    )

    meta = build_metadata(
        clean
    )

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
    print("=" * 135)
    print(
        "GROUP 4E — HISTORY MODALITY SENSITIVITY"
    )
    print("=" * 135)

    print(
        "condition       rho seed       G      N "
        "A_state  A_action    A_both "
        "jointState jointAction jointBoth"
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

                _, history = (
                    classify_endpoints(
                        mask,
                        meta,
                    )
                )

                # Same frozen history sampling rule used by
                # clean-policy sensitivity analysis.
                seed_base = (
                    910_000
                    + condition_index * 100_000
                    + int(rho * 10_000) * 10
                    + seed
                )

                endpoints = (
                    deterministic_sample(
                        history,
                        seed=seed_base + 2,
                    )
                )

                variants = {
                    "state": hybrid(
                        clean,
                        poison,
                        "state_only",
                    ),
                    "action": hybrid(
                        clean,
                        poison,
                        "action_only",
                    ),
                    "both": hybrid(
                        clean,
                        poison,
                        "both",
                    ),
                }

                dt = shifts(
                    model=dt_model,
                    clean=clean,
                    variants=variants,
                    endpoints=endpoints,
                    meta=meta,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                )

                joint = shifts(
                    model=joint_model,
                    clean=clean,
                    variants=variants,
                    endpoints=endpoints,
                    meta=meta,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                )

                result = {}

                for name in (
                    "state",
                    "action",
                    "both",
                ):
                    dt_mean = float(
                        np.mean(dt[name])
                    )

                    joint_mean = float(
                        np.mean(joint[name])
                    )

                    result[name] = {
                        "dt_mean": dt_mean,
                        "joint_mean": joint_mean,
                        "A": (
                            joint_mean
                            - dt_mean
                        ),
                        "fraction_joint_gt_dt": (
                            float(
                                np.mean(
                                    joint[name]
                                    > dt[name]
                                )
                            )
                        ),
                    }

                g = G[
                    (
                        condition,
                        rho,
                        seed,
                    )
                ]

                row = {
                    "condition": condition,
                    "rho": rho,
                    "seed": seed,
                    "G": g,
                    "n": int(
                        len(endpoints)
                    ),
                    "modalities": result,
                }

                rows.append(row)

                print(
                    f"{condition:15s} "
                    f"{rho:.2f} "
                    f"{seed:4d} "
                    f"{g:+8.4f} "
                    f"{len(endpoints):6d} "
                    f"{result['state']['A']:+8.5f} "
                    f"{result['action']['A']:+9.5f} "
                    f"{result['both']['A']:+9.5f} "
                    f"{result['state']['joint_mean']:10.5f} "
                    f"{result['action']['joint_mean']:11.5f} "
                    f"{result['both']['joint_mean']:9.5f}"
                )

        del dt_model
        del joint_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=" * 135)
    print("MODALITY SUMMARY")
    print("=" * 135)

    gs = [
        row["G"]
        for row in rows
    ]

    for modality in (
        "state",
        "action",
        "both",
    ):
        values = np.asarray(
            [
                row["modalities"][
                    modality
                ]["A"]
                for row in rows
            ],
            dtype=np.float64,
        )

        fractions = np.asarray(
            [
                row["modalities"][
                    modality
                ][
                    "fraction_joint_gt_dt"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        print(
            f"{modality:8s} "
            f"mean_A={values.mean():+.6f} "
            f"median_A={np.median(values):+.6f} "
            f"A>0={np.sum(values > 0)}/12 "
            f"corr(G,A)="
            f"{pearson(gs,values):+.4f} "
            f"mean_frac_joint_gt="
            f"{fractions.mean():.4f}"
        )

    out = Path(
        "experiments/dt_mtm_stress/"
        "group4e_history_modality.json"
    )

    out.write_text(
        json.dumps(
            {
                "analysis": (
                    "group4e_history_modality"
                ),
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
