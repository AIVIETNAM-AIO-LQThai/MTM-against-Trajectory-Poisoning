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
)


SENSITIVITY_RESULT = Path(
    "experiments/dt_mtm_stress/"
    "group4e_clean_policy_sensitivity.json"
)

BATCH_SIZE = 4096
EPS = 1e-12


def changed_state_indices(clean, poison, used_n):
    return np.flatnonzero(
        np.any(
            clean["observations"][:used_n]
            != poison["observations"][:used_n],
            axis=1,
        )
    )


def representation_shifts(
    model,
    clean,
    poison,
    indices,
    local_timestep,
    state_mean,
    state_std,
    device,
):
    raw_values = []
    token_values = []

    model.eval()

    with torch.inference_mode():

        for start in range(
            0,
            len(indices),
            BATCH_SIZE,
        ):
            idx = indices[
                start:start + BATCH_SIZE
            ]

            s_clean = (
                clean["observations"][idx]
                - state_mean
            ) / state_std

            s_poison = (
                poison["observations"][idx]
                - state_mean
            ) / state_std

            t = local_timestep[idx].copy()

            t[t >= 1000] = 999

            s_clean = torch.as_tensor(
                s_clean,
                dtype=torch.float32,
                device=device,
            )

            s_poison = torch.as_tensor(
                s_poison,
                dtype=torch.float32,
                device=device,
            )

            t = torch.as_tensor(
                t,
                dtype=torch.long,
                device=device,
            )

            z_clean = model.embed_state(
                s_clean
            )

            z_poison = model.embed_state(
                s_poison
            )

            time = model.embed_timestep(
                t
            )

            h_clean = model.embed_ln(
                z_clean + time
            )

            h_poison = model.embed_ln(
                z_poison + time
            )

            raw = torch.linalg.vector_norm(
                z_poison - z_clean,
                dim=1,
            )

            token = torch.linalg.vector_norm(
                h_poison - h_clean,
                dim=1,
            )

            raw_values.append(
                raw.cpu().numpy()
            )

            token_values.append(
                token.cpu().numpy()
            )

    return {
        "raw": np.concatenate(raw_values),
        "token": np.concatenate(token_values),
    }


def comparison(dt, joint):
    result = {}

    for level in ("raw", "token"):

        x = dt[level]
        y = joint[level]

        result[level] = {
            "dt_mean": float(
                np.mean(x)
            ),

            "joint_mean": float(
                np.mean(y)
            ),

            "B_joint_minus_dt": float(
                np.mean(y) - np.mean(x)
            ),

            "ratio_joint_over_dt": float(
                np.mean(y)
                / (np.mean(x) + EPS)
            ),

            "fraction_joint_gt_dt": float(
                np.mean(y > x)
            ),
        }

    return result


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

    sensitivity = json.loads(
        SENSITIVITY_RESULT.read_text(
            encoding="utf-8"
        )
    )

    history_A = {}

    for row in sensitivity["rows"]:

        history_A[
            (
                row["condition"],
                float(row["rho"]),
                int(row["seed"]),
            )
        ] = float(
            row["history_only"][
                "A_mean_joint_minus_dt"
            ]
        )

    rows = []

    print()
    print("=" * 135)
    print(
        "GROUP 4E — STATE REPRESENTATION AMPLIFICATION"
    )
    print("=" * 135)

    print(
        "condition       rho seed changed "
        "A_history "
        "B_raw rawRatio "
        "B_token tokenRatio "
        "tokenJ>DT"
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

        for condition, root in ROOTS.items():

            for rho_slug, rho_key in RHOS:

                rho = float(
                    rho_key
                )

                poison = load_dataset(
                    root
                    / f"rho_{rho_slug}_seed_{seed}.hdf5"
                )

                indices = changed_state_indices(
                    clean,
                    poison,
                    meta["used_n"],
                )

                dt_shift = (
                    representation_shifts(
                        dt_model,
                        clean,
                        poison,
                        indices,
                        meta["local_timestep"],
                        state_mean,
                        state_std,
                        device,
                    )
                )

                joint_shift = (
                    representation_shifts(
                        joint_model,
                        clean,
                        poison,
                        indices,
                        meta["local_timestep"],
                        state_mean,
                        state_std,
                        device,
                    )
                )

                metrics = comparison(
                    dt_shift,
                    joint_shift,
                )

                ah = history_A[
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
                    "changed_states": int(
                        len(indices)
                    ),
                    "A_history": ah,
                    "representations": (
                        metrics
                    ),
                }

                rows.append(
                    row
                )

                print(
                    f"{condition:15s} "
                    f"{rho:.2f} "
                    f"{seed:4d} "
                    f"{len(indices):7d} "
                    f"{ah:+9.5f} "
                    f"{metrics['raw']['B_joint_minus_dt']:+8.5f} "
                    f"{metrics['raw']['ratio_joint_over_dt']:8.4f} "
                    f"{metrics['token']['B_joint_minus_dt']:+9.5f} "
                    f"{metrics['token']['ratio_joint_over_dt']:10.4f} "
                    f"{metrics['token']['fraction_joint_gt_dt']:9.4f}"
                )

        del dt_model
        del joint_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=" * 135)
    print("REPRESENTATION SUMMARY")
    print("=" * 135)

    history = [
        row["A_history"]
        for row in rows
    ]

    for level in (
        "raw",
        "token",
    ):

        B = np.asarray(
            [
                row[
                    "representations"
                ][level][
                    "B_joint_minus_dt"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        ratios = np.asarray(
            [
                row[
                    "representations"
                ][level][
                    "ratio_joint_over_dt"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        fractions = np.asarray(
            [
                row[
                    "representations"
                ][level][
                    "fraction_joint_gt_dt"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        print(
            f"{level:6s} "
            f"mean_B={B.mean():+.6f} "
            f"median_B={np.median(B):+.6f} "
            f"B>0={np.sum(B > 0)}/12 "
            f"mean_ratio={ratios.mean():.4f} "
            f"mean_frac_joint_gt="
            f"{fractions.mean():.4f} "
            f"corr(A_history,B)="
            f"{corr(history,B):+.4f}"
        )

    out = Path(
        "experiments/dt_mtm_stress/"
        "group4e_state_representation.json"
    )

    out.write_text(
        json.dumps(
            {
                "analysis": (
                    "group4e_state_representation"
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
