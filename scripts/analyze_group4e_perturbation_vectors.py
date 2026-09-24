from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np


CLEAN = Path(
    "data/raw/walker2d-medium-v2/walker2d_medium-v2.hdf5"
)

ROOTS = {
    "canonical": Path(
        "data/poisoned/csdpc/walker2d-medium-v2"
    ),
    "s2_overlap_r0": Path(
        "data/poisoned/csdpc_s2_overlap_r0_v1/walker2d-medium-v2"
    ),
}

GROUP4D = Path(
    "experiments/dt_mtm_stress/group4d_stress_comparison.json"
)

RHOS = (("001", "0.01"), ("005", "0.05"))
SEEDS = (0, 1, 2)
EPS = 1e-12


def load(path: Path):
    with h5py.File(path, "r") as f:
        return {
            "observations": f["observations"][:].astype(np.float64),
            "actions": f["actions"][:].astype(np.float64),
        }


def delta(clean, poison):
    return {
        "observations": (
            poison["observations"] - clean["observations"]
        ),
        "actions": (
            poison["actions"] - clean["actions"]
        ),
    }


def changed_mask(d):
    obs = np.any(d["observations"] != 0.0, axis=1)
    act = np.any(d["actions"] != 0.0, axis=1)
    return obs | act


def cosine_rows(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)

    valid = (na > EPS) & (nb > EPS)

    if not np.any(valid):
        return np.asarray([], dtype=np.float64)

    dots = np.sum(a[valid] * b[valid], axis=1)

    return dots / (na[valid] * nb[valid])


def flattened_cosine(a, b):
    x = a.reshape(-1)
    y = b.reshape(-1)

    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)

    if nx <= EPS or ny <= EPS:
        return None

    return float(np.dot(x, y) / (nx * ny))


def sign_agreement(a, b):
    # Compare only entries where both perturbations are non-zero.
    valid = (a != 0.0) & (b != 0.0)

    if not np.any(valid):
        return None

    return float(
        np.mean(np.sign(a[valid]) == np.sign(b[valid]))
    )


def normalized_distance(a, b):
    direct = np.linalg.norm((a - b).reshape(-1))

    na = np.linalg.norm(a.reshape(-1))
    nb = np.linalg.norm(b.reshape(-1))

    mean_norm = 0.5 * (na + nb)

    if mean_norm <= EPS:
        return None

    return float(direct / mean_norm)


def compare_modalities(a, b, overlap):
    result = {}

    for key in ("observations", "actions"):
        x = a[key][overlap]
        y = b[key][overlap]

        row_cos = cosine_rows(x, y)

        result[key] = {
            "mean_row_cosine": (
                float(np.mean(row_cos))
                if len(row_cos) else None
            ),
            "median_row_cosine": (
                float(np.median(row_cos))
                if len(row_cos) else None
            ),
            "global_cosine": flattened_cosine(x, y),
            "sign_agreement": sign_agreement(x, y),
            "normalized_direct_distance": normalized_distance(x, y),
            "valid_row_cosines": int(len(row_cos)),
        }

    return result


def corr(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    if len(xs) < 2:
        return None

    return float(np.corrcoef(xs, ys)[0, 1])


def main():
    clean = load(CLEAN)

    perf = json.loads(
        GROUP4D.read_text(encoding="utf-8")
    )

    G = {}

    for condition in ROOTS:
        for _, rho_key in RHOS:
            for row in perf["conditions"][condition][rho_key]["rows"]:
                G[
                    (
                        condition,
                        float(rho_key),
                        int(row["seed"]),
                    )
                ] = float(row["stress_response_gap"])

    deltas = {}
    masks = {}

    for condition, root in ROOTS.items():
        for rho_slug, rho_key in RHOS:
            rho = float(rho_key)

            for seed in SEEDS:
                poison = load(
                    root / f"rho_{rho_slug}_seed_{seed}.hdf5"
                )

                d = delta(clean, poison)

                deltas[(condition, rho, seed)] = d
                masks[(condition, rho, seed)] = changed_mask(d)

    within = []

    print("=" * 125)
    print("WITHIN-CONDITION ATTACK-SEED PERTURBATION DIRECTION")
    print("=" * 125)

    print(
        "condition       rho pair overlap  |dG|  "
        "obsRowCos obsGlobal actRowCos actGlobal"
    )

    for condition in ROOTS:
        for _, rho_key in RHOS:
            rho = float(rho_key)

            for a, b in combinations(SEEDS, 2):
                ka = (condition, rho, a)
                kb = (condition, rho, b)

                overlap = masks[ka] & masks[kb]
                n_overlap = int(overlap.sum())

                metrics = compare_modalities(
                    deltas[ka],
                    deltas[kb],
                    overlap,
                )

                dg = abs(G[ka] - G[kb])

                row = {
                    "condition": condition,
                    "rho": rho,
                    "seed_a": a,
                    "seed_b": b,
                    "overlap": n_overlap,
                    "absolute_G_difference": dg,
                    "metrics": metrics,
                }

                within.append(row)

                print(
                    f"{condition:15s} "
                    f"{rho:.2f} "
                    f"{a}-{b} "
                    f"{n_overlap:7d} "
                    f"{dg:6.3f} "
                    f"{metrics['observations']['mean_row_cosine']:+9.4f} "
                    f"{metrics['observations']['global_cosine']:+9.4f} "
                    f"{metrics['actions']['mean_row_cosine']:+9.4f} "
                    f"{metrics['actions']['global_cosine']:+9.4f}"
                )

    cross = []

    print()
    print("=" * 125)
    print("CANONICAL vs S2 — SAME RHO / SAME SEED")
    print("=" * 125)

    print(
        "rho seed overlap  |dG|  "
        "obsRowCos obsGlobal actRowCos actGlobal"
    )

    for _, rho_key in RHOS:
        rho = float(rho_key)

        for seed in SEEDS:
            kc = ("canonical", rho, seed)
            ks = ("s2_overlap_r0", rho, seed)

            overlap = masks[kc] & masks[ks]
            n_overlap = int(overlap.sum())

            metrics = compare_modalities(
                deltas[kc],
                deltas[ks],
                overlap,
            )

            dg = abs(G[kc] - G[ks])

            row = {
                "rho": rho,
                "seed": seed,
                "overlap": n_overlap,
                "absolute_G_difference": dg,
                "metrics": metrics,
            }

            cross.append(row)

            print(
                f"{rho:.2f} "
                f"{seed:4d} "
                f"{n_overlap:7d} "
                f"{dg:6.3f} "
                f"{metrics['observations']['mean_row_cosine']:+9.4f} "
                f"{metrics['observations']['global_cosine']:+9.4f} "
                f"{metrics['actions']['mean_row_cosine']:+9.4f} "
                f"{metrics['actions']['global_cosine']:+9.4f}"
            )

    print()
    print("=" * 125)
    print("DESCRIPTIVE CORRELATION WITH |dG|")
    print("=" * 125)

    for label, rows in (
        ("within-seed-pairs", within),
        ("canonical-vs-s2", cross),
    ):
        y = [r["absolute_G_difference"] for r in rows]

        print()
        print(label)

        for modality in ("observations", "actions"):
            row_cos = [
                1.0 - r["metrics"][modality]["mean_row_cosine"]
                for r in rows
            ]

            global_cos = [
                1.0 - r["metrics"][modality]["global_cosine"]
                for r in rows
            ]

            distance = [
                r["metrics"][modality]["normalized_direct_distance"]
                for r in rows
            ]

            print(
                f"  {modality:12s} "
                f"corr(|dG|,1-rowCos)="
                f"{corr(y, row_cos):+.4f} "
                f"corr(|dG|,1-globalCos)="
                f"{corr(y, global_cos):+.4f} "
                f"corr(|dG|,normDistance)="
                f"{corr(y, distance):+.4f}"
            )

    out = Path(
        "experiments/dt_mtm_stress/"
        "group4e_perturbation_vectors.json"
    )

    out.write_text(
        json.dumps(
            {
                "analysis": "group4e_perturbation_vectors",
                "within_condition_seed_pairs": within,
                "canonical_vs_s2": cross,
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
