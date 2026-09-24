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

RHOS = (("001", 0.01), ("005", 0.05))
SEEDS = (0, 1, 2)


def load(path):
    with h5py.File(path, "r") as f:
        return {
            "observations": f["observations"][:],
            "actions": f["actions"][:],
            "rewards": f["rewards"][:],
            "terminals": f["terminals"][:].astype(bool),
            "timeouts": f["timeouts"][:].astype(bool),
        }


def trajectories(data):
    done = data["terminals"] | data["timeouts"]

    result = []
    start = 0

    for i, x in enumerate(done):
        if x:
            end = i + 1
            result.append((start, end))
            start = end

    return result


def changed_mask(clean, poison):
    obs = np.any(
        clean["observations"] != poison["observations"],
        axis=1,
    )
    act = np.any(
        clean["actions"] != poison["actions"],
        axis=1,
    )
    return obs | act


def build_transition_metadata(clean, trajs):
    n = len(clean["rewards"])

    traj_id = np.full(n, -1, dtype=np.int64)
    traj_return = np.full(n, np.nan, dtype=np.float64)
    traj_length = np.zeros(n, dtype=np.int64)
    relative_pos = np.full(n, np.nan, dtype=np.float64)
    distance_to_end = np.zeros(n, dtype=np.int64)

    for tid, (start, end) in enumerate(trajs):
        length = end - start
        ret = float(np.sum(clean["rewards"][start:end]))

        indices = np.arange(start, end)
        local = np.arange(length)

        traj_id[indices] = tid
        traj_return[indices] = ret
        traj_length[indices] = length

        if length > 1:
            relative_pos[indices] = local / (length - 1)
        else:
            relative_pos[indices] = 0.0

        distance_to_end[indices] = (length - 1) - local

    return {
        "traj_id": traj_id,
        "traj_return": traj_return,
        "traj_length": traj_length,
        "relative_pos": relative_pos,
        "distance_to_end": distance_to_end,
    }


def stats(x):
    x = np.asarray(x, dtype=np.float64)

    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
    }


def jaccard(a, b):
    a = set(a)
    b = set(b)

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


def main():
    clean = load(CLEAN)
    trajs = trajectories(clean)

    meta = build_transition_metadata(
        clean,
        trajs,
    )

    used_n = trajs[-1][1]

    action_norm = np.linalg.norm(
        clean["actions"],
        axis=1,
    )

    obs_norm = np.linalg.norm(
        clean["observations"],
        axis=1,
    )

    rows = []
    masks = {}

    print("=" * 125)
    print("GROUP 4E — POISON SELECTION LOCALIZATION")
    print("=" * 125)

    print(
        "condition       rho seed changed "
        "trajReturn  trajLen  relPos "
        "reward  actionNorm  obsNorm"
    )

    for condition, root in ROOTS.items():
        for rho_slug, rho in RHOS:
            for seed in SEEDS:

                poison = load(
                    root
                    / f"rho_{rho_slug}_seed_{seed}.hdf5"
                )

                mask = changed_mask(
                    clean,
                    poison,
                )[:used_n]

                idx = np.flatnonzero(mask)

                masks[(condition, rho, seed)] = idx

                row = {
                    "condition": condition,
                    "rho": rho,
                    "seed": seed,
                    "changed_transitions": int(len(idx)),

                    "trajectory_return": stats(
                        meta["traj_return"][idx]
                    ),

                    "trajectory_length": stats(
                        meta["traj_length"][idx]
                    ),

                    "relative_position": stats(
                        meta["relative_pos"][idx]
                    ),

                    "distance_to_end": stats(
                        meta["distance_to_end"][idx]
                    ),

                    "clean_reward": stats(
                        clean["rewards"][idx]
                    ),

                    "clean_action_norm": stats(
                        action_norm[idx]
                    ),

                    "clean_observation_norm": stats(
                        obs_norm[idx]
                    ),
                }

                rows.append(row)

                print(
                    f"{condition:15s} "
                    f"{rho:.2f} "
                    f"{seed:4d} "
                    f"{len(idx):7d} "
                    f"{row['trajectory_return']['mean']:10.2f} "
                    f"{row['trajectory_length']['mean']:8.1f} "
                    f"{row['relative_position']['mean']:7.3f} "
                    f"{row['clean_reward']['mean']:7.3f} "
                    f"{row['clean_action_norm']['mean']:10.3f} "
                    f"{row['clean_observation_norm']['mean']:8.3f}"
                )

    print()
    print("=" * 125)
    print("ATTACK-SEED OVERLAP")
    print("=" * 125)

    overlaps = []

    for condition in ROOTS:
        for _, rho in RHOS:
            print()
            print(condition, "rho=", rho)

            for a, b in combinations(SEEDS, 2):
                ia = masks[(condition, rho, a)]
                ib = masks[(condition, rho, b)]

                ta = np.unique(
                    meta["traj_id"][ia]
                )
                tb = np.unique(
                    meta["traj_id"][ib]
                )

                transition_j = jaccard(ia, ib)
                trajectory_j = jaccard(ta, tb)

                print(
                    f"seed {a} vs {b}: "
                    f"transition J={transition_j:.4f} | "
                    f"trajectory J={trajectory_j:.4f}"
                )

                overlaps.append(
                    {
                        "condition": condition,
                        "rho": rho,
                        "seed_a": a,
                        "seed_b": b,
                        "transition_jaccard": transition_j,
                        "trajectory_jaccard": trajectory_j,
                    }
                )

    out = Path(
        "experiments/dt_mtm_stress/"
        "group4e_selection_localization.json"
    )

    out.write_text(
        json.dumps(
            {
                "analysis": "group4e_selection_localization",
                "rows": rows,
                "seed_overlaps": overlaps,
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
