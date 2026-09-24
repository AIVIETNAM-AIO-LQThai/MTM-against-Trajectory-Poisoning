from __future__ import annotations

import json
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

DT_K = 20
MTM_K = 4


def load(path):
    with h5py.File(path, "r") as f:
        return {
            "observations": f["observations"][:],
            "actions": f["actions"][:],
            "terminals": f["terminals"][:].astype(bool),
            "timeouts": f["timeouts"][:].astype(bool),
        }


def completed_trajectories(data):
    done = data["terminals"] | data["timeouts"]

    slices = []
    start = 0

    for i, flag in enumerate(done):
        if flag:
            slices.append((start, i + 1))
            start = i + 1

    return slices, start


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


def runs_of_true(x):
    runs = []
    current = 0

    for value in x:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0

    if current:
        runs.append(current)

    return runs


def analyze(clean, poison, trajectories, used_n):
    mask = changed_mask(clean, poison)[:used_n]

    per_traj = []
    runs = []

    dt_contexts = 0
    dt_exposed = 0
    dt_clean_endpoint_exposed = 0
    dt_poison_counts = []

    mtm_windows = 0
    mtm_exposed = 0
    mtm_poison_counts = []

    for start, end in trajectories:
        local = mask[start:end]
        length = len(local)

        changed_n = int(local.sum())

        if changed_n:
            per_traj.append(changed_n)

        runs.extend(runs_of_true(local))

        # Prefix sum for O(1) window poison counts.
        prefix = np.concatenate(
            ([0], np.cumsum(local.astype(np.int64)))
        )

        # DT: causal context ending at every transition.
        for endpoint in range(length):
            left = max(0, endpoint - DT_K + 1)
            right = endpoint + 1

            count = int(
                prefix[right] - prefix[left]
            )

            dt_contexts += 1

            if count > 0:
                dt_exposed += 1
                dt_poison_counts.append(count)

                if not local[endpoint]:
                    dt_clean_endpoint_exposed += 1

        # MTM: complete contiguous fixed windows.
        if length >= MTM_K:
            for left in range(length - MTM_K + 1):
                right = left + MTM_K

                count = int(
                    prefix[right] - prefix[left]
                )

                mtm_windows += 1

                if count > 0:
                    mtm_exposed += 1
                    mtm_poison_counts.append(count)

    per_traj = np.asarray(per_traj, dtype=np.float64)
    runs = np.asarray(runs, dtype=np.float64)

    return {
        "changed_transitions": int(mask.sum()),
        "affected_trajectories": int(len(per_traj)),

        "changed_per_affected_trajectory_mean": (
            float(per_traj.mean())
            if len(per_traj) else 0.0
        ),

        "changed_per_affected_trajectory_median": (
            float(np.median(per_traj))
            if len(per_traj) else 0.0
        ),

        "changed_per_affected_trajectory_max": (
            int(per_traj.max())
            if len(per_traj) else 0
        ),

        "poison_run_count": int(len(runs)),

        "poison_run_mean": (
            float(runs.mean())
            if len(runs) else 0.0
        ),

        "poison_run_median": (
            float(np.median(runs))
            if len(runs) else 0.0
        ),

        "poison_run_max": (
            int(runs.max())
            if len(runs) else 0
        ),

        "dt_context_count": dt_contexts,

        "dt_exposed_context_fraction": (
            dt_exposed / dt_contexts
        ),

        "dt_clean_endpoint_exposed_fraction": (
            dt_clean_endpoint_exposed
            / max(1, used_n - int(mask.sum()))
        ),

        "dt_mean_poison_tokens_per_exposed_context": (
            float(np.mean(dt_poison_counts))
            if dt_poison_counts else 0.0
        ),

        "mtm_window_count": mtm_windows,

        "mtm_exposed_window_fraction": (
            mtm_exposed / mtm_windows
        ),

        "mtm_mean_poison_tokens_per_exposed_window": (
            float(np.mean(mtm_poison_counts))
            if mtm_poison_counts else 0.0
        ),
    }


def main():
    clean = load(CLEAN)

    trajectories, used_n = completed_trajectories(clean)

    print(
        f"completed trajectories={len(trajectories)} "
        f"used transitions={used_n} "
        f"trailing={len(clean['terminals']) - used_n}"
    )

    if len(trajectories) != 1190:
        raise RuntimeError(
            f"Expected 1190 trajectories, got {len(trajectories)}"
        )

    if used_n != 999995:
        raise RuntimeError(
            f"Expected 999995 used transitions, got {used_n}"
        )

    rows = []

    print()
    print("=" * 130)
    print("GROUP 4E — POISON SEQUENCE / WINDOW EXPOSURE")
    print("=" * 130)

    print(
        "condition       rho seed changed trajs "
        "chg/traj  runMean runMax "
        "DTexposed cleanDTexposed "
        "MTMexposed"
    )

    for condition, root in ROOTS.items():
        for rho_slug, rho in RHOS:
            for seed in SEEDS:
                poison_path = (
                    root
                    / f"rho_{rho_slug}_seed_{seed}.hdf5"
                )

                poison = load(poison_path)

                result = analyze(
                    clean,
                    poison,
                    trajectories,
                    used_n,
                )

                rows.append(
                    {
                        "condition": condition,
                        "rho": rho,
                        "seed": seed,
                        **result,
                    }
                )

                print(
                    f"{condition:15s} "
                    f"{rho:.2f} "
                    f"{seed:4d} "
                    f"{result['changed_transitions']:7d} "
                    f"{result['affected_trajectories']:5d} "
                    f"{result['changed_per_affected_trajectory_mean']:8.2f} "
                    f"{result['poison_run_mean']:7.3f} "
                    f"{result['poison_run_max']:6d} "
                    f"{result['dt_exposed_context_fraction']:9.4f} "
                    f"{result['dt_clean_endpoint_exposed_fraction']:14.4f} "
                    f"{result['mtm_exposed_window_fraction']:10.4f}"
                )

    out = Path(
        "experiments/dt_mtm_stress/"
        "group4e_window_exposure.json"
    )

    out.write_text(
        json.dumps(
            {
                "analysis": "group4e_window_exposure",
                "dt_context_length": DT_K,
                "mtm_window_length": MTM_K,
                "completed_trajectories": len(trajectories),
                "used_transitions": used_n,
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
