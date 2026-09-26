from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


POISON_ROOTS = {
    "canonical": Path(
        "data/poisoned/csdpc/walker2d-medium-v2"
    ),
    "s2_overlap_r0": Path(
        "data/poisoned/csdpc_s2_overlap_r0_v1/walker2d-medium-v2"
    ),
}

RHOS = (
    ("001", 0.01),
    ("005", 0.05),
)

SEEDS = (0, 1, 2)


def load_hdf5(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    out = {}

    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj[...]

        f.visititems(visit)

    return out


def row_change_mask(a, b):
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch: {a.shape} vs {b.shape}"
        )

    if a.ndim == 0:
        return np.asarray([bool(a != b)])

    diff = a != b

    if a.ndim == 1:
        return diff

    axes = tuple(range(1, a.ndim))
    return np.any(diff, axis=axes)


def numeric_stats(clean, poison, changed_rows):
    if not (
        np.issubdtype(clean.dtype, np.number)
        and np.issubdtype(poison.dtype, np.number)
    ):
        return {}

    if not np.any(changed_rows):
        return {
            "max_abs_change": 0.0,
            "mean_abs_change_changed_elements": 0.0,
        }

    delta = poison.astype(np.float64) - clean.astype(np.float64)
    abs_delta = np.abs(delta)

    changed_elements = abs_delta > 0

    vals = abs_delta[changed_elements]

    return {
        "max_abs_change": float(vals.max()) if vals.size else 0.0,
        "mean_abs_change_changed_elements": (
            float(vals.mean()) if vals.size else 0.0
        ),
    }


def trajectory_ids(clean):
    n = None

    for x in clean.values():
        if np.ndim(x) >= 1:
            n = len(x)
            break

    if n is None:
        raise RuntimeError("No transition arrays found.")

    done = np.zeros(n, dtype=bool)

    for key in ("terminals", "timeouts"):
        if key in clean:
            arr = np.asarray(clean[key]).reshape(-1)
            if len(arr) == n:
                done |= arr.astype(bool)

    traj_id = np.empty(n, dtype=np.int64)

    tid = 0
    for i in range(n):
        traj_id[i] = tid

        if done[i]:
            tid += 1

    return traj_id


def analyze_pair(clean, poison):
    clean_keys = set(clean)
    poison_keys = set(poison)

    if clean_keys != poison_keys:
        raise RuntimeError(
            "Dataset-key mismatch.\n"
            f"clean only: {sorted(clean_keys - poison_keys)}\n"
            f"poison only: {sorted(poison_keys - clean_keys)}"
        )

    tids = trajectory_ids(clean)

    result = {
        "keys": {},
        "union_changed_transitions": 0,
        "changed_trajectories": 0,
    }

    union = np.zeros(len(tids), dtype=bool)

    for key in sorted(clean_keys):
        a = np.asarray(clean[key])
        b = np.asarray(poison[key])

        if a.ndim == 0:
            changed = np.asarray([bool(a != b)])
        else:
            changed = row_change_mask(a, b)

        count = int(changed.sum())

        # Only transition-length arrays participate in union counts.
        if a.ndim >= 1 and len(a) == len(tids):
            union |= changed

        info = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "changed_rows": count,
        }

        info.update(
            numeric_stats(a, b, changed)
        )

        result["keys"][key] = info

    result["union_changed_transitions"] = int(union.sum())

    if np.any(union):
        result["changed_trajectories"] = int(
            np.unique(tids[union]).size
        )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clean",
        type=Path,
        required=True,
        help="Path to the frozen clean walker2d-medium-v2 HDF5.",
    )

    args = parser.parse_args()

    clean = load_hdf5(args.clean)

    output = {
        "clean_dataset": str(args.clean),
        "runs": [],
    }

    print("=" * 110)
    print("GROUP 4E — RAW POISON LOCALIZATION")
    print("=" * 110)

    for condition, root in POISON_ROOTS.items():
        for rho_slug, rho in RHOS:
            for seed in SEEDS:
                path = (
                    root
                    / f"rho_{rho_slug}_seed_{seed}.hdf5"
                )

                poison = load_hdf5(path)

                result = analyze_pair(
                    clean,
                    poison,
                )

                changed_keys = [
                    key
                    for key, info in result["keys"].items()
                    if info["changed_rows"] > 0
                ]

                print()
                print(
                    f"{condition:14s} "
                    f"rho={rho:.2f} seed={seed} | "
                    f"changed transitions="
                    f"{result['union_changed_transitions']} | "
                    f"changed trajectories="
                    f"{result['changed_trajectories']}"
                )

                for key in changed_keys:
                    info = result["keys"][key]

                    print(
                        f"    {key:20s} "
                        f"rows={info['changed_rows']:7d} "
                        f"max_abs="
                        f"{info.get('max_abs_change', 0.0):.8g} "
                        f"mean_abs="
                        f"{info.get('mean_abs_change_changed_elements', 0.0):.8g}"
                    )

                output["runs"].append(
                    {
                        "condition": condition,
                        "rho": rho,
                        "seed": seed,
                        "dataset": str(path),
                        **result,
                    }
                )

    out = Path(
        "experiments/dt_mtm_stress/"
        "group4e_poison_localization.json"
    )

    out.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("output ->", out)


if __name__ == "__main__":
    main()