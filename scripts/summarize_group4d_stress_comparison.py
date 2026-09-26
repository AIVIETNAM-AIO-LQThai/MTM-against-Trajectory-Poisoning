from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CONDITIONS = ("canonical", "s2_overlap_r0")
RHOS = (("001", 0.01), ("005", 0.05))
SEEDS = (0, 1, 2)

# Frozen fresh Group-4C vanilla-DT clean controls.
CLEAN_DT = {
    0: 73.716021030119,
    1: 65.341193981272,
    2: 67.012623942427,
}

# Frozen Group-4B DT+MTM clean controls.
CLEAN_JOINT = {
    0: 69.97396976583133,
    1: 64.23227166791362,
    2: 68.22817744783433,
}


def load_score(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["normalized_return_mean"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dt-root",
        type=Path,
        default=Path("experiments/dt_stress/walker2d_medium"),
    )
    parser.add_argument(
        "--joint-root",
        type=Path,
        default=Path("experiments/dt_mtm_stress/walker2d_medium"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/dt_mtm_stress/group4d_stress_comparison.json"
        ),
    )
    args = parser.parse_args()

    output = {
        "clean_dt": CLEAN_DT,
        "clean_joint": CLEAN_JOINT,
        "conditions": {},
    }

    print("=" * 96)
    print("GROUP 4D MATCHED DT vs DT+MTM STRESS COMPARISON")
    print("=" * 96)
    print()
    print("G = Delta_DT - Delta_joint")
    print("Delta = clean return - poisoned return")
    print("Positive G => joint model loses less relative to its own clean baseline.")
    print("Negative G => joint model loses more relative to its own clean baseline.")
    print()

    for condition in CONDITIONS:
        output["conditions"][condition] = {}

        for rho_slug, rho in RHOS:
            rows = []

            for seed in SEEDS:
                dt_summary = (
                    args.dt_root
                    / condition
                    / f"rho_{rho_slug}"
                    / f"attack_seed_{seed}"
                    / f"train_seed_{seed}"
                    / "eval_5000"
                    / "summary.json"
                )

                joint_summary = (
                    args.joint_root
                    / condition
                    / f"rho_{rho_slug}"
                    / f"attack_seed_{seed}"
                    / f"train_seed_{seed}"
                    / "eval_5000"
                    / "summary.json"
                )

                poison_dt = load_score(dt_summary)
                poison_joint = load_score(joint_summary)

                delta_dt = CLEAN_DT[seed] - poison_dt
                delta_joint = CLEAN_JOINT[seed] - poison_joint

                gap = delta_dt - delta_joint
                poison_return_gap = poison_joint - poison_dt

                rows.append(
                    {
                        "seed": seed,
                        "clean_dt": CLEAN_DT[seed],
                        "clean_joint": CLEAN_JOINT[seed],
                        "poison_dt": poison_dt,
                        "poison_joint": poison_joint,
                        "delta_dt": delta_dt,
                        "delta_joint": delta_joint,
                        "stress_response_gap": gap,
                        "poison_return_gap_joint_minus_dt": poison_return_gap,
                    }
                )

            gaps = np.asarray(
                [x["stress_response_gap"] for x in rows],
                dtype=np.float64,
            )

            delta_dt = np.asarray(
                [x["delta_dt"] for x in rows],
                dtype=np.float64,
            )

            delta_joint = np.asarray(
                [x["delta_joint"] for x in rows],
                dtype=np.float64,
            )

            poisoned_dt = np.asarray(
                [x["poison_dt"] for x in rows],
                dtype=np.float64,
            )

            poisoned_joint = np.asarray(
                [x["poison_joint"] for x in rows],
                dtype=np.float64,
            )

            summary = {
                "rho": rho,
                "rows": rows,
                "mean_delta_dt": float(delta_dt.mean()),
                "mean_delta_joint": float(delta_joint.mean()),
                "mean_stress_response_gap": float(gaps.mean()),
                "median_stress_response_gap": float(np.median(gaps)),
                "positive_gap_count": int(np.sum(gaps > 0.0)),
                "mean_poison_dt_return": float(poisoned_dt.mean()),
                "mean_poison_joint_return": float(poisoned_joint.mean()),
            }

            output["conditions"][condition][str(rho)] = summary

            print(
                f"{condition:14s} rho={rho:.2f} | "
                f"Delta_DT={summary['mean_delta_dt']:+.4f} | "
                f"Delta_joint={summary['mean_delta_joint']:+.4f} | "
                f"G={summary['mean_stress_response_gap']:+.4f} | "
                f"median_G={summary['median_stress_response_gap']:+.4f} | "
                f"G>0 {summary['positive_gap_count']}/3 | "
                f"poison DT={summary['mean_poison_dt_return']:.4f} | "
                f"poison joint={summary['mean_poison_joint_return']:.4f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print("summary ->", args.output)


if __name__ == "__main__":
    main()
