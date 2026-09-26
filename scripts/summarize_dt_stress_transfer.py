from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CONDITIONS = ("canonical", "s2_overlap_r0")
RHO_SLUGS = ("001", "005")
ATTACK_SEEDS = (0, 1, 2)
TRAIN_SEEDS = (0, 1, 2)
GROUP1_MEAN = 69.80938768165248
GROUP1_COMPATIBILITY_FLOOR = 0.90 * GROUP1_MEAN


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def rho_value(slug: str) -> float:
    return 0.01 if slug == "001" else 0.05


def classify(mean_degradation: float, positive_count: int, practical_floor: float) -> str:
    if mean_degradation >= practical_floor and positive_count == 3:
        return "consistent degradation"
    if mean_degradation > 0.0:
        return "weak/inconsistent degradation"
    return "no detectable degradation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/dt_stress/walker2d_medium"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/dt_stress/stress_transfer_summary.json"),
    )
    args = parser.parse_args()

    clean = {}
    for seed in TRAIN_SEEDS:
        path = args.root / "clean" / f"train_seed_{seed}" / "eval_5000" / "summary.json"
        payload = load_json(path)
        clean[seed] = float(payload["normalized_return_mean"])

    clean_values = np.asarray([clean[s] for s in TRAIN_SEEDS], dtype=np.float64)
    clean_mean = float(clean_values.mean())
    if clean_mean < GROUP1_COMPATIBILITY_FLOOR:
        raise SystemExit(
            "Group 4C clean bridge failed: "
            f"mean={clean_mean:.6f} floor={GROUP1_COMPATIBILITY_FLOOR:.6f}"
        )

    practical_floor = 0.05 * clean_mean
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for condition in CONDITIONS:
        for rho_slug in RHO_SLUGS:
            for seed in TRAIN_SEEDS:
                attack_seed = seed
                train_seed = seed

                path = (
                    args.root
                    / condition
                    / f"rho_{rho_slug}"
                    / f"attack_seed_{attack_seed}"
                    / f"train_seed_{train_seed}"
                    / "eval_5000"
                    / "summary.json"
                )
                payload = load_json(path)
                score = float(payload["normalized_return_mean"])
                degradation = clean[train_seed] - score
                groups[(condition, rho_slug)].append(
                    {
                        "attack_seed": attack_seed,
                        "training_seed": train_seed,
                        "poison_return": score,
                        "clean_return": clean[train_seed],
                        "degradation": degradation,
                        "summary_path": str(path),
                    }
                )

    result = {
        "group4c_clean": clean,
        "group4c_clean_mean": clean_mean,
        "group1_reference_mean": GROUP1_MEAN,
        "group1_90pct_floor": GROUP1_COMPATIBILITY_FLOOR,
        "practical_effect_floor_fraction": 0.05,
        "practical_effect_floor_points": practical_floor,
        "conditions": [],
    }

    print("=" * 80)
    print("GROUP 4C VANILLA-DT STRESS TRANSFER")
    print("=" * 80)
    print(f"fresh clean mean: {clean_mean:.6f}")
    print(f"5% practical-effect floor: {practical_floor:.6f}")
    print()

    for condition in CONDITIONS:
        for rho_slug in RHO_SLUGS:
            rows = groups[(condition, rho_slug)]
            degradations = np.asarray([r["degradation"] for r in rows], dtype=np.float64)
            positive_count = int(np.count_nonzero(degradations > 0.0))
            mean_d = float(degradations.mean())
            median_d = float(np.median(degradations))
            std_d = float(degradations.std())
            label = classify(mean_d, positive_count, practical_floor)
            record = {
                "condition": condition,
                "rho": rho_value(rho_slug),
                "rho_slug": rho_slug,
                "num_pairs": len(rows),
                "positive_degradation_pairs": positive_count,
                "mean_degradation": mean_d,
                "median_degradation": median_d,
                "std_degradation": std_d,
                "classification": label,
                "eligible_for_defense_claim": label == "consistent degradation",
                "pairs": rows,
            }
            result["conditions"].append(record)
            print(
                f"{condition:14s} rho={rho_value(rho_slug):.2f} | "
                f"mean_D={mean_d:+.4f} median_D={median_d:+.4f} "
                f"positive={positive_count}/3 | {label}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print()
    print(f"summary -> {args.output}")


if __name__ == "__main__":
    main()
