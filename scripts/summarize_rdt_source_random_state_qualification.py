from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/attack_qualification/rdt_source_random_state_corruption.json"
OUTROOT = ROOT / "experiments/attack_qualification/rdt_source_random_state_corruption"
PREFLIGHT = OUTROOT / "preflight.json"
OUTPUT = OUTROOT / "qualification_summary.json"


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_eval(path: Path, model_seed: int, condition: str, attack_seed: int) -> float:
    record = read_json(path)

    if record.get("condition") != condition:
        raise RuntimeError(f"condition mismatch: {path}")
    if int(record["training_seed"]) != model_seed:
        raise RuntimeError(f"model seed mismatch: {path}")
    if int(record["attack_seed"]) != attack_seed:
        raise RuntimeError(f"attack/corruption seed mismatch: {path}")
    if int(record["training_update"]) != 100000:
        raise RuntimeError(f"checkpoint is not 100k: {path}")
    if int(record["num_episodes"]) != 100:
        raise RuntimeError(f"episode count mismatch: {path}")
    if int(record["eval_seed_base"]) != 30000:
        raise RuntimeError(f"eval seed mismatch: {path}")

    return float(record["normalized_return_mean"])


def main():
    cfg = read_json(CONFIG)
    preflight = read_json(PREFLIGHT)

    if preflight.get("status") != "PREFLIGHT_PASS":
        raise RuntimeError("preflight is not PASS")

    model_seeds = [int(x) for x in cfg["training"]["model_seeds"]]
    corruption_seeds = [int(x) for x in cfg["corruption"]["corruption_seeds"]]

    if model_seeds != [0, 1, 2]:
        raise RuntimeError("model seeds changed")
    if corruption_seeds != [2023, 2024, 2025]:
        raise RuntimeError("corruption seeds changed")

    clean = {}
    for model_seed in model_seeds:
        path = OUTROOT / "clean" / f"model_seed_{model_seed}" / "eval" / "summary.json"
        clean[model_seed] = read_eval(
            path,
            model_seed,
            "rdt_source_clean_ratio_0p02",
            1234,
        )

    clean_mean = float(np.mean([clean[s] for s in model_seeds]))
    gate = cfg["qualification_gate"]
    floor_fraction = float(
        gate["mean_degradation_floor_fraction_of_fresh_clean_mean"]
    )
    degradation_floor = float(clean_mean * floor_fraction)

    rows = []
    matrix = {}

    for corruption_seed in corruption_seeds:
        matrix[str(corruption_seed)] = {}

        for model_seed in model_seeds:
            path = (
                OUTROOT
                / "corrupted"
                / f"corruption_seed_{corruption_seed}"
                / f"model_seed_{model_seed}"
                / "eval"
                / "summary.json"
            )

            corrupted_return = read_eval(
                path,
                model_seed,
                "rdt_source_random_state_corruption_ratio_0p02",
                corruption_seed,
            )

            delta = float(clean[model_seed] - corrupted_return)

            row = {
                "corruption_seed": corruption_seed,
                "model_seed": model_seed,
                "clean_return": float(clean[model_seed]),
                "corrupted_return": corrupted_return,
                "degradation": delta,
                "positive_degradation": bool(delta > 0.0),
            }

            rows.append(row)
            matrix[str(corruption_seed)][str(model_seed)] = row

    deltas = np.asarray([r["degradation"] for r in rows], dtype=np.float64)
    mean_delta = float(deltas.mean())
    median_delta = float(np.median(deltas))
    positive_cells = int(np.sum(deltas > 0.0))

    model_seed_means = {
        str(model_seed): float(
            np.mean([
                r["degradation"]
                for r in rows
                if r["model_seed"] == model_seed
            ])
        )
        for model_seed in model_seeds
    }

    corruption_seed_means = {
        str(corruption_seed): float(
            np.mean([
                r["degradation"]
                for r in rows
                if r["corruption_seed"] == corruption_seed
            ])
        )
        for corruption_seed in corruption_seeds
    }

    effect_pass = bool(mean_delta >= degradation_floor)
    positive_cells_pass = bool(
        positive_cells >= int(gate["minimum_positive_cells"])
    )
    model_seed_pass = bool(
        all(v > 0.0 for v in model_seed_means.values())
    )
    corruption_seed_pass = bool(
        all(v > 0.0 for v in corruption_seed_means.values())
    )

    qualification_pass = bool(
        effect_pass
        and positive_cells_pass
        and model_seed_pass
        and corruption_seed_pass
    )

    result = {
        "experiment": "RDT_SOURCE_RANDOM_STATE_CORRUPTION_DT_QUALIFICATION",
        "attack_name": cfg["name"],
        "fresh_clean_controls": True,
        "data_ratio": float(cfg["downsampling"]["ratio"]),
        "clean_returns": {str(k): v for k, v in clean.items()},
        "clean_mean": clean_mean,
        "degradation_floor_fraction": floor_fraction,
        "degradation_floor": degradation_floor,
        "mean_degradation": mean_delta,
        "median_degradation": median_delta,
        "positive_cells": positive_cells,
        "total_cells": len(rows),
        "model_seed_mean_degradation": model_seed_means,
        "corruption_seed_mean_degradation": corruption_seed_means,
        "gate_components": {
            "effect_floor_pass": effect_pass,
            "positive_cells_pass": positive_cells_pass,
            "model_seed_consistency_pass": model_seed_pass,
            "corruption_seed_consistency_pass": corruption_seed_pass,
        },
        "qualification_pass": qualification_pass,
        "matrix": matrix,
    }

    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("RDT-SOURCE RANDOM STATE CORRUPTION — VANILLA DT QUALIFICATION")
    print("=" * 108)
    print("fresh clean 2% returns:", {k: round(v, 6) for k, v in clean.items()})
    print(f"fresh clean mean: {clean_mean:.6f}")
    print()
    print("corruption_seed model_seed clean corrupted degradation")

    for row in rows:
        print(
            f"{row['corruption_seed']:15d} "
            f"{row['model_seed']:10d} "
            f"{row['clean_return']:7.3f} "
            f"{row['corrupted_return']:9.3f} "
            f"{row['degradation']:+11.3f}"
        )

    print()
    print(f"mean degradation:   {mean_delta:+.6f}")
    print(f"median degradation: {median_delta:+.6f}")
    print(f"required floor:     {degradation_floor:.6f}")
    print(f"positive cells:     {positive_cells}/9")
    print("model-seed means:", model_seed_means)
    print("corruption-seed means:", corruption_seed_means)
    print()
    print(
        "ATTACK QUALIFICATION:",
        "PASS" if qualification_pass else "FAIL",
    )
    print("output ->", OUTPUT)


if __name__ == "__main__":
    main()
