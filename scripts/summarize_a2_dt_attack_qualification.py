from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "attack_qualification" / "a2_dt_attack_qualification.json"
OUTROOT = ROOT / "experiments" / "attack_qualification" / "a2_dt"
OUTPUT = OUTROOT / "qualification_summary.json"


def read_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))

    clean = {}
    for seed in cfg["model_seeds"]:
        path = OUTROOT / "clean" / f"model_seed_{seed}" / "eval" / "summary.json"
        rec = read_summary(path)
        clean[int(seed)] = float(rec["normalized_return_mean"])

    clean_values = np.asarray([clean[s] for s in cfg["model_seeds"]], dtype=np.float64)
    clean_mean = float(clean_values.mean())

    clean_bridge_pass = bool(
        clean_mean >= float(cfg["clean_bridge"]["compatibility_floor"])
    )

    rows = []
    matrix = {}

    for attack_seed in cfg["attack_seeds"]:
        matrix[str(attack_seed)] = {}
        for model_seed in cfg["model_seeds"]:
            path = (
                OUTROOT
                / "poison"
                / f"attack_seed_{attack_seed}"
                / f"model_seed_{model_seed}"
                / "eval"
                / "summary.json"
            )
            rec = read_summary(path)
            poison_return = float(rec["normalized_return_mean"])
            delta = float(clean[int(model_seed)] - poison_return)

            row = {
                "attack_seed": int(attack_seed),
                "model_seed": int(model_seed),
                "clean_return": float(clean[int(model_seed)]),
                "poison_return": poison_return,
                "degradation": delta,
                "positive_degradation": bool(delta > 0.0),
            }
            rows.append(row)
            matrix[str(attack_seed)][str(model_seed)] = row

    deltas = np.asarray([r["degradation"] for r in rows], dtype=np.float64)

    mean_delta = float(deltas.mean())
    median_delta = float(np.median(deltas))
    positive_cells = int(np.sum(deltas > 0.0))

    model_seed_means = {}
    for model_seed in cfg["model_seeds"]:
        values = [
            r["degradation"]
            for r in rows
            if r["model_seed"] == model_seed
        ]
        model_seed_means[str(model_seed)] = float(np.mean(values))

    attack_seed_means = {}
    for attack_seed in cfg["attack_seeds"]:
        values = [
            r["degradation"]
            for r in rows
            if r["attack_seed"] == attack_seed
        ]
        attack_seed_means[str(attack_seed)] = float(np.mean(values))

    gate = cfg["qualification_gate"]
    floor = float(
        gate["mean_degradation_floor_fraction_of_matched_clean_mean"]
        * clean_mean
    )

    effect_pass = bool(mean_delta >= floor)
    positive_cells_pass = bool(
        positive_cells >= int(gate["minimum_positive_cells"])
    )
    model_seed_pass = bool(all(v > 0.0 for v in model_seed_means.values()))
    attack_seed_pass = bool(all(v > 0.0 for v in attack_seed_means.values()))

    qualification_pass = bool(
        clean_bridge_pass
        and effect_pass
        and positive_cells_pass
        and model_seed_pass
        and attack_seed_pass
    )

    result = {
        "experiment": cfg["experiment"],
        "clean_returns": {str(k): v for k, v in clean.items()},
        "clean_mean": clean_mean,
        "clean_bridge_floor": float(cfg["clean_bridge"]["compatibility_floor"]),
        "clean_bridge_pass": clean_bridge_pass,
        "degradation_floor": floor,
        "mean_degradation": mean_delta,
        "median_degradation": median_delta,
        "positive_cells": positive_cells,
        "total_cells": len(rows),
        "model_seed_mean_degradation": model_seed_means,
        "attack_seed_mean_degradation": attack_seed_means,
        "gate_components": {
            "effect_floor_pass": effect_pass,
            "positive_cells_pass": positive_cells_pass,
            "model_seed_consistency_pass": model_seed_pass,
            "attack_seed_consistency_pass": attack_seed_pass,
        },
        "qualification_pass": qualification_pass,
        "matrix": matrix,
    }

    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=" * 96)
    print("A2 VANILLA DT ATTACK QUALIFICATION")
    print("=" * 96)
    print("clean:", {k: round(v, 6) for k, v in clean.items()})
    print(f"clean mean: {clean_mean:.6f}")
    print(f"clean bridge: {'PASS' if clean_bridge_pass else 'FAIL'}")
    print()
    print("attack_seed model_seed clean poison degradation")
    for row in rows:
        print(
            f"{row['attack_seed']:11d} "
            f"{row['model_seed']:10d} "
            f"{row['clean_return']:7.3f} "
            f"{row['poison_return']:7.3f} "
            f"{row['degradation']:+11.3f}"
        )
    print()
    print(f"mean degradation:   {mean_delta:+.6f}")
    print(f"median degradation: {median_delta:+.6f}")
    print(f"required floor:     {floor:.6f}")
    print(f"positive cells:     {positive_cells}/9")
    print("model-seed means:", model_seed_means)
    print("attack-seed means:", attack_seed_means)
    print()
    print("A2 ATTACK QUALIFICATION:", "PASS" if qualification_pass else "FAIL")
    print("output ->", OUTPUT)


if __name__ == "__main__":
    main()
