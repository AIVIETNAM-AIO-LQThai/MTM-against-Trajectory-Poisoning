from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import numpy as np


CLEAN_ROOT = Path("experiments/dt_mtm/walker2d_medium_clean")
POISON_ROOT = Path("experiments/dt_mtm_stress/walker2d_medium")

GROUP4D = Path(
    "experiments/dt_mtm_stress/group4d_stress_comparison.json"
)

CONDITIONS = ("canonical", "s2_overlap_r0")
RHOS = (("001", "0.01"), ("005", "0.05"))
SEEDS = (0, 1, 2)


def read_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(path)

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def summarize_run(run_dir):
    train = [
        r for r in read_jsonl(run_dir / "training_metrics.jsonl")
        if r.get("type") == "train"
        and int(r.get("step", -1)) >= 10000
    ]

    probe = read_jsonl(run_dir / "clean_probe.jsonl")

    if not train:
        raise RuntimeError(f"No post-warmup records: {run_dir}")
    if not probe:
        raise RuntimeError(f"No clean probe records: {run_dir}")

    cosine = [
        float(r["shared_grad_cosine"])
        for r in train
        if r.get("shared_grad_cosine") is not None
        and np.isfinite(float(r["shared_grad_cosine"]))
    ]

    ratios = []
    zero_mtm = 0
    diag_count = 0

    for r in train:
        dt = r.get("shared_dt_grad_norm")
        mtm = r.get("shared_mtm_scaled_grad_norm")

        if dt is None or mtm is None:
            continue

        dt = float(dt)
        mtm = float(mtm)

        if not np.isfinite(dt) or not np.isfinite(mtm):
            continue

        diag_count += 1

        if mtm == 0.0:
            zero_mtm += 1

        if dt > 0.0:
            ratios.append(mtm / dt)

    last_probe = max(probe, key=lambda r: int(r["step"]))

    return {
        "median_cosine": median(cosine) if cosine else None,
        "negative_cosine_fraction": (
            sum(x < 0.0 for x in cosine) / len(cosine)
            if cosine else None
        ),
        "zero_shared_mtm_fraction": (
            zero_mtm / diag_count if diag_count else None
        ),
        "median_scaled_mtm_to_dt_grad_ratio": (
            median(ratios) if ratios else None
        ),
        "median_dt_loss": median(float(r["dt_loss"]) for r in train),
        "median_mtm_loss": median(float(r["mtm_loss"]) for r in train),
        "final_probe_step": int(last_probe["step"]),
        "final_probe_dt_action_mse": float(last_probe["dt_action_mse"]),
        "final_probe_mtm_total_loss": float(last_probe["mtm_total_loss"]),
    }


def difference(poison, clean):
    out = {}
    for key in (
        "median_cosine",
        "negative_cosine_fraction",
        "zero_shared_mtm_fraction",
        "median_scaled_mtm_to_dt_grad_ratio",
        "median_dt_loss",
        "median_mtm_loss",
        "final_probe_dt_action_mse",
        "final_probe_mtm_total_loss",
    ):
        a = poison[key]
        b = clean[key]
        out[key + "_delta"] = (
            None if a is None or b is None else a - b
        )
    return out


def main():
    group4d = json.loads(GROUP4D.read_text(encoding="utf-8"))

    clean = {}

    for seed in SEEDS:
        run = CLEAN_ROOT / f"seed_{seed}" / "lambda_1"
        clean[seed] = summarize_run(run)

    rows = []

    for condition in CONDITIONS:
        for rho_slug, rho_key in RHOS:
            performance = group4d["conditions"][condition][rho_key]

            by_seed = {
                int(x["seed"]): x
                for x in performance["rows"]
            }

            for seed in SEEDS:
                run = (
                    POISON_ROOT
                    / condition
                    / f"rho_{rho_slug}"
                    / f"attack_seed_{seed}"
                    / f"train_seed_{seed}"
                )

                poison = summarize_run(run)
                delta = difference(poison, clean[seed])

                row = {
                    "condition": condition,
                    "rho": float(rho_key),
                    "seed": seed,
                    "G": float(
                        by_seed[seed]["stress_response_gap"]
                    ),
                    "clean": clean[seed],
                    "poison": poison,
                    "poison_minus_clean": delta,
                }

                rows.append(row)

    rows.sort(key=lambda r: r["G"])

    print("=" * 120)
    print("GROUP 4E — MECHANISM SCREEN, SORTED BY G")
    print("=" * 120)

    for r in rows:
        d = r["poison_minus_clean"]

        print(
            f"{r['condition']:14s} "
            f"rho={r['rho']:.2f} "
            f"seed={r['seed']} | "
            f"G={r['G']:+8.4f} | "
            f"dCos={d['median_cosine_delta']:+.4f} | "
            f"dNegCos={d['negative_cosine_fraction_delta']:+.4f} | "
            f"dGradRatio={d['median_scaled_mtm_to_dt_grad_ratio_delta']:+.4f} | "
            f"dProbeDT={d['final_probe_dt_action_mse_delta']:+.6f} | "
            f"dProbeMTM={d['final_probe_mtm_total_loss_delta']:+.6f}"
        )

    output = {
        "analysis": "group4e_mechanism_screen",
        "rows": rows,
    }

    out = Path(
        "experiments/dt_mtm_stress/group4e_mechanism_screen.json"
    )

    out.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print("output ->", out)


if __name__ == "__main__":
    main()
