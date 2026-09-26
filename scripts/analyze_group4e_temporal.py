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

# Post-warmup temporal bins.
# [10000,20000), ..., [90000,100001)
BINS = [
    (10000, 20000),
    (20000, 30000),
    (30000, 40000),
    (40000, 50000),
    (50000, 60000),
    (60000, 70000),
    (70000, 80000),
    (80000, 90000),
    (90000, 100001),
]


def read_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def find_clean_run(seed: int) -> Path:
    candidates = []

    for metrics_path in CLEAN_ROOT.rglob("training_metrics.jsonl"):
        run_dir = metrics_path.parent
        summary_path = run_dir / "summary.json"

        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        if int(summary.get("seed", -1)) != seed:
            continue

        if abs(float(summary.get("lambda_mtm", -1.0)) - 1.0) > 1e-12:
            continue

        if (
            summary.get("status") == "complete"
            and int(summary.get("final_step", -1)) == 100000
        ):
            candidates.append(run_dir)

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one clean lambda=1 run for seed {seed}; "
            f"found {candidates}"
        )

    return candidates[0]


def safe_median(values):
    values = [
        float(x)
        for x in values
        if x is not None and np.isfinite(float(x))
    ]
    return median(values) if values else None


def summarize_bin(train, probes, lo: int, hi: int):
    train_bin = [
        r for r in train
        if r.get("type") == "train"
        and lo <= int(r["step"]) < hi
    ]

    probe_bin = [
        r for r in probes
        if lo <= int(r["step"]) < hi
    ]

    cosines = [
        float(r["shared_grad_cosine"])
        for r in train_bin
        if r.get("shared_grad_cosine") is not None
        and np.isfinite(float(r["shared_grad_cosine"]))
    ]

    ratios = []
    zero_mtm = 0
    diag_count = 0

    for r in train_bin:
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

    # Use the latest fixed clean-probe observation inside each bin.
    last_probe = (
        max(probe_bin, key=lambda r: int(r["step"]))
        if probe_bin else None
    )

    return {
        "bin_start": lo,
        "bin_end_exclusive": hi,
        "n_train_records": len(train_bin),
        "n_defined_cosines": len(cosines),

        "median_cosine": safe_median(cosines),

        "negative_cosine_fraction": (
            sum(x < 0.0 for x in cosines) / len(cosines)
            if cosines else None
        ),

        "zero_shared_mtm_fraction": (
            zero_mtm / diag_count
            if diag_count else None
        ),

        "median_grad_ratio": safe_median(ratios),

        "median_dt_loss": safe_median(
            r["dt_loss"] for r in train_bin
        ),

        "median_mtm_loss": safe_median(
            r["mtm_loss"] for r in train_bin
        ),

        "probe_step": (
            int(last_probe["step"])
            if last_probe else None
        ),

        "probe_dt_mse": (
            float(last_probe["dt_action_mse"])
            if last_probe else None
        ),

        "probe_mtm_loss": (
            float(last_probe["mtm_total_loss"])
            if last_probe else None
        ),
    }


def summarize_temporal(run_dir: Path):
    train = read_jsonl(run_dir / "training_metrics.jsonl")
    probes = read_jsonl(run_dir / "clean_probe.jsonl")

    return [
        summarize_bin(train, probes, lo, hi)
        for lo, hi in BINS
    ]


def subtract(poison, clean):
    fields = (
        "median_cosine",
        "negative_cosine_fraction",
        "zero_shared_mtm_fraction",
        "median_grad_ratio",
        "median_dt_loss",
        "median_mtm_loss",
        "probe_dt_mse",
        "probe_mtm_loss",
    )

    result = {
        "bin_start": poison["bin_start"],
        "bin_end_exclusive": poison["bin_end_exclusive"],
    }

    for field in fields:
        a = poison[field]
        b = clean[field]

        result["d_" + field] = (
            None
            if a is None or b is None
            else float(a - b)
        )

    return result


def fmt(x, digits=4):
    if x is None:
        return "   n/a"
    return f"{x:+.{digits}f}"


def main():
    group4d = json.loads(GROUP4D.read_text(encoding="utf-8"))

    clean_temporal = {}

    for seed in SEEDS:
        clean_run = find_clean_run(seed)
        print(f"clean seed {seed} -> {clean_run}")
        clean_temporal[seed] = summarize_temporal(clean_run)

    rows = []

    for condition in CONDITIONS:
        for rho_slug, rho_key in RHOS:
            perf_rows = {
                int(r["seed"]): r
                for r in group4d["conditions"][condition][rho_key]["rows"]
            }

            for seed in SEEDS:
                poison_run = (
                    POISON_ROOT
                    / condition
                    / f"rho_{rho_slug}"
                    / f"attack_seed_{seed}"
                    / f"train_seed_{seed}"
                )

                poison_temporal = summarize_temporal(poison_run)

                delta_temporal = [
                    subtract(p, c)
                    for p, c in zip(
                        poison_temporal,
                        clean_temporal[seed],
                    )
                ]

                rows.append(
                    {
                        "condition": condition,
                        "rho": float(rho_key),
                        "seed": seed,
                        "G": float(
                            perf_rows[seed]["stress_response_gap"]
                        ),
                        "bins": delta_temporal,
                    }
                )

    rows.sort(key=lambda x: x["G"])

    print()
    print("=" * 125)
    print("GROUP 4E TEMPORAL MECHANISM ANALYSIS")
    print("=" * 125)

    # Print all 12 runs.
    for row in rows:
        print()
        print(
            f"{row['condition']} rho={row['rho']:.2f} "
            f"seed={row['seed']} | G={row['G']:+.4f}"
        )

        print(
            "bin      "
            " dCos     dNegCos  dGradRatio "
            " dDTLoss   dMTMLoss  dProbeDT   dProbeMTM"
        )

        for b in row["bins"]:
            lo = b["bin_start"] // 1000
            hi = b["bin_end_exclusive"] // 1000

            print(
                f"{lo:02d}-{hi:02d}k "
                f"{fmt(b['d_median_cosine']):>9} "
                f"{fmt(b['d_negative_cosine_fraction']):>9} "
                f"{fmt(b['d_median_grad_ratio']):>11} "
                f"{fmt(b['d_median_dt_loss'], 5):>10} "
                f"{fmt(b['d_median_mtm_loss'], 5):>10} "
                f"{fmt(b['d_probe_dt_mse'], 6):>10} "
                f"{fmt(b['d_probe_mtm_loss'], 6):>11}"
            )

    # Exploratory extreme-G contrast.
    negative_extreme = rows[:3]
    positive_extreme = rows[-3:]

    def group_bin_mean(group, field, index):
        vals = [
            r["bins"][index][field]
            for r in group
            if r["bins"][index][field] is not None
        ]
        return float(np.mean(vals)) if vals else None

    fields = (
        "d_median_cosine",
        "d_negative_cosine_fraction",
        "d_median_grad_ratio",
        "d_median_dt_loss",
        "d_median_mtm_loss",
        "d_probe_dt_mse",
        "d_probe_mtm_loss",
    )

    contrast = []

    print()
    print("=" * 125)
    print("EXPLORATORY EXTREME-G CONTRAST")
    print("=" * 125)

    print("Most negative G:")
    for r in negative_extreme:
        print(
            f"  {r['condition']} rho={r['rho']:.2f} "
            f"seed={r['seed']} G={r['G']:+.4f}"
        )

    print("Most positive G:")
    for r in positive_extreme:
        print(
            f"  {r['condition']} rho={r['rho']:.2f} "
            f"seed={r['seed']} G={r['G']:+.4f}"
        )

    print()
    print(
        "bin       dGradRatio_bad-good  "
        "dNegCos_bad-good  dProbeDT_bad-good"
    )

    for i, (lo, hi) in enumerate(BINS):
        bad = {
            f: group_bin_mean(negative_extreme, f, i)
            for f in fields
        }

        good = {
            f: group_bin_mean(positive_extreme, f, i)
            for f in fields
        }

        diff = {
            f: (
                None
                if bad[f] is None or good[f] is None
                else bad[f] - good[f]
            )
            for f in fields
        }

        contrast.append(
            {
                "bin_start": lo,
                "bin_end_exclusive": hi,
                "negative_G_mean": bad,
                "positive_G_mean": good,
                "negative_minus_positive": diff,
            }
        )

        print(
            f"{lo//1000:02d}-{hi//1000:02d}k "
            f"{fmt(diff['d_median_grad_ratio']):>19} "
            f"{fmt(diff['d_negative_cosine_fraction']):>18} "
            f"{fmt(diff['d_probe_dt_mse'], 6):>19}"
        )

    output = {
        "analysis": "group4e_temporal_mechanisms",
        "note": (
            "Extreme-G comparison is exploratory and selected after "
            "Group 4D outcomes were observed."
        ),
        "runs": rows,
        "extreme_contrast": contrast,
    }

    out = Path(
        "experiments/dt_mtm_stress/"
        "group4e_temporal_mechanisms.json"
    )

    out.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print("output ->", out)


if __name__ == "__main__":
    main()
