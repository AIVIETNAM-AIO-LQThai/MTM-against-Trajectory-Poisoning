from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch


CLEAN_ROOT = Path("experiments/dt_mtm/walker2d_medium_clean")
POISON_ROOT = Path("experiments/dt_mtm_stress/walker2d_medium")

GROUP4D = Path(
    "experiments/dt_mtm_stress/group4d_stress_comparison.json"
)

CONDITIONS = ("canonical", "s2_overlap_r0")
RHOS = (("001", "0.01"), ("005", "0.05"))
SEEDS = (0, 1, 2)
# Intermediate clean checkpoints were removed during storage cleanup.
# Final-state parameter divergence remains available.
STEPS = (100000,)


def load_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_clean_run(seed: int) -> Path:
    candidates = []

    for summary_path in CLEAN_ROOT.rglob("summary.json"):
        try:
            summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
        except Exception:
            continue

        if int(summary.get("seed", -1)) != seed:
            continue

        if abs(
            float(summary.get("lambda_mtm", -1.0)) - 1.0
        ) > 1e-12:
            continue

        if (
            summary.get("status") == "complete"
            and int(summary.get("final_step", -1)) == 100000
        ):
            candidates.append(summary_path.parent)

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one clean seed={seed} run; "
            f"found {candidates}"
        )

    return candidates[0]


def group_for_key(key: str):
    if (
        key.startswith("dt.embed_state.")
        or key.startswith("dt.embed_action.")
    ):
        return "shared_embeddings"

    if key.startswith("dt."):
        return "dt_backbone"

    if (
        key.startswith("state_bridge.")
        or key.startswith("action_bridge.")
    ):
        return "bridges"

    if key.startswith("mtm."):
        return "mtm"

    return "other"


def accumulate(clean_sd, poison_sd):
    groups = {
        "shared_embeddings": [],
        "dt_backbone": [],
        "bridges": [],
        "mtm": [],
        "whole_dt": [],
    }

    if set(clean_sd) != set(poison_sd):
        missing_clean = sorted(set(poison_sd) - set(clean_sd))
        missing_poison = sorted(set(clean_sd) - set(poison_sd))

        raise RuntimeError(
            "State-dict key mismatch.\n"
            f"missing clean={missing_clean[:10]}\n"
            f"missing poison={missing_poison[:10]}"
        )

    for key in clean_sd:
        clean = clean_sd[key]
        poison = poison_sd[key]

        if not torch.is_tensor(clean):
            continue

        if not torch.is_floating_point(clean):
            continue

        c = clean.detach().to(torch.float64).reshape(-1)
        p = poison.detach().to(torch.float64).reshape(-1)

        item = (c, p)

        group = group_for_key(key)

        if group in groups:
            groups[group].append(item)

        if key.startswith("dt."):
            groups["whole_dt"].append(item)

    output = {}

    for name, items in groups.items():
        if not items:
            output[name] = None
            continue

        diff_sq = 0.0
        clean_sq = 0.0
        count = 0

        for clean, poison in items:
            diff = poison - clean

            diff_sq += float(torch.sum(diff * diff))
            clean_sq += float(torch.sum(clean * clean))
            count += int(clean.numel())

        l2 = math.sqrt(diff_sq)
        clean_l2 = math.sqrt(clean_sq)

        output[name] = {
            "parameter_count": count,
            "l2": l2,
            "rms": l2 / math.sqrt(count),
            "relative_l2": l2 / (clean_l2 + 1e-12),
        }

    return output


def main():
    group4d = json.loads(
        GROUP4D.read_text(encoding="utf-8")
    )

    clean_runs = {
        seed: find_clean_run(seed)
        for seed in SEEDS
    }

    for seed, path in clean_runs.items():
        print(f"clean seed {seed} -> {path}")

    rows = []

    for condition in CONDITIONS:
        for rho_slug, rho_key in RHOS:

            perf = {
                int(r["seed"]): r
                for r in group4d[
                    "conditions"
                ][condition][rho_key]["rows"]
            }

            for seed in SEEDS:

                poison_run = (
                    POISON_ROOT
                    / condition
                    / f"rho_{rho_slug}"
                    / f"attack_seed_{seed}"
                    / f"train_seed_{seed}"
                )

                temporal = []

                for step in STEPS:
                    clean_ckpt = (
                        clean_runs[seed]
                        / "checkpoints"
                        / f"joint_step_{step:06d}.pt"
                    )

                    poison_ckpt = (
                        poison_run
                        / "checkpoints"
                        / f"joint_step_{step:06d}.pt"
                    )

                    clean = load_checkpoint(clean_ckpt)
                    poison = load_checkpoint(poison_ckpt)

                    clean_sd = clean["model_state_dict"]
                    poison_sd = poison["model_state_dict"]

                    metrics = accumulate(
                        clean_sd,
                        poison_sd,
                    )

                    temporal.append(
                        {
                            "step": step,
                            "groups": metrics,
                        }
                    )

                rows.append(
                    {
                        "condition": condition,
                        "rho": float(rho_key),
                        "seed": seed,
                        "G": float(
                            perf[seed][
                                "stress_response_gap"
                            ]
                        ),
                        "temporal": temporal,
                    }
                )

    rows.sort(key=lambda x: x["G"])

    print()
    print("=" * 115)
    print("GROUP 4E PARAMETER DIVERGENCE — FINAL 100K")
    print("=" * 115)

    print(
        "condition       rho seed       G"
        "     shared_rel    DT_rel"
        "    bridge_rel     MTM_rel"
    )

    for row in rows:
        final = row["temporal"][-1]["groups"]

        print(
            f"{row['condition']:15s} "
            f"{row['rho']:.2f} "
            f"{row['seed']:4d} "
            f"{row['G']:+8.4f} "
            f"{final['shared_embeddings']['relative_l2']:12.6f} "
            f"{final['whole_dt']['relative_l2']:9.6f} "
            f"{final['bridges']['relative_l2']:11.6f} "
            f"{final['mtm']['relative_l2']:11.6f}"
        )

    # Simple descriptive correlations across the 12 runs.
    G = np.asarray(
        [r["G"] for r in rows],
        dtype=np.float64,
    )

    print()
    print("Pearson correlation with G at 100K:")

    for group in (
        "shared_embeddings",
        "whole_dt",
        "bridges",
        "mtm",
    ):
        values = np.asarray(
            [
                r["temporal"][-1]["groups"][group][
                    "relative_l2"
                ]
                for r in rows
            ],
            dtype=np.float64,
        )

        corr = float(np.corrcoef(G, values)[0, 1])

        print(
            f"  {group:20s}: {corr:+.4f}"
        )

    output = {
        "analysis": "group4e_parameter_divergence",
        "note": (
            "Exploratory analysis selected after "
            "Group-4D outcomes."
        ),
        "rows": rows,
    }

    out = Path(
        "experiments/dt_mtm_stress/"
        "group4e_parameter_divergence.json"
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
