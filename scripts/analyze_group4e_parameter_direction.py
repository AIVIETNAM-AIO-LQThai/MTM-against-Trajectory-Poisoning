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

RHOS = (("001", "0.01"), ("005", "0.05"))
SEEDS = (0, 1, 2)
STEP = 100000


def load_state(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    return ckpt["model_state_dict"]


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
            f"Expected exactly one completed clean run "
            f"for seed={seed}; found {candidates}"
        )

    return candidates[0]


def groups_for_key(key: str):
    groups = []

    shared = (
        key.startswith("dt.embed_state.")
        or key.startswith("dt.embed_action.")
    )

    if shared:
        groups.append("shared_embeddings")

    if key.startswith("dt."):
        groups.append("whole_dt")

        if not shared:
            groups.append("dt_backbone_nonshared")

    if (
        key.startswith("state_bridge.")
        or key.startswith("action_bridge.")
    ):
        groups.append("bridges")

    if key.startswith("mtm."):
        groups.append("mtm")

    return groups


def compare_directions(clean_sd, can_sd, s2_sd):
    if set(clean_sd) != set(can_sd) or set(clean_sd) != set(s2_sd):
        raise RuntimeError("State-dict keys do not match.")

    names = (
        "shared_embeddings",
        "dt_backbone_nonshared",
        "whole_dt",
        "bridges",
        "mtm",
    )

    acc = {
        name: {
            "dot": 0.0,
            "can_sq": 0.0,
            "s2_sq": 0.0,
            "between_sq": 0.0,
            "count": 0,
        }
        for name in names
    }

    for key, clean_value in clean_sd.items():
        if not torch.is_tensor(clean_value):
            continue

        if not torch.is_floating_point(clean_value):
            continue

        relevant = groups_for_key(key)
        if not relevant:
            continue

        clean = clean_value.detach().to(torch.float64)
        canonical = can_sd[key].detach().to(torch.float64)
        s2 = s2_sd[key].detach().to(torch.float64)

        dc = canonical - clean
        ds = s2 - clean
        between = canonical - s2

        dot = float(torch.sum(dc * ds))
        can_sq = float(torch.sum(dc * dc))
        s2_sq = float(torch.sum(ds * ds))
        between_sq = float(torch.sum(between * between))
        count = int(clean.numel())

        for name in relevant:
            a = acc[name]
            a["dot"] += dot
            a["can_sq"] += can_sq
            a["s2_sq"] += s2_sq
            a["between_sq"] += between_sq
            a["count"] += count

    out = {}

    for name, a in acc.items():
        can_norm = math.sqrt(a["can_sq"])
        s2_norm = math.sqrt(a["s2_sq"])
        between_norm = math.sqrt(a["between_sq"])

        denom = can_norm * s2_norm

        cosine = (
            a["dot"] / denom
            if denom > 0.0 else None
        )

        if cosine is not None:
            cosine = max(-1.0, min(1.0, cosine))
            angle = math.degrees(math.acos(cosine))
        else:
            angle = None

        mean_displacement = 0.5 * (can_norm + s2_norm)

        out[name] = {
            "parameter_count": a["count"],
            "cosine": cosine,
            "angle_degrees": angle,
            "canonical_displacement_norm": can_norm,
            "s2_displacement_norm": s2_norm,
            "canonical_to_s2_norm_ratio": (
                can_norm / (s2_norm + 1e-12)
            ),
            "canonical_s2_direct_distance": between_norm,
            "direct_distance_over_mean_displacement": (
                between_norm / (mean_displacement + 1e-12)
            ),
        }

    return out


def main():
    perf = json.loads(
        GROUP4D.read_text(encoding="utf-8")
    )

    clean_runs = {
        seed: find_clean_run(seed)
        for seed in SEEDS
    }

    clean_states = {}

    for seed, run in clean_runs.items():
        path = (
            run
            / "checkpoints"
            / f"joint_step_{STEP:06d}.pt"
        )

        print(f"clean seed {seed} -> {path}")
        clean_states[seed] = load_state(path)

    rows = []

    for rho_slug, rho_key in RHOS:
        can_perf = {
            int(r["seed"]): r
            for r in perf["conditions"][
                "canonical"
            ][rho_key]["rows"]
        }

        s2_perf = {
            int(r["seed"]): r
            for r in perf["conditions"][
                "s2_overlap_r0"
            ][rho_key]["rows"]
        }

        for seed in SEEDS:
            can_run = (
                POISON_ROOT
                / "canonical"
                / f"rho_{rho_slug}"
                / f"attack_seed_{seed}"
                / f"train_seed_{seed}"
            )

            s2_run = (
                POISON_ROOT
                / "s2_overlap_r0"
                / f"rho_{rho_slug}"
                / f"attack_seed_{seed}"
                / f"train_seed_{seed}"
            )

            can_path = (
                can_run
                / "checkpoints"
                / f"joint_step_{STEP:06d}.pt"
            )

            s2_path = (
                s2_run
                / "checkpoints"
                / f"joint_step_{STEP:06d}.pt"
            )

            can_sd = load_state(can_path)
            s2_sd = load_state(s2_path)

            directions = compare_directions(
                clean_states[seed],
                can_sd,
                s2_sd,
            )

            g_can = float(
                can_perf[seed]["stress_response_gap"]
            )

            g_s2 = float(
                s2_perf[seed]["stress_response_gap"]
            )

            rows.append(
                {
                    "rho": float(rho_key),
                    "seed": seed,
                    "G_canonical": g_can,
                    "G_s2": g_s2,
                    "G_difference_s2_minus_canonical": (
                        g_s2 - g_can
                    ),
                    "absolute_G_difference": abs(
                        g_s2 - g_can
                    ),
                    "groups": directions,
                }
            )

    print()
    print("=" * 125)
    print(
        "GROUP 4E — CANONICAL vs S2 PARAMETER-DISPLACEMENT DIRECTION"
    )
    print("=" * 125)

    print(
        "rho seed    G_can     G_s2   |dG|"
        "    sharedCos     DTcos"
        "   bridgeCos     MTMcos"
    )

    for r in rows:
        g = r["groups"]

        print(
            f"{r['rho']:.2f} "
            f"{r['seed']:4d} "
            f"{r['G_canonical']:+8.4f} "
            f"{r['G_s2']:+8.4f} "
            f"{r['absolute_G_difference']:7.4f} "
            f"{g['shared_embeddings']['cosine']:+12.6f} "
            f"{g['whole_dt']['cosine']:+9.6f} "
            f"{g['bridges']['cosine']:+11.6f} "
            f"{g['mtm']['cosine']:+10.6f}"
        )

    print()
    print("Descriptive correlation:")
    print(
        "corr(|G_s2-G_can|, 1-cosine)"
    )

    y = np.asarray(
        [r["absolute_G_difference"] for r in rows],
        dtype=np.float64,
    )

    for group in (
        "shared_embeddings",
        "whole_dt",
        "bridges",
        "mtm",
    ):
        x = np.asarray(
            [
                1.0 - r["groups"][group]["cosine"]
                for r in rows
            ],
            dtype=np.float64,
        )

        corr = float(np.corrcoef(y, x)[0, 1])

        print(f"  {group:20s}: {corr:+.4f}")

    output = {
        "analysis": "group4e_parameter_direction",
        "note": (
            "Exploratory same-rho same-seed comparison "
            "selected after Group-4D outcomes."
        ),
        "rows": rows,
    }

    out = Path(
        "experiments/dt_mtm_stress/"
        "group4e_parameter_direction.json"
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
