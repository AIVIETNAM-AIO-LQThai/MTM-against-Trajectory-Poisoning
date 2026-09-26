from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analyze_group4e_clean_policy_sensitivity import (
    CLEAN_DATA,
    ROOTS,
    RHOS,
    SEEDS,
    load_dataset,
    build_metadata,
)

from scripts.e1b_min_overlap_core import (
    REPLICATES,
    build_min_overlap_control,
    validate_min_overlap_control,
)


OUTPUT = Path(
    "experiments/postfinal_controls/"
    "e1b_min_overlap_preflight.json"
)


def main():
    clean = load_dataset(
        CLEAN_DATA
    )

    meta = build_metadata(
        clean
    )

    rows = []

    print()
    print("=" * 120)
    print(
        "E1B MINIMUM-OVERLAP "
        "CONSTRUCTION PREFLIGHT"
    )
    print("=" * 120)

    print(
        "condition       rho seed rep "
        "overlap moved runs delta_error"
    )

    for seed in SEEDS:
        for condition_index, (
            condition,
            root,
        ) in enumerate(
            ROOTS.items()
        ):
            for (
                rho_slug,
                rho_key,
            ) in RHOS:
                rho = float(
                    rho_key
                )

                poison = load_dataset(
                    root
                    / (
                        f"rho_{rho_slug}_"
                        f"seed_{seed}.hdf5"
                    )
                )

                artifact_seed = (
                    condition_index
                    * 100
                    + int(
                        rho * 1000
                    )
                    + seed
                )

                for replicate in REPLICATES:
                    control, records = (
                        build_min_overlap_control(
                            clean,
                            poison,
                            meta["used_n"],
                            replicate=replicate,
                            artifact_seed=artifact_seed,
                        )
                    )

                    invariants = (
                        validate_min_overlap_control(
                            clean,
                            poison,
                            control,
                            records,
                            meta["used_n"],
                        )
                    )

                    row = {
                        "condition": condition,
                        "rho": rho,
                        "seed": int(
                            seed
                        ),
                        "replicate": int(
                            replicate
                        ),
                        "invariants": (
                            invariants
                        ),
                    }

                    rows.append(row)

                    print(
                        f"{condition:15s} "
                        f"{rho:.2f} "
                        f"{seed:4d} "
                        f"{replicate:3d} "
                        f"{invariants['source_target_overlap_fraction']:7.4f} "
                        f"{invariants['moved_away_fraction']:7.4f} "
                        f"{invariants['total_run_count']:5d} "
                        f"{invariants['max_recovered_delta_error']:.3e}"
                    )

    if len(rows) != 60:
        raise RuntimeError(
            f"Expected 60 preflight rows, got {len(rows)}."
        )

    overlaps = np.asarray(
        [
            r["invariants"][
                "source_target_overlap_fraction"
            ]
            for r in rows
        ],
        dtype=np.float64,
    )

    moved_runs = np.asarray(
        [
            r["invariants"][
                "moved_run_fraction"
            ]
            for r in rows
        ],
        dtype=np.float64,
    )

    summary = {
        "rows": int(
            len(rows)
        ),
        "mean_overlap_fraction": float(
            overlaps.mean()
        ),
        "max_overlap_fraction": float(
            overlaps.max()
        ),
        "mean_moved_away_fraction": float(
            1.0 - overlaps.mean()
        ),
        "mean_moved_run_fraction": float(
            moved_runs.mean()
        ),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            {
                "analysis": (
                    "e1b_min_overlap_preflight"
                ),
                "status": "PASS",
                "rows": rows,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("PREFLIGHT PASS")
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )
    print()
    print("output ->", OUTPUT)


if __name__ == "__main__":
    main()
