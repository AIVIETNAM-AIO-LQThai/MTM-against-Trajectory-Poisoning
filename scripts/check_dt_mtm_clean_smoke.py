from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-dt-probe-ratio", type=float, default=1.5)
    parser.add_argument("--max-mtm-probe-ratio", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summary.exists():
        raise FileNotFoundError(args.summary)

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    initial = payload["initial_probe"]
    final = payload["final_probe"]

    initial_dt = float(initial["dt_action_mse"])
    final_dt = float(final["dt_action_mse"])
    initial_mtm = float(initial["mtm_total_loss"])
    final_mtm = float(final["mtm_total_loss"])

    values = (initial_dt, final_dt, initial_mtm, final_mtm)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise RuntimeError("clean probe contains non-finite/negative losses")

    dt_ratio = final_dt / max(initial_dt, 1e-12)
    mtm_ratio = final_mtm / max(initial_mtm, 1e-12)

    print("GROUP 4B CLEAN SMOKE CHECK")
    print("final_step:", payload["final_step"])
    print(f"DT probe ratio:  {dt_ratio:.6f}")
    print(f"MTM probe ratio: {mtm_ratio:.6f}")
    print(f"DT allowed max:  {args.max_dt_probe_ratio:.6f}")
    print(f"MTM allowed max: {args.max_mtm_probe_ratio:.6f}")

    failures = []
    if dt_ratio > args.max_dt_probe_ratio:
        failures.append(
            "fixed clean DT action-MSE probe inflated beyond the predeclared "
            "catastrophic-smoke threshold"
        )
    if mtm_ratio > args.max_mtm_probe_ratio:
        failures.append(
            "fixed clean MTM reconstruction probe inflated beyond the "
            "predeclared catastrophic-smoke threshold"
        )

    if failures:
        print("SMOKE: FAIL")
        for failure in failures:
            print(" -", failure)
        raise SystemExit(2)

    print("SMOKE: PASS")
    print(
        "This is only a no-catastrophic-collapse smoke gate; it is NOT the "
        "final clean-policy compatibility gate."
    )


if __name__ == "__main__":
    main()
