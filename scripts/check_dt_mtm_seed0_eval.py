from __future__ import annotations

import argparse
import json
from pathlib import Path


GROUP1_SEED0_MEAN = 69.32201154593943
MIN_FRACTION = 0.80
MIN_NORMALIZED_RETURN = GROUP1_SEED0_MEAN * MIN_FRACTION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    if not args.summary.exists():
        raise FileNotFoundError(args.summary)

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    value = float(payload["normalized_return_mean"])

    print("GROUP 4B SEED-0 CLEAN POLICY SCREEN")
    print(f"joint normalized-return mean: {value:.6f}")
    print(f"Group-1 seed-0 baseline:      {GROUP1_SEED0_MEAN:.6f}")
    print(f"predeclared 80% threshold:    {MIN_NORMALIZED_RETURN:.6f}")

    if value < MIN_NORMALIZED_RETURN:
        print("SEED-0 SCREEN: FAIL — severe clean collapse")
        raise SystemExit(2)

    print("SEED-0 SCREEN: PASS")
    print("This is not the final three-seed clean compatibility gate.")


if __name__ == "__main__":
    main()
