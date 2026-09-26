from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


CLEAN_PATH = Path(
    "data/raw/walker2d-medium-v2/walker2d_medium-v2.hdf5"
)

CONDITIONS = {
    "canonical": Path("data/poisoned/csdpc/walker2d-medium-v2"),
    "s2_overlap_r0": Path(
        "data/poisoned/csdpc_s2_overlap_r0_v1/walker2d-medium-v2"
    ),
}
RHO_SLUGS = ("001", "005")
ATTACK_SEEDS = (0, 1, 2)
REQUIRED_KEYS = (
    "observations",
    "actions",
    "rewards",
    "terminals",
    "timeouts",
)


@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    sha256: str
    shapes: dict[str, tuple[int, ...]]
    changed_observations: int
    changed_actions: int
    changed_rewards: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_artifacts() -> list[tuple[str, str, int, Path]]:
    rows = []
    for condition, root in CONDITIONS.items():
        for rho_slug in RHO_SLUGS:
            for attack_seed in ATTACK_SEEDS:
                rows.append(
                    (
                        condition,
                        rho_slug,
                        attack_seed,
                        root / f"rho_{rho_slug}_seed_{attack_seed}.hdf5",
                    )
                )
    return rows


def _load_key(handle: h5py.File, key: str) -> np.ndarray:
    if key not in handle:
        raise KeyError(f"missing required key {key!r}")
    return handle[key][:]


def inspect_dataset(path: Path, clean: dict[str, np.ndarray]) -> DatasetInfo:
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as handle:
        arrays = {key: _load_key(handle, key) for key in REQUIRED_KEYS}

    shapes = {key: tuple(value.shape) for key, value in arrays.items()}
    for key in REQUIRED_KEYS:
        if arrays[key].shape != clean[key].shape:
            raise RuntimeError(
                f"shape mismatch for {path} key={key}: "
                f"poison={arrays[key].shape} clean={clean[key].shape}"
            )

    # Poisoning must not alter episode segmentation. This is necessary for the
    # frozen Group-1 trajectory sampler to remain semantically identical.
    for key in ("terminals", "timeouts"):
        if not np.array_equal(arrays[key], clean[key]):
            raise RuntimeError(f"episode-boundary mismatch in {path}: {key}")

    obs_changed = int(np.count_nonzero(arrays["observations"] != clean["observations"]))
    actions_changed = int(np.count_nonzero(arrays["actions"] != clean["actions"]))
    rewards_changed = int(np.count_nonzero(arrays["rewards"] != clean["rewards"]))

    if obs_changed + actions_changed + rewards_changed == 0:
        raise RuntimeError(f"poison artifact is bytewise equivalent on train fields: {path}")

    return DatasetInfo(
        path=path,
        sha256=sha256_file(path),
        shapes=shapes,
        changed_observations=obs_changed,
        changed_actions=actions_changed,
        changed_rewards=rewards_changed,
    )


def main() -> None:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(CLEAN_PATH)

    with h5py.File(CLEAN_PATH, "r") as handle:
        clean = {key: _load_key(handle, key) for key in REQUIRED_KEYS}

    rows = []
    for condition, rho_slug, attack_seed, path in expected_artifacts():
        info = inspect_dataset(path, clean)
        row = {
            "condition": condition,
            "rho": 0.01 if rho_slug == "001" else 0.05,
            "rho_slug": rho_slug,
            "attack_seed": attack_seed,
            "path": str(path),
            "sha256": info.sha256,
            "shapes": {k: list(v) for k, v in info.shapes.items()},
            "changed_observations": info.changed_observations,
            "changed_actions": info.changed_actions,
            "changed_rewards": info.changed_rewards,
        }
        rows.append(row)
        print(
            f"PASS {condition:14s} rho={row['rho']:.2f} attack_seed={attack_seed} "
            f"obs_changed={info.changed_observations} "
            f"act_changed={info.changed_actions} reward_changed={info.changed_rewards}"
        )

    output = Path("experiments/dt_stress/preflight_artifacts.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "clean_path": str(CLEAN_PATH),
                "clean_sha256": sha256_file(CLEAN_PATH),
                "artifacts": rows,
                "status": "pass",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print(f"GROUP 4C DATASET PREFLIGHT: PASS ({len(rows)} poisoned artifacts)")
    print(f"manifest -> {output}")


if __name__ == "__main__":
    main()
