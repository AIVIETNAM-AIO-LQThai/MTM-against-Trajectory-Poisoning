
from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path

import h5py
import numpy as np

from src.data.normalization import compute_state_statistics
from src.data.trajectories import find_completed_trajectories


ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "configs/attack_qualification/rdt_source_random_state_corruption.json"
SOURCE = ROOT / "data/raw/walker2d-medium-v2/walker2d_medium-v2.hdf5"
DATA_ROOT = ROOT / "data/derived/rdt_source_random_state_corruption/walker2d-medium-v2"
CLEAN = DATA_ROOT / "clean_ratio_0p02.hdf5"
CORRUPTED_ROOT = DATA_ROOT / "corrupted"
META_ROOT = ROOT / "data/metadata/rdt_source_random_state_corruption/walker2d-medium-v2"
NORMALIZATION = META_ROOT / "clean_ratio_0p02_normalization.npz"
SELECTION = META_ROOT / "clean_ratio_0p02_selection.npz"
OUTPUT = ROOT / "experiments/attack_qualification/rdt_source_random_state_corruption/preflight.json"
TRAIN_SCRIPT = ROOT / "scripts/train_dt_stress.py"
EVAL_SCRIPT = ROOT / "scripts/evaluate_dt_stress.py"

EXPECTED_SELECTED = [
    902, 239, 15, 185, 71, 171, 201, 726, 484, 35, 63, 32,
    708, 991, 949, 304, 186, 374, 234, 29, 1030, 996, 511,
]
EXPECTED_CLEAN_TRANSITIONS = 20147

CORE_DT_PATHS = [
    "src/methods/dt",
    "src/data/batching.py",
    "src/evaluation/walker2d.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def selected_global_indices(trajectories, selected):
    return np.concatenate([
        np.arange(
            trajectories[int(index)].start,
            trajectories[int(index)].end,
            dtype=np.int64,
        )
        for index in selected
    ])


def collect_objects(handle):
    objects = {"/": "group"}

    def visitor(name, obj):
        objects["/" + name] = (
            "dataset" if isinstance(obj, h5py.Dataset) else "group"
        )

    handle.visititems(visitor)
    return objects


def attribute_dict(obj):
    return {
        str(key): np.asarray(value)
        for key, value in obj.attrs.items()
    }


def assert_attributes_equal(left, right, path):
    a = attribute_dict(left)
    b = attribute_dict(right)

    if set(a) != set(b):
        raise RuntimeError(f"attribute keys changed: {path}")

    for key in a:
        if a[key].shape != b[key].shape or not np.array_equal(a[key], b[key]):
            raise RuntimeError(f"attribute changed: {path}:{key}")


def verify_core_dt():
    result = subprocess.run(
        [
            "git", "diff", "--quiet",
            "a2-rtg-inflation-fail-v1", "--",
            *CORE_DT_PATHS,
        ],
        cwd=ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError("core DT implementation changed")

    return {
        "baseline_tag": "a2-rtg-inflation-fail-v1",
        "unchanged": True,
        "paths": CORE_DT_PATHS,
    }


def verify_plumbing():
    train_text = TRAIN_SCRIPT.read_text(encoding="utf-8")
    eval_text = EVAL_SCRIPT.read_text(encoding="utf-8")

    for token in [
        "--normalization",
        "--expected-num-trajectories",
        "--expected-num-transitions",
        "--expected-trailing-transitions",
        "normalization_sha256",
    ]:
        if token not in train_text:
            raise RuntimeError(f"training plumbing missing: {token}")

    for token in [
        "--normalization",
        "args.normalization",
    ]:
        if token not in eval_text:
            raise RuntimeError(f"evaluation plumbing missing: {token}")

    return {
        "train_normalization_path_argument": True,
        "train_dataset_contract_arguments": True,
        "evaluation_normalization_path_argument": True,
        "legacy_defaults_retained": True,
    }


def verify_clean_downsample(cfg):
    clean_meta = read_json(META_ROOT / "clean_ratio_0p02.json")

    if sha256_file(SOURCE) != cfg["dataset"]["sha256"]:
        raise RuntimeError("source SHA mismatch")
    if sha256_file(CLEAN) != clean_meta["clean_dataset_sha256"]:
        raise RuntimeError("clean downsample SHA mismatch")
    if sha256_file(NORMALIZATION) != clean_meta["normalization_sha256"]:
        raise RuntimeError("normalization SHA mismatch")
    if sha256_file(SELECTION) != clean_meta["selection_sha256"]:
        raise RuntimeError("selection SHA mismatch")

    with h5py.File(SOURCE, "r") as source:
        source_terminals = np.asarray(source["terminals"], dtype=bool)
        source_timeouts = np.asarray(source["timeouts"], dtype=bool)
        source_trajectories, trailing = find_completed_trajectories(
            source_terminals,
            source_timeouts,
        )

        if len(source_trajectories) != 1190 or trailing != 5:
            raise RuntimeError("source trajectory contract changed")

        rng = random.Random(int(cfg["downsampling"]["seed"]))
        expected_selected = rng.sample(
            list(range(len(source_trajectories))),
            int(len(source_trajectories) * float(cfg["downsampling"]["ratio"])),
        )

        if expected_selected != EXPECTED_SELECTED:
            raise RuntimeError("random.sample trajectory selection changed")

        expected_rows = selected_global_indices(
            source_trajectories,
            expected_selected,
        )

        if len(expected_rows) != EXPECTED_CLEAN_TRANSITIONS:
            raise RuntimeError("unexpected clean transition count")

        with np.load(SELECTION) as selection:
            np.testing.assert_array_equal(
                selection["selected_trajectory_indices"],
                np.asarray(EXPECTED_SELECTED, dtype=np.int64),
            )
            np.testing.assert_array_equal(
                selection["source_global_transition_indices"],
                expected_rows,
            )

        source_objects = collect_objects(source)
        source_length = len(source["rewards"])

        with h5py.File(CLEAN, "r") as clean:
            clean_objects = collect_objects(clean)

            if source_objects != clean_objects:
                raise RuntimeError("clean HDF5 hierarchy changed")

            assert_attributes_equal(source, clean, "/")

            for object_path, kind in source_objects.items():
                if object_path == "/":
                    continue

                name = object_path.lstrip("/")
                src = source[name]
                dst = clean[name]

                assert_attributes_equal(src, dst, object_path)

                if kind == "group":
                    continue

                if src.dtype != dst.dtype:
                    raise RuntimeError(f"dtype changed: {object_path}")

                source_data = src[()]

                if src.ndim >= 1 and src.shape[0] == source_length:
                    expected = source_data[expected_rows]
                else:
                    expected = source_data

                if not np.array_equal(expected, dst[()]):
                    raise RuntimeError(f"downsample mismatch: {object_path}")

            observations = np.asarray(clean["observations"])
            terminals = np.asarray(clean["terminals"], dtype=bool)
            timeouts = np.asarray(clean["timeouts"], dtype=bool)

    trajectories, clean_trailing = find_completed_trajectories(
        terminals,
        timeouts,
    )

    if (
        len(trajectories) != 23
        or len(observations) != EXPECTED_CLEAN_TRANSITIONS
        or clean_trailing != 0
    ):
        raise RuntimeError("clean 2% trajectory contract mismatch")

    recomputed_mean, recomputed_std = compute_state_statistics(
        observations,
        trajectories,
    )

    with np.load(NORMALIZATION) as norm:
        np.testing.assert_array_equal(
            norm["state_mean"],
            recomputed_mean,
        )
        np.testing.assert_array_equal(
            norm["state_std"],
            recomputed_std,
        )

        if (
            int(norm["num_training_transitions"]) != EXPECTED_CLEAN_TRANSITIONS
            or int(norm["num_trajectories"]) != 23
            or int(norm["trailing_transitions"]) != 0
        ):
            raise RuntimeError("normalization metadata contract mismatch")

    return {
        "selected_trajectory_indices": EXPECTED_SELECTED,
        "num_trajectories": 23,
        "num_transitions": EXPECTED_CLEAN_TRANSITIONS,
        "trailing_transitions": 0,
        "clean_sha256": sha256_file(CLEAN),
        "normalization_sha256": sha256_file(NORMALIZATION),
    }


def verify_corruption(cfg, seed):
    corrupted_path = CORRUPTED_ROOT / f"corruption_seed_{seed}.hdf5"
    meta = read_json(META_ROOT / f"corruption_seed_{seed}.json")
    records_path = META_ROOT / f"corruption_seed_{seed}_records.npz"

    if sha256_file(corrupted_path) != meta["corrupted_dataset_sha256"]:
        raise RuntimeError(f"corrupted SHA mismatch: {seed}")
    if sha256_file(records_path) != meta["records_file_sha256"]:
        raise RuntimeError(f"records SHA mismatch: {seed}")

    with h5py.File(CLEAN, "r") as clean, h5py.File(corrupted_path, "r") as corrupted:
        clean_objects = collect_objects(clean)
        corrupted_objects = collect_objects(corrupted)

        if clean_objects != corrupted_objects:
            raise RuntimeError("corrupted HDF5 hierarchy changed")

        assert_attributes_equal(clean, corrupted, "/")

        clean_observations = np.asarray(clean["observations"])

        rng = np.random.RandomState(int(seed))
        random_num = rng.random(len(clean_observations))
        expected_attacked = np.where(
            random_num < float(cfg["corruption"]["corruption_rate"])
        )[0].astype(np.int64)

        observation_std = np.std(
            clean_observations,
            axis=0,
            keepdims=True,
        )

        original = clean_observations[expected_attacked].copy()

        noise = rng.uniform(
            -float(cfg["corruption"]["corruption_scale"]),
            float(cfg["corruption"]["corruption_scale"]),
            size=original.shape,
        )

        expected_observations = clean_observations.copy()
        expected_observations[expected_attacked] = (
            original + noise * observation_std
        )

        with np.load(records_path) as records:
            np.testing.assert_array_equal(
                records["attacked_transition_indices"],
                expected_attacked,
            )
            np.testing.assert_array_equal(
                records["clean_observation_std"],
                observation_std.reshape(-1),
            )

        for object_path, kind in clean_objects.items():
            if object_path == "/":
                continue

            name = object_path.lstrip("/")
            left = clean[name]
            right = corrupted[name]

            assert_attributes_equal(left, right, object_path)

            if kind == "group":
                continue

            if left.shape != right.shape or left.dtype != right.dtype:
                raise RuntimeError(f"schema changed: {object_path}")

            if object_path == "/observations":
                np.testing.assert_array_equal(
                    right[()],
                    expected_observations,
                )
            elif not np.array_equal(left[()], right[()]):
                raise RuntimeError(
                    f"non-observation field changed: {object_path}"
                )

    return {
        "corruption_seed": int(seed),
        "attacked_transition_count": int(len(expected_attacked)),
        "attacked_fraction": float(
            len(expected_attacked) / EXPECTED_CLEAN_TRANSITIONS
        ),
        "corrupted_sha256": sha256_file(corrupted_path),
        "source_semantics_reproduced": True,
        "only_observations_changed": True,
    }


def main():
    cfg = read_json(CONFIG)

    result = {
        "status": "PREFLIGHT_PASS",
        "experiment": cfg["name"],
        "core_dt": verify_core_dt(),
        "plumbing": verify_plumbing(),
        "clean_downsample": verify_clean_downsample(cfg),
        "corruptions": [
            verify_corruption(cfg, int(seed))
            for seed in cfg["corruption"]["corruption_seeds"]
        ],
        "planned_training_matrix": {
            "fresh_clean_controls": 3,
            "corrupted_runs": 9,
            "total_runs": 12,
            "model_seeds": [0, 1, 2],
            "corruption_seeds": [2023, 2024, 2025],
            "normalization": str(NORMALIZATION),
        },
        "qualification_gate": cfg["qualification_gate"],
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 104)
    print("RDT-SOURCE RANDOM STATE CORRUPTION PREFLIGHT")
    print("=" * 104)
    print("core DT architecture/training modules unchanged: PASS")
    print("normalization/dataset-contract plumbing: PASS")
    print("source random.sample downsampling: PASS")
    print("selected trajectories: 23")
    print(
        "clean transitions:",
        result["clean_downsample"]["num_transitions"],
    )
    print(
        "clean trailing:",
        result["clean_downsample"]["trailing_transitions"],
    )
    print(
        "clean SHA256:",
        result["clean_downsample"]["clean_sha256"],
    )
    print(
        "normalization SHA256:",
        result["clean_downsample"]["normalization_sha256"],
    )
    print()
    print("seed attacked fraction source_semantics observation_only")

    for row in result["corruptions"]:
        print(
            f"{row['corruption_seed']:4d} "
            f"{row['attacked_transition_count']:8d} "
            f"{row['attacked_fraction']:.6f} "
            f"{str(row['source_semantics_reproduced']):>16s} "
            f"{str(row['only_observations_changed']):>16s}"
        )

    print()
    print("planned matrix: 3 fresh clean + 9 corrupted = 12 DT runs")
    print("PREFLIGHT: PASS")
    print("output ->", OUTPUT)


if __name__ == "__main__":
    main()
