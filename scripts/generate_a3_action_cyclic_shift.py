from __future__ import annotations

import argparse, hashlib, json, math, shutil
from pathlib import Path

import h5py
import numpy as np

from src.data.trajectories import find_completed_trajectories

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/attack_qualification/a3_action_cyclic_shift.json"
POISON_ROOT = ROOT / "data/poisoned/action_cyclic_shift/walker2d-medium-v2"
META_ROOT = ROOT / "data/metadata/action_cyclic_shift/walker2d-medium-v2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    checks = [
        (c["schema_version"], "a3-action-cyclic-shift-protocol-v1", "schema"),
        (c["status"], "predeclared", "status"),
        (c["stage"], "A3", "stage"),
        (c["name"], "high_return_action_cyclic_shift", "name"),
        (c["attack"]["attack_seeds"], [20, 21, 22], "attack_seeds"),
        (c["attack"]["transition_budget_fraction"], 0.05, "budget"),
        (c["attack"]["candidate_pool"]["trajectory_return_quantile_min"], 0.70, "qmin"),
        (c["attack"]["candidate_pool"]["minimum_trajectory_length"], 4, "min_len"),
        (c["attack"]["selection"]["minimum_budget_utilization"], 0.98, "min_util"),
        (
            c["attack"]["selection"]["procedure"],
            "seeded_shuffle_then_greedy_whole_trajectory_packing",
            "selection_procedure",
        ),
        (c["attack"]["transformation"]["type"], "within_trajectory_cyclic_action_shift", "transform"),
        (c["attack"]["transformation"]["modified_dataset_field"], "actions", "modified_field"),
        (
            c["attack"]["transformation"]["mapping"],
            "poisoned_action[t] = clean_action[(t + k) mod L]",
            "mapping",
        ),
        (
            c["attack"]["transformation"]["offset_rule"]["sampling"],
            "uniform_integer_inclusive",
            "offset_sampling",
        ),
        (
            c["attack"]["transformation"]["offset_rule"]["trajectory_seed_rule"],
            "attack_seed * 1000003 + trajectory_id",
            "offset_seed_rule",
        ),
        (
            c["attack"]["transformation"]["preserve_action_multiset_within_selected_trajectory"],
            True,
            "preserve_action_multiset",
        ),
        (
            c["attack"]["transformation"]["trailing_fragment_unchanged"],
            True,
            "trailing_fragment_unchanged",
        ),
    ]
    for actual, expected, name in checks:
        if actual != expected:
            raise ValueError(f"A3 frozen field changed: {name}: {actual!r} != {expected!r}")
    return c


def trajectory_returns(rewards, trajectories):
    r = np.asarray(rewards)
    return np.asarray(
        [float(np.sum(r[t.start:t.end], dtype=np.float64)) for t in trajectories],
        dtype=np.float64,
    )


def select_trajectories(
    trajectories,
    returns,
    *,
    threshold,
    min_length,
    requested_budget,
    attack_seed,
):
    eligible = np.asarray(
        [
            i for i, (t, ret) in enumerate(zip(trajectories, returns))
            if float(ret) >= threshold and int(t.length) >= min_length
        ],
        dtype=np.int64,
    )
    if len(eligible) == 0:
        raise RuntimeError("empty high-return candidate pool")

    rng = np.random.default_rng(int(attack_seed))
    order = eligible.copy()
    rng.shuffle(order)

    selected, used = [], 0
    for idx in order:
        length = int(trajectories[int(idx)].length)
        if used + length > requested_budget:
            continue
        selected.append(int(idx))
        used += length
        if used == requested_budget:
            break

    return selected, int(used), [int(x) for x in order], [int(x) for x in eligible]


def shift_bounds(length: int):
    if length < 4:
        raise ValueError("A3 requires trajectory length >= 4")
    lo = int(math.ceil(length / 4.0))
    hi = int(math.floor(3.0 * length / 4.0))
    if lo > hi:
        raise RuntimeError("invalid shift interval")
    return lo, hi


def choose_shift_offset(*, length, attack_seed, trajectory_index):
    lo, hi = shift_bounds(int(length))
    seed = int(attack_seed) * 1_000_003 + int(trajectory_index)
    rng = np.random.default_rng(seed)
    return int(rng.integers(lo, hi + 1))


def apply_action_shift(actions, trajectories, selected, *, attack_seed):
    clean = np.asarray(actions)
    poisoned = clean.copy()
    records = []

    for idx in selected:
        t = trajectories[int(idx)]
        start, end, length = int(t.start), int(t.end), int(t.length)
        lo, hi = shift_bounds(length)
        k = choose_shift_offset(
            length=length,
            attack_seed=attack_seed,
            trajectory_index=idx,
        )
        src = clean[start:end]
        dst = np.roll(src, -k, axis=0)
        poisoned[start:end] = dst

        diff = dst.astype(np.float64) - src.astype(np.float64)
        l2 = np.linalg.norm(diff, axis=1)
        records.append(
            {
                "trajectory_index": int(idx),
                "start": start,
                "end": end,
                "length": length,
                "shift_lower_bound": lo,
                "shift_upper_bound": hi,
                "shift_offset": k,
                "trajectory_seed": int(attack_seed) * 1_000_003 + int(idx),
                "changed_transition_count": int(np.count_nonzero(np.any(dst != src, axis=1))),
                "mean_action_l2_change": float(np.mean(l2)),
                "max_action_l2_change": float(np.max(l2)),
            }
        )

    return poisoned, records


def validate(
    clean_actions,
    poisoned_actions,
    trajectories,
    selected,
    *,
    returns,
    threshold,
    min_length,
    used_transitions,
    requested_budget,
    actual_budget,
    min_util,
    attack_seed,
):
    clean = np.asarray(clean_actions)
    poison = np.asarray(poisoned_actions)
    if clean.shape != poison.shape or clean.dtype != poison.dtype:
        raise RuntimeError("action shape/dtype changed")
    if actual_budget > requested_budget:
        raise RuntimeError("budget exceeded")

    util = actual_budget / requested_budget
    if util < min_util:
        raise RuntimeError(f"budget utilization below frozen minimum: {util:.6f}")

    mask = np.zeros(len(clean), dtype=bool)
    changed = 0
    for idx in selected:
        t = trajectories[int(idx)]
        start, end, length = int(t.start), int(t.end), int(t.length)
        if length < min_length:
            raise RuntimeError("selected trajectory too short")
        if float(returns[int(idx)]) < threshold - 1e-10:
            raise RuntimeError("selected trajectory outside Q70 pool")

        k = choose_shift_offset(
            length=length,
            attack_seed=attack_seed,
            trajectory_index=idx,
        )
        expected = np.roll(clean[start:end], -k, axis=0)
        if not np.array_equal(expected, poison[start:end]):
            raise RuntimeError("frozen cyclic mapping violated")
        if not np.array_equal(np.roll(poison[start:end], k, axis=0), clean[start:end]):
            raise RuntimeError("selected action multiset not preserved")

        mask[start:end] = True
        changed += int(np.count_nonzero(np.any(poison[start:end] != clean[start:end], axis=1)))

    if not np.array_equal(clean[~mask], poison[~mask]):
        raise RuntimeError("unselected action changed")
    if not np.array_equal(clean[used_transitions:], poison[used_transitions:]):
        raise RuntimeError("trailing fragment changed")
    if changed == 0:
        raise RuntimeError("no action targets changed")

    return {
        "requested_transition_budget": int(requested_budget),
        "actual_transition_budget": int(actual_budget),
        "budget_utilization": float(util),
        "selected_trajectory_count": int(len(selected)),
        "changed_transition_count": int(changed),
        "changed_fraction_of_selected": float(changed / actual_budget),
        "selected_within_candidate_pool": True,
        "unselected_actions_identical": True,
        "trailing_fragment_unchanged": True,
        "action_multiset_preserved_per_selected_trajectory": True,
    }


def write_action_only(clean_path: Path, poison_path: Path, poisoned_actions):
    poison_path.parent.mkdir(parents=True, exist_ok=True)
    if poison_path.exists():
        raise FileExistsError(poison_path)

    shutil.copy2(clean_path, poison_path)
    try:
        with h5py.File(poison_path, "r+") as h:
            ds = h["actions"]
            arr = np.asarray(poisoned_actions)
            if ds.shape != arr.shape or ds.dtype != arr.dtype:
                raise RuntimeError("HDF5 action shape/dtype mismatch")
            ds[...] = arr
            h.flush()
    except Exception:
        poison_path.unlink(missing_ok=True)
        raise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--poison-root", type=Path, default=POISON_ROOT)
    p.add_argument("--metadata-root", type=Path, default=META_ROOT)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    clean_path = (ROOT / cfg["dataset"]["path"]).resolve()
    clean_sha = sha256_file(clean_path)
    if clean_sha != cfg["dataset"]["sha256"]:
        raise RuntimeError("frozen clean dataset SHA256 mismatch")
    print("Frozen clean dataset SHA256: PASS")

    with h5py.File(clean_path, "r") as h:
        actions = np.asarray(h["actions"])
        rewards = np.asarray(h["rewards"])
        terminals = np.asarray(h["terminals"], dtype=bool)
        timeouts = np.asarray(h["timeouts"], dtype=bool)

    trajectories, trailing = find_completed_trajectories(terminals, timeouts)
    used = int(sum(t.length for t in trajectories))
    dcfg = cfg["dataset"]
    if (
        len(trajectories) != dcfg["completed_trajectories"]
        or used != dcfg["used_transitions"]
        or trailing != dcfg["trailing_transitions"]
    ):
        raise RuntimeError("frozen trajectory contract changed")

    returns = trajectory_returns(rewards, trajectories)
    acfg = cfg["attack"]
    ccfg = acfg["candidate_pool"]
    q70 = float(np.quantile(returns, ccfg["trajectory_return_quantile_min"]))
    requested = int(acfg["transition_budget_fraction"] * used)
    min_len = int(ccfg["minimum_trajectory_length"])
    min_util = float(acfg["selection"]["minimum_budget_utilization"])

    args.poison_root.mkdir(parents=True, exist_ok=True)
    args.metadata_root.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 104)
    print("A3 - HIGH-RETURN ACTION CYCLIC-SHIFT POISONING")
    print("=" * 104)
    print("seed q70 candidates selected budget/request utilization changed_fraction")

    rows = []
    for attack_seed in acfg["attack_seeds"]:
        selected, actual, order, eligible = select_trajectories(
            trajectories,
            returns,
            threshold=q70,
            min_length=min_len,
            requested_budget=requested,
            attack_seed=attack_seed,
        )
        poisoned, records = apply_action_shift(
            actions,
            trajectories,
            selected,
            attack_seed=attack_seed,
        )
        integrity = validate(
            actions,
            poisoned,
            trajectories,
            selected,
            returns=returns,
            threshold=q70,
            min_length=min_len,
            used_transitions=used,
            requested_budget=requested,
            actual_budget=actual,
            min_util=min_util,
            attack_seed=attack_seed,
        )

        poison_path = args.poison_root / f"attack_seed_{attack_seed}.hdf5"
        meta_path = args.metadata_root / f"attack_seed_{attack_seed}.json"
        write_action_only(clean_path, poison_path, poisoned)
        poison_sha = sha256_file(poison_path)

        meta = {
            "schema_version": "a3-action-cyclic-shift-artifact-v1",
            "stage": "A3",
            "attack_name": cfg["name"],
            "attack_seed": int(attack_seed),
            "clean_dataset": str(clean_path),
            "clean_dataset_sha256": clean_sha,
            "poisoned_dataset": str(poison_path.resolve()),
            "poisoned_dataset_sha256": poison_sha,
            "completed_trajectories": len(trajectories),
            "used_transitions": used,
            "trailing_transitions": trailing,
            "candidate_quantile_min": float(ccfg["trajectory_return_quantile_min"]),
            "candidate_return_threshold_q70": q70,
            "candidate_trajectory_count": len(eligible),
            "requested_budget_fraction": float(acfg["transition_budget_fraction"]),
            "integrity": integrity,
            "selected_trajectory_indices": selected,
            "candidate_shuffle_order": order,
            "selected_trajectories": records,
            "claim_boundary": [
                "Action-only A3 artifact generation.",
                "No victim performance used.",
                "Attack effectiveness is not established by A3.",
                "Effectiveness must be qualified separately in A4.",
            ],
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        row = {
            "attack_seed": int(attack_seed),
            "q70": q70,
            "candidate_trajectory_count": len(eligible),
            "selected_trajectory_count": len(selected),
            **integrity,
            "poisoned_dataset_sha256": poison_sha,
        }
        rows.append(row)

        print(
            f"{attack_seed:4d} {q70:10.3f} {len(eligible):10d} {len(selected):8d} "
            f"{actual:6d}/{requested:<6d} {integrity['budget_utilization']:.6f} "
            f"{integrity['changed_fraction_of_selected']:.6f}"
        )

    summary = {
        "schema_version": "a3-action-cyclic-shift-summary-v1",
        "stage": "A3",
        "status": "ARTIFACT_GENERATION_PASS",
        "attack_name": cfg["name"],
        "clean_dataset_sha256": clean_sha,
        "candidate_return_threshold_q70": q70,
        "rows": rows,
    }
    (args.metadata_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print("A3 ARTIFACT GENERATION: PASS")
    print("metadata ->", args.metadata_root)


if __name__ == "__main__":
    main()
