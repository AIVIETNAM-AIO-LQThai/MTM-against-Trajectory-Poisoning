from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

EXPECTED_OBS_DIM = 17
EXPECTED_ACTION_DIM = 6
EXPECTED_TRANSITIONS = 1_000_000

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest().upper()

def stats(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/raw/walker2d-medium-v2/"
            "walker2d_medium-v2.hdf5"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/metadata/walker2d_medium_dataset_audit.json"
        ),
    )

    args = parser.parse_args()
    path = args.dataset

    if not path.exists():
        raise FileNotFoundError(path)

    print("=" * 70)
    print("WALKER2D-MEDIUM-V2 DATASET-AUDIT")
    print("=" * 70)

    file_hash = sha256_file(path)
    file_size = path.stat().st_size

    print(f"File:   {path}")
    print(f"Bytes:  {file_size}")
    print(f"SHA256: {file_hash}")

    with h5py.File(path, "r") as f:
        observations = f["observations"][:]
        next_observations = f["next_observations"][:]
        actions = f["actions"][:]
        rewards = f["rewards"][:]
        terminals = f["terminals"][:].astype(bool)
        timeouts = f["timeouts"][:].astype(bool)

    # ------------------------------------------------------------
    # Basic shape checks
    # ------------------------------------------------------------
    n = len(observations)
    arrays = {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
    }

    for name, array in arrays.items():
        if len(array) != n:
            raise ValueError(
                f"{name} has {len(array)} entries, expected {n}"
            )

    assert n == EXPECTED_TRANSITIONS, (
        f"Expected {EXPECTED_TRANSITIONS} transitions, got {n}"
    )
    assert observations.shape[1] == EXPECTED_OBS_DIM
    assert next_observations.shape[1] == EXPECTED_OBS_DIM
    assert actions.shape[1] == EXPECTED_ACTION_DIM

    print("\n[PASS] Shapes")

    # ------------------------------------------------------------
    # NaN / Inf checks
    # ------------------------------------------------------------
    numeric_arrays = {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "rewards": rewards,
    }

    validity = {}
    for name, array in numeric_arrays.items():
        nan_count = int(np.isnan(array).sum())
        inf_count = int(np.isinf(array).sum())
        validity[name] = {
            "nan_count": nan_count,
            "inf_count": inf_count,
        }
        print(
            f"{name:20s} "
            f"NaN={nan_count} "
            f"Inf={inf_count}"
        )
        if nan_count != 0 or inf_count != 0:
            raise ValueError(
                f"Invalid values detected in {name}"
            )

    print("[PASS] No NaN / Inf")

    # ------------------------------------------------------------
    # Action checks
    # ------------------------------------------------------------
    action_min = float(actions.min())
    action_max = float(actions.max())
    print("\nAction range:")
    print(f"  min = {action_min:.8f}")
    print(f"  max = {action_max:.8f}")
    action_bound_violations = int(
        np.sum((actions < -1.00001) | (actions > 1.00001))
    )

    print(f"Action-bound violations: {action_bound_violations}")

    # ------------------------------------------------------------
    # Terminal / timeout audit
    # ------------------------------------------------------------
    terminal_count = int(terminals.sum())
    timeout_count = int(timeouts.sum())
    both_count = int(
        np.logical_and(terminals, timeouts).sum()
    )

    print("\nBoundaries:")
    print(f"  terminals = {terminal_count}")
    print(f"  timeouts  = {timeout_count}")
    print(f"  both      = {both_count}")
    boundaries = np.logical_or(terminals, timeouts)

    # ------------------------------------------------------------
    # Reconstruct trajectories
    # ------------------------------------------------------------
    trajectory_lengths = []
    trajectory_returns = []
    start = 0
    running_return = 0.0

    for i in range(n):
        running_return += float(rewards[i])
        if terminals[i] or timeouts[i]:
            trajectory_lengths.append(i - start + 1)
            trajectory_returns.append(running_return)
            start = i + 1
            running_return = 0.0

    trailing_transitions = n - start
    trajectory_lengths = np.asarray(
        trajectory_lengths,
        dtype=np.int64,
    )
    trajectory_returns = np.asarray(
        trajectory_returns,
        dtype=np.float64,
    )

    num_trajectories = len(trajectory_lengths)
    used_transitions = int(trajectory_lengths.sum())

    print("\nTrajectories:")
    print(f"  count = {num_trajectories}")
    print(f"  raw transitions = {n}")
    print(f"  reference-used transitions = {used_transitions}")
    print(f"  trailing unfinished transitions = {trailing_transitions}")

    # Internal consistency
    assert used_transitions + trailing_transitions == n

    # Every stored trajectory must end at an explicit terminal/timeout.
    assert num_trajectories == int(
        np.logical_or(terminals, timeouts).sum()
    )

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------
    reward_stats = stats(rewards)
    length_stats = stats(trajectory_lengths)
    return_stats = stats(trajectory_returns)

    print("\nReward statistics:")
    for key, value in reward_stats.items():
        print(f"  {key:5s}: {value:.6f}")

    print("\nTrajectory-length statistics:")
    for key, value in length_stats.items():
        print(f"  {key:5s}: {value:.6f}")

    print("\nEpisode-return statistics:")
    for key, value in return_stats.items():
        print(f"  {key:5s}: {value:.6f}")

    # ------------------------------------------------------------
    # Optional sequential consistency diagnostic
    # ------------------------------------------------------------
    non_boundary_indices = np.where(~boundaries[:-1])[0]
    next_obs_error = float(np.max(
        np.abs(
            next_observations[non_boundary_indices]
            - observations[non_boundary_indices + 1]
        )
    ))

    print("\nSequential observation consistency:")
    print(
        f"  max |next_obs[t] - obs[t+1]| "
        f"= {next_obs_error:.10f}"
    )

    # ------------------------------------------------------------
    # Audit check results
    # ------------------------------------------------------------

    total_nan_count = sum(
        item["nan_count"]
        for item in validity.values()
    )

    total_inf_count = sum(
        item["inf_count"]
        for item in validity.values()
    )

    dataset_validity_pass = (
        total_nan_count == 0
        and total_inf_count == 0
        and action_bound_violations == 0
    )

    trajectory_boundary_pass = (
        used_transitions + trailing_transitions == n
        and num_trajectories == int(boundaries.sum())
        and next_obs_error <= 1e-6
    )

    checks = {
        "dataset_validity": {
            "status": (
                "PASS"
                if dataset_validity_pass
                else "FAIL"
            ),
            "evidence": {
                "num_raw_transitions": int(n),
                "observation_dim": int(
                    observations.shape[1]
                ),
                "action_dim": int(
                    actions.shape[1]
                ),
                "nan_count": int(total_nan_count),
                "inf_count": int(total_inf_count),
                "action_bound_violations": int(
                    action_bound_violations
                ),
            },
        },

        "trajectory_boundaries": {
            "status": (
                "PASS"
                if trajectory_boundary_pass
                else "FAIL"
            ),
            "evidence": {
                "terminal_count": int(
                    terminal_count
                ),
                "timeout_count": int(
                    timeout_count
                ),
                "num_trajectories": int(
                    num_trajectories
                ),
                "num_training_transitions": int(
                    used_transitions
                ),
                "trailing_unfinished_transitions": int(
                    trailing_transitions
                ),
                "max_nonboundary_next_obs_error": float(
                    next_obs_error
                ),
            },
        },
    }

    print("\nAudit checks:")

    for check_name, check in checks.items():
        print(
            f"  {check_name}: "
            f"{check['status']}"
        )

    # ------------------------------------------------------------
    # Save audit
    # ------------------------------------------------------------
    result = {
        "dataset_id": "walker2d-medium-v2",
        "filename": path.name,
        "size_bytes": file_size,
        "sha256": file_hash,

        "num_raw_transitions": int(n),
        "num_training_transitions": int(used_transitions),
        "num_trajectories": int(num_trajectories),
        "trailing_unfinished_transitions": int(trailing_transitions),

        "observation_shape": list(observations.shape),
        "action_shape": list(actions.shape),

        "terminal_count": terminal_count,
        "timeout_count": timeout_count,
        "terminal_and_timeout_count": both_count,

        "action_min": action_min,
        "action_max": action_max,
        "action_bound_violations": action_bound_violations,

        "reward_statistics": reward_stats,
        "trajectory_length_statistics": length_stats,
        "episode_return_statistics": return_stats,

        "validity": validity,

        "max_nonboundary_next_obs_error": float(next_obs_error),
        "checks": checks,
        "audit_complete": (
            dataset_validity_pass 
            and trajectory_boundary_pass
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print("DATASET AUDIT COMPLETE")
    print(f"Saved -> {args.output}")
    print("=" * 70)

if __name__ == "__main__":
    main()