from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REPLICATES = tuple(range(5))
TIE_SEED_BASE = 6_100_000
ATOL = 1e-5
PRIMARY_COST_SCALE = 1_000_000.0
TIE_COST_SCALE = 1e-3


def trajectory_ranges(clean, used_n):
    done = (
        np.asarray(clean["terminals"][:used_n], dtype=bool)
        | np.asarray(clean["timeouts"][:used_n], dtype=bool)
    )

    ranges = []
    start = 0

    for i, flag in enumerate(done):
        if not flag:
            continue

        end = i + 1
        ranges.append((start, end))
        start = end

    if start != used_n:
        raise RuntimeError(
            f"Completed trajectory range ends at {start}, "
            f"expected {used_n}."
        )

    return ranges


def contiguous_runs(indices):
    values = np.asarray(indices, dtype=np.int64)

    if len(values) == 0:
        return []

    split_at = np.flatnonzero(np.diff(values) != 1) + 1

    return [
        np.asarray(run, dtype=np.int64)
        for run in np.split(values, split_at)
    ]


def ordered_runs_from_mask(mask):
    return contiguous_runs(np.flatnonzero(mask))


def ordered_run_lengths(mask):
    return [
        int(len(run))
        for run in ordered_runs_from_mask(mask)
    ]


def _interval_overlap_costs(source_mask, length):
    source_int = np.asarray(source_mask, dtype=np.int64)
    prefix = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(source_int),
        )
    )

    starts = np.arange(
        0,
        len(source_mask) - length + 1,
        dtype=np.int64,
    )

    costs = (
        prefix[starts + length]
        - prefix[starts]
    )

    return costs.astype(np.int64, copy=False)


def minimum_overlap_layout(
    source_mask,
    run_lengths,
    *,
    tie_seed,
):
    source_mask = np.asarray(source_mask, dtype=bool)
    run_lengths = [
        int(x)
        for x in run_lengths
    ]

    L = int(len(source_mask))
    m = len(run_lengths)

    if m == 0:
        return {
            "starts": [],
            "overlap_count": 0,
        }

    minimum_required = (
        sum(run_lengths)
        + max(m - 1, 0)
    )

    if minimum_required > L:
        raise RuntimeError(
            "Run sequence is not linearly packable: "
            f"trajectory_length={L}, "
            f"run_lengths={run_lengths}"
        )

    rng = np.random.default_rng(tie_seed)

    dp_levels = []
    back_levels = []

    first_len = run_lengths[0]
    first_overlap = _interval_overlap_costs(
        source_mask,
        first_len,
    )

    first_jitter = (
        rng.random(len(first_overlap))
        * TIE_COST_SCALE
    )

    first_dp = (
        first_overlap.astype(np.float64)
        * PRIMARY_COST_SCALE
        + first_jitter
    )

    dp_levels.append(first_dp)
    back_levels.append(
        np.full(
            len(first_dp),
            -1,
            dtype=np.int64,
        )
    )

    for j in range(1, m):
        current_len = run_lengths[j]
        previous_len = run_lengths[j - 1]

        overlap = _interval_overlap_costs(
            source_mask,
            current_len,
        )

        jitter = (
            rng.random(len(overlap))
            * TIE_COST_SCALE
        )

        local_cost = (
            overlap.astype(np.float64)
            * PRIMARY_COST_SCALE
            + jitter
        )

        previous_dp = dp_levels[-1]

        prefix_best_score = np.empty_like(
            previous_dp
        )

        prefix_best_index = np.empty(
            len(previous_dp),
            dtype=np.int64,
        )

        best_score = np.inf
        best_index = -1

        for p, value in enumerate(previous_dp):
            if value < best_score:
                best_score = float(value)
                best_index = int(p)

            prefix_best_score[p] = best_score
            prefix_best_index[p] = best_index

        current_dp = np.full(
            len(local_cost),
            np.inf,
            dtype=np.float64,
        )

        current_back = np.full(
            len(local_cost),
            -1,
            dtype=np.int64,
        )

        for s in range(len(local_cost)):
            max_previous_start = (
                s - previous_len - 1
            )

            if max_previous_start < 0:
                continue

            if max_previous_start >= len(previous_dp):
                max_previous_start = (
                    len(previous_dp) - 1
                )

            prev_index = int(
                prefix_best_index[
                    max_previous_start
                ]
            )

            if prev_index < 0:
                continue

            prev_score = float(
                prefix_best_score[
                    max_previous_start
                ]
            )

            if not np.isfinite(prev_score):
                continue

            current_dp[s] = (
                prev_score
                + local_cost[s]
            )

            current_back[s] = (
                prev_index
            )

        if not np.any(np.isfinite(current_dp)):
            raise RuntimeError(
                "Dynamic program found no feasible "
                f"placement at run {j}: "
                f"trajectory_length={L}, "
                f"run_lengths={run_lengths}"
            )

        dp_levels.append(current_dp)
        back_levels.append(current_back)

    last_start = int(
        np.argmin(dp_levels[-1])
    )

    if not np.isfinite(
        dp_levels[-1][last_start]
    ):
        raise RuntimeError(
            "No finite final E1B placement."
        )

    starts = [0] * m
    starts[-1] = last_start

    for j in range(m - 1, 0, -1):
        prev = int(
            back_levels[j][
                starts[j]
            ]
        )

        if prev < 0:
            raise RuntimeError(
                "Broken E1B DP backpointer."
            )

        starts[j - 1] = prev

    target_mask = np.zeros(
        L,
        dtype=bool,
    )

    for start, length in zip(
        starts,
        run_lengths,
    ):
        target_mask[
            start:start + length
        ] = True

    overlap_count = int(
        np.sum(
            source_mask
            & target_mask
        )
    )

    return {
        "starts": [
            int(x)
            for x in starts
        ],
        "overlap_count": overlap_count,
    }


def build_min_overlap_control(
    clean,
    poison,
    used_n,
    *,
    replicate,
    artifact_seed,
):
    clean_obs = np.asarray(
        clean["observations"]
    )

    poison_obs = np.asarray(
        poison["observations"]
    )

    source_delta = (
        poison_obs[:used_n]
        - clean_obs[:used_n]
    )

    source_mask = np.any(
        source_delta != 0.0,
        axis=1,
    )

    control_obs = clean_obs.copy()
    records = []

    for trajectory_id, (
        traj_start,
        traj_end,
    ) in enumerate(
        trajectory_ranges(
            clean,
            used_n,
        )
    ):
        source_mask_local = (
            source_mask[
                traj_start:traj_end
            ]
        )

        runs_local = (
            ordered_runs_from_mask(
                source_mask_local
            )
        )

        if not runs_local:
            continue

        lengths = [
            int(len(run))
            for run in runs_local
        ]

        tie_seed = (
            TIE_SEED_BASE
            + artifact_seed * 100_000
            + replicate * 10_000
            + trajectory_id
        )

        layout = minimum_overlap_layout(
            source_mask_local,
            lengths,
            tie_seed=tie_seed,
        )

        target_starts = (
            layout["starts"]
        )

        trajectory_records = []

        for run_index, (
            source_run,
            target_start_local,
        ) in enumerate(
            zip(
                runs_local,
                target_starts,
            )
        ):
            source_start_local = int(
                source_run[0]
            )

            length = int(
                len(source_run)
            )

            source_start = (
                traj_start
                + source_start_local
            )

            source_end = (
                source_start
                + length
            )

            target_start = (
                traj_start
                + int(
                    target_start_local
                )
            )

            target_end = (
                target_start
                + length
            )

            delta_sequence = (
                source_delta[
                    source_start:source_end
                ]
            )

            control_obs[
                target_start:target_end
            ] = (
                clean_obs[
                    target_start:target_end
                ]
                + delta_sequence
            )

            trajectory_records.append(
                {
                    "run_index": int(
                        run_index
                    ),
                    "source_start": int(
                        source_start
                    ),
                    "source_end": int(
                        source_end
                    ),
                    "target_start": int(
                        target_start
                    ),
                    "target_end": int(
                        target_end
                    ),
                    "length": int(
                        length
                    ),
                }
            )

        records.append(
            {
                "trajectory_id": int(
                    trajectory_id
                ),
                "trajectory_start": int(
                    traj_start
                ),
                "trajectory_end": int(
                    traj_end
                ),
                "trajectory_length": int(
                    traj_end - traj_start
                ),
                "source_modified_count": int(
                    source_mask_local.sum()
                ),
                "source_run_lengths": lengths,
                "minimum_overlap_count": int(
                    layout[
                        "overlap_count"
                    ]
                ),
                "runs": (
                    trajectory_records
                ),
            }
        )

    control = {
        "observations": control_obs,
        "actions": clean["actions"],
        "rewards": clean["rewards"],
        "terminals": clean["terminals"],
        "timeouts": clean["timeouts"],
    }

    return control, records


def validate_min_overlap_control(
    clean,
    poison,
    control,
    records,
    used_n,
):
    clean_obs = np.asarray(
        clean["observations"]
    )

    poison_obs = np.asarray(
        poison["observations"]
    )

    control_obs = np.asarray(
        control["observations"]
    )

    source_delta = (
        poison_obs[:used_n]
        - clean_obs[:used_n]
    )

    target_delta = (
        control_obs[:used_n]
        - clean_obs[:used_n]
    )

    source_mask = np.any(
        source_delta != 0.0,
        axis=1,
    )

    target_mask = np.any(
        target_delta != 0.0,
        axis=1,
    )

    source_count = int(
        source_mask.sum()
    )

    target_count = int(
        target_mask.sum()
    )

    if source_count != target_count:
        raise RuntimeError(
            "Modified-count mismatch: "
            f"source={source_count}, "
            f"target={target_count}"
        )

    record_by_trajectory = {
        int(row["trajectory_id"]): row
        for row in records
    }

    max_recovered_delta_error = 0.0
    overlap_total = 0
    moved_run_count = 0
    total_run_count = 0

    ranges = trajectory_ranges(
        clean,
        used_n,
    )

    for trajectory_id, (
        traj_start,
        traj_end,
    ) in enumerate(ranges):
        source_local = (
            source_mask[
                traj_start:traj_end
            ]
        )

        target_local = (
            target_mask[
                traj_start:traj_end
            ]
        )

        source_n = int(
            source_local.sum()
        )

        target_n = int(
            target_local.sum()
        )

        if source_n == 0:
            if target_n != 0:
                raise RuntimeError(
                    "Clean source trajectory became "
                    "modified."
                )
            continue

        if source_n != target_n:
            raise RuntimeError(
                "Per-trajectory count mismatch: "
                f"trajectory={trajectory_id}, "
                f"source={source_n}, "
                f"target={target_n}"
            )

        source_lengths = (
            ordered_run_lengths(
                source_local
            )
        )

        target_lengths = (
            ordered_run_lengths(
                target_local
            )
        )

        if source_lengths != target_lengths:
            raise RuntimeError(
                "Ordered run-length mismatch: "
                f"trajectory={trajectory_id}, "
                f"source={source_lengths}, "
                f"target={target_lengths}"
            )

        record = (
            record_by_trajectory[
                trajectory_id
            ]
        )

        if (
            source_lengths
            != record[
                "source_run_lengths"
            ]
        ):
            raise RuntimeError(
                "Record/source run-length mismatch."
            )

        trajectory_overlap = int(
            np.sum(
                source_local
                & target_local
            )
        )

        if (
            trajectory_overlap
            != int(
                record[
                    "minimum_overlap_count"
                ]
            )
        ):
            raise RuntimeError(
                "Recorded minimum-overlap count "
                "does not match constructed mask."
            )

        overlap_total += (
            trajectory_overlap
        )

        for run in record["runs"]:
            s0 = int(
                run["source_start"]
            )
            s1 = int(
                run["source_end"]
            )
            t0 = int(
                run["target_start"]
            )
            t1 = int(
                run["target_end"]
            )

            expected_observation = (
                clean_obs[t0:t1]
                + source_delta[s0:s1]
            )

            if not np.array_equal(
                control_obs[t0:t1],
                expected_observation,
            ):
                raise RuntimeError(
                    "Stored target observation "
                    "does not equal clean target "
                    "+ source delta."
                )

            recovered_delta = (
                control_obs[t0:t1]
                - clean_obs[t0:t1]
            )

            error = float(
                np.max(
                    np.abs(
                        recovered_delta
                        - source_delta[s0:s1]
                    )
                )
            )

            max_recovered_delta_error = max(
                max_recovered_delta_error,
                error,
            )

            total_run_count += 1

            if s0 != t0:
                moved_run_count += 1

    if (
        max_recovered_delta_error
        > ATOL
    ):
        raise RuntimeError(
            "Recovered float32 delta error "
            "exceeds tolerance: "
            f"{max_recovered_delta_error}"
        )

    if not np.array_equal(
        control["actions"],
        clean["actions"],
    ):
        raise RuntimeError(
            "E1B modified actions."
        )

    if not np.array_equal(
        control["rewards"],
        clean["rewards"],
    ):
        raise RuntimeError(
            "E1B modified rewards."
        )

    if not np.array_equal(
        control["terminals"],
        clean["terminals"],
    ):
        raise RuntimeError(
            "E1B modified terminals."
        )

    if not np.array_equal(
        control["timeouts"],
        clean["timeouts"],
    ):
        raise RuntimeError(
            "E1B modified timeouts."
        )

    return {
        "source_modified_count": (
            source_count
        ),
        "target_modified_count": (
            target_count
        ),
        "source_target_overlap_count": int(
            overlap_total
        ),
        "source_target_overlap_fraction": float(
            overlap_total
            / max(source_count, 1)
        ),
        "moved_away_fraction": float(
            1.0
            - overlap_total
            / max(source_count, 1)
        ),
        "total_run_count": int(
            total_run_count
        ),
        "moved_run_count": int(
            moved_run_count
        ),
        "moved_run_fraction": float(
            moved_run_count
            / max(total_run_count, 1)
        ),
        "max_recovered_delta_error": float(
            max_recovered_delta_error
        ),
    }
