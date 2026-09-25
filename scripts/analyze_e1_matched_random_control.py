from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_group4e_clean_policy_sensitivity import (
    CLEAN_DATA,
    NORMALIZATION,
    ROOTS,
    RHOS,
    SEEDS,
    load_clean_dt,
    load_clean_joint_policy,
    load_dataset,
    build_metadata,
    classify_endpoints,
    deterministic_sample,
    sensitivity_for_endpoints,
)

from scripts.analyze_group4e_layerwise_history import (
    LEVELS,
    frozen_sample as layerwise_sample,
    state_only_dataset,
    model_shifts,
)

from scripts.analyze_group4e_attention_routing import (
    LAYERS as ATTENTION_LAYERS,
    frozen_sample as attention_sample,
    state_changed_mask,
    rerouting_for_model,
)


OUTPUT = Path(
    "experiments/postfinal_controls/"
    "e1_matched_random_control.json"
)

CONTROL_SEED_BASE = 3_100_000

MATCH_ATOL = 2e-6


def matched_random_sign_control(
    clean,
    poison,
    used_n: int,
    seed: int,
):
    """
    Build a state-only counterfactual control.

    For every CSDPC-modified state component:

        delta_control = sign * delta_csdpc

    where sign is independently sampled from {-1, +1}.

    Therefore component-wise absolute perturbation magnitude
    is preserved exactly up to floating-point storage error.
    """

    rng = np.random.default_rng(seed)

    clean_obs = np.asarray(
        clean["observations"]
    )

    poison_obs = np.asarray(
        poison["observations"]
    )

    delta = (
        poison_obs[:used_n]
        - clean_obs[:used_n]
    )

    changed = np.any(
        delta != 0.0,
        axis=1,
    )

    control_obs = clean_obs.copy()

    signs = rng.choice(
        np.asarray(
            [-1.0, 1.0],
            dtype=delta.dtype,
        ),
        size=delta.shape,
        replace=True,
    )

    control_delta = (
        delta * signs
    )

    control_obs[:used_n][changed] = (
        clean_obs[:used_n][changed]
        + control_delta[changed]
    )

    return {
        "observations": control_obs,
        "actions": clean["actions"],
        "rewards": clean["rewards"],
        "terminals": clean["terminals"],
        "timeouts": clean["timeouts"],
    }


def validate_control(
    clean,
    poison,
    control,
    used_n: int,
):
    real_delta = (
        poison["observations"][:used_n]
        - clean["observations"][:used_n]
    )

    control_delta = (
        control["observations"][:used_n]
        - clean["observations"][:used_n]
    )

    real_mask = np.any(
        real_delta != 0.0,
        axis=1,
    )

    control_mask = np.any(
        control_delta != 0.0,
        axis=1,
    )

    same_mask = bool(
        np.array_equal(
            real_mask,
            control_mask,
        )
    )

    if not same_mask:
        raise RuntimeError(
            "Matched control changed the attacked state mask."
        )

    idx = np.flatnonzero(
        real_mask
    )

    abs_component_error = float(
        np.max(
            np.abs(
                np.abs(real_delta[idx])
                - np.abs(control_delta[idx])
            )
        )
    )

    if abs_component_error > MATCH_ATOL:
        raise RuntimeError(
            "Matched-control component magnitudes differ: "
            f"{abs_component_error}"
        )

    real_l1 = np.linalg.norm(
        real_delta[idx],
        ord=1,
        axis=1,
    )

    control_l1 = np.linalg.norm(
        control_delta[idx],
        ord=1,
        axis=1,
    )

    real_l2 = np.linalg.norm(
        real_delta[idx],
        ord=2,
        axis=1,
    )

    control_l2 = np.linalg.norm(
        control_delta[idx],
        ord=2,
        axis=1,
    )

    real_linf = np.linalg.norm(
        real_delta[idx],
        ord=np.inf,
        axis=1,
    )

    control_linf = np.linalg.norm(
        control_delta[idx],
        ord=np.inf,
        axis=1,
    )

    denominator = (
        real_l2 * control_l2
    )

    cosine = np.sum(
        real_delta[idx]
        * control_delta[idx],
        axis=1,
    ) / np.maximum(
        denominator,
        1e-12,
    )

    return {
        "n_state_modified": int(
            len(idx)
        ),
        "same_state_modified_indices": (
            same_mask
        ),
        "max_component_abs_magnitude_error": (
            abs_component_error
        ),
        "max_l1_error": float(
            np.max(
                np.abs(
                    real_l1 - control_l1
                )
            )
        ),
        "max_l2_error": float(
            np.max(
                np.abs(
                    real_l2 - control_l2
                )
            )
        ),
        "max_linf_error": float(
            np.max(
                np.abs(
                    real_linf
                    - control_linf
                )
            )
        ),
        "mean_delta_cosine": float(
            np.mean(cosine)
        ),
        "median_delta_cosine": float(
            np.median(cosine)
        ),
        "fraction_identical_direction": float(
            np.mean(
                np.all(
                    real_delta[idx]
                    == control_delta[idx],
                    axis=1,
                )
            )
        ),
    }


def summarize_action_shift(
    dt_shift,
    joint_shift,
):
    return {
        "n": int(
            len(dt_shift)
        ),
        "dt_mean": float(
            np.mean(dt_shift)
        ),
        "joint_mean": float(
            np.mean(joint_shift)
        ),
        "A_joint_minus_dt": float(
            np.mean(joint_shift)
            - np.mean(dt_shift)
        ),
        "fraction_joint_gt_dt": float(
            np.mean(
                joint_shift
                > dt_shift
            )
        ),
    }


def action_sensitivity(
    *,
    dt_model,
    joint_model,
    clean,
    perturbation,
    direct_endpoints,
    history_endpoints,
    meta,
    state_mean,
    state_std,
    device,
):
    result = {}

    for name, endpoints in (
        (
            "direct",
            direct_endpoints,
        ),
        (
            "history_only",
            history_endpoints,
        ),
    ):
        dt_shift, joint_shift = (
            sensitivity_for_endpoints(
                dt_model=dt_model,
                joint_model=joint_model,
                clean=clean,
                poison=perturbation,
                endpoints=endpoints,
                meta=meta,
                state_mean=state_mean,
                state_std=state_std,
                device=device,
            )
        )

        result[name] = (
            summarize_action_shift(
                dt_shift,
                joint_shift,
            )
        )

    return result


def layerwise_metrics(
    *,
    dt_model,
    joint_model,
    clean,
    perturbation,
    endpoints,
    meta,
    state_mean,
    state_std,
    device,
):
    dt = model_shifts(
        model=dt_model,
        clean=clean,
        poison_state_only=perturbation,
        endpoints=endpoints,
        meta=meta,
        state_mean=state_mean,
        state_std=state_std,
        device=device,
    )

    joint = model_shifts(
        model=joint_model,
        clean=clean,
        poison_state_only=perturbation,
        endpoints=endpoints,
        meta=meta,
        state_mean=state_mean,
        state_std=state_std,
        device=device,
    )

    output = {}

    for level in LEVELS:
        dt_mean = float(
            np.mean(
                dt[level]
            )
        )

        joint_mean = float(
            np.mean(
                joint[level]
            )
        )

        output[level] = {
            "dt_mean": dt_mean,
            "joint_mean": joint_mean,
            "C_joint_minus_dt": (
                joint_mean - dt_mean
            ),
            "fraction_joint_gt_dt": float(
                np.mean(
                    joint[level]
                    > dt[level]
                )
            ),
        }

    return output


def attention_metrics(
    *,
    dt_model,
    joint_model,
    clean,
    perturbation,
    endpoints,
    changed,
    meta,
    state_mean,
    state_std,
    device,
):
    dt = rerouting_for_model(
        model=dt_model,
        clean=clean,
        poison_state=perturbation,
        endpoints=endpoints,
        changed=changed,
        meta=meta,
        state_mean=state_mean,
        state_std=state_std,
        device=device,
    )

    joint = rerouting_for_model(
        model=joint_model,
        clean=clean,
        poison_state=perturbation,
        endpoints=endpoints,
        changed=changed,
        meta=meta,
        state_mean=state_mean,
        state_std=state_std,
        device=device,
    )

    output = {}

    for level in ATTENTION_LAYERS:
        dt_mean = float(
            np.mean(
                dt[level]
            )
        )

        joint_mean = float(
            np.mean(
                joint[level]
            )
        )

        output[level] = {
            "dt_reroute": dt_mean,
            "joint_reroute": joint_mean,
            "D_joint_minus_dt": (
                joint_mean - dt_mean
            ),
        }

    return output


def source_summary(rows, source):
    selected = [
        row["sources"][source]
        for row in rows
    ]

    direct = np.asarray(
        [
            row[
                "action_sensitivity"
            ]["direct"][
                "A_joint_minus_dt"
            ]
            for row in selected
        ],
        dtype=np.float64,
    )

    history = np.asarray(
        [
            row[
                "action_sensitivity"
            ]["history_only"][
                "A_joint_minus_dt"
            ]
            for row in selected
        ],
        dtype=np.float64,
    )

    block2 = np.asarray(
        [
            row["layerwise"][
                "block2"
            ]["C_joint_minus_dt"]
            for row in selected
        ],
        dtype=np.float64,
    )

    attention2 = np.asarray(
        [
            row["attention"][
                "block2"
            ]["D_joint_minus_dt"]
            for row in selected
        ],
        dtype=np.float64,
    )

    return {
        "direct_A": {
            "mean": float(
                np.mean(direct)
            ),
            "positive": int(
                np.sum(direct > 0)
            ),
        },
        "history_A": {
            "mean": float(
                np.mean(history)
            ),
            "positive": int(
                np.sum(history > 0)
            ),
        },
        "block2_C": {
            "mean": float(
                np.mean(block2)
            ),
            "median": float(
                np.median(block2)
            ),
            "positive": int(
                np.sum(block2 > 0)
            ),
        },
        "block2_attention_D": {
            "mean": float(
                np.mean(attention2)
            ),
            "positive": int(
                np.sum(attention2 > 0)
            ),
        },
    }


def seed_level_block2(rows, source):
    result = {}

    for seed in SEEDS:
        values = [
            row["sources"][source][
                "layerwise"
            ]["block2"][
                "C_joint_minus_dt"
            ]
            for row in rows
            if row["seed"] == seed
        ]

        result[str(seed)] = {
            "n_artifacts": int(
                len(values)
            ),
            "mean_C_block2": float(
                np.mean(values)
            ),
        }

    return result


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)

    clean = load_dataset(
        CLEAN_DATA
    )

    meta = build_metadata(
        clean
    )

    with np.load(
        NORMALIZATION
    ) as f:
        state_mean = (
            f["state_mean"]
            .astype(np.float32)
        )

        state_std = (
            f["state_std"]
            .astype(np.float32)
        )

    rows = []

    print()
    print("=" * 145)
    print(
        "E1 — MATCHED RANDOM-SIGN "
        "HISTORICAL-STATE CONTROL"
    )
    print("=" * 145)

    print(
        "condition       rho seed source          "
        "A_direct  A_history   C_block2    D_block2"
    )

    for seed in SEEDS:
        dt_model, _ = (
            load_clean_dt(
                seed,
                device,
            )
        )

        joint_model, _ = (
            load_clean_joint_policy(
                seed,
                device,
            )
        )

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

                changed = (
                    state_changed_mask(
                        clean,
                        poison,
                        meta["used_n"],
                    )
                )

                direct, history = (
                    classify_endpoints(
                        changed,
                        meta,
                    )
                )

                control_seed = (
                    CONTROL_SEED_BASE
                    + condition_index
                    * 100_000
                    + int(
                        rho * 10_000
                    ) * 10
                    + seed
                )

                control = (
                    matched_random_sign_control(
                        clean,
                        poison,
                        meta["used_n"],
                        control_seed,
                    )
                )

                invariants = (
                    validate_control(
                        clean,
                        poison,
                        control,
                        meta["used_n"],
                    )
                )

                # Original Group-4E clean-policy
                # sampling contract.
                sensitivity_seed = (
                    910_000
                    + condition_index
                    * 100_000
                    + int(
                        rho * 10_000
                    ) * 10
                    + seed
                )

                direct_sample = (
                    deterministic_sample(
                        direct,
                        seed=(
                            sensitivity_seed
                            + 1
                        ),
                    )
                )

                history_sample = (
                    deterministic_sample(
                        history,
                        seed=(
                            sensitivity_seed
                            + 2
                        ),
                    )
                )

                # Original layerwise sampling contract.
                layer_seed = (
                    1_700_000
                    + seed * 100_000
                    + condition_index
                    * 10_000
                    + int(
                        rho * 1000
                    )
                )

                layer_endpoints = (
                    layerwise_sample(
                        history,
                        layer_seed,
                    )
                )

                # Original attention sampling contract.
                attention_seed = (
                    2_100_000
                    + seed * 100_000
                    + condition_index
                    * 10_000
                    + int(
                        rho * 1000
                    )
                )

                attention_endpoints = (
                    attention_sample(
                        history,
                        attention_seed,
                    )
                )

                csdpc_state = (
                    state_only_dataset(
                        clean,
                        poison,
                    )
                )

                sources = {}

                for (
                    source_name,
                    perturbation,
                ) in (
                    (
                        "csdpc_state",
                        csdpc_state,
                    ),
                    (
                        "matched_random_sign",
                        control,
                    ),
                ):
                    sensitivity = (
                        action_sensitivity(
                            dt_model=dt_model,
                            joint_model=joint_model,
                            clean=clean,
                            perturbation=perturbation,
                            direct_endpoints=direct_sample,
                            history_endpoints=history_sample,
                            meta=meta,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    )

                    layers = (
                        layerwise_metrics(
                            dt_model=dt_model,
                            joint_model=joint_model,
                            clean=clean,
                            perturbation=perturbation,
                            endpoints=layer_endpoints,
                            meta=meta,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    )

                    attention = (
                        attention_metrics(
                            dt_model=dt_model,
                            joint_model=joint_model,
                            clean=clean,
                            perturbation=perturbation,
                            endpoints=attention_endpoints,
                            changed=changed,
                            meta=meta,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    )

                    sources[source_name] = {
                        "action_sensitivity": (
                            sensitivity
                        ),
                        "layerwise": layers,
                        "attention": attention,
                    }

                    print(
                        f"{condition:15s} "
                        f"{rho:.2f} "
                        f"{seed:4d} "
                        f"{source_name:19s} "
                        f"{sensitivity['direct']['A_joint_minus_dt']:+9.6f} "
                        f"{sensitivity['history_only']['A_joint_minus_dt']:+10.6f} "
                        f"{layers['block2']['C_joint_minus_dt']:+10.6f} "
                        f"{attention['block2']['D_joint_minus_dt']:+11.8f}"
                    )

                rows.append(
                    {
                        "condition": (
                            condition
                        ),
                        "rho": rho,
                        "seed": seed,
                        "control_seed": (
                            control_seed
                        ),
                        "invariants": (
                            invariants
                        ),
                        "sources": (
                            sources
                        ),
                    }
                )

        del dt_model
        del joint_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summaries = {
        source: source_summary(
            rows,
            source,
        )
        for source in (
            "csdpc_state",
            "matched_random_sign",
        )
    }

    seed_block2 = {
        source: seed_level_block2(
            rows,
            source,
        )
        for source in (
            "csdpc_state",
            "matched_random_sign",
        )
    }

    csdpc_c2 = (
        summaries[
            "csdpc_state"
        ]["block2_C"]["mean"]
    )

    control_c2 = (
        summaries[
            "matched_random_sign"
        ]["block2_C"]["mean"]
    )

    ratio = (
        None
        if abs(csdpc_c2) < 1e-12
        else float(
            control_c2
            / csdpc_c2
        )
    )

    paired_c2 = np.asarray(
        [
            row["sources"][
                "matched_random_sign"
            ]["layerwise"][
                "block2"
            ][
                "C_joint_minus_dt"
            ]
            - row["sources"][
                "csdpc_state"
            ]["layerwise"][
                "block2"
            ][
                "C_joint_minus_dt"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    comparison = {
        "mean_control_minus_csdpc_C_block2": (
            float(
                np.mean(
                    paired_c2
                )
            )
        ),
        "median_control_minus_csdpc_C_block2": (
            float(
                np.median(
                    paired_c2
                )
            )
        ),
        "control_over_csdpc_mean_C_block2_ratio": (
            ratio
        ),
        "control_C_block2_positive_artifacts": int(
            summaries[
                "matched_random_sign"
            ]["block2_C"]["positive"]
        ),
        "csdpc_C_block2_positive_artifacts": int(
            summaries[
                "csdpc_state"
            ]["block2_C"]["positive"]
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
                    "e1_matched_random_control"
                ),
                "control": (
                    "componentwise_random_sign"
                ),
                "rows": rows,
                "summary": summaries,
                "seed_level_block2": (
                    seed_block2
                ),
                "comparison": (
                    comparison
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 145)
    print("E1 SUMMARY")
    print("=" * 145)

    for source, summary in (
        summaries.items()
    ):
        print()
        print(source)
        print(
            "  A_direct mean:",
            f"{summary['direct_A']['mean']:+.6f}",
            "positive:",
            f"{summary['direct_A']['positive']}/12",
        )
        print(
            "  A_history mean:",
            f"{summary['history_A']['mean']:+.6f}",
            "positive:",
            f"{summary['history_A']['positive']}/12",
        )
        print(
            "  C_block2 mean:",
            f"{summary['block2_C']['mean']:+.6f}",
            "positive:",
            f"{summary['block2_C']['positive']}/12",
        )
        print(
            "  D_block2 mean:",
            f"{summary['block2_attention_D']['mean']:+.8f}",
            "positive:",
            f"{summary['block2_attention_D']['positive']}/12",
        )

    print()
    print("comparison:")
    print(
        json.dumps(
            comparison,
            indent=2,
        )
    )

    print()
    print("output ->", OUTPUT)


if __name__ == "__main__":
    main()