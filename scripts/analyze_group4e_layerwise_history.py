from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_group4e_clean_policy_sensitivity import (
    CLEAN_DATA,
    NORMALIZATION,
    GROUP4D,
    ROOTS,
    RHOS,
    SEEDS,
    K,
    load_clean_dt,
    load_clean_joint_policy,
    load_dataset,
    build_metadata,
    changed_mask,
    classify_endpoints,
    build_context_batch,
)


MAX_ENDPOINTS = 5000
BATCH = 256

LEVELS = (
    "input",
    "block1",
    "block2",
    "block3",
    "action",
)


def frozen_sample(values, seed):
    values = np.asarray(
        values,
        dtype=np.int64,
    )

    if len(values) <= MAX_ENDPOINTS:
        return values

    rng = np.random.default_rng(seed)

    chosen = rng.choice(
        values,
        size=MAX_ENDPOINTS,
        replace=False,
    )

    return np.sort(chosen)


def state_only_dataset(clean, poison):
    return {
        "observations": poison["observations"],
        "actions": clean["actions"],
        "rewards": clean["rewards"],
        "terminals": clean["terminals"],
        "timeouts": clean["timeouts"],
    }


def endpoint_representations(
    model,
    batch,
    device,
):
    (
        states,
        actions,
        rtg,
        timesteps,
        attention,
    ) = batch

    states = torch.as_tensor(
        states,
        dtype=torch.float32,
        device=device,
    )

    actions = torch.as_tensor(
        actions,
        dtype=torch.float32,
        device=device,
    )

    rtg = torch.as_tensor(
        rtg,
        dtype=torch.float32,
        device=device,
    )

    timesteps = torch.as_tensor(
        timesteps,
        dtype=torch.long,
        device=device,
    )

    attention = torch.as_tensor(
        attention,
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():

        time_emb = model.embed_timestep(
            timesteps
        )

        return_emb = (
            model.embed_return(rtg)
            + time_emb
        )

        state_emb = (
            model.embed_state(states)
            + time_emb
        )

        action_emb = (
            model.embed_action(actions)
            + time_emb
        )

        stacked = torch.stack(
            (
                return_emb,
                state_emb,
                action_emb,
            ),
            dim=2,
        )

        stacked = stacked.reshape(
            states.shape[0],
            3 * K,
            model.hidden_size,
        )

        stacked = model.embed_ln(
            stacked
        )

        stacked_mask = (
            attention.unsqueeze(-1)
            .expand(
                attention.shape[0],
                K,
                3,
            )
            .reshape(
                attention.shape[0],
                3 * K,
            )
        )

        output = model.transformer(
            inputs_embeds=stacked,
            attention_mask=stacked_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden = output.hidden_states

        if len(hidden) != 4:
            raise RuntimeError(
                "Expected 3 GPT blocks -> 4 hidden-state levels, "
                f"got {len(hidden)}"
            )

        # Final timestep is always position K-1 because contexts
        # are left padded. State token ordering is:
        #
        # R_t, s_t, a_t
        #
        endpoint_state_token = (
            3 * (K - 1) + 1
        )

        reps = {
            "input": hidden[0][
                :, endpoint_state_token, :
            ],

            "block1": hidden[1][
                :, endpoint_state_token, :
            ],

            "block2": hidden[2][
                :, endpoint_state_token, :
            ],

            "block3": hidden[3][
                :, endpoint_state_token, :
            ],
        }

        reps["action"] = (
            model.predict_action(
                reps["block3"]
            )
        )

        return {
            key: value.detach().cpu().numpy()
            for key, value in reps.items()
        }


def model_shifts(
    *,
    model,
    clean,
    poison_state_only,
    endpoints,
    meta,
    state_mean,
    state_std,
    device,
):
    collected = {
        level: []
        for level in LEVELS
    }

    for start in range(
        0,
        len(endpoints),
        BATCH,
    ):
        idx = endpoints[
            start:start + BATCH
        ]

        clean_batch = build_context_batch(
            clean,
            idx,
            meta,
            state_mean,
            state_std,
        )

        poison_batch = build_context_batch(
            poison_state_only,
            idx,
            meta,
            state_mean,
            state_std,
        )

        clean_rep = endpoint_representations(
            model,
            clean_batch,
            device,
        )

        poison_rep = endpoint_representations(
            model,
            poison_batch,
            device,
        )

        for level in LEVELS:
            shift = np.linalg.norm(
                poison_rep[level]
                - clean_rep[level],
                axis=1,
            )

            collected[level].append(
                shift
            )

    return {
        level: np.concatenate(values)
        for level, values
        in collected.items()
    }


def pearson(x, y):
    return float(
        np.corrcoef(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
        )[0, 1]
    )


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

    with np.load(NORMALIZATION) as f:
        state_mean = (
            f["state_mean"]
            .astype(np.float32)
        )

        state_std = (
            f["state_std"]
            .astype(np.float32)
        )

    group4d = json.loads(
        GROUP4D.read_text(
            encoding="utf-8"
        )
    )

    G = {}

    for condition in ROOTS:
        for _, rho_key in RHOS:
            for row in group4d[
                "conditions"
            ][condition][rho_key]["rows"]:

                G[
                    (
                        condition,
                        float(rho_key),
                        int(row["seed"]),
                    )
                ] = float(
                    row["stress_response_gap"]
                )

    rows = []

    print()
    print("=" * 145)
    print(
        "GROUP 4E — LAYERWISE HISTORICAL-STATE PROPAGATION"
    )
    print("=" * 145)

    print(
        "condition       rho seed       G "
        "C_input   C_block1  C_block2  C_block3   C_action"
    )

    for seed in SEEDS:

        dt_model, _ = load_clean_dt(
            seed,
            device,
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
        ) in enumerate(ROOTS.items()):

            for rho_slug, rho_key in RHOS:

                rho = float(
                    rho_key
                )

                poison = load_dataset(
                    root
                    / f"rho_{rho_slug}_seed_{seed}.hdf5"
                )

                mask = changed_mask(
                    clean,
                    poison,
                    meta["used_n"],
                )

                _, history = (
                    classify_endpoints(
                        mask,
                        meta,
                    )
                )

                analysis_seed = (
                    1_700_000
                    + seed * 100_000
                    + condition_index * 10_000
                    + int(rho * 1000)
                )

                endpoints = frozen_sample(
                    history,
                    analysis_seed,
                )

                state_poison = (
                    state_only_dataset(
                        clean,
                        poison,
                    )
                )

                dt = model_shifts(
                    model=dt_model,
                    clean=clean,
                    poison_state_only=state_poison,
                    endpoints=endpoints,
                    meta=meta,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                )

                joint = model_shifts(
                    model=joint_model,
                    clean=clean,
                    poison_state_only=state_poison,
                    endpoints=endpoints,
                    meta=meta,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                )

                levels = {}

                for level in LEVELS:

                    dt_mean = float(
                        np.mean(dt[level])
                    )

                    joint_mean = float(
                        np.mean(joint[level])
                    )

                    levels[level] = {
                        "dt_mean": dt_mean,
                        "joint_mean": joint_mean,
                        "C_joint_minus_dt": (
                            joint_mean
                            - dt_mean
                        ),
                        "fraction_joint_gt_dt": float(
                            np.mean(
                                joint[level]
                                > dt[level]
                            )
                        ),
                    }

                g = G[
                    (
                        condition,
                        rho,
                        seed,
                    )
                ]

                rows.append(
                    {
                        "condition": condition,
                        "rho": rho,
                        "seed": seed,
                        "G": g,
                        "n": int(
                            len(endpoints)
                        ),
                        "levels": levels,
                    }
                )

                print(
                    f"{condition:15s} "
                    f"{rho:.2f} "
                    f"{seed:4d} "
                    f"{g:+8.4f} "
                    f"{levels['input']['C_joint_minus_dt']:+9.6f} "
                    f"{levels['block1']['C_joint_minus_dt']:+10.6f} "
                    f"{levels['block2']['C_joint_minus_dt']:+10.6f} "
                    f"{levels['block3']['C_joint_minus_dt']:+10.6f} "
                    f"{levels['action']['C_joint_minus_dt']:+10.6f}"
                )

        del dt_model
        del joint_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=" * 145)
    print("LAYER SUMMARY")
    print("=" * 145)

    gs = [
        row["G"]
        for row in rows
    ]

    for level in LEVELS:

        C = np.asarray(
            [
                row["levels"][level][
                    "C_joint_minus_dt"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        fractions = np.asarray(
            [
                row["levels"][level][
                    "fraction_joint_gt_dt"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        print(
            f"{level:7s} "
            f"mean_C={C.mean():+.6f} "
            f"median_C={np.median(C):+.6f} "
            f"C>0={np.sum(C > 0)}/12 "
            f"mean_frac_joint_gt="
            f"{fractions.mean():.4f} "
            f"corr(G,C)="
            f"{pearson(gs,C):+.4f}"
        )

    out = Path(
        "experiments/dt_mtm_stress/"
        "group4e_layerwise_history.json"
    )

    out.write_text(
        json.dumps(
            {
                "analysis": (
                    "group4e_layerwise_history"
                ),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("output ->", out)


if __name__ == "__main__":
    main()
