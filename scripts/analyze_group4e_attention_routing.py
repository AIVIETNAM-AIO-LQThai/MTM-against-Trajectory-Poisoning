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
    build_context_batch,
)


MAX_ENDPOINTS = 5000
BATCH_SIZE = 128
LAYERS = ("block1", "block2", "block3")


def state_changed_mask(clean, poison, used_n):
    return np.any(
        clean["observations"][:used_n]
        != poison["observations"][:used_n],
        axis=1,
    )


def history_endpoints(mask, meta):
    out = []

    prefix = np.concatenate(
        ([0], np.cumsum(mask.astype(np.int64)))
    )

    for endpoint in range(meta["used_n"]):

        # Endpoint itself must remain clean.
        if mask[endpoint]:
            continue

        left = max(
            int(meta["traj_start"][endpoint]),
            endpoint - K + 1,
        )

        count = int(
            prefix[endpoint]
            - prefix[left]
        )

        if count > 0:
            out.append(endpoint)

    return np.asarray(
        out,
        dtype=np.int64,
    )


def frozen_sample(values, seed):
    if len(values) <= MAX_ENDPOINTS:
        return values

    rng = np.random.default_rng(seed)

    values = rng.choice(
        values,
        size=MAX_ENDPOINTS,
        replace=False,
    )

    return np.sort(values)


def state_only_dataset(clean, poison):
    return {
        "observations": poison["observations"],
        "actions": clean["actions"],
        "rewards": clean["rewards"],
        "terminals": clean["terminals"],
        "timeouts": clean["timeouts"],
    }


def poison_position_mask(
    endpoints,
    changed,
    meta,
):
    result = np.zeros(
        (len(endpoints), K),
        dtype=bool,
    )

    for row, endpoint in enumerate(endpoints):

        left = max(
            int(meta["traj_start"][endpoint]),
            int(endpoint) - K + 1,
        )

        indices = np.arange(
            left,
            int(endpoint) + 1,
        )

        pad = K - len(indices)

        result[row, pad:] = (
            changed[indices]
        )

        # History-only contract.
        result[row, -1] = False

    return result


def model_attentions(
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

        time = model.embed_timestep(
            timesteps
        )

        returns = (
            model.embed_return(rtg)
            + time
        )

        state = (
            model.embed_state(states)
            + time
        )

        action = (
            model.embed_action(actions)
            + time
        )

        stacked = torch.stack(
            (returns, state, action),
            dim=2,
        ).reshape(
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
            output_attentions=True,
            return_dict=True,
        )

    if (
        output.attentions is None
        or len(output.attentions) != 3
    ):
        raise RuntimeError(
            "Expected attention tensors from all 3 GPT blocks."
        )

    return [
        x.detach().cpu().numpy()
        for x in output.attentions
    ]


def rerouting_for_model(
    *,
    model,
    clean,
    poison_state,
    endpoints,
    changed,
    meta,
    state_mean,
    state_std,
    device,
):
    collected = {
        level: []
        for level in LAYERS
    }

    for start in range(
        0,
        len(endpoints),
        BATCH_SIZE,
    ):

        batch_endpoints = endpoints[
            start:start + BATCH_SIZE
        ]

        poison_ts = poison_position_mask(
            batch_endpoints,
            changed,
            meta,
        )

        clean_batch = build_context_batch(
            clean,
            batch_endpoints,
            meta,
            state_mean,
            state_std,
        )

        poison_batch = build_context_batch(
            poison_state,
            batch_endpoints,
            meta,
            state_mean,
            state_std,
        )

        clean_attn = model_attentions(
            model,
            clean_batch,
            device,
        )

        poison_attn = model_attentions(
            model,
            poison_batch,
            device,
        )

        # Token order:
        # R0,s0,a0,R1,s1,a1,...
        query = 3 * (K - 1) + 1

        token_mask = np.zeros(
            (
                len(batch_endpoints),
                3 * K,
            ),
            dtype=np.float64,
        )

        token_mask[:, 1::3] = (
            poison_ts.astype(
                np.float64
            )
        )

        n_poison = np.maximum(
            poison_ts.sum(
                axis=1
            ).astype(
                np.float64
            ),
            1.0,
        )

        for layer_index, level in enumerate(
            LAYERS
        ):

            # Shape:
            # [B, heads, query, key]
            #
            # n_head=1 for frozen DT.
            ca = clean_attn[
                layer_index
            ][:, 0, query, :]

            pa = poison_attn[
                layer_index
            ][:, 0, query, :]

            clean_mass = np.sum(
                ca * token_mask,
                axis=1,
            )

            poison_mass = np.sum(
                pa * token_mask,
                axis=1,
            )

            # Change in attention assigned to attacked
            # historical state positions, normalized by
            # number of attacked state tokens.
            reroute = (
                poison_mass - clean_mass
            ) / n_poison

            collected[level].append(
                reroute
            )

    return {
        level: np.concatenate(
            values
        )
        for level, values
        in collected.items()
    }


def safe_corr(x, y):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if (
        np.std(x) == 0.0
        or np.std(y) == 0.0
    ):
        return None

    return float(
        np.corrcoef(x, y)[0, 1]
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
    print("=" * 125)
    print(
        "GROUP 4E — HISTORICAL-STATE ATTENTION ROUTING"
    )
    print("=" * 125)

    print(
        "condition       rho seed       G "
        "D_block1    D_block2    D_block3"
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

                rho = float(rho_key)

                poison = load_dataset(
                    root
                    / f"rho_{rho_slug}_seed_{seed}.hdf5"
                )

                changed = state_changed_mask(
                    clean,
                    poison,
                    meta["used_n"],
                )

                population = history_endpoints(
                    changed,
                    meta,
                )

                sample_seed = (
                    2_100_000
                    + seed * 100_000
                    + condition_index * 10_000
                    + int(rho * 1000)
                )

                endpoints = frozen_sample(
                    population,
                    sample_seed,
                )

                poison_state = (
                    state_only_dataset(
                        clean,
                        poison,
                    )
                )

                dt = rerouting_for_model(
                    model=dt_model,
                    clean=clean,
                    poison_state=poison_state,
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
                    poison_state=poison_state,
                    endpoints=endpoints,
                    changed=changed,
                    meta=meta,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                )

                layers = {}

                for level in LAYERS:

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

                    layers[level] = {
                        "dt_reroute": (
                            dt_mean
                        ),
                        "joint_reroute": (
                            joint_mean
                        ),
                        "D_joint_minus_dt": (
                            joint_mean
                            - dt_mean
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
                        "layers": layers,
                    }
                )

                print(
                    f"{condition:15s} "
                    f"{rho:.2f} "
                    f"{seed:4d} "
                    f"{g:+8.4f} "
                    f"{layers['block1']['D_joint_minus_dt']:+11.7f} "
                    f"{layers['block2']['D_joint_minus_dt']:+11.7f} "
                    f"{layers['block3']['D_joint_minus_dt']:+11.7f}"
                )

        del dt_model
        del joint_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=" * 125)
    print("ATTENTION ROUTING SUMMARY")
    print("=" * 125)

    gs = [
        row["G"]
        for row in rows
    ]

    for level in LAYERS:

        D = np.asarray(
            [
                row["layers"][level][
                    "D_joint_minus_dt"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        correlation = safe_corr(
            gs,
            D,
        )

        corr_text = (
            "undefined"
            if correlation is None
            else f"{correlation:+.4f}"
        )

        print(
            f"{level:7s} "
            f"mean_D={D.mean():+.8f} "
            f"median_D={np.median(D):+.8f} "
            f"D>0={np.sum(D > 0)}/12 "
            f"corr(G,D)={corr_text}"
        )

    out = Path(
        "experiments/dt_mtm_stress/"
        "group4e_attention_routing.json"
    )

    out.write_text(
        json.dumps(
            {
                "analysis": (
                    "group4e_attention_routing"
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
