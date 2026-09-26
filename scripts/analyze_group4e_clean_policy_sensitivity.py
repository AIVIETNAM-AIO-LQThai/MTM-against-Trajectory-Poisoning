from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from src.methods.dt.model import DecisionTransformer


CLEAN_DATA = Path(
    "data/raw/walker2d-medium-v2/walker2d_medium-v2.hdf5"
)

NORMALIZATION = Path(
    "data/metadata/walker2d_medium_normalization.npz"
)

GROUP4D = Path(
    "experiments/dt_mtm_stress/group4d_stress_comparison.json"
)

ROOTS = {
    "canonical": Path(
        "data/poisoned/csdpc/walker2d-medium-v2"
    ),
    "s2_overlap_r0": Path(
        "data/poisoned/csdpc_s2_overlap_r0_v1/walker2d-medium-v2"
    ),
}

RHOS = (("001", "0.01"), ("005", "0.05"))
SEEDS = (0, 1, 2)

K = 20
MAX_PER_CLASS = 10_000
BATCH_SIZE = 512
RTG_SCALE = 1000.0
EPS = 1e-12


def make_dt() -> DecisionTransformer:
    return DecisionTransformer(
        state_dim=17,
        action_dim=6,
        hidden_size=128,
        max_ep_len=1000,
        n_layer=3,
        n_head=1,
        n_inner=512,
        activation_function="relu",
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        embd_pdrop=0.1,
        action_tanh=True,
    )


def torch_load(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def load_clean_dt(seed: int, device):
    path = Path(
        f"experiments/dt_stress/walker2d_medium/"
        f"clean/train_seed_{seed}/checkpoints/"
        f"checkpoint_step_100000.pt"
    )

    ckpt = torch_load(path)

    model = make_dt()

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model, path


def load_clean_joint_policy(seed: int, device):
    path = Path(
        f"experiments/dt_mtm/walker2d_medium_clean/"
        f"seed_{seed}/primary/checkpoints/"
        f"joint_step_100000.pt"
    )

    ckpt = torch_load(path)

    full = ckpt["model_state_dict"]

    dt_state = {
        key[len("dt."):]: value
        for key, value in full.items()
        if key.startswith("dt.")
    }

    model = make_dt()

    missing, unexpected = model.load_state_dict(
        dt_state,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "Joint DT extraction mismatch:\n"
            f"missing={missing}\n"
            f"unexpected={unexpected}"
        )

    model.to(device)
    model.eval()

    return model, path


def load_dataset(path: Path):
    with h5py.File(path, "r") as f:
        return {
            "observations": f["observations"][:],
            "actions": f["actions"][:],
            "rewards": f["rewards"][:],
            "terminals": f["terminals"][:].astype(bool),
            "timeouts": f["timeouts"][:].astype(bool),
        }


def build_metadata(data):
    done = data["terminals"] | data["timeouts"]

    n = len(done)

    traj_start = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    local_timestep = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    rtg = np.zeros(
        n,
        dtype=np.float64,
    )

    completed_end = 0
    start = 0
    trajectories = 0

    for i, flag in enumerate(done):
        if not flag:
            continue

        end = i + 1
        length = end - start

        idx = np.arange(start, end)

        traj_start[idx] = start
        local_timestep[idx] = np.arange(length)

        rewards = np.asarray(
            data["rewards"][start:end],
            dtype=np.float64,
        )

        rtg[start:end] = np.cumsum(
            rewards[::-1]
        )[::-1]

        start = end
        completed_end = end
        trajectories += 1

    if trajectories != 1190:
        raise RuntimeError(
            f"expected 1190 trajectories, got {trajectories}"
        )

    if completed_end != 999_995:
        raise RuntimeError(
            f"expected 999995 completed transitions, "
            f"got {completed_end}"
        )

    return {
        "traj_start": traj_start,
        "local_timestep": local_timestep,
        "rtg": rtg,
        "used_n": completed_end,
    }


def changed_mask(clean, poison, used_n):
    obs = np.any(
        clean["observations"][:used_n]
        != poison["observations"][:used_n],
        axis=1,
    )

    act = np.any(
        clean["actions"][:used_n]
        != poison["actions"][:used_n],
        axis=1,
    )

    return obs | act


def classify_endpoints(mask, meta):
    used_n = meta["used_n"]

    direct = np.flatnonzero(mask)

    history_only = []

    prefix = np.concatenate(
        ([0], np.cumsum(mask.astype(np.int64)))
    )

    for endpoint in range(used_n):
        if mask[endpoint]:
            continue

        start = max(
            int(meta["traj_start"][endpoint]),
            endpoint - K + 1,
        )

        count = int(
            prefix[endpoint + 1] - prefix[start]
        )

        if count > 0:
            history_only.append(endpoint)

    return (
        direct.astype(np.int64),
        np.asarray(
            history_only,
            dtype=np.int64,
        ),
    )


def deterministic_sample(
    values,
    *,
    seed,
):
    values = np.asarray(
        values,
        dtype=np.int64,
    )

    if len(values) <= MAX_PER_CLASS:
        return values.copy()

    rng = np.random.default_rng(seed)

    chosen = rng.choice(
        values,
        size=MAX_PER_CLASS,
        replace=False,
    )

    return np.sort(chosen)


def build_context_batch(
    data,
    endpoints,
    meta,
    state_mean,
    state_std,
):
    b = len(endpoints)

    states = np.zeros(
        (b, K, 17),
        dtype=np.float32,
    )

    actions = np.full(
        (b, K, 6),
        -10.0,
        dtype=np.float32,
    )

    rtg = np.zeros(
        (b, K, 1),
        dtype=np.float32,
    )

    timesteps = np.zeros(
        (b, K),
        dtype=np.int64,
    )

    attention = np.zeros(
        (b, K),
        dtype=np.float32,
    )

    for j, endpoint in enumerate(endpoints):
        traj_start = int(
            meta["traj_start"][endpoint]
        )

        start = max(
            traj_start,
            int(endpoint) - K + 1,
        )

        idx = np.arange(
            start,
            int(endpoint) + 1,
        )

        tlen = len(idx)
        pad = K - tlen

        s = (
            data["observations"][idx]
            - state_mean
        ) / state_std

        states[j, pad:] = s.astype(
            np.float32,
            copy=False,
        )

        actions[j, pad:] = data[
            "actions"
        ][idx].astype(
            np.float32,
            copy=False,
        )

        rtg[j, pad:, 0] = (
            meta["rtg"][idx]
            / RTG_SCALE
        ).astype(
            np.float32,
            copy=False,
        )

        local = meta[
            "local_timestep"
        ][idx].copy()

        local[local >= 1000] = 999

        timesteps[j, pad:] = local
        attention[j, pad:] = 1.0

    return (
        states,
        actions,
        rtg,
        timesteps,
        attention,
    )


def predict(
    model,
    batch,
    device,
):
    states, actions, rtg, timesteps, attention = batch

    with torch.inference_mode():
        states_t = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=device,
        )

        actions_t = torch.as_tensor(
            actions,
            dtype=torch.float32,
            device=device,
        )

        rtg_t = torch.as_tensor(
            rtg,
            dtype=torch.float32,
            device=device,
        )

        timestep_t = torch.as_tensor(
            timesteps,
            dtype=torch.long,
            device=device,
        )

        attention_t = torch.as_tensor(
            attention,
            dtype=torch.long,
            device=device,
        )

        _, action_pred, _ = model(
            states_t,
            actions_t,
            rtg_t,
            timestep_t,
            attention_t,
        )

        # Last valid position is always the endpoint because
        # contexts are left-padded.
        out = action_pred[:, -1, :]

    return out.detach().cpu().numpy()


def sensitivity_for_endpoints(
    *,
    dt_model,
    joint_model,
    clean,
    poison,
    endpoints,
    meta,
    state_mean,
    state_std,
    device,
):
    dt_values = []
    joint_values = []

    for start in range(
        0,
        len(endpoints),
        BATCH_SIZE,
    ):
        e = endpoints[
            start:start + BATCH_SIZE
        ]

        clean_batch = build_context_batch(
            clean,
            e,
            meta,
            state_mean,
            state_std,
        )

        poison_batch = build_context_batch(
            poison,
            e,
            meta,
            state_mean,
            state_std,
        )

        dt_clean = predict(
            dt_model,
            clean_batch,
            device,
        )

        dt_poison = predict(
            dt_model,
            poison_batch,
            device,
        )

        joint_clean = predict(
            joint_model,
            clean_batch,
            device,
        )

        joint_poison = predict(
            joint_model,
            poison_batch,
            device,
        )

        dt_shift = np.linalg.norm(
            dt_poison - dt_clean,
            axis=1,
        )

        joint_shift = np.linalg.norm(
            joint_poison - joint_clean,
            axis=1,
        )

        dt_values.append(dt_shift)
        joint_values.append(joint_shift)

    return (
        np.concatenate(dt_values),
        np.concatenate(joint_values),
    )


def summarize(dt, joint):
    if len(dt) == 0:
        return None

    return {
        "n": int(len(dt)),

        "dt_mean": float(np.mean(dt)),
        "dt_median": float(np.median(dt)),
        "dt_q90": float(np.quantile(dt, 0.90)),

        "joint_mean": float(np.mean(joint)),
        "joint_median": float(np.median(joint)),
        "joint_q90": float(
            np.quantile(joint, 0.90)
        ),

        "A_mean_joint_minus_dt": float(
            np.mean(joint) - np.mean(dt)
        ),

        "mean_ratio_joint_over_dt": float(
            np.mean(joint)
            / (np.mean(dt) + EPS)
        ),

        "fraction_joint_gt_dt": float(
            np.mean(joint > dt)
        ),
    }


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

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

    clean = load_dataset(CLEAN_DATA)

    meta = build_metadata(clean)

    with np.load(NORMALIZATION) as f:
        state_mean = f[
            "state_mean"
        ].astype(np.float32)

        state_std = f[
            "state_std"
        ].astype(np.float32)

    group4d = json.loads(
        GROUP4D.read_text(
            encoding="utf-8"
        )
    )

    G = {}

    for condition in ROOTS:
        for _, rho_key in RHOS:
            for r in group4d[
                "conditions"
            ][condition][rho_key]["rows"]:

                G[
                    (
                        condition,
                        float(rho_key),
                        int(r["seed"]),
                    )
                ] = float(
                    r["stress_response_gap"]
                )

    rows = []

    print()
    print("=" * 145)
    print(
        "GROUP 4E — FROZEN CLEAN-POLICY "
        "POISON SENSITIVITY"
    )
    print("=" * 145)

    print(
        "condition       rho seed       G   "
        "N_all   DTmean JointMean       A "
        "J>DT    "
        "A_direct   A_history"
    )

    for seed in SEEDS:
        dt_model, dt_path = load_clean_dt(
            seed,
            device,
        )

        joint_model, joint_path = (
            load_clean_joint_policy(
                seed,
                device,
            )
        )

        print()
        print(
            f"seed {seed}: DT={dt_path}"
        )
        print(
            f"seed {seed}: joint={joint_path}"
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

                mask = changed_mask(
                    clean,
                    poison,
                    meta["used_n"],
                )

                direct, history = (
                    classify_endpoints(
                        mask,
                        meta,
                    )
                )

                # Frozen deterministic analysis RNG.
                seed_base = (
                    910_000
                    + condition_index * 100_000
                    + int(rho * 10_000) * 10
                    + seed
                )

                direct_sample = (
                    deterministic_sample(
                        direct,
                        seed=seed_base + 1,
                    )
                )

                history_sample = (
                    deterministic_sample(
                        history,
                        seed=seed_base + 2,
                    )
                )

                all_sample = np.concatenate(
                    (
                        direct_sample,
                        history_sample,
                    )
                )

                # Evaluate each class separately so the
                # balanced aggregate can also be reproduced.
                dt_d, joint_d = (
                    sensitivity_for_endpoints(
                        dt_model=dt_model,
                        joint_model=joint_model,
                        clean=clean,
                        poison=poison,
                        endpoints=direct_sample,
                        meta=meta,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                )

                dt_h, joint_h = (
                    sensitivity_for_endpoints(
                        dt_model=dt_model,
                        joint_model=joint_model,
                        clean=clean,
                        poison=poison,
                        endpoints=history_sample,
                        meta=meta,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                )

                dt_all = np.concatenate(
                    (dt_d, dt_h)
                )

                joint_all = np.concatenate(
                    (joint_d, joint_h)
                )

                summary_all = summarize(
                    dt_all,
                    joint_all,
                )

                summary_direct = summarize(
                    dt_d,
                    joint_d,
                )

                summary_history = summarize(
                    dt_h,
                    joint_h,
                )

                g = G[
                    (condition, rho, seed)
                ]

                row = {
                    "condition": condition,
                    "rho": rho,
                    "seed": seed,
                    "G": g,

                    "direct_population": int(
                        len(direct)
                    ),

                    "history_only_population": int(
                        len(history)
                    ),

                    "direct_sample_size": int(
                        len(direct_sample)
                    ),

                    "history_only_sample_size": int(
                        len(history_sample)
                    ),

                    "all": summary_all,
                    "direct": summary_direct,
                    "history_only": summary_history,
                }

                rows.append(row)

                print(
                    f"{condition:15s} "
                    f"{rho:.2f} "
                    f"{seed:4d} "
                    f"{g:+8.4f} "
                    f"{summary_all['n']:7d} "
                    f"{summary_all['dt_mean']:8.5f} "
                    f"{summary_all['joint_mean']:9.5f} "
                    f"{summary_all['A_mean_joint_minus_dt']:+8.5f} "
                    f"{summary_all['fraction_joint_gt_dt']:5.3f} "
                    f"{summary_direct['A_mean_joint_minus_dt']:+10.5f} "
                    f"{summary_history['A_mean_joint_minus_dt']:+10.5f}"
                )

        # Free previous seed's models before loading next.
        del dt_model
        del joint_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=" * 145)
    print("DESCRIPTIVE CORRELATION WITH GROUP-4D G")
    print("=" * 145)

    G_values = [
        r["G"]
        for r in rows
    ]

    for scope in (
        "all",
        "direct",
        "history_only",
    ):
        A = [
            r[scope]["A_mean_joint_minus_dt"]
            for r in rows
        ]

        ratio = [
            r[scope]["mean_ratio_joint_over_dt"]
            for r in rows
        ]

        fraction = [
            r[scope]["fraction_joint_gt_dt"]
            for r in rows
        ]

        print(
            f"{scope:12s} "
            f"corr(G,A)="
            f"{pearson(G_values, A):+.4f} "
            f"corr(G,ratio)="
            f"{pearson(G_values, ratio):+.4f} "
            f"corr(G,fracJointGT)="
            f"{pearson(G_values, fraction):+.4f}"
        )

    out = Path(
        "experiments/dt_mtm_stress/"
        "group4e_clean_policy_sensitivity.json"
    )

    out.write_text(
        json.dumps(
            {
                "analysis": (
                    "group4e_clean_policy_sensitivity"
                ),
                "context_length": K,
                "max_sample_per_endpoint_class": (
                    MAX_PER_CLASS
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
