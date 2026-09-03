from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.attacks.csdpc.metadata import logical_dataset_sha256
from src.baselines.cql.dataset_adapter import build_cql_training_view
from src.data.hdf5_io import load_hdf5_dataset
from src.baselines.cql.compat import (
    COMPATIBILITY_FIX_ID,
    make_device_compatible_cql_trainer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "cql"
    / "gate_b_walker2d_medium_v2.json"
)

DEFAULT_CQL_ROOT = (
    PROJECT_ROOT.parent
    / "CQL-official"
)

EXPECTED_CQL_COMMIT = (
    "d67dbe9cf5d2b96e3b462b6146f249b3d6569796"
)


def _json_sha256(data: Any) -> str:
    payload = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def read_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_cql_reference(cql_root: Path) -> str:
    cql_root = Path(cql_root).expanduser().resolve()

    if not cql_root.exists():
        raise FileNotFoundError(
            f"CQL reference repo not found: {cql_root}"
        )

    commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(cql_root),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    if commit != EXPECTED_CQL_COMMIT:
        raise RuntimeError(
            "Unexpected CQL reference commit.\n"
            f"Expected: {EXPECTED_CQL_COMMIT}\n"
            f"Actual:   {commit}"
        )

    return commit


def install_cql_reference_on_path(cql_root: Path) -> Path:
    cql_root = Path(cql_root).expanduser().resolve()

    d4rl_source_root = cql_root / "d4rl"

    if not d4rl_source_root.exists():
        raise FileNotFoundError(
            f"Missing frozen CQL d4rl source: {d4rl_source_root}"
        )

    source = str(d4rl_source_root)

    if source not in sys.path:
        sys.path.insert(0, source)

    return d4rl_source_root


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_environment(env, seed: int) -> None:
    if hasattr(env, "seed"):
        env.seed(seed)

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


def load_verified_training_view(
    dataset_path: Path,
):
    raw_dataset = load_hdf5_dataset(
        Path(dataset_path)
    )

    training_view = build_cql_training_view(
        raw_dataset
    )

    return raw_dataset, training_view


def populate_replay_buffer(
    replay_buffer,
    training_view,
) -> None:
    learner_keys = {
        "observations",
        "actions",
        "next_observations",
        "rewards",
        "terminals",
    }

    missing = learner_keys - set(training_view)

    if missing:
        raise KeyError(
            f"Missing CQL learner fields: {sorted(missing)}"
        )

    replay_buffer._observations = np.asarray(
        training_view["observations"]
    )

    replay_buffer._next_obs = np.asarray(
        training_view["next_observations"]
    )

    replay_buffer._actions = np.asarray(
        training_view["actions"]
    )

    replay_buffer._rewards = np.asarray(
        training_view["rewards"]
    ).reshape(-1, 1)

    replay_buffer._terminals = np.asarray(
        training_view["terminals"]
    ).reshape(-1, 1)

    replay_buffer._size = int(
        len(training_view["terminals"])
    )

    replay_buffer._top = replay_buffer._size


def build_components(
    *,
    dataset_path: Path,
    seed: int,
    config_path: Path = DEFAULT_CONFIG,
    cql_root: Path = DEFAULT_CQL_ROOT,
    use_gpu: bool = True,
    gpu_id: int = 0,
):
    cql_root = Path(cql_root)

    cql_commit = verify_cql_reference(
        cql_root
    )

    install_cql_reference_on_path(
        cql_root
    )

    import gym
    import d4rl  # noqa: F401

    import rlkit.torch.pytorch_util as ptu

    from rlkit.data_management.env_replay_buffer import (
        EnvReplayBuffer,
    )

    from rlkit.samplers.data_collector import (
        CustomMDPPathCollector,
        MdpPathCollector,
    )

    from rlkit.torch.networks import FlattenMlp

    from rlkit.torch.sac.cql import CQLTrainer as OfficialCQLTrainer

    from rlkit.torch.sac.policies import (
        MakeDeterministic,
        TanhGaussianPolicy,
    )

    from rlkit.torch.torch_rl_algorithm import (
        TorchBatchRLAlgorithm,
    )

    CQLTrainer = make_device_compatible_cql_trainer(OfficialCQLTrainer)

    config = read_config(
        Path(config_path)
    )

    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU requested but torch.cuda.is_available() is False."
        )

    ptu.set_gpu_mode(
        use_gpu,
        gpu_id=gpu_id,
    )

    if use_gpu:
        torch.cuda.set_device(gpu_id)

    # Project reproducibility fix.
    seed_everything(seed)

    env = gym.make(
        config["env_name"]
    )

    seed_environment(
        env,
        seed,
    )

    # Ensure network initialization is fixed even if
    # environment construction consumed global RNG state.
    seed_everything(seed)

    obs_dim = int(
        env.observation_space.low.size
    )

    action_dim = int(
        env.action_space.low.size
    )

    hidden = int(
        config["layer_size"]
    )

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[
            hidden,
            hidden,
            hidden,
        ],
    )

    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[
            hidden,
            hidden,
            hidden,
        ],
    )

    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[
            hidden,
            hidden,
            hidden,
        ],
    )

    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[
            hidden,
            hidden,
            hidden,
        ],
    )

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[
            hidden,
            hidden,
            hidden,
        ],
    )

    eval_policy = MakeDeterministic(
        policy
    )

    eval_collector = MdpPathCollector(
        env,
        eval_policy,
    )

    expl_collector = CustomMDPPathCollector(
        env,
    )

    raw_dataset, training_view = (
        load_verified_training_view(
            Path(dataset_path)
        )
    )

    replay_buffer = EnvReplayBuffer(
        config["replay_buffer_size"],
        env,
    )

    populate_replay_buffer(
        replay_buffer,
        training_view,
    )

    trainer = CQLTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **config["trainer_kwargs"],
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_collector,
        evaluation_data_collector=eval_collector,
        replay_buffer=replay_buffer,
        eval_both=True,
        batch_rl=True,
        **config["algorithm_kwargs"],
    )

    algorithm.to(
        ptu.device
    )

    metadata = {
        "cql_reference_commit": cql_commit,
        "config": config,
        "config_sha256": _json_sha256(
            config
        ),
        "compatibility_fixes": [
            COMPATIBILITY_FIX_ID,
        ],
        "dataset_path": str(
            Path(dataset_path).resolve()
        ),
        "dataset_logical_sha256": (
            logical_dataset_sha256(
                raw_dataset
            )
        ),
        "raw_transition_count": int(
            len(
                raw_dataset[
                    "observations"
                ]
            )
        ),
        "cql_training_transition_count": int(
            len(
                training_view[
                    "observations"
                ]
            )
        ),
        "seed": int(seed),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "device": str(
            ptu.device
        ),
    }

    return {
        "env": env,
        "policy": policy,
        "eval_policy": eval_policy,
        "qf1": qf1,
        "qf2": qf2,
        "target_qf1": target_qf1,
        "target_qf2": target_qf2,
        "trainer": trainer,
        "algorithm": algorithm,
        "replay_buffer": replay_buffer,
        "training_view": training_view,
        "metadata": metadata,
        "ptu": ptu,
    }


def evaluate_policy(
    *,
    env,
    policy,
    episodes: int = 10,
    max_path_length: int = 1000,
):
    returns = []

    for _ in range(episodes):
        obs = env.reset()

        if isinstance(obs, tuple):
            obs = obs[0]

        episode_return = 0.0

        for _ in range(max_path_length):
            action, _ = policy.get_action(
                obs
            )

            step_result = env.step(
                action
            )

            if len(step_result) == 5:
                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = step_result

                done = bool(
                    terminated
                    or truncated
                )

            else:
                (
                    obs,
                    reward,
                    done,
                    _,
                ) = step_result

            episode_return += float(
                reward
            )

            if done:
                break

        returns.append(
            episode_return
        )

    returns_array = np.asarray(
        returns,
        dtype=np.float64,
    )

    return {
        "episodes": int(episodes),
        "returns": [
            float(value)
            for value in returns_array
        ],
        "mean_raw_return": float(
            returns_array.mean()
        ),
        "std_raw_return": float(
            returns_array.std()
        ),
    }


def run_cql(
    *,
    dataset_path: Path,
    seed: int,
    condition: str,
    rho: float,
    attack_seed: Optional[int],
    output_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    cql_root: Path = DEFAULT_CQL_ROOT,
    gpu_id: int = 0,
):
    components = build_components(
        dataset_path=dataset_path,
        seed=seed,
        config_path=config_path,
        cql_root=cql_root,
        use_gpu=True,
        gpu_id=gpu_id,
    )

    config = components[
        "metadata"
    ]["config"]

    if condition == "clean":
        run_name = (
            f"clean_model_seed_{seed}"
        )

    elif condition == "csdpc":
        if attack_seed is None:
            raise ValueError(
                "CSDPC run requires attack_seed."
            )

        rho_code = (
            f"{int(round(rho * 100)):03d}"
        )

        run_name = (
            f"csdpc_rho_{rho_code}"
            f"_attack_seed_{attack_seed}"
            f"_model_seed_{seed}"
        )

    else:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    run_dir = (
        Path(output_root)
        / run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        **components["metadata"],
        "condition": condition,
        "rho": float(rho),
        "attack_seed": attack_seed,
        "run_name": run_name,
        "run_dir": str(
            run_dir.resolve()
        ),
        "started_unix_time": time.time(),
    }

    manifest_path = (
        run_dir
        / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    from rlkit.launchers.launcher_util import (
        setup_logger,
    )

    setup_logger(
        run_name,
        variant=manifest,
        log_dir=str(run_dir),
    )

    components[
        "algorithm"
    ].train()

    evaluation = evaluate_policy(
        env=components["env"],
        policy=components[
            "eval_policy"
        ],
        episodes=10,
        max_path_length=config[
            "algorithm_kwargs"
        ][
            "max_path_length"
        ],
    )

    import d4rl

    normalized = float(
        d4rl.get_normalized_score(
            config["env_name"],
            evaluation[
                "mean_raw_return"
            ],
        )
        * 100.0
    )

    result = {
        **manifest,
        "finished_unix_time": (
            time.time()
        ),
        "evaluation": {
            **evaluation,
            "d4rl_normalized_return": (
                normalized
            ),
        },
    }

    (
        run_dir
        / "result.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    torch.save(
        {
            "policy": components[
                "policy"
            ].state_dict(),
            "qf1": components[
                "qf1"
            ].state_dict(),
            "qf2": components[
                "qf2"
            ].state_dict(),
            "target_qf1": components[
                "target_qf1"
            ].state_dict(),
            "target_qf2": components[
                "target_qf2"
            ].state_dict(),
        },
        run_dir
        / "final_checkpoint.pt",
    )

    components["env"].close()

    return result