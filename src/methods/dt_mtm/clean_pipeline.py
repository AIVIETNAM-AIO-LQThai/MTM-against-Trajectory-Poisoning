from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Mapping

import h5py
import numpy as np
import torch

from src.data.batching import DTBatch, sample_dt_batch
from src.data.mtm_batching import (
    MTMBatch,
    MTMShuffledWindowSampler,
    MTMWindowRanges,
    build_split_window_ranges,
    get_mtm_batch,
    sample_mtm_batch,
)
from src.data.mtm_dataset import ReferenceMTMDataset
from src.data.mtm_split import MTMTrajectorySplit, reference_trajectory_split
from src.data.mtm_statistics import compute_reference_mtm_statistics
from src.data.trajectories import TrajectorySlice, find_completed_trajectories
from src.methods.dt.losses import masked_action_mse
from src.methods.dt_mtm.model import DTMTMModel
from src.methods.dt_mtm.trainer import DTMTMTrainBatch
from src.methods.dt_mtm.walker2d import (
    ACTION_DIM,
    DT_CONTEXT_LENGTH,
    MAX_EP_LEN,
    MTM_DISCOUNT,
    MTM_TRAIN_FRACTION,
    MTM_TRAJECTORY_LENGTH,
    STATE_DIM,
    build_walker2d_dt_mtm,
)
from src.methods.mtm.losses import reference_mtm_loss
from src.methods.mtm.masking import ReferenceAutoMaskSample, sample_reference_auto_mask
from src.methods.mtm.tokenizers import ContinuousTokenizer


RTG_SCALE = 1000.0

DATA_SHAPES = {
    "states": (1, STATE_DIM),
    "actions": (1, ACTION_DIM),
    "returns": (1, 1),
}

DEFAULT_DATASET_PATH = Path(
    "data/raw/walker2d-medium-v2/walker2d_medium-v2.hdf5"
)
DEFAULT_NORMALIZATION_PATH = Path(
    "data/metadata/walker2d_medium_normalization.npz"
)


@dataclass(frozen=True)
class CleanJointData:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    trajectories: tuple[TrajectorySlice, ...]
    trailing_transitions: int
    state_mean: np.ndarray
    state_std: np.ndarray
    mtm_dataset: ReferenceMTMDataset
    mtm_split: MTMTrajectorySplit
    mtm_ranges: MTMWindowRanges


@dataclass(frozen=True)
class JointBatchAudit:
    dt_trajectory_indices: np.ndarray
    dt_start_indices: np.ndarray
    mtm_dataset_indices: np.ndarray
    mtm_trajectory_ids: np.ndarray
    mtm_global_starts: np.ndarray
    mask_mode: str
    mask_position: int


@dataclass(frozen=True)
class CleanProbeMetrics:
    dt_action_mse: float
    mtm_total_loss: float
    mtm_state_loss: float
    mtm_action_loss: float
    mtm_return_loss: float


def prepare_clean_joint_data_from_arrays(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    *,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    train_fraction: float = MTM_TRAIN_FRACTION,
) -> CleanJointData:
    observations = np.asarray(observations)
    actions = np.asarray(actions)
    rewards = np.asarray(rewards)
    terminals = np.asarray(terminals, dtype=bool)
    timeouts = np.asarray(timeouts, dtype=bool)
    state_mean = np.asarray(state_mean)
    state_std = np.asarray(state_std)

    if observations.ndim != 2 or observations.shape[1] != STATE_DIM:
        raise ValueError(
            f"observations must have shape [N,{STATE_DIM}], got {observations.shape}"
        )
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(
            f"actions must have shape [N,{ACTION_DIM}], got {actions.shape}"
        )
    if state_mean.shape != (STATE_DIM,) or state_std.shape != (STATE_DIM,):
        raise ValueError("state normalization has unexpected shape")
    if np.any(state_std <= 0.0):
        raise ValueError("state_std must be strictly positive")

    trajectories, trailing = find_completed_trajectories(
        terminals,
        timeouts,
    )

    mtm_dataset = ReferenceMTMDataset(
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
        trajectory_length=MTM_TRAJECTORY_LENGTH,
        max_path_length=MAX_EP_LEN,
        discount=MTM_DISCOUNT,
    )
    mtm_split = reference_trajectory_split(
        mtm_dataset.trajectories,
        train_fraction=train_fraction,
    )
    mtm_ranges = build_split_window_ranges(
        mtm_dataset,
        mtm_split,
    )

    return CleanJointData(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
        trajectories=tuple(trajectories),
        trailing_transitions=int(trailing),
        state_mean=state_mean.astype(np.float32, copy=False),
        state_std=state_std.astype(np.float32, copy=False),
        mtm_dataset=mtm_dataset,
        mtm_split=mtm_split,
        mtm_ranges=mtm_ranges,
    )


def load_clean_walker2d(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    normalization_path: Path = DEFAULT_NORMALIZATION_PATH,
) -> CleanJointData:
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if not normalization_path.exists():
        raise FileNotFoundError(normalization_path)

    with h5py.File(dataset_path, "r") as handle:
        observations = handle["observations"][:]
        actions = handle["actions"][:]
        rewards = handle["rewards"][:]
        terminals = handle["terminals"][:].astype(bool)
        timeouts = handle["timeouts"][:].astype(bool)

    with np.load(normalization_path) as handle:
        state_mean = handle["state_mean"].copy()
        state_std = handle["state_std"].copy()
        norm_transitions = int(handle["num_training_transitions"])
        norm_trajectories = int(handle["num_trajectories"])

    data = prepare_clean_joint_data_from_arrays(
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
        state_mean=state_mean,
        state_std=state_std,
        train_fraction=MTM_TRAIN_FRACTION,
    )

    used_transitions = sum(item.length for item in data.trajectories)

    # Frozen Walker2d-medium-v2 identity checks from Groups 1 and 3.
    if len(data.trajectories) != 1190:
        raise RuntimeError(
            f"expected 1190 completed trajectories, got {len(data.trajectories)}"
        )
    if used_transitions != 999_995:
        raise RuntimeError(
            f"expected 999995 used transitions, got {used_transitions}"
        )
    if data.trailing_transitions != 5:
        raise RuntimeError(
            f"expected 5 trailing transitions, got {data.trailing_transitions}"
        )
    if norm_transitions != used_transitions:
        raise RuntimeError("normalization transition count mismatch")
    if norm_trajectories != len(data.trajectories):
        raise RuntimeError("normalization trajectory count mismatch")
    if len(data.mtm_split.train_trajectories) != 1130:
        raise RuntimeError("expected 1130 MTM training trajectories")
    if len(data.mtm_split.validation_trajectories) != 60:
        raise RuntimeError("expected 60 MTM validation trajectories")

    return data


def build_mtm_tokenizers(
    data: CleanJointData,
    *,
    device: torch.device,
) -> dict[str, ContinuousTokenizer]:
    statistics = compute_reference_mtm_statistics(
        data.observations,
        data.actions,
        data.mtm_dataset.returns,
        data.mtm_split.train_trajectories,
        max_path_length=MAX_EP_LEN,
    )

    return {
        key: ContinuousTokenizer.from_statistics(statistics[key]).to(device)
        for key in ("states", "actions", "returns")
    }


def build_joint_model(
    *,
    seed: int,
    device: torch.device,
) -> DTMTMModel:
    # DT is constructed first after seeding inside the factory. Therefore,
    # for a fixed seed, its initialization matches Group 1 before the MTM
    # branch is added. No pretrained Group-3 checkpoint is loaded.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return build_walker2d_dt_mtm().to(device)

def encode_mtm_batch(
    batch: MTMBatch,
    tokenizers: Mapping[str, ContinuousTokenizer],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    raw = {
        "states": torch.as_tensor(
            batch.states, dtype=torch.float32, device=device
        ),
        "actions": torch.as_tensor(
            batch.actions, dtype=torch.float32, device=device
        ),
        "returns": torch.as_tensor(
            batch.returns, dtype=torch.float32, device=device
        ),
    }

    return {
        key: tokenizers[key].encode(value)
        for key, value in raw.items()
    }


def _compose_train_batch(
    dt_batch: DTBatch,
    mtm_batch: MTMBatch,
    mask_sample: ReferenceAutoMaskSample,
    encoded_mtm: Mapping[str, torch.Tensor],
) -> tuple[DTMTMTrainBatch, JointBatchAudit]:
    batch = DTMTMTrainBatch(
        states=dt_batch.states,
        actions=dt_batch.actions,
        returns_to_go=dt_batch.rtg[:, :-1],
        timesteps=dt_batch.timesteps,
        attention_mask=dt_batch.attention_mask,
        mtm_trajectories=encoded_mtm,
        mtm_masks=mask_sample.masks,
    )

    audit = JointBatchAudit(
        dt_trajectory_indices=dt_batch.trajectory_indices.copy(),
        dt_start_indices=dt_batch.start_indices.copy(),
        mtm_dataset_indices=mtm_batch.dataset_indices.copy(),
        mtm_trajectory_ids=mtm_batch.trajectory_ids.copy(),
        mtm_global_starts=mtm_batch.global_starts.copy(),
        mask_mode=mask_sample.selected_mode,
        mask_position=int(mask_sample.selected_position),
    )
    return batch, audit


def sample_clean_joint_train_batch(
    data: CleanJointData,
    tokenizers: Mapping[str, ContinuousTokenizer],
    *,
    dt_np_rng: np.random.Generator,
    dt_py_rng: Random,
    mtm_sampler: MTMShuffledWindowSampler,
    mask_rng: np.random.RandomState,
    device: torch.device,
    dt_batch_size: int = 64,
    mtm_batch_size: int = 64,
) -> tuple[DTMTMTrainBatch, JointBatchAudit]:
    dt_batch = sample_dt_batch(
        data.observations,
        data.actions,
        data.rewards,
        data.terminals,
        list(data.trajectories),
        data.state_mean,
        data.state_std,
        batch_size=dt_batch_size,
        context_length=DT_CONTEXT_LENGTH,
        max_ep_len=MAX_EP_LEN,
        rtg_scale=RTG_SCALE,
        np_rng=dt_np_rng,
        py_rng=dt_py_rng,
    )

    mtm_indices = mtm_sampler.next_indices(mtm_batch_size)
    mtm_batch = get_mtm_batch(data.mtm_dataset, mtm_indices)

    train_count = len(data.mtm_split.train_ids)
    if not bool(np.all(mtm_batch.trajectory_ids < train_count)):
        raise RuntimeError("MTM validation leakage into joint training batch")

    encoded_mtm = encode_mtm_batch(
        mtm_batch,
        tokenizers,
        device=device,
    )
    mask_sample = sample_reference_auto_mask(
        DATA_SHAPES,
        traj_length=MTM_TRAJECTORY_LENGTH,
        device=device,
        rng=mask_rng,
    )

    return _compose_train_batch(
        dt_batch,
        mtm_batch,
        mask_sample,
        encoded_mtm,
    )


def build_fixed_clean_probe(
    data: CleanJointData,
    tokenizers: Mapping[str, ContinuousTokenizer],
    *,
    seed: int,
    device: torch.device,
    dt_batch_size: int = 256,
    mtm_batch_size: int = 256,
) -> DTMTMTrainBatch:
    dt_np_rng = np.random.default_rng(seed + 300_000)
    dt_py_rng = Random(seed + 300_000)

    dt_batch = sample_dt_batch(
        data.observations,
        data.actions,
        data.rewards,
        data.terminals,
        list(data.trajectories),
        data.state_mean,
        data.state_std,
        batch_size=dt_batch_size,
        context_length=DT_CONTEXT_LENGTH,
        max_ep_len=MAX_EP_LEN,
        rtg_scale=RTG_SCALE,
        np_rng=dt_np_rng,
        py_rng=dt_py_rng,
    )

    mtm_batch_rng = np.random.RandomState(seed + 400_000)
    mtm_batch = sample_mtm_batch(
        data.mtm_dataset,
        data.mtm_ranges.validation,
        batch_size=mtm_batch_size,
        rng=mtm_batch_rng,
    )
    encoded_mtm = encode_mtm_batch(
        mtm_batch,
        tokenizers,
        device=device,
    )

    mask_rng = np.random.RandomState(seed + 500_000)
    mask_sample = sample_reference_auto_mask(
        DATA_SHAPES,
        traj_length=MTM_TRAJECTORY_LENGTH,
        device=device,
        rng=mask_rng,
    )

    batch, _ = _compose_train_batch(
        dt_batch,
        mtm_batch,
        mask_sample,
        encoded_mtm,
    )
    return batch


def evaluate_clean_probe(
    model: DTMTMModel,
    batch: DTMTMTrainBatch,
    *,
    device: torch.device,
) -> CleanProbeMetrics:
    was_training = model.training
    model.eval()

    with torch.no_grad():
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=device)
        returns_to_go = torch.as_tensor(
            batch.returns_to_go, dtype=torch.float32, device=device
        )
        timesteps = torch.as_tensor(batch.timesteps, dtype=torch.long, device=device)
        attention_mask = torch.as_tensor(
            batch.attention_mask, dtype=torch.long, device=device
        )

        _, action_predictions, _ = model.forward_dt(
            states,
            actions,
            returns_to_go,
            timesteps,
            attention_mask,
        )
        dt_loss = masked_action_mse(
            action_predictions,
            actions,
            attention_mask,
        )

        trajectories = {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in batch.mtm_trajectories.items()
        }
        masks = {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in batch.mtm_masks.items()
        }
        mtm_predictions = model.forward_mtm(trajectories, masks)
        mtm_output = reference_mtm_loss(
            trajectories,
            mtm_predictions,
            masks,
        )

    if was_training:
        model.train()

    return CleanProbeMetrics(
        dt_action_mse=float(dt_loss.detach().cpu()),
        mtm_total_loss=float(mtm_output.total_loss.detach().cpu()),
        mtm_state_loss=float(mtm_output.full_losses["states"].detach().cpu()),
        mtm_action_loss=float(mtm_output.full_losses["actions"].detach().cpu()),
        mtm_return_loss=float(mtm_output.full_losses["returns"].detach().cpu()),
    )
