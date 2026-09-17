from __future__ import annotations

from random import Random

import numpy as np
import torch

from src.data.mtm_batching import MTMShuffledWindowSampler
from src.methods.dt_mtm.calibration import choose_lambda_from_gradient_ratios
from src.methods.dt_mtm.clean_pipeline import (
    build_joint_model,
    build_mtm_tokenizers,
    prepare_clean_joint_data_from_arrays,
    sample_clean_joint_train_batch,
)
from src.methods.dt_mtm.walker2d import build_walker2d_dt


def _tiny_data():
    rng = np.random.default_rng(7)
    lengths = (6, 6, 6, 6)
    n = sum(lengths)

    observations = rng.normal(size=(n, 17)).astype(np.float32)
    actions = rng.normal(size=(n, 6)).astype(np.float32)
    rewards = rng.normal(size=n).astype(np.float32)
    terminals = np.zeros(n, dtype=bool)
    timeouts = np.zeros(n, dtype=bool)

    offset = 0
    for length in lengths:
        terminals[offset + length - 1] = True
        offset += length

    state_mean = observations.mean(axis=0)
    state_std = observations.std(axis=0) + 0.5

    return prepare_clean_joint_data_from_arrays(
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
        state_mean=state_mean,
        state_std=state_std,
        train_fraction=0.75,
    )


def test_joint_factory_preserves_dt_initialization_order():
    seed = 1234

    torch.manual_seed(seed)
    standalone = build_walker2d_dt()
    standalone_state = {
        key: value.detach().clone()
        for key, value in standalone.state_dict().items()
    }

    joint = build_joint_model(
        seed=seed,
        device=torch.device("cpu"),
    )

    assert standalone_state.keys() == joint.dt.state_dict().keys()
    for key, expected in standalone_state.items():
        torch.testing.assert_close(
            joint.dt.state_dict()[key],
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_clean_joint_batch_preserves_both_stream_shapes():
    data = _tiny_data()
    device = torch.device("cpu")
    tokenizers = build_mtm_tokenizers(data, device=device)

    mtm_sampler = MTMShuffledWindowSampler(
        data.mtm_ranges.train,
        seed=1000,
    )

    batch, audit = sample_clean_joint_train_batch(
        data,
        tokenizers,
        dt_np_rng=np.random.default_rng(0),
        dt_py_rng=Random(0),
        mtm_sampler=mtm_sampler,
        mask_rng=np.random.RandomState(2000),
        device=device,
        dt_batch_size=3,
        mtm_batch_size=2,
    )

    assert batch.states.shape == (3, 20, 17)
    assert batch.actions.shape == (3, 20, 6)
    assert batch.returns_to_go.shape == (3, 20, 1)
    assert batch.timesteps.shape == (3, 20)
    assert batch.attention_mask.shape == (3, 20)

    assert tuple(batch.mtm_trajectories["states"].shape) == (2, 4, 1, 17)
    assert tuple(batch.mtm_trajectories["actions"].shape) == (2, 4, 1, 6)
    assert tuple(batch.mtm_trajectories["returns"].shape) == (2, 4, 1, 1)

    assert audit.mtm_trajectory_ids.shape == (2,)
    assert np.all(audit.mtm_trajectory_ids < len(data.mtm_split.train_ids))


def test_lambda_selection_uses_largest_candidate_under_cap():
    result = choose_lambda_from_gradient_ratios(
        [2.0, 2.0, 2.0],
        [-0.2, 0.0, 0.2],
        candidates=(0.01, 0.03, 0.1, 0.3),
        target_max_scaled_ratio=0.25,
    )

    # median raw ratio = 2.0, so scaled ratios are
    # 0.02, 0.06, 0.20, 0.60. Largest admissible is 0.1.
    assert result.recommended_lambda == 0.1
    assert result.median_raw_gradient_ratio == 2.0
    assert result.candidate_scaled_ratios[0.1] == 0.2

def test_lambda_selection_preserves_zero_shared_gradient_samples():
    result = choose_lambda_from_gradient_ratios(
        [0.0, 0.0, 2.0, 2.0],
        [-0.2, 0.2],
        candidates=(0.1, 0.3),
        target_max_scaled_ratio=0.25,
    )

    # Zeros are valid mask outcomes and must participate in the median.
    # median([0, 0, 2, 2]) = 1.0. If zeros were discarded this would be 2.0.
    assert result.median_raw_gradient_ratio == 1.0
    assert result.recommended_lambda == 0.1


def test_lambda_selection_accepts_all_zero_shared_mtm_ratios():
    result = choose_lambda_from_gradient_ratios(
        [0.0, 0.0, 0.0],
        [],
        candidates=(0.1, 0.3, 1.0),
        target_max_scaled_ratio=0.25,
    )

    assert result.median_raw_gradient_ratio == 0.0
    assert result.recommended_lambda == 1.0
    assert np.isnan(result.median_cosine_similarity)

