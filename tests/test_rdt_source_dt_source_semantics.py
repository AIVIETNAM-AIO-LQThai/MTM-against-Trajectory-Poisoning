from __future__ import annotations

import numpy as np
import torch

from src.methods.rdt_source_dt.data import (
    SourceSequenceDataset,
    SourceTrajectory,
    discounted_cumsum,
)
from src.methods.rdt_source_dt.losses import (
    source_masked_action_mse,
)
from src.methods.rdt_source_dt.model import (
    make_source_compatible_dt,
)


def make_toy_trajectory(
) -> SourceTrajectory:
    observations = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )

    actions = np.asarray(
        [
            [0.1],
            [0.2],
            [0.3],
        ],
        dtype=np.float32,
    )

    rewards = np.asarray(
        [
            1.0,
            2.0,
            3.0,
        ],
        dtype=np.float32,
    )

    returns = discounted_cumsum(
        rewards,
        gamma=1.0,
    )

    return SourceTrajectory(
        observations=(
            observations
        ),
        actions=actions,
        rewards=rewards,
        returns=returns,
    )


def test_discounted_cumsum_source_semantics():
    actual = discounted_cumsum(
        np.asarray(
            [
                1.0,
                2.0,
                3.0,
            ],
            dtype=np.float32,
        ),
        gamma=1.0,
    )

    np.testing.assert_array_equal(
        actual,
        np.asarray(
            [
                6.0,
                5.0,
                3.0,
            ],
            dtype=np.float32,
        ),
    )


def test_right_padding_and_timestep_semantics():
    trajectory = (
        make_toy_trajectory()
    )

    dataset = (
        SourceSequenceDataset(
            [
                trajectory
            ],
            seq_len=4,
            episode_len=10,
            reward_scale=0.001,
        )
    )

    (
        states,
        actions,
        returns,
        time_steps,
        mask,
    ) = dataset.prepare_sample(
        0,
        2,
    )

    np.testing.assert_array_equal(
        states,
        np.asarray(
            [
                [5.0, 6.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        actions,
        np.asarray(
            [
                [0.3],
                [0.0],
                [0.0],
                [0.0],
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_allclose(
        returns[:, 0],
        np.asarray(
            [
                0.003,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        time_steps,
        np.asarray(
            [
                2,
                3,
                4,
                5,
            ],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        mask,
        np.asarray(
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),
    )


def test_source_loss_keeps_padding_in_denominator():
    predictions = torch.ones(
        (
            1,
            2,
            2,
        ),
        dtype=torch.float32,
    )

    targets = torch.zeros_like(
        predictions
    )

    mask = torch.tensor(
        [
            [
                1.0,
                0.0,
            ]
        ],
        dtype=torch.float32,
    )

    loss = (
        source_masked_action_mse(
            predictions,
            targets,
            mask,
        )
    )

    # Two valid squared errors of 1 are divided by all
    # four B*K*A elements -> 0.5.
    assert torch.isclose(
        loss,
        torch.tensor(
            0.5
        ),
    )


def test_source_model_contract_and_forward():
    torch.manual_seed(
        0
    )

    model = (
        make_source_compatible_dt()
    )

    assert len(
        model.blocks
    ) == 3

    assert (
        model.embedding_dim
        == 128
    )

    assert (
        model.seq_len
        == 20
    )

    assert (
        model.timestep_embedding.num_embeddings
        == 1020
    )

    assert not hasattr(
        model,
        "embedding_dropout",
    )

    assert (
        model.predict_dropout.p
        == 0.1
    )

    for block in model.blocks:
        assert (
            block.attention.num_heads
            == 1
        )

        assert (
            block.attention.dropout
            == 0.0
        )

        assert (
            block.attention_residual_dropout.p
            == 0.1
        )

    batch = 2
    seq = 20

    states = torch.zeros(
        (
            batch,
            seq,
            17,
        )
    )

    actions = torch.zeros(
        (
            batch,
            seq,
            6,
        )
    )

    returns = torch.zeros(
        (
            batch,
            seq,
            1,
        )
    )

    time_steps = (
        torch.arange(
            seq
        )
        .repeat(
            batch,
            1,
        )
    )

    padding_mask = (
        torch.zeros(
            (
                batch,
                seq,
            ),
            dtype=torch.bool,
        )
    )

    predicted_actions = (
        model(
            states=states,
            actions=actions,
            returns_to_go=(
                returns
            ),
            time_steps=(
                time_steps
            ),
            padding_mask=(
                padding_mask
            ),
        )
    )

    assert (
        predicted_actions.shape
        == (
            batch,
            seq,
            6,
        )
    )

    assert bool(
        torch.isfinite(
            predicted_actions
        ).all()
    )


def test_source_linear_biases_and_layernorm_parameters():
    torch.manual_seed(
        0
    )

    model = (
        make_source_compatible_dt()
    )

    for module in (
        model.modules()
    ):
        if isinstance(
            module,
            torch.nn.Linear,
        ):
            if (
                module.bias
                is not None
            ):
                torch.testing.assert_close(
                    module.bias,
                    torch.zeros_like(
                        module.bias
                    ),
                )

        if isinstance(
            module,
            torch.nn.LayerNorm,
        ):
            torch.testing.assert_close(
                module.bias,
                torch.zeros_like(
                    module.bias
                ),
            )

            torch.testing.assert_close(
                module.weight,
                torch.ones_like(
                    module.weight
                ),
            )
