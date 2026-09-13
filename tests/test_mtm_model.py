import numpy as np
import pytest
import torch

from src.methods.mtm.masking import (
    sample_reference_auto_mask,
)

from src.methods.mtm.model import (
    MTMConfig,
    ReferenceMTM,
    make_1d_sincos_position_embedding,
)


DATA_SHAPES = {
    "states": (1, 17),
    "actions": (1, 6),
    "returns": (1, 1),
}


def make_tiny_model():
    torch.manual_seed(0)

    return ReferenceMTM(
        DATA_SHAPES,
        traj_length=4,
        config=MTMConfig(
            n_embd=32,
            n_head=4,
            n_enc_layer=1,
            n_dec_layer=1,
            dropout=0.0,
        ),
    )


def make_trajectories(
    batch_size=2,
):
    torch.manual_seed(1)

    return {
        "states": torch.randn(
            batch_size,
            4,
            1,
            17,
        ),
        "actions": torch.randn(
            batch_size,
            4,
            1,
            6,
        ),
        "returns": torch.randn(
            batch_size,
            4,
            1,
            1,
        ),
    }


def make_all_visible_masks():
    return {
        "states": torch.ones(
            4,
            1,
        ),
        "actions": torch.ones(
            4,
            1,
        ),
        "returns": torch.ones(
            4,
            1,
        ),
    }


def test_reference_default_config():
    config = MTMConfig()

    assert config.n_embd == 512
    assert config.n_head == 4
    assert config.n_enc_layer == 2
    assert config.n_dec_layer == 1
    assert config.dropout == 0.1


def test_position_embedding_shape():
    embedding = (
        make_1d_sincos_position_embedding(
            32,
            4,
        )
    )

    assert embedding.shape == (
        1,
        4,
        1,
        32,
    )


def test_position_zero_values():
    embedding = (
        make_1d_sincos_position_embedding(
            32,
            4,
        )
    )

    torch.testing.assert_allclose(
        embedding[
            0,
            0,
            0,
            :16,
        ],
        torch.zeros(
            16
        ),
    )

    torch.testing.assert_allclose(
        embedding[
            0,
            0,
            0,
            16:,
        ],
        torch.full(
            (16,),
            0.5,
        ),
    )


def test_all_visible_forward_shapes():
    model = make_tiny_model()
    model.eval()

    trajectories = (
        make_trajectories()
    )

    masks = (
        make_all_visible_masks()
    )

    with torch.no_grad():
        outputs = model(
            trajectories,
            masks,
        )

    assert set(
        outputs.keys()
    ) == set(
        DATA_SHAPES.keys()
    )

    for key in DATA_SHAPES:
        assert (
            outputs[key].shape
            == trajectories[key].shape
        )


def test_mixed_mask_forward_shapes():
    model = make_tiny_model()
    model.eval()

    trajectories = (
        make_trajectories()
    )

    masks = {
        "states": torch.tensor(
            [
                [0.0],
                [1.0],
                [1.0],
                [0.0],
            ]
        ),
        "actions": torch.tensor(
            [
                [1.0],
                [1.0],
                [0.0],
                [0.0],
            ]
        ),
        "returns": torch.tensor(
            [
                [1.0],
                [0.0],
                [1.0],
                [0.0],
            ]
        ),
    }

    with torch.no_grad():
        outputs = model(
            trajectories,
            masks,
        )

    for key in DATA_SHAPES:
        assert (
            outputs[key].shape
            == trajectories[key].shape
        )


def test_visible_token_restore_indices():
    embeddings = torch.tensor(
        [
            [
                [0.0],
                [1.0],
                [2.0],
                [3.0],
            ]
        ]
    )

    mask = torch.tensor(
        [
            1.0,
            0.0,
            1.0,
            0.0,
        ]
    )

    (
        visible,
        restore_indices,
        visible_count,
    ) = (
        ReferenceMTM
        ._select_visible_tokens(
            embeddings,
            mask,
        )
    )

    assert visible_count == 2

    torch.testing.assert_allclose(
        visible[
            0,
            :,
            0,
        ],
        torch.tensor(
            [
                0.0,
                2.0,
            ]
        ),
    )

    torch.testing.assert_close(
        restore_indices,
        torch.tensor(
            [
                0,
                2,
                1,
                3,
            ]
        ),
    )

    visible_plus_masks = (
        torch.tensor(
            [
                [
                    [0.0],
                    [2.0],
                    [-1.0],
                    [-1.0],
                ]
            ]
        )
    )

    restored = torch.gather(
        visible_plus_masks,
        dim=1,
        index=(
            restore_indices[
                None,
                :,
                None,
            ]
        ),
    )

    torch.testing.assert_allclose(
        restored[
            0,
            :,
            0,
        ],
        torch.tensor(
            [
                0.0,
                -1.0,
                2.0,
                -1.0,
            ]
        ),
    )


def test_reference_auto_mask_runs():
    model = make_tiny_model()
    model.eval()

    trajectories = (
        make_trajectories(
            batch_size=1
        )
    )

    sample = (
        sample_reference_auto_mask(
            DATA_SHAPES,
            traj_length=4,
            rng=np.random.RandomState(
                123
            ),
        )
    )

    with torch.no_grad():
        outputs = model(
            trajectories,
            sample.masks,
        )

    for key in DATA_SHAPES:
        assert (
            outputs[key].shape
            == trajectories[key].shape
        )


def test_all_hidden_mask_runs():
    """
    AUTO_MASK can generate a realization with no visible tokens.

    The decoder should still be able to run from learned mask
    tokens.
    """

    model = make_tiny_model()
    model.eval()

    trajectories = (
        make_trajectories(
            batch_size=1
        )
    )

    masks = {
        key: torch.zeros(
            4,
            1,
        )
        for key in DATA_SHAPES
    }

    with torch.no_grad():
        outputs = model(
            trajectories,
            masks,
        )

    for key in DATA_SHAPES:
        assert (
            outputs[key].shape
            == trajectories[key].shape
        )

        assert torch.isfinite(
            outputs[key]
        ).all()


def test_hidden_raw_value_does_not_leak():
    """
    Changing a token whose mask is zero must not affect the model.

    It is removed before the encoder and replaced by a learned
    mask token before decoding.
    """

    model = make_tiny_model()
    model.eval()

    trajectories_a = (
        make_trajectories(
            batch_size=1
        )
    )

    trajectories_b = {
        key: value.clone()
        for key, value
        in trajectories_a.items()
    }

    trajectories_b[
        "states"
    ][
        :,
        3,
        :,
        :,
    ] += 1000.0

    masks = {
        "states": torch.tensor(
            [
                [1.0],
                [1.0],
                [1.0],
                [0.0],
            ]
        ),
        "actions": torch.ones(
            4,
            1,
        ),
        "returns": torch.ones(
            4,
            1,
        ),
    }

    with torch.no_grad():
        output_a = model(
            trajectories_a,
            masks,
        )

        output_b = model(
            trajectories_b,
            masks,
        )

    for key in DATA_SHAPES:
        torch.testing.assert_allclose(
            output_a[key],
            output_b[key],
            rtol=0.0,
            atol=1e-6,
        )


def test_gradients_reach_all_major_components():
    model = make_tiny_model()
    model.train()

    trajectories = (
        make_trajectories(
            batch_size=2
        )
    )

    masks = {
        "states": torch.tensor(
            [
                [0.0],
                [1.0],
                [1.0],
                [1.0],
            ]
        ),
        "actions": torch.ones(
            4,
            1,
        ),
        "returns": torch.ones(
            4,
            1,
        ),
    }

    outputs = model(
        trajectories,
        masks,
    )

    temporary_loss = sum(
        output.square().mean()
        for output
        in outputs.values()
    )

    temporary_loss.backward()

    parameters = [
        model.encoder_embed[
            "states"
        ].weight,
        model.encoder.layers[
            0
        ].self_attn.in_proj_weight,
        model.decoder.layers[
            0
        ].self_attn.in_proj_weight,
        model.output_heads[
            "actions"
        ][-1].weight,
        model.mask_tokens[
            "states"
        ],
    ]

    for parameter in parameters:
        assert (
            parameter.grad
            is not None
        )

        assert (
            parameter.grad
            .abs()
            .sum()
            .item()
            > 0.0
        )


def test_wrong_modalities_raise():
    model = make_tiny_model()

    trajectories = (
        make_trajectories()
    )

    trajectories.pop(
        "returns"
    )

    masks = {
        "states": torch.ones(
            4,
            1,
        ),
        "actions": torch.ones(
            4,
            1,
        ),
    }

    with pytest.raises(
        ValueError
    ):
        model(
            trajectories,
            masks,
        )