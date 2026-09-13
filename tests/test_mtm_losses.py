import math

import pytest
import torch

from src.methods.mtm.losses import (
    reference_mtm_loss,
)


def test_full_loss_matches_hand_calculation():
    """
    One modality:

    t0 prediction error:
        [1,2] -> squares [1,4]

    t1 prediction error:
        [3,4] -> squares [9,16]

    total squared error:
        30

    four elements:
        full MSE = 30 / 4 = 7.5
    """

    targets = {
        "states": torch.zeros(
            1,
            2,
            1,
            2,
        )
    }

    predictions = {
        "states": torch.tensor(
            [
                [
                    [
                        [1.0, 2.0]
                    ],
                    [
                        [3.0, 4.0]
                    ],
                ]
            ]
        )
    }

    masks = {
        "states": torch.tensor(
            [
                [1.0],
                [0.0],
            ]
        )
    }

    result = reference_mtm_loss(
        targets,
        predictions,
        masks,
    )

    torch.testing.assert_close(
        result.full_losses[
            "states"
        ],
        torch.tensor(
            7.5
        ),
    )

    torch.testing.assert_close(
        result.total_loss,
        torch.tensor(
            7.5
        ),
    )


def test_visible_and_masked_diagnostics_match_reference_scaling():
    """
    Same example:

    visible t0 squared errors:
        1 + 4 = 5

    visible TOKEN count:
        1

    reference visible diagnostic:
        5 / 1 = 5

    hidden t1 squared errors:
        9 + 16 = 25

    hidden TOKEN count:
        1

    reference masked diagnostic:
        25 / 1 = 25

    Note that the denominator does NOT include D=2.
    """

    targets = {
        "states": torch.zeros(
            1,
            2,
            1,
            2,
        )
    }

    predictions = {
        "states": torch.tensor(
            [
                [
                    [
                        [1.0, 2.0]
                    ],
                    [
                        [3.0, 4.0]
                    ],
                ]
            ]
        )
    }

    masks = {
        "states": torch.tensor(
            [
                [1.0],
                [0.0],
            ]
        )
    }

    result = reference_mtm_loss(
        targets,
        predictions,
        masks,
    )

    torch.testing.assert_close(
        result.visible_losses[
            "states"
        ],
        torch.tensor(
            5.0
        ),
    )

    torch.testing.assert_close(
        result.masked_losses[
            "states"
        ],
        torch.tensor(
            25.0
        ),
    )


def test_total_loss_is_sum_of_modality_full_losses():
    targets = {
        "states": torch.zeros(
            1,
            2,
            1,
            2,
        ),

        "actions": torch.zeros(
            1,
            2,
            1,
            1,
        ),

        "returns": torch.zeros(
            1,
            2,
            1,
            1,
        ),
    }

    predictions = {
        # Full loss:
        # (1 + 4 + 9 + 16) / 4 = 7.5
        "states": torch.tensor(
            [
                [
                    [[1.0, 2.0]],
                    [[3.0, 4.0]],
                ]
            ]
        ),

        # Full loss:
        # (1 + 9) / 2 = 5
        "actions": torch.tensor(
            [
                [
                    [[1.0]],
                    [[3.0]],
                ]
            ]
        ),

        # Full loss:
        # (4 + 4) / 2 = 4
        "returns": torch.tensor(
            [
                [
                    [[2.0]],
                    [[2.0]],
                ]
            ]
        ),
    }

    masks = {
        key: torch.tensor(
            [
                [1.0],
                [0.0],
            ]
        )
        for key in targets
    }

    result = reference_mtm_loss(
        targets,
        predictions,
        masks,
    )

    torch.testing.assert_close(
        result.full_losses[
            "states"
        ],
        torch.tensor(
            7.5
        ),
    )

    torch.testing.assert_close(
        result.full_losses[
            "actions"
        ],
        torch.tensor(
            5.0
        ),
    )

    torch.testing.assert_close(
        result.full_losses[
            "returns"
        ],
        torch.tensor(
            4.0
        ),
    )

    torch.testing.assert_close(
        result.total_loss,
        torch.tensor(
            16.5
        ),
    )


def test_training_objective_includes_hidden_and_visible_positions():
    """
    Official training objective is FULL reconstruction.

    Therefore both visible and hidden prediction values must
    receive gradients.
    """

    target = torch.zeros(
        1,
        2,
        1,
        1,
    )

    prediction = torch.tensor(
        [
            [
                [[1.0]],
                [[2.0]],
            ]
        ],
        requires_grad=True,
    )

    result = reference_mtm_loss(
        {
            "actions": target
        },
        {
            "actions": prediction
        },
        {
            "actions": torch.tensor(
                [
                    [1.0],
                    [0.0],
                ]
            )
        },
    )

    result.total_loss.backward()

    assert prediction.grad is not None

    # Visible position.
    assert (
        prediction.grad[
            0,
            0,
            0,
            0,
        ].abs().item()
        > 0.0
    )

    # Hidden position.
    assert (
        prediction.grad[
            0,
            1,
            0,
            0,
        ].abs().item()
        > 0.0
    )


def test_all_visible_has_undefined_masked_diagnostic():
    target = torch.zeros(
        1,
        2,
        1,
        1,
    )

    prediction = torch.ones_like(
        target
    )

    result = reference_mtm_loss(
        {
            "returns": target
        },
        {
            "returns": prediction
        },
        {
            "returns": torch.ones(
                2,
                1,
            )
        },
    )

    assert math.isnan(
        result.masked_losses[
            "returns"
        ].item()
    )

    assert torch.isfinite(
        result.total_loss
    )


def test_all_hidden_has_undefined_visible_diagnostic():
    target = torch.zeros(
        1,
        2,
        1,
        1,
    )

    prediction = torch.ones_like(
        target
    )

    result = reference_mtm_loss(
        {
            "returns": target
        },
        {
            "returns": prediction
        },
        {
            "returns": torch.zeros(
                2,
                1,
            )
        },
    )

    assert math.isnan(
        result.visible_losses[
            "returns"
        ].item()
    )

    assert torch.isfinite(
        result.total_loss
    )


def test_loss_keys_select_training_objective_only():
    targets = {
        "states": torch.zeros(
            1,
            1,
            1,
            1,
        ),

        "actions": torch.zeros(
            1,
            1,
            1,
            1,
        ),
    }

    predictions = {
        "states": torch.tensor(
            [[[[2.0]]]]
        ),

        "actions": torch.tensor(
            [[[[3.0]]]]
        ),
    }

    masks = {
        "states": torch.tensor(
            [[1.0]]
        ),

        "actions": torch.tensor(
            [[1.0]]
        ),
    }

    result = reference_mtm_loss(
        targets,
        predictions,
        masks,
        loss_keys=(
            "actions",
        ),
    )

    # state full loss = 4
    # action full loss = 9
    #
    # selected objective = action only
    torch.testing.assert_close(
        result.total_loss,
        torch.tensor(
            9.0
        ),
    )

    # Diagnostics/full loss are still computed
    # for both modalities.
    assert set(
        result.full_losses.keys()
    ) == {
        "states",
        "actions",
    }


def test_wrong_prediction_shape_raises():
    targets = {
        "actions": torch.zeros(
            1,
            2,
            1,
            6,
        )
    }

    predictions = {
        "actions": torch.zeros(
            1,
            2,
            1,
            5,
        )
    }

    masks = {
        "actions": torch.ones(
            2,
            1,
        )
    }

    with pytest.raises(
        ValueError
    ):
        reference_mtm_loss(
            targets,
            predictions,
            masks,
        )


def test_unknown_loss_key_raises():
    target = torch.zeros(
        1,
        1,
        1,
        1,
    )

    with pytest.raises(
        ValueError
    ):
        reference_mtm_loss(
            {
                "returns": target
            },
            {
                "returns": target.clone()
            },
            {
                "returns": torch.ones(
                    1,
                    1,
                )
            },
            loss_keys=(
                "not_a_modality",
            ),
        )

def test_reference_model_output_can_use_reference_loss():
    from src.methods.mtm.model import (
        MTMConfig,
        ReferenceMTM,
    )

    torch.manual_seed(
        123
    )

    data_shapes = {
        "states": (1, 17),
        "actions": (1, 6),
        "returns": (1, 1),
    }

    model = ReferenceMTM(
        data_shapes,
        traj_length=4,
        config=MTMConfig(
            n_embd=32,
            n_head=4,
            n_enc_layer=1,
            n_dec_layer=1,
            dropout=0.0,
        ),
    )

    trajectories = {
        "states": torch.randn(
            2,
            4,
            1,
            17,
        ),

        "actions": torch.randn(
            2,
            4,
            1,
            6,
        ),

        "returns": torch.randn(
            2,
            4,
            1,
            1,
        ),
    }

    masks = {
        "states": torch.tensor(
            [
                [1.0],
                [0.0],
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

    predictions = model(
        trajectories,
        masks,
    )

    result = reference_mtm_loss(
        trajectories,
        predictions,
        masks,
    )

    assert torch.isfinite(
        result.total_loss
    )

    result.total_loss.backward()

    assert (
        model.encoder_embed[
            "states"
        ].weight.grad
        is not None
    )

    assert (
        model.decoder.layers[
            0
        ].self_attn.in_proj_weight.grad
        is not None
    )

    assert (
        model.output_heads[
            "actions"
        ][-1].weight.grad
        is not None
    )