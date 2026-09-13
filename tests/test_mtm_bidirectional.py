import torch

from src.methods.mtm.model import (
    MTMConfig,
    ReferenceMTM,
)


DATA_SHAPES = {
    "states": (1, 17),
    "actions": (1, 6),
    "returns": (1, 1),
}


def test_visible_future_changes_past_reconstruction():
    """
    MTM reconstruction must be bidirectional.

    State at t=0 is hidden.

    State at t=3 is visible.

    We modify only the visible future state.

    Reconstruction of the hidden state at t=0 should change.
    """

    torch.manual_seed(
        1234
    )

    model = ReferenceMTM(
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

    model.eval()

    torch.manual_seed(
        99
    )

    trajectories_a = {
        "states": torch.randn(
            1,
            4,
            1,
            17,
        ),
        "actions": torch.randn(
            1,
            4,
            1,
            6,
        ),
        "returns": torch.randn(
            1,
            4,
            1,
            1,
        ),
    }

    trajectories_b = {
        key: value.clone()
        for key, value
        in trajectories_a.items()
    }

    # Change only a visible FUTURE token.
    trajectories_b[
        "states"
    ][
        0,
        3,
        0,
        :,
    ] += 10.0

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

    with torch.no_grad():
        output_a = model(
            trajectories_a,
            masks,
        )

        output_b = model(
            trajectories_b,
            masks,
        )

    past_prediction_a = (
        output_a[
            "states"
        ][
            0,
            0,
            0,
            :,
        ]
    )

    past_prediction_b = (
        output_b[
            "states"
        ][
            0,
            0,
            0,
            :,
        ]
    )

    max_difference = (
        (
            past_prediction_a
            - past_prediction_b
        )
        .abs()
        .max()
        .item()
    )

    assert (
        max_difference
        > 1e-6
    ), (
        "Visible future information did not affect past "
        "reconstruction. The MTM path may accidentally "
        "have become causal."
    )