import numpy as np
import pytest
import torch

from src.methods.mtm.masking import (
    REFERENCE_MASK_RATIOS,
    REFERENCE_MODE_WEIGHTS,
    apply_reference_auto_frontier,
    create_reference_auto_mask,
    create_reference_full_random_mask,
    sample_reference_auto_mask,
)


def reference_data_shapes():
    # Exact insertion order used by the official
    # d4rl_cont tokenizer configuration:
    #
    # states -> actions -> returns
    #
    # ContinuousTokenizer creates one token per
    # timestep for every modality.
    return {
        "states": (1, 17),
        "actions": (1, 6),
        "returns": (1, 1),
    }


def test_full_random_mask_shape():
    rng = np.random.RandomState(
        0
    )

    mask = (
        create_reference_full_random_mask(
            (1, 17),
            traj_length=4,
            mask_ratios=0.5,
            rng=rng,
        )
    )

    assert mask.shape == (
        4,
        1,
    )


def test_full_random_ratio_is_initial_visible_fraction():
    rng = np.random.RandomState(
        0
    )

    mask = (
        create_reference_full_random_mask(
            (2, 17),
            traj_length=4,
            mask_ratios=0.5,
            rng=rng,
        )
    )

    # L = 4
    # tokens/time = 2
    # total = 8
    #
    # int(8 * 0.5) = 4 ONES.
    #
    # Since 1 means visible, the official
    # "mask_ratio" is initially a visible ratio.
    assert mask.shape == (
        4,
        2,
    )

    assert int(
        mask.sum().item()
    ) == 4


def test_full_random_mask_exact_seed():
    rng = np.random.RandomState(
        123
    )

    mask = (
        create_reference_full_random_mask(
            (1, 17),
            traj_length=4,
            mask_ratios=0.5,
            rng=rng,
        )
    )

    expected = torch.tensor(
        [
            [0.0],
            [1.0],
            [1.0],
            [0.0],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_allclose(
        mask,
        expected,
    )


def make_all_visible_masks():
    return {
        "states": torch.ones(
            4,
            1,
            dtype=torch.float64,
        ),
        "actions": torch.ones(
            4,
            1,
            dtype=torch.float64,
        ),
        "returns": torch.ones(
            4,
            1,
            dtype=torch.float64,
        ),
    }


def test_auto_frontier_when_action_is_selected():
    masks = (
        apply_reference_auto_frontier(
            make_all_visible_masks(),
            selected_mode="actions",
            selected_position=2,
        )
    )

    # Order is:
    # states -> returns -> actions
    #
    # states and returns are before action,
    # so they may stay visible AT position 2.
    #
    # action is hidden starting at position 2.

    expected_before = torch.tensor(
        [
            [1.0],
            [1.0],
            [1.0],
            [0.0],
        ],
        dtype=torch.float64,
    )

    expected_action = torch.tensor(
        [
            [1.0],
            [1.0],
            [0.0],
            [0.0],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_allclose(
        masks["states"],
        expected_before,
    )

    torch.testing.assert_allclose(
        masks["returns"],
        expected_before,
    )

    torch.testing.assert_allclose(
        masks["actions"],
        expected_action,
    )


def test_auto_frontier_when_return_is_selected():
    masks = (
        apply_reference_auto_frontier(
            make_all_visible_masks(),
            selected_mode="returns",
            selected_position=2,
        )
    )

    expected_states = torch.tensor(
        [
            [1.0],
            [1.0],
            [1.0],
            [0.0],
        ],
        dtype=torch.float64,
    )

    expected_hidden_from_current = (
        torch.tensor(
            [
                [1.0],
                [1.0],
                [0.0],
                [0.0],
            ],
            dtype=torch.float64,
        )
    )

    torch.testing.assert_allclose(
        masks["states"],
        expected_states,
    )

    torch.testing.assert_allclose(
        masks["returns"],
        expected_hidden_from_current,
    )

    torch.testing.assert_allclose(
        masks["actions"],
        expected_hidden_from_current,
    )


def test_auto_frontier_when_state_is_selected():
    masks = (
        apply_reference_auto_frontier(
            make_all_visible_masks(),
            selected_mode="states",
            selected_position=2,
        )
    )

    expected = torch.tensor(
        [
            [1.0],
            [1.0],
            [0.0],
            [0.0],
        ],
        dtype=torch.float64,
    )

    for key in (
        "states",
        "returns",
        "actions",
    ):
        torch.testing.assert_allclose(
            masks[key],
            expected,
        )


def test_auto_mask_exact_seed_123():
    rng = np.random.RandomState(
        123
    )

    result = (
        sample_reference_auto_mask(
            reference_data_shapes(),
            traj_length=4,
            mask_ratios=(
                REFERENCE_MASK_RATIOS
            ),
            mode_weights=(
                REFERENCE_MODE_WEIGHTS
            ),
            rng=rng,
        )
    )

    assert (
        result.selected_mode
        == "actions"
    )

    assert (
        result.selected_position
        == 2
    )

    expected_states = torch.tensor(
        [
            [1.0],
            [0.0],
            [0.0],
            [0.0],
        ],
        dtype=torch.float64,
    )

    expected_actions = torch.tensor(
        [
            [1.0],
            [1.0],
            [0.0],
            [0.0],
        ],
        dtype=torch.float64,
    )

    expected_returns = torch.tensor(
        [
            [0.0],
            [1.0],
            [1.0],
            [0.0],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_allclose(
        result.masks["states"],
        expected_states,
    )

    torch.testing.assert_allclose(
        result.masks["actions"],
        expected_actions,
    )

    torch.testing.assert_allclose(
        result.masks["returns"],
        expected_returns,
    )


def test_same_seed_gives_same_auto_mask():
    result_a = (
        sample_reference_auto_mask(
            reference_data_shapes(),
            traj_length=4,
            rng=np.random.RandomState(
                42
            ),
        )
    )

    result_b = (
        sample_reference_auto_mask(
            reference_data_shapes(),
            traj_length=4,
            rng=np.random.RandomState(
                42
            ),
        )
    )

    assert (
        result_a.selected_mode
        == result_b.selected_mode
    )

    assert (
        result_a.selected_position
        == result_b.selected_position
    )

    for key in (
        result_a.masks.keys()
    ):
        torch.testing.assert_allclose(
            result_a.masks[key],
            result_b.masks[key],
        )


def test_global_rng_matches_explicit_random_state():
    # np.random.seed() controls the same legacy RNG
    # family that RandomState reproduces.

    np.random.seed(
        123
    )

    global_result = (
        sample_reference_auto_mask(
            reference_data_shapes(),
            traj_length=4,
            rng=None,
        )
    )

    explicit_result = (
        sample_reference_auto_mask(
            reference_data_shapes(),
            traj_length=4,
            rng=np.random.RandomState(
                123
            ),
        )
    )

    assert (
        global_result.selected_mode
        == explicit_result.selected_mode
    )

    assert (
        global_result.selected_position
        == explicit_result.selected_position
    )

    for key in (
        global_result.masks.keys()
    ):
        torch.testing.assert_allclose(
            global_result.masks[key],
            explicit_result.masks[key],
        )


def test_auto_masks_have_reference_shapes():
    masks = (
        create_reference_auto_mask(
            reference_data_shapes(),
            traj_length=4,
            rng=np.random.RandomState(
                5
            ),
        )
    )

    assert set(
        masks.keys()
    ) == {
        "states",
        "actions",
        "returns",
    }

    for mask in masks.values():
        assert mask.shape == (
            4,
            1,
        )

        unique_values = set(
            mask.detach()
            .cpu()
            .numpy()
            .ravel()
            .tolist()
        )

        assert unique_values.issubset(
            {
                0.0,
                1.0,
            }
        )


@pytest.mark.parametrize(
    "ratio",
    [
        -0.1,
        1.1,
        float("nan"),
    ],
)
def test_invalid_mask_ratio_raises(
    ratio,
):
    with pytest.raises(
        ValueError
    ):
        create_reference_full_random_mask(
            (1, 17),
            traj_length=4,
            mask_ratios=ratio,
        )


def test_invalid_mode_weights_raise():
    with pytest.raises(
        ValueError
    ):
        sample_reference_auto_mask(
            reference_data_shapes(),
            traj_length=4,
            mode_weights=(
                0.2,
                0.1,
                0.6,
            ),
        )