import copy

import numpy as np

from scripts.analyze_e1_matched_random_control import (
    matched_random_sign_control,
    validate_control,
)


def fake_dataset():
    observations = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    actions = np.asarray(
        [
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.4, 0.5],
        ],
        dtype=np.float32,
    )

    rewards = np.asarray(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float32,
    )

    terminals = np.asarray(
        [False, False, False, True],
    )

    timeouts = np.asarray(
        [False, False, False, False],
    )

    clean = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
    }

    poison = copy.deepcopy(
        clean
    )

    poison["observations"] = (
        observations.copy()
    )

    poison["observations"][1] += (
        np.asarray(
            [0.2, -0.4, 0.1],
            dtype=np.float32,
        )
    )

    poison["observations"][3] += (
        np.asarray(
            [-0.5, 0.3, 0.25],
            dtype=np.float32,
        )
    )

    return clean, poison


def test_control_preserves_absolute_component_magnitude():
    clean, poison = fake_dataset()

    control = matched_random_sign_control(
        clean,
        poison,
        used_n=4,
        seed=12345,
    )

    real_delta = (
        poison["observations"]
        - clean["observations"]
    )

    control_delta = (
        control["observations"]
        - clean["observations"]
    )

    np.testing.assert_allclose(
        np.abs(control_delta),
        np.abs(real_delta),
        atol=2e-6,
        rtol=0.0,
    )


def test_control_preserves_modified_indices():
    clean, poison = fake_dataset()

    control = matched_random_sign_control(
        clean,
        poison,
        used_n=4,
        seed=12345,
    )

    real_mask = np.any(
        poison["observations"]
        != clean["observations"],
        axis=1,
    )

    control_mask = np.any(
        control["observations"]
        != clean["observations"],
        axis=1,
    )

    np.testing.assert_array_equal(
        control_mask,
        real_mask,
    )


def test_control_is_deterministic():
    clean, poison = fake_dataset()

    first = matched_random_sign_control(
        clean,
        poison,
        used_n=4,
        seed=9876,
    )

    second = matched_random_sign_control(
        clean,
        poison,
        used_n=4,
        seed=9876,
    )

    np.testing.assert_array_equal(
        first["observations"],
        second["observations"],
    )


def test_control_does_not_modify_other_modalities():
    clean, poison = fake_dataset()

    control = matched_random_sign_control(
        clean,
        poison,
        used_n=4,
        seed=12345,
    )

    np.testing.assert_array_equal(
        control["actions"],
        clean["actions"],
    )

    np.testing.assert_array_equal(
        control["rewards"],
        clean["rewards"],
    )

    np.testing.assert_array_equal(
        control["terminals"],
        clean["terminals"],
    )

    np.testing.assert_array_equal(
        control["timeouts"],
        clean["timeouts"],
    )


def test_control_validation_passes():
    clean, poison = fake_dataset()

    control = matched_random_sign_control(
        clean,
        poison,
        used_n=4,
        seed=12345,
    )

    result = validate_control(
        clean,
        poison,
        control,
        used_n=4,
    )

    assert (
        result[
            "same_state_modified_indices"
        ]
        is True
    )

    assert (
        result[
            "max_component_abs_magnitude_error"
        ]
        <= 2e-6
    )

    assert (
        result["max_l2_error"]
        <= 2e-6
    )

    assert (
        result["max_linf_error"]
        <= 2e-6
    )