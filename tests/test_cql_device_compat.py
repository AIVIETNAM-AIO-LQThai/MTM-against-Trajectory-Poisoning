from __future__ import annotations

import pytest
import torch

from src.baselines.cql.compat import (
    make_device_compatible_cql_trainer,
)


class DummyOfficialTrainer:
    def _get_tensor_values(
        self,
        obs,
        actions,
        network=None,
    ):
        assert actions.device == obs.device

        return network(
            obs,
            actions,
        )


class DummyNetwork:
    def __call__(
        self,
        obs,
        actions,
    ):
        return (
            obs.sum()
            + actions.sum()
        )


def test_compat_preserves_cpu_behavior():
    Trainer = (
        make_device_compatible_cql_trainer(
            DummyOfficialTrainer
        )
    )

    trainer = Trainer()

    obs = torch.tensor(
        [[1.0, 2.0]]
    )

    actions = torch.tensor(
        [[3.0, 4.0]]
    )

    result = trainer._get_tensor_values(
        obs,
        actions,
        network=DummyNetwork(),
    )

    assert result.item() == 10.0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_compat_moves_actions_to_cuda():
    Trainer = (
        make_device_compatible_cql_trainer(
            DummyOfficialTrainer
        )
    )

    trainer = Trainer()

    obs = torch.tensor(
        [[1.0, 2.0]],
        device="cuda",
    )

    actions = torch.tensor(
        [[3.0, 4.0]],
        device="cpu",
    )

    original_actions = actions.clone()

    result = trainer._get_tensor_values(
        obs,
        actions,
        network=DummyNetwork(),
    )

    assert result.device.type == "cuda"

    # The original CPU tensor is not mutated.
    assert actions.device.type == "cpu"

    torch.testing.assert_close(
        actions,
        original_actions,
    )

    assert result.item() == 10.0