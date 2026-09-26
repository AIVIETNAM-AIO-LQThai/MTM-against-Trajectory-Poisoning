import pytest
import torch

from src.methods.mtm.optim import (
    create_reference_mtm_optimizer_and_scheduler,
    reference_mtm_lr_multiplier,
)


def test_reference_schedule_start():
    assert (
        reference_mtm_lr_multiplier(
            0,
            warmup_steps=40_000,
            num_train_steps=140_010,
        )
        == 0.0
    )


def test_reference_schedule_half_warmup():
    assert (
        reference_mtm_lr_multiplier(
            20_000,
            warmup_steps=40_000,
            num_train_steps=140_010,
        )
        == pytest.approx(
            0.5
        )
    )


def test_reference_schedule_peak():
    assert (
        reference_mtm_lr_multiplier(
            40_000,
            warmup_steps=40_000,
            num_train_steps=140_010,
        )
        == pytest.approx(
            1.0
        )
    )


def test_reference_schedule_half_cosine():
    # Remaining decay:
    #
    # 140010 - 40000 = 100010
    #
    # Half:
    # 50005
    #
    # Global step:
    # 90005

    assert (
        reference_mtm_lr_multiplier(
            90_005,
            warmup_steps=40_000,
            num_train_steps=140_010,
        )
        == pytest.approx(
            0.5,
            abs=1e-12,
        )
    )


def test_reference_schedule_end():
    assert (
        reference_mtm_lr_multiplier(
            140_010,
            warmup_steps=40_000,
            num_train_steps=140_010,
        )
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )


def test_scheduler_is_lambda_lr_over_adamw():
    parameter = torch.nn.Parameter(
        torch.tensor(
            1.0
        )
    )

    (
        optimizer,
        scheduler,
    ) = (
        create_reference_mtm_optimizer_and_scheduler(
            [parameter],
            learning_rate=1e-4,
            weight_decay=0.005,
            warmup_steps=4,
            num_train_steps=10,
        )
    )

    # LambdaLR applies lambda(0) when initialized.
    assert (
        optimizer.param_groups[
            0
        ]["lr"]
        == pytest.approx(
            0.0
        )
    )

    optimizer.step()
    scheduler.step()

    assert (
        optimizer.param_groups[
            0
        ]["lr"]
        == pytest.approx(
            2.5e-5
        )
    )


@pytest.mark.parametrize(
    "warmup,num_steps",
    [
        (
            0,
            100,
        ),
        (
            100,
            100,
        ),
        (
            101,
            100,
        ),
    ],
)
def test_invalid_schedule_raises(
    warmup,
    num_steps,
):
    with pytest.raises(
        ValueError
    ):
        reference_mtm_lr_multiplier(
            0,
            warmup_steps=warmup,
            num_train_steps=num_steps,
        )