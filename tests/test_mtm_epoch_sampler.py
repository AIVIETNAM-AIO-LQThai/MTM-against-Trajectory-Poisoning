import numpy as np

from src.data.mtm_batching import (
    MTMShuffledWindowSampler,
    MTMWindowRange,
)


def test_shuffled_epoch_visits_every_index_once():
    sampler = MTMShuffledWindowSampler(
        MTMWindowRange(
            start=10,
            end=20,
        ),
        seed=123,
    )

    first = sampler.next_indices(4)
    second = sampler.next_indices(4)
    third = sampler.next_indices(4)

    assert len(first) == 4
    assert len(second) == 4

    # drop_last=False behavior:
    # final epoch batch has only 2 samples.
    assert len(third) == 2

    combined = np.concatenate(
        [
            first,
            second,
            third,
        ]
    )

    assert sorted(
        combined.tolist()
    ) == list(
        range(10, 20)
    )

    assert len(
        np.unique(
            combined
        )
    ) == 10

    assert sampler.epoch == 1
    assert sampler.position == 0


def test_shuffled_epoch_is_deterministic():
    left = MTMShuffledWindowSampler(
        MTMWindowRange(
            start=0,
            end=17,
        ),
        seed=456,
    )

    right = MTMShuffledWindowSampler(
        MTMWindowRange(
            start=0,
            end=17,
        ),
        seed=456,
    )

    for _ in range(8):
        assert np.array_equal(
            left.next_indices(5),
            right.next_indices(5),
        )


def test_sampler_state_restores_exactly():
    original = MTMShuffledWindowSampler(
        MTMWindowRange(
            start=0,
            end=23,
        ),
        seed=789,
    )

    original.next_indices(7)
    original.next_indices(7)

    state = (
        original.state_dict()
    )

    restored = MTMShuffledWindowSampler(
        MTMWindowRange(
            start=0,
            end=23,
        ),
        seed=999,
    )

    restored.load_state_dict(
        state
    )

    for _ in range(10):
        assert np.array_equal(
            original.next_indices(6),
            restored.next_indices(6),
        )