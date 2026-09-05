import numpy as np
import pytest
import torch

from src.data.mtm_statistics import (
    DataStatistics,
)

from src.methods.mtm.tokenizers import (
    ContinuousTokenizer,
)


def make_stats():
    return DataStatistics(
        mean=np.asarray(
            [10.0, -5.0],
            dtype=np.float32,
        ),
        std=np.asarray(
            [2.0, 4.0],
            dtype=np.float32,
        ),
        min=np.asarray(
            [0.0, -20.0],
            dtype=np.float32,
        ),
        max=np.asarray(
            [20.0, 10.0],
            dtype=np.float32,
        ),
    )


def test_encode_shape():
    tokenizer = (
        ContinuousTokenizer.from_statistics(
            make_stats()
        )
    )

    x = torch.zeros(
        3,
        4,
        2,
    )

    encoded = tokenizer.encode(
        x
    )

    assert encoded.shape == (
        3,
        4,
        1,
        2,
    )


def test_round_trip():
    tokenizer = (
        ContinuousTokenizer.from_statistics(
            make_stats()
        )
    )

    x = torch.tensor(
        [
            [
                [10.0, -5.0],
                [12.0, -1.0],
                [8.0, -9.0],
            ]
        ],
        dtype=torch.float32,
    )

    encoded = tokenizer.encode(
        x
    )

    decoded = tokenizer.decode(
        encoded
    )

    torch.testing.assert_allclose(
        decoded,
        x,
        rtol=1e-6,
        atol=1e-6,
    )


def test_normalization_values():
    tokenizer = (
        ContinuousTokenizer.from_statistics(
            make_stats()
        )
    )

    x = torch.tensor(
        [
            [
                [12.0, 3.0],
            ]
        ],
        dtype=torch.float32,
    )

    encoded = tokenizer.encode(
        x
    )

    # (12 - 10) / 2 = 1
    # (3 - -5) / 4  = 2
    expected = torch.tensor(
        [
            [
                [
                    [1.0, 2.0]
                ]
            ]
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_allclose(
        encoded,
        expected,
    )


def test_small_std_is_replaced_by_one():
    tokenizer = ContinuousTokenizer(
        data_mean=np.asarray(
            [0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        data_std=np.asarray(
            [
                0.0,
                0.099,
                0.1,
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_allclose(
        tokenizer.data_std
        .detach()
        .cpu()
        .numpy(),
        np.asarray(
            [
                1.0,
                1.0,
                0.1,
            ],
            dtype=np.float32,
        ),
    )


def test_normalize_false_is_identity_except_token_axis():
    tokenizer = ContinuousTokenizer(
        data_mean=np.asarray(
            [100.0, 100.0],
            dtype=np.float32,
        ),
        data_std=np.asarray(
            [20.0, 20.0],
            dtype=np.float32,
        ),
        normalize=False,
    )

    x = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ],
        dtype=torch.float32,
    )

    encoded = tokenizer.encode(
        x
    )

    torch.testing.assert_allclose(
        encoded.squeeze(2),
        x,
    )

    decoded = tokenizer.decode(
        encoded
    )

    torch.testing.assert_allclose(
        decoded,
        x,
    )


def test_wrong_input_rank_raises():
    tokenizer = (
        ContinuousTokenizer.from_statistics(
            make_stats()
        )
    )

    with pytest.raises(ValueError):
        tokenizer.encode(
            torch.zeros(
                4,
                2,
            )
        )


def test_wrong_feature_dim_raises():
    tokenizer = (
        ContinuousTokenizer.from_statistics(
            make_stats()
        )
    )

    with pytest.raises(ValueError):
        tokenizer.encode(
            torch.zeros(
                1,
                4,
                3,
            )
        )