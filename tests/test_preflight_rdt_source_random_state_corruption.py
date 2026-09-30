from scripts.preflight_rdt_source_random_state_corruption import (
    EXPECTED_CLEAN_TRANSITIONS,
    EXPECTED_SELECTED,
)


def test_frozen_downsample_contract():
    assert len(EXPECTED_SELECTED) == 23
    assert EXPECTED_CLEAN_TRANSITIONS == 20147
    assert EXPECTED_SELECTED == [
        902, 239, 15, 185, 71, 171, 201, 726, 484, 35, 63, 32,
        708, 991, 949, 304, 186, 374, 234, 29, 1030, 996, 511,
    ]
