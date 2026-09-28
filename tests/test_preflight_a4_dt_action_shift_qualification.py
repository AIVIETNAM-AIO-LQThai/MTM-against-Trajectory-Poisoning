from scripts.preflight_a4_dt_action_shift_qualification import shift_bounds, choose_shift_offset

def test_shift_bounds():
    assert shift_bounds(4) == (1,3)
    assert shift_bounds(5) == (2,3)
    assert shift_bounds(100) == (25,75)

def test_shift_offset_deterministic():
    a = choose_shift_offset(100,20,42)
    b = choose_shift_offset(100,20,42)
    assert a == b
    assert 25 <= a <= 75
