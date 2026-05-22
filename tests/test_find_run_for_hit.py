import pytest

import straxion

NS = 1_000_000_000


def test_hit_at_run_start_returns_that_run():
    runs = ["1756824965", "1756830000", "1756900000"]
    assert straxion.find_run_for_hit(1756824965 * NS, runs) == "1756824965"


def test_hit_inside_run_returns_that_run():
    runs = ["1756824965", "1756830000", "1756900000"]
    assert straxion.find_run_for_hit(1756824965 * NS + NS, runs) == "1756824965"
    assert straxion.find_run_for_hit(1756830500 * NS, runs) == "1756830000"


def test_hit_after_last_run_open_ended():
    runs = ["1756824965", "1756830000", "1756900000"]
    # With no explicit ends, the last run is treated as open-ended.
    assert straxion.find_run_for_hit(1757000000 * NS, runs) == "1756900000"


def test_hit_before_first_run_returns_none():
    runs = ["1756824965", "1756830000", "1756900000"]
    assert straxion.find_run_for_hit(1756000000 * NS, runs) is None


def test_unsorted_input_handled():
    runs = ["1756900000", "1756824965", "1756830000"]
    assert straxion.find_run_for_hit(1756830500 * NS, runs) == "1756830000"
    assert straxion.find_run_for_hit(1756824965 * NS, runs) == "1756824965"


def test_explicit_run_ends_inside_window():
    runs = ["1756824965", "1756830000", "1756900000"]
    ends = [(1756824965 + 100) * NS, (1756830000 + 100) * NS, (1756900000 + 100) * NS]
    assert straxion.find_run_for_hit((1756824965 + 50) * NS, runs, ends) == "1756824965"


def test_explicit_run_ends_inter_run_gap_returns_none():
    runs = ["1756824965", "1756830000", "1756900000"]
    ends = [(1756824965 + 100) * NS, (1756830000 + 100) * NS, (1756900000 + 100) * NS]
    # 500s after run0 start is past run0 end and before run1 start.
    assert straxion.find_run_for_hit((1756824965 + 500) * NS, runs, ends) is None


def test_explicit_run_ends_past_last_end_returns_none():
    runs = ["1756824965", "1756830000", "1756900000"]
    ends = [(1756824965 + 100) * NS, (1756830000 + 100) * NS, (1756900000 + 100) * NS]
    assert straxion.find_run_for_hit((1756900000 + 200) * NS, runs, ends) is None


def test_explicit_run_ends_unsorted_input():
    runs = ["1756900000", "1756824965", "1756830000"]
    ends = [(1756900000 + 100) * NS, (1756824965 + 100) * NS, (1756830000 + 100) * NS]
    # Hit inside run1's window.
    assert straxion.find_run_for_hit((1756830000 + 50) * NS, runs, ends) == "1756830000"
    # Hit in the gap after run0.
    assert straxion.find_run_for_hit((1756824965 + 500) * NS, runs, ends) is None


def test_empty_run_list_returns_none():
    assert straxion.find_run_for_hit(1756824965 * NS, []) is None


def test_integer_run_ids_accepted():
    runs = [1756824965, 1756830000, 1756900000]
    assert straxion.find_run_for_hit(1756830500 * NS, runs) == "1756830000"


def test_hit_exactly_at_run_end_excluded():
    # End is exclusive: hit_time == end falls into the next run (or None if gap).
    runs = ["1756824965", "1756830000"]
    ends = [(1756824965 + 100) * NS, (1756830000 + 100) * NS]
    # hit exactly at run0's end, before run1 -> None (in gap).
    assert straxion.find_run_for_hit((1756824965 + 100) * NS, runs, ends) is None


def test_single_run():
    runs = ["1756824965"]
    assert straxion.find_run_for_hit(1756824965 * NS, runs) == "1756824965"
    assert straxion.find_run_for_hit(1756824965 * NS + NS, runs) == "1756824965"
    assert straxion.find_run_for_hit(1756824964 * NS, runs) is None


def test_single_run_with_end():
    runs = ["1756824965"]
    ends = [(1756824965 + 10) * NS]
    assert straxion.find_run_for_hit((1756824965 + 5) * NS, runs, ends) == "1756824965"
    assert straxion.find_run_for_hit((1756824965 + 20) * NS, runs, ends) is None


@pytest.mark.parametrize(
    "hit_seconds, expected",
    [
        (1756824964, None),
        (1756824965, "1756824965"),
        (1756829999, "1756824965"),
        (1756830000, "1756830000"),
        (1756899999, "1756830000"),
        (1756900000, "1756900000"),
        (1800000000, "1756900000"),
    ],
)
def test_parametrized_boundaries(hit_seconds, expected):
    runs = ["1756824965", "1756830000", "1756900000"]
    assert straxion.find_run_for_hit(hit_seconds * NS, runs) == expected
