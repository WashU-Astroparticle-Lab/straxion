import numpy as np
import pytest

import straxion

DT_NS = 26_316  # ~38 kHz, close to qualiphide_thz_offline sampling.


def _make_records(starts, length=1000, dt=DT_NS, channels=None):
    """Build a minimal records-like structured array for testing."""
    dtype = np.dtype(
        [
            ("time", np.int64),
            ("endtime", np.int64),
            ("length", np.int32),
            ("dt", np.int64),
            ("channel", np.int16),
        ]
    )
    n = len(starts)
    if channels is None:
        channels = np.arange(n, dtype=np.int16)
    records = np.zeros(n, dtype=dtype)
    records["time"] = starts
    records["dt"] = dt
    records["length"] = length
    records["endtime"] = np.asarray(starts) + length * dt
    records["channel"] = channels
    return records


def test_hit_at_record_start_is_sample_zero():
    records = _make_records([1_000_000_000], length=500, channels=[3])
    assert straxion.find_sample_for_hit(1_000_000_000, records) == 0


def test_hit_inside_record_returns_floor_division():
    records = _make_records([1_000_000_000], length=500, channels=[3])
    # 10.5 samples past the start -> sample 10.
    hit_time = 1_000_000_000 + 10 * DT_NS + DT_NS // 2
    assert straxion.find_sample_for_hit(hit_time, records) == 10


def test_hit_at_last_sample():
    records = _make_records([1_000_000_000], length=500, channels=[3])
    hit_time = 1_000_000_000 + 499 * DT_NS
    assert straxion.find_sample_for_hit(hit_time, records) == 499


def test_hit_at_endtime_excluded():
    records = _make_records([1_000_000_000], length=500, channels=[3])
    # endtime is exclusive.
    hit_time = int(records[0]["endtime"])
    assert straxion.find_sample_for_hit(hit_time, records) is None


def test_hit_before_record_returns_none():
    records = _make_records([1_000_000_000], length=500, channels=[3])
    assert straxion.find_sample_for_hit(999_999_999, records) is None


def test_all_channels_give_same_sample_index():
    # All channels share the same timing, so the sample index is well-defined
    # regardless of which channel the matching record happens to belong to.
    records = _make_records(
        [1_000_000_000, 1_000_000_000, 1_000_000_000],
        length=500,
        channels=[3, 7, 11],
    )
    hit_time = 1_000_000_000 + 5 * DT_NS
    assert straxion.find_sample_for_hit(hit_time, records) == 5


def test_empty_records_returns_none():
    records = _make_records([], length=500, channels=[])
    assert straxion.find_sample_for_hit(1_000_000_000, records) is None


def test_multiple_chunks_same_channel():
    # Two consecutive chunks for the same channel.
    start0 = 1_000_000_000
    length = 500
    start1 = start0 + length * DT_NS
    records = _make_records([start0, start1], length=length, channels=[3, 3])
    hit_time = start1 + 17 * DT_NS
    assert straxion.find_sample_for_hit(hit_time, records) == 17


@pytest.mark.parametrize(
    "offset_samples, expected_sample",
    [(0, 0), (1, 1), (123, 123), (499, 499)],
)
def test_parametrized_sample_offsets(offset_samples, expected_sample):
    records = _make_records([1_000_000_000], length=500, channels=[3])
    hit_time = 1_000_000_000 + offset_samples * DT_NS
    assert straxion.find_sample_for_hit(hit_time, records) == expected_sample
