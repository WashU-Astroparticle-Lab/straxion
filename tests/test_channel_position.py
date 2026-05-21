import numpy as np
import pytest

import straxion


def test_returns_array_of_shape_2():
    pos = straxion.get_channel_position(0)
    assert isinstance(pos, np.ndarray)
    assert pos.shape == (2,)


def test_all_channels_return_valid_positions():
    positions = np.array([straxion.get_channel_position(i) for i in range(41)])
    assert positions.shape == (41, 2)
    # y is either bottom row (0) or top row (0.779 mm).
    assert set(np.unique(positions[:, 1])) == {0.0, 0.779}
    # x spans the 22-column layout at 0.9 mm pitch.
    assert positions[:, 0].min() == pytest.approx(0.0)
    assert positions[:, 0].max() == pytest.approx(19.35)


def test_all_positions_unique():
    positions = [tuple(straxion.get_channel_position(i)) for i in range(41)]
    assert len(set(positions)) == 41


def test_invalid_channel_raises():
    with pytest.raises(ValueError):
        straxion.get_channel_position(41)
    with pytest.raises(ValueError):
        straxion.get_channel_position(-1)


def test_returns_copy_not_view():
    pos = straxion.get_channel_position(0)
    pos[0] = 999.0
    # A second call should not be affected by mutation of the first result.
    assert straxion.get_channel_position(0)[0] == 0.0


def test_array_input_returns_n_by_2():
    channels = np.array(
        [7, 16, 30, 32, 38, 10, 17, 19, 25, 27, 33, 34, 35, 39, 4, 11, 12, 22, 8, 28, 15],
        dtype=np.int16,
    )
    out = straxion.get_channel_position(channels)
    assert out.shape == (len(channels), 2)
    # Each row matches the scalar result for that channel.
    for i, ch in enumerate(channels):
        assert np.array_equal(out[i], straxion.get_channel_position(int(ch)))


def test_array_input_list_and_dtypes():
    # Plain list input works.
    out_list = straxion.get_channel_position([0, 5, 40])
    assert out_list.shape == (3, 2)
    # int16 (the realistic CHANNEL_DTYPE) works.
    out_int16 = straxion.get_channel_position(np.array([0, 5, 40], dtype=np.int16))
    assert np.array_equal(out_list, out_int16)


def test_array_input_invalid_raises():
    with pytest.raises(ValueError):
        straxion.get_channel_position(np.array([0, 41]))
