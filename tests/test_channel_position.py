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
