import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import straxion  # noqa: E402

straxion.register_xenon_colors()

NUM_CHANNELS = 41
N_SAMPLES = 2000
SAMPLING_RATE = 38e3


def _make_fake_records(num_channels=NUM_CHANNELS, n_samples=N_SAMPLES, seed=0):
    """Build a (num_channels, n_samples) fake `data_dx` block.

    Each channel has a Gaussian pulse with peak amplitude ~1e-6, plus low-level
    noise. The pulse amplitude is channel-dependent so that traces remain
    distinguishable after plotting.
    """
    rng = np.random.default_rng(seed)
    data_dx = rng.normal(0.0, 1e-8, size=(num_channels, n_samples))
    t = np.arange(n_samples)
    for ch in range(num_channels):
        peak = 1e-6 * (1.0 + 0.01 * ch)
        data_dx[ch] += peak * np.exp(-((t - n_samples // 2) ** 2) / (50.0**2))
    return {"data_dx": data_dx}


def _expected_x_order(num_channels=NUM_CHANNELS):
    x_positions = straxion.get_channel_position(np.arange(num_channels))[:, 0]
    return np.argsort(x_positions, kind="stable")


def test_plot_staggered_traces_basic_returns_fig_ax():
    records = _make_fake_records()
    fig, ax = straxion.plot_staggered_traces(
        records, zooml=0, zoomr=N_SAMPLES, sampling_rate=SAMPLING_RATE
    )
    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == NUM_CHANNELS
    plt.close(fig)


def test_plot_staggered_traces_uses_existing_axes():
    records = _make_fake_records()
    fig, ax = plt.subplots()
    out_fig, out_ax = straxion.plot_staggered_traces(records, zooml=0, zoomr=N_SAMPLES, ax=ax)
    assert out_ax is ax
    assert out_fig is fig
    plt.close(fig)


def test_plot_staggered_traces_orders_by_x_position():
    """Rank in the staggered offset must match left-to-right x order."""
    records = _make_fake_records()
    time_offset = 3e-4
    amp_offset = 5e-7

    fig, ax = straxion.plot_staggered_traces(
        records,
        zooml=0,
        zoomr=N_SAMPLES,
        sampling_rate=SAMPLING_RATE,
        time_offset=time_offset,
        amp_offset=amp_offset,
    )

    expected_order = _expected_x_order()
    for rank, line in enumerate(ax.lines):
        # x offset baked into the line equals time_offset * rank exactly,
        # since t[0] = 0.
        assert line.get_xdata()[0] == pytest.approx(time_offset * rank)
        # Subtracting the rank-dependent amplitude offset recovers the
        # original channel trace, identifying which channel was plotted at
        # this rank.
        ydata = line.get_ydata() - amp_offset * rank
        ch_expected = expected_order[rank]
        np.testing.assert_allclose(ydata, records["data_dx"][ch_expected], rtol=0, atol=1e-15)

    # Sanity: the x positions of the channels in plotted order are
    # monotonically non-decreasing.
    x_positions = straxion.get_channel_position(np.arange(NUM_CHANNELS))[:, 0]
    plotted_x = x_positions[expected_order]
    assert np.all(np.diff(plotted_x) >= 0)
    plt.close(fig)


def test_plot_staggered_traces_not_sorted_by_amplitude():
    """Regression: ordering must reflect geometry, not amplitude.

    Channels here have monotonically increasing amplitude with channel index,
    so amplitude-sorting would produce a different rank order than
    x-position sorting (the array layout is not channel-monotonic in x).
    """
    records = _make_fake_records()
    fig, ax = straxion.plot_staggered_traces(
        records, zooml=0, zoomr=N_SAMPLES, sampling_rate=SAMPLING_RATE
    )

    expected_order = _expected_x_order()
    # Amplitude-descending order, then the butterfly fold the old function used.
    amps = np.array([np.nanmax(records["data_dx"][ch]) for ch in range(NUM_CHANNELS)])
    amp_order = np.argsort(amps)[::-1]
    butterfly = np.concatenate((amp_order[::2][::-1], amp_order[1::2]))

    assert not np.array_equal(expected_order, butterfly)
    plt.close(fig)


def test_plot_staggered_traces_highlight_colors():
    records = _make_fake_records()
    hit_ch = 16
    far_ch = 0
    fig, ax = straxion.plot_staggered_traces(
        records,
        zooml=0,
        zoomr=N_SAMPLES,
        single_si_phonon_hits_channels=hit_ch,
        no_hit_far_channels=[far_ch],
        trace_color="xenon_jet",
        color_single_hit="xenon_blue",
        color_no_hit="xenon_red",
    )

    expected_order = _expected_x_order()
    hit_rank = int(np.where(expected_order == hit_ch)[0][0])
    far_rank = int(np.where(expected_order == far_ch)[0][0])

    # Verify color, alpha, and zorder for highlighted vs. background traces.
    hit_line = ax.lines[hit_rank]
    far_line = ax.lines[far_rank]

    assert matplotlib.colors.to_hex(hit_line.get_color()) == matplotlib.colors.to_hex("xenon_blue")
    assert hit_line.get_alpha() == pytest.approx(0.6)
    assert hit_line.get_zorder() == 3

    assert matplotlib.colors.to_hex(far_line.get_color()) == matplotlib.colors.to_hex("xenon_red")
    assert far_line.get_alpha() == pytest.approx(0.6)
    assert far_line.get_zorder() == 2

    # A background channel (not in either highlight set) should be the
    # default trace color with low alpha.
    bg_ch = next(c for c in range(NUM_CHANNELS) if c not in (hit_ch, far_ch))
    bg_rank = int(np.where(expected_order == bg_ch)[0][0])
    bg_line = ax.lines[bg_rank]
    assert matplotlib.colors.to_hex(bg_line.get_color()) == matplotlib.colors.to_hex("xenon_jet")
    assert bg_line.get_alpha() == pytest.approx(0.2)
    assert bg_line.get_zorder() == 1
    plt.close(fig)


def test_plot_staggered_traces_accepts_int_hit_channel():
    records = _make_fake_records()
    # Should not raise when a bare int is passed instead of a list.
    fig, _ = straxion.plot_staggered_traces(
        records, zooml=0, zoomr=N_SAMPLES, single_si_phonon_hits_channels=5
    )
    plt.close(fig)


def test_plot_staggered_traces_respects_zoom_slice():
    records = _make_fake_records()
    zooml, zoomr = 200, 700
    fig, ax = straxion.plot_staggered_traces(
        records, zooml=zooml, zoomr=zoomr, sampling_rate=SAMPLING_RATE
    )
    # Each line should carry zoomr - zooml samples.
    for line in ax.lines:
        assert len(line.get_xdata()) == zoomr - zooml
    plt.close(fig)
