import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import straxion  # noqa: E402


def test_register_xenon_colors():
    straxion.register_xenon_colors()
    assert matplotlib.colors.to_hex("xenon_blue") == "#4067b1"
    assert matplotlib.colors.to_hex("xenon_red") == "#b9123e"


def test_plot_channels_basic():
    values = np.arange(41, dtype=float)
    fig, ax, sc = straxion.plot_channels(values)
    assert fig is not None
    assert ax is not None
    assert sc.get_cmap().name == "magma"
    plt.close(fig)


def test_plot_channels_wrong_length_raises():
    with pytest.raises(ValueError):
        straxion.plot_channels(np.arange(10))


def test_plot_channels_existing_axes():
    values = np.arange(41, dtype=float)
    fig, ax = plt.subplots()
    out_fig, out_ax, _ = straxion.plot_channels(values, ax=ax)
    assert out_ax is ax
    assert out_fig is fig
    plt.close(fig)


def test_plot_channels_no_highlight_central():
    values = np.arange(41, dtype=float)
    fig_with, _, _ = straxion.plot_channels(values, highlight_central=True)
    n_patches_with = len(fig_with.axes[0].patches)
    plt.close(fig_with)

    fig_without, _, _ = straxion.plot_channels(values, highlight_central=False)
    n_patches_without = len(fig_without.axes[0].patches)
    plt.close(fig_without)

    assert n_patches_with - n_patches_without == 4


def test_plot_channels_plot_center_false():
    values = np.arange(41, dtype=float)
    fig, _, _ = straxion.plot_channels(values, plot_center=False)
    plt.close(fig)


def test_plot_channels_save_pdf_at():
    values = np.arange(41, dtype=float)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "out.pdf")
        fig, _, _ = straxion.plot_channels(values, save_pdf_at=out)
        plt.close(fig)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        with open(out, "rb") as f:
            assert f.read(4) == b"%PDF"


def test_plot_channels_vertical_colorbar_and_title():
    values = np.arange(41, dtype=float)
    fig, ax, _ = straxion.plot_channels(
        values,
        colorbar_orientation="vertical",
        colorbar_label="test",
        title="hello",
        vmin=0,
        vmax=40,
    )
    assert ax.get_title() == "hello"
    plt.close(fig)


def test_plot_channels_log_scale_positive_values():
    values = np.logspace(0, 3, 41)
    fig, _, sc = straxion.plot_channels(values, log_scale=True)
    assert isinstance(sc.norm, matplotlib.colors.LogNorm)
    assert sc.norm.vmin == pytest.approx(values.min())
    assert sc.norm.vmax == pytest.approx(values.max())
    plt.close(fig)


def test_plot_channels_log_scale_with_vmin_vmax():
    values = np.logspace(0, 3, 41)
    fig, _, sc = straxion.plot_channels(values, log_scale=True, vmin=10, vmax=100)
    assert isinstance(sc.norm, matplotlib.colors.LogNorm)
    assert sc.norm.vmin == pytest.approx(10)
    assert sc.norm.vmax == pytest.approx(100)
    plt.close(fig)


def _scatter_collections(ax):
    from matplotlib.collections import PathCollection

    return [c for c in ax.collections if isinstance(c, PathCollection)]


def test_plot_channels_log_scale_handles_non_positive():
    # Mix of positive, zero, and negative values must not raise.
    values = np.linspace(-5, 5, 41)
    fig, ax, sc = straxion.plot_channels(values, log_scale=True)
    assert isinstance(sc.norm, matplotlib.colors.LogNorm)
    positive = values[values > 0]
    assert sc.norm.vmin == pytest.approx(positive.min())
    assert sc.norm.vmax == pytest.approx(positive.max())
    # Two scatter collections: one for bad (non-positive), one for positive.
    cols = _scatter_collections(ax)
    assert len(cols) == 2
    plt.close(fig)


def test_plot_channels_log_scale_bad_color_actually_rendered():
    # Regression: bad_color must be visible in the rendered facecolors.
    values = np.linspace(-1, 5, 41)
    fig, ax, _ = straxion.plot_channels(values, log_scale=True, bad_color="red")
    fig.canvas.draw()
    cols = _scatter_collections(ax)
    assert len(cols) == 2
    # The collection drawn first (the "bad" one) should be solid red and opaque.
    bad_col = cols[0]
    expected = np.array(matplotlib.colors.to_rgba("red"))
    facecolors = bad_col.get_facecolors()
    assert facecolors.shape[0] > 0
    for c in facecolors:
        assert np.allclose(c, expected)
    plt.close(fig)


def test_plot_channels_log_scale_no_bad_collection_when_all_positive():
    values = np.logspace(0, 3, 41)
    fig, ax, _ = straxion.plot_channels(values, log_scale=True)
    cols = _scatter_collections(ax)
    # Only the positive scatter exists; no separate bad collection.
    assert len(cols) == 1
    plt.close(fig)


def test_plot_channels_log_scale_bad_count_matches_non_positive():
    values = np.array([0.0] * 20 + [1.0] * 21)  # 20 non-positive, 21 positive
    fig, ax, _ = straxion.plot_channels(values, log_scale=True)
    fig.canvas.draw()
    cols = _scatter_collections(ax)
    # All 20 non-positive values fall on plotted (non-missing) channels.
    assert len(cols[0].get_offsets()) == 20
    assert len(cols[1].get_offsets()) == 21
    plt.close(fig)


def test_plot_channels_log_scale_all_non_positive_raises():
    values = np.zeros(41)
    with pytest.raises(ValueError):
        straxion.plot_channels(values, log_scale=True)


def test_plot_channels_linear_scale_unchanged():
    # Default behavior must still use vmin/vmax (not a LogNorm).
    values = np.arange(41, dtype=float)
    fig, _, sc = straxion.plot_channels(values, vmin=0, vmax=40)
    assert not isinstance(sc.norm, matplotlib.colors.LogNorm)
    plt.close(fig)
