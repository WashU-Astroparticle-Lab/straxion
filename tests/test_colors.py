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


def test_plot_channels_log_scale_handles_non_positive():
    # Mix of positive, zero, and negative values must not raise.
    values = np.linspace(-5, 5, 41)
    fig, _, sc = straxion.plot_channels(values, log_scale=True)
    assert isinstance(sc.norm, matplotlib.colors.LogNorm)
    # Norm should derive from the positive subset.
    positive = values[values > 0]
    assert sc.norm.vmin == pytest.approx(positive.min())
    assert sc.norm.vmax == pytest.approx(positive.max())
    # The cmap should have a custom 'bad' color set (not fully transparent).
    bad_rgba = sc.get_cmap().get_bad()
    assert bad_rgba[3] > 0  # alpha > 0 means a visible bad color
    plt.close(fig)


def test_plot_channels_log_scale_bad_color():
    values = np.linspace(-1, 5, 41)
    fig, _, sc = straxion.plot_channels(values, log_scale=True, bad_color="red")
    expected = matplotlib.colors.to_rgba("red")
    assert sc.get_cmap().get_bad() == pytest.approx(expected)
    plt.close(fig)


def test_plot_channels_log_scale_all_non_positive_raises():
    values = np.zeros(41)
    with pytest.raises(ValueError):
        straxion.plot_channels(values, log_scale=True)


def test_plot_channels_log_scale_does_not_mutate_global_cmap():
    # Ensure setting bad_color on the cmap does not leak into the global registry.
    import matplotlib as mpl

    original_bad = mpl.colormaps.get_cmap("magma").get_bad().copy()
    values = np.linspace(-1, 5, 41)
    fig, _, _ = straxion.plot_channels(values, log_scale=True, bad_color="red")
    plt.close(fig)
    assert (mpl.colormaps.get_cmap("magma").get_bad() == original_bad).all()


def test_plot_channels_linear_scale_unchanged():
    # Default behavior must still use vmin/vmax (not a LogNorm).
    values = np.arange(41, dtype=float)
    fig, _, sc = straxion.plot_channels(values, vmin=0, vmax=40)
    assert not isinstance(sc.norm, matplotlib.colors.LogNorm)
    plt.close(fig)
