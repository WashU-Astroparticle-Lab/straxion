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
