import numpy as np
import matplotlib.colors as mcolors


def register_xenon_colors():
    """Register Xenon color palette as named colors in matplotlib and set as default color cycle.

    This allows you to use colors like color="xenon_blue" in matplotlib plots.
    It also sets the Xenon colors as the default color cycle for automatic coloring.
    Call this function after importing matplotlib but before creating plots.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> import straxion
        >>> straxion.register_xenon_colors()
        >>> plt.plot([1, 2, 3], color="xenon_blue")  # Named color
        >>> plt.plot([1, 2, 3], [1, 4, 2])  # Uses first color in cycle
    """
    import matplotlib.pyplot as plt
    from cycler import cycler

    xenon_colors = {
        "xenon_black": "#000000",
        "xenon_blue": "#4067b1",
        "xenon_lightblue": "#6ccef5",
        "xenon_red": "#B9123E",
        "xenon_yellow": "#ffc74e",
        "xenon_green": "#39a974",
        "xenon_purple": "#8A1859",
        "xenon_silver": "#BFC2C7",
        "xenon_salmon": "#FFB0A8",
        "xenon_violet": "#B580CA",
        "xenon_darkblue": "#203769",
        "xenon_grey": "#909090",
        "xenon_gray": "#909090",
        "xenon_1sigma_green": "#83C369",
        "xenon_2sigma_yellow": "#FDED95",
        "xenon_mint": "#85F7C2",
        "xenon_forestgreen": "#105D20",
        "xenon_orange": "#E77D4D",
        "xenon_darkred": "#9D0008",
        "xenon_sand": "#EDDAB7",
        "xenon_lightgrey": "#DCDCDC",
        "xenon_lightgray": "#DCDCDC",
        "xenon_jet": "#393939",
        "xenon_teal": "#149A9A",
    }

    # Register each color in matplotlib's color registry
    for name, hex_color in xenon_colors.items():
        mcolors.get_named_colors_mapping()[name] = hex_color

    # Set the Xenon colors as the default color cycle
    color_list = [
        "#000000",
        "#4067b1",
        "#6ccef5",
        "#B9123E",
        "#ffc74e",
        "#39a974",
        "#8A1859",
        "#bfc2c7",
        "#FFB0A8",
        "#85F7C2",
        "#EDDAB7",
    ]
    plt.rcParams["axes.prop_cycle"] = cycler("color", color_list)


def plot_channels(
    values,
    missing=[17, 16, 7],
    plot_center=True,
    central=[1, 16, 27, 36],
    ax=None,
    figsize=(4, 3),
    s=250,
    cmap=None,
    vmin=None,
    vmax=None,
    xlim=(-1, 21),
    ylim=(-0.5, 1.4),
    xlabel="mm",
    ylabel="mm",
    title=None,
    colorbar_label=None,
    colorbar_orientation="horizontal",
    **kwargs,
):
    """
    Plot channel values in the QUALIPHIDE-FIR KID array layout.

    Parameters
    ----------
    values : array-like
        Array of length 41 containing values indexed by frequency
        channel (0-40).
    missing : list, optional
        List of position indices to exclude from plot. Default is
        [17, 16, 7] (H1, K1, N1).
    plot_center : bool, optional
        If False, also exclude central channels. Default is True.
    central : list, optional
        List of frequency indices for central channels. Default is
        [1, 16, 27, 36].
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple, optional
        Figure size if creating new axes. Default is (8, 6).
    s : float, optional
        Size of scatter points. Default is 100.
    cmap : str or Colormap, optional
        Colormap to use. Default is None (uses default).
    vmin, vmax : float, optional
        Color scale limits.
    xlim, ylim : tuple, optional
        Axis limits.
    xlabel, ylabel : str, optional
        Axis labels.
    title : str, optional
        Plot title.
    colorbar_label : str, optional
        Colorbar label.
    colorbar_orientation : str, optional
        Orientation of colorbar. 'horizontal' or 'vertical'.
        Default is 'horizontal'.
    **kwargs
        Additional arguments passed to scatter.

    Returns
    -------
    fig, ax, sc
        Figure, axes, and scatter plot object.
    """
    import matplotlib.pyplot as plt

    freq_to_position_mapping = {
        0: 0,  # A1
        1: 22,  # A2
        2: 13,  # B1
        3: 35,  # B2
        4: 5,  # C1
        5: 27,  # C2
        6: 8,  # D1
        7: 30,  # D2
        8: 4,  # E1
        9: 26,  # E2
        10: 21,  # F1
        11: 43,  # F2
        12: 12,  # G1
        13: 34,  # G2
        14: 38,  # H2
        15: 3,  # I1
        16: 25,  # I2
        17: 20,  # J1
        18: 42,  # J2
        19: 39,  # K2
        20: 31,  # L2
        21: 9,  # L1
        22: 15,  # M1
        23: 37,  # M2
        24: 29,  # N2
        25: 36,  # O2
        26: 14,  # O1
        27: 23,  # P2
        28: 1,  # P1
        29: 6,  # Q1
        30: 28,  # Q2
        31: 10,  # R1
        32: 32,  # R2
        33: 19,  # S1
        34: 41,  # S2
        35: 2,  # T1
        36: 24,  # T2
        37: 11,  # U1
        38: 33,  # U2
        39: 18,  # V1
        40: 40,  # V2
    }

    # Validate input
    values = np.array(values)
    if len(values) != 41:
        raise ValueError(f"values must have length 41, got {len(values)}")

    # Create inverse mapping: position -> frequency
    position_to_freq_mapping = {pos: freq for freq, pos in freq_to_position_mapping.items()}

    # Create channel centers
    centers = np.zeros((2, 2, 22))  # x/y, row, col
    centers[1, 0, :] = 0.779  # top row y, bottom y = 0
    for i in range(22):
        centers[0, 0, i] = i * 0.9 + 0.45  # top row x
        centers[0, 1, i] = i * 0.9  # bottom row x

    centers = centers.reshape(2, 44)
    sorted_indices = np.argsort(centers[0, :])
    centers = centers[:, sorted_indices]

    # Determine which channels to exclude
    missing_positions = list(missing)
    if not plot_center:
        missing_positions.extend([freq_to_position_mapping[ji] for ji in central])

    # Get yielded (non-missing) channels
    yielded = [i for i in range(44) if i not in missing_positions]

    # Map position indices to frequency indices and get values
    color_vals = values[np.array([position_to_freq_mapping[ji] for ji in yielded])]

    # Create figure/axes if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Create scatter plot
    sc = ax.scatter(
        centers[0, yielded],
        centers[1, yielded],
        s=s,
        c=color_vals,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )

    # Add circles on central channels (always drawn)
    from matplotlib.patches import Circle

    central_positions = [freq_to_position_mapping[freq_idx] for freq_idx in central]
    for pos_idx in central_positions:
        circle = Circle(
            (centers[0, pos_idx], centers[1, pos_idx]),
            radius=0.4,
            fill=False,
            edgecolor="xenon_red",
            linewidth=1,
            zorder=10,
        )
        ax.add_patch(circle)

    # Add colorbar
    if colorbar_label is None:
        colorbar_label = "Value"
    fig.colorbar(sc, ax=ax, label=colorbar_label, orientation=colorbar_orientation)

    # Set axis properties
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.tick_params(length=1)

    if title is not None:
        ax.set_title(title)

    fig.show()
