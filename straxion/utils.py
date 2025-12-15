import numpy as np
import matplotlib.colors as mcolors
from pathlib import Path
import pickle
import os

# Common senses.
SECOND_TO_NANOSECOND = 1_000_000_000

# Common dtype constants for waveform records.
TIME_DTYPE = np.int64
LENGTH_DTYPE = np.int64
CHANNEL_DTYPE = np.int16
DATA_DTYPE = np.dtype("f4")
INDEX_DTYPE = np.int32

# Placeholder index for not found items (will raise IndexError if used).
NOT_FOUND_INDEX = 999_999_999

# Baseline monitor interval.
N_BASELINE_MONITOR_INTERVAL = 100

# Pulse template with sampling rate of 38 kHz.
PULSE_TEMPLATE_38kHz = np.load(Path(__file__).parent / "msc" / "pulse_template_38kHz.npy")
PULSE_TEMPLATE_LENGTH = len(PULSE_TEMPLATE_38kHz)
PULSE_TEMPLATE_ARGMAX = np.argmax(PULSE_TEMPLATE_38kHz)

# Hit waveform recording window length, from the maximum of the hit waveform.
HIT_WINDOW_LENGTH_LEFT = 200
HIT_WINDOW_LENGTH_RIGHT = 600

# Energy resolution constants for truth generation.
# Reference photon energy and dx values for 25um wavelength
PHOTON_25um_meV = 50  # meV
PHOTON_25um_DX = 1.54e-6  # dx units
# Energy resolution in dx units (optimistic and conservative modes)
DX_RESOL_OPTIMISTIC = 186835.48206306322e-12
DX_RESOL_CONSERVATIVE = 267423.0098878706e-12

# Noise PSD
# Assumed (of_window_left, of_window_right) = (100, 300) samples
# This is a placeholder for the noise PSD,
# and only used when we cannot compute the noise PSD from the noise bank in a data-driven way.
NOISE_PSD_38kHz = np.load(Path(__file__).parent / "msc" / "noise_psd_38kHz.npy").tolist()

# Default path to template interpolation file
# This constructs path relative to this module's location
DEFAULT_TEMPLATE_INTERP_PATH = str(Path(__file__).parent / "msc" / "template_interp.pkl")


def load_interpolation(load_path="template_interp.pkl"):
    """
    Load saved interpolation function.

    Parameters:
    -----------
    load_path : str
        Path to saved interpolation function

    Returns:
    --------
    At_interp : interp1d
        Interpolation function
    t_max : float
        Time of maximum value in template (in seconds)
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"Interpolation file not found: {load_path}. "
            "Please run build_and_save_interpolation() first."
        )

    with open(load_path, "rb") as f:
        data = pickle.load(f)

    return data["interp"], data["t_max"]


def base_waveform_dtype():
    """Return the base dtype list for a waveform record, without the data fields.

    Returns:
        list: List of dtype tuples for the base waveform record fields.

    """
    return [
        (("Start time since unix epoch [ns]", "time"), TIME_DTYPE),
        (("Exclusive end time since unix epoch [ns]", "endtime"), TIME_DTYPE),
        (("Length of the interval in samples", "length"), LENGTH_DTYPE),
        (
            ("Width of one sample [ns], which is not exact due to the int conversion", "dt"),
            TIME_DTYPE,
        ),
        (("Channel number defined by channel_map", "channel"), CHANNEL_DTYPE),
    ]


def timestamp_to_nanoseconds(timestamp_str):
    """Convert a timestamp string in format YYYYMMDDHHmmSS to nanoseconds since Unix epoch.

    Args:
        timestamp_str (str): Timestamp string in format YYYYMMDDHHmmSS

    Returns:
        int: Timestamp in nanoseconds since Unix epoch

    Example:
        >>> timestamp_to_nanoseconds("19980717223000")
        900714600000000000

    """
    from datetime import datetime

    # Parse the timestamp string
    dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")

    # Convert to Unix timestamp (seconds since epoch)
    unix_timestamp = dt.timestamp()

    # Convert to nanoseconds
    nanoseconds = int(unix_timestamp * 1_000_000_000)

    return nanoseconds


def circfit(x, y):
    """Least squares fit of X-Y data to a circle.

    Adapted from the Matlab implementation of Andrew D. Horchler (horchler@gmail.com).

    Args:
        x (array-like): 1D array of x position data.
        y (array-like): 1D array of y position data.

    Returns:
        tuple: (x_center, y_center, radius, rms_error)
            x_center (float): X-position of center of fitted circle.
            y_center (float): Y-position of center of fitted circle.
            radius (float): Radius of fitted circle.
            rms_error (float): Root mean squared error of the fit.

    Raises:
        ValueError: If x and y are not the same length, have less than three points,
            or are collinear.

    """
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()

    # Sanity checks.
    if x.size != y.size:
        raise ValueError(
            "x and y must be the same length. "
            f"Got x.shape={x.shape}, y.shape={y.shape}, x.size={x.size}, y.size={y.size}"
        )
    if x.size < 3:
        raise ValueError(
            f"At least three points are required. Got x.size={x.size}, y.size={y.size}"
        )

    # Collinearity check.
    collinearity_matrix = np.column_stack([x[: min(50, len(x))], y[: min(50, len(y))]])
    diff_matrix = np.diff(collinearity_matrix, axis=0)
    rank = np.linalg.matrix_rank(diff_matrix)
    if rank == 1:
        raise ValueError(
            f"Points are collinear or nearly collinear.\n"
            f"First 50 (or fewer) x: {x[:min(50, len(x))]}\n"
            f"First 50 (or fewer) y: {y[:min(50, len(y))]}\n"
            f"Collinearity diff matrix shape: {diff_matrix.shape}, rank: {rank}"
        )

    x2 = x * x
    y2 = y * y
    xy = x * y
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x2)
    sum_y2 = np.sum(y2)
    sum_xy = np.sum(xy)
    sum_x2y = np.sum((x2 + y2) * y)
    sum_x2x = np.sum((x2 + y2) * x)
    sum_x2y2 = np.sum(x2 + y2)
    n_points = len(x)

    # Solve Ax=b.
    a_matrix = np.array(
        [[sum_x, sum_y, n_points], [sum_xy, sum_y2, sum_y], [sum_x2, sum_xy, sum_x]]
    )
    b_vector = np.array([sum_x2y2, sum_x2y, sum_x2x])
    try:
        solution = np.linalg.solve(a_matrix, b_vector)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"Failed to solve linear system in circfit.\n"
            f"a_matrix=\n{a_matrix}\n"
            f"b_vector={b_vector}\n"
            f"Error: {e}"
        )
    x_center = 0.5 * solution[0]
    y_center = 0.5 * solution[1]
    radius = np.sqrt(x_center**2 + y_center**2 + solution[2])

    # Root mean squared error.
    # Calculate the distance from each point to the fitted circle center.
    distances = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    # Compute the RMS error between these distances and the fitted radius.
    rms_error = np.sqrt(np.mean((distances - radius) ** 2))
    return x_center, y_center, radius, rms_error


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
        "xenon_light_blue": "#6ccef5",
        "xenon_red": "#B9123E",
        "xenon_yellow": "#ffc74e",
        "xenon_green": "#39a974",
        "xenon_purple": "#8A1859",
        "xenon_silver": "#bfc2c7",
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
