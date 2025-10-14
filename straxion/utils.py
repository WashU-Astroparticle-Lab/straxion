import numpy as np
import matplotlib.colors as mcolors

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
PULSE_TEMPLATE_38kHz = np.array(
    [
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        7.67229793e-11,
        6.30287041e-10,
        1.18385110e-09,
        1.73741517e-09,
        3.49320597e-09,
        5.31580764e-09,
        7.13840932e-09,
        8.96101099e-09,
        1.07836127e-08,
        1.26062143e-08,
        1.44288160e-08,
        1.62514177e-08,
        1.80740194e-08,
        1.98966210e-08,
        2.17192227e-08,
        2.35418244e-08,
        9.13963040e-08,
        3.33344214e-07,
        5.75292124e-07,
        8.17240034e-07,
        1.05918794e-06,
        1.30113585e-06,
        1.54000000e-06,
        1.52620304e-06,
        1.51240607e-06,
        1.49860911e-06,
        1.48481215e-06,
        1.45185813e-06,
        1.41864280e-06,
        1.39845364e-06,
        1.37826449e-06,
        1.34238276e-06,
        1.27737916e-06,
        1.19468282e-06,
        1.13885250e-06,
        1.11129745e-06,
        1.08374241e-06,
        1.05552810e-06,
        1.02314470e-06,
        9.90761297e-07,
        9.70761563e-07,
        9.52296825e-07,
        9.33832088e-07,
        9.15367351e-07,
        8.86649130e-07,
        8.52636824e-07,
        8.19743770e-07,
        8.04672082e-07,
        7.89600394e-07,
        7.74528706e-07,
        7.59457019e-07,
        7.35894247e-07,
        7.12242341e-07,
        6.88590435e-07,
        6.68378415e-07,
        6.50581378e-07,
        6.32784342e-07,
        6.14987305e-07,
        5.97190268e-07,
        5.81916113e-07,
        5.67469597e-07,
        5.53023080e-07,
        5.38576564e-07,
        5.24130047e-07,
        5.12299760e-07,
        5.00472055e-07,
        4.88644349e-07,
        4.76816643e-07,
        4.64988938e-07,
        4.53161232e-07,
        4.41575446e-07,
        4.30009100e-07,
        4.18442754e-07,
        4.06876408e-07,
        3.95310062e-07,
        3.83743716e-07,
        3.73052682e-07,
        3.64325144e-07,
        3.55597605e-07,
        3.46870066e-07,
        3.38142528e-07,
        3.29414989e-07,
        3.20687450e-07,
        3.11959912e-07,
        3.02686261e-07,
        2.92725198e-07,
        2.82764135e-07,
        2.72803072e-07,
        2.62842009e-07,
        2.52880946e-07,
        2.42919883e-07,
        2.34039966e-07,
        2.29742020e-07,
        2.25444073e-07,
        2.21146126e-07,
        2.16848180e-07,
        2.12550233e-07,
        2.08252286e-07,
        2.03954340e-07,
        1.99656393e-07,
        1.95358446e-07,
        1.91060500e-07,
        1.86762553e-07,
        1.83491666e-07,
        1.80594526e-07,
        1.77697385e-07,
        1.74800245e-07,
        1.71903104e-07,
        1.69005964e-07,
        1.66108823e-07,
        1.63211683e-07,
        1.60314542e-07,
        1.57417402e-07,
        1.54520261e-07,
        1.51623121e-07,
        1.48725980e-07,
        1.45716230e-07,
        1.42022358e-07,
        1.38328487e-07,
        1.34634616e-07,
        1.30940744e-07,
        1.27246873e-07,
        1.23553001e-07,
        1.19859130e-07,
        1.16165258e-07,
        1.12471387e-07,
        1.08777515e-07,
        1.05083644e-07,
        1.01389772e-07,
        9.80586797e-08,
        9.58204444e-08,
        9.35822092e-08,
        9.13439740e-08,
        8.91057387e-08,
        8.68675035e-08,
        8.46292682e-08,
        8.23910330e-08,
        8.01527978e-08,
        7.79145625e-08,
        7.56763273e-08,
        7.34380921e-08,
        7.11998568e-08,
        6.90501312e-08,
        6.70597473e-08,
        6.50693634e-08,
        6.30789795e-08,
        6.10885955e-08,
        5.90982116e-08,
        5.71078277e-08,
        5.51174438e-08,
        5.31270599e-08,
        5.11366759e-08,
        4.91462920e-08,
        4.71559081e-08,
        4.51655242e-08,
        4.37894263e-08,
        4.31201107e-08,
        4.24507950e-08,
        4.17814793e-08,
        4.11121637e-08,
        4.04428480e-08,
        3.97735324e-08,
        3.91042167e-08,
        3.84349010e-08,
        3.77655854e-08,
        3.70962697e-08,
        3.64269540e-08,
        3.57576384e-08,
        3.50827527e-08,
        3.44037141e-08,
        3.37246755e-08,
        3.30456369e-08,
        3.23665983e-08,
        3.16875598e-08,
        3.10085212e-08,
        3.03294826e-08,
        2.96504440e-08,
        2.89714054e-08,
        2.82923668e-08,
        2.76133283e-08,
        2.69342897e-08,
        2.68937332e-08,
        2.71525903e-08,
        2.74114474e-08,
        2.76703045e-08,
        2.79291616e-08,
        2.81880187e-08,
        2.84468758e-08,
        2.87057329e-08,
        2.89645900e-08,
        2.92234471e-08,
        2.94823042e-08,
        2.97411613e-08,
        3.00000184e-08,
        2.95897779e-08,
        2.90002202e-08,
        2.84106625e-08,
        2.78211048e-08,
        2.72315471e-08,
        2.66419894e-08,
        2.60524316e-08,
        2.54628739e-08,
        2.48733162e-08,
        2.42837585e-08,
        2.36942008e-08,
        2.31046431e-08,
        2.25150854e-08,
        2.19377042e-08,
        2.13617284e-08,
        2.07857526e-08,
        2.02097768e-08,
        1.96338010e-08,
        1.90578251e-08,
        1.84818493e-08,
        1.79058735e-08,
        1.73298977e-08,
        1.67539219e-08,
        1.61779461e-08,
        1.56019702e-08,
        1.50256153e-08,
        1.43637284e-08,
        1.37018416e-08,
        1.30399547e-08,
        1.23780678e-08,
        1.17161809e-08,
        1.10542940e-08,
        1.03924071e-08,
        9.73052025e-09,
        9.06863337e-09,
        8.40674648e-09,
        7.74485960e-09,
        7.08297272e-09,
        6.48802803e-09,
        6.42225935e-09,
        6.35649066e-09,
        6.29072198e-09,
        6.22495330e-09,
        6.15918462e-09,
        6.09341593e-09,
        6.02764725e-09,
        5.96187857e-09,
        5.89610989e-09,
        5.83034120e-09,
        5.76457252e-09,
        5.69880384e-09,
        5.55058914e-09,
        5.11037391e-09,
        4.67015868e-09,
        4.22994345e-09,
        3.78972822e-09,
        3.34951299e-09,
        2.90929776e-09,
        2.46908253e-09,
        2.02886730e-09,
        1.58865207e-09,
        1.14843684e-09,
        7.08221605e-10,
        2.68006374e-10,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
    ]
)
PULSE_TEMPLATE_LENGTH = len(PULSE_TEMPLATE_38kHz)

# Hit waveform recording window length.
HIT_WINDOW_LENGTH_LEFT = 200
HIT_WINDOW_LENGTH_RIGHT = 600

# Energy resolution constants for truth generation.
# Reference photon energy and dx values for 25um wavelength
PHOTON_25um_meV = 50  # meV
PHOTON_25um_DX = 1.54e6  # dx units
# Energy resolution in dx units (optimistic and conservative modes)
DX_RESOL_OPTIMISTIC = 186835.48206306322
DX_RESOL_CONSERVATIVE = 267423.0098878706


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
