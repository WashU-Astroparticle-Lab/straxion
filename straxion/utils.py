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
        7.67229793e01,
        6.30287041e02,
        1.18385110e03,
        1.73741517e03,
        3.49320597e03,
        5.31580764e03,
        7.13840932e03,
        8.96101099e03,
        1.07836127e04,
        1.26062143e04,
        1.44288160e04,
        1.62514177e04,
        1.80740194e04,
        1.98966210e04,
        2.17192227e04,
        2.35418244e04,
        9.13963040e04,
        3.33344214e05,
        5.75292124e05,
        8.17240034e05,
        1.05918794e06,
        1.30113585e06,
        1.54000000e06,
        1.52620304e06,
        1.51240607e06,
        1.49860911e06,
        1.48481215e06,
        1.45185813e06,
        1.41864280e06,
        1.39845364e06,
        1.37826449e06,
        1.34238276e06,
        1.27737916e06,
        1.19468282e06,
        1.13885250e06,
        1.11129745e06,
        1.08374241e06,
        1.05552810e06,
        1.02314470e06,
        9.90761297e05,
        9.70761563e05,
        9.52296825e05,
        9.33832088e05,
        9.15367351e05,
        8.86649130e05,
        8.52636824e05,
        8.19743770e05,
        8.04672082e05,
        7.89600394e05,
        7.74528706e05,
        7.59457019e05,
        7.35894247e05,
        7.12242341e05,
        6.88590435e05,
        6.68378415e05,
        6.50581378e05,
        6.32784342e05,
        6.14987305e05,
        5.97190268e05,
        5.81916113e05,
        5.67469597e05,
        5.53023080e05,
        5.38576564e05,
        5.24130047e05,
        5.12299760e05,
        5.00472055e05,
        4.88644349e05,
        4.76816643e05,
        4.64988938e05,
        4.53161232e05,
        4.41575446e05,
        4.30009100e05,
        4.18442754e05,
        4.06876408e05,
        3.95310062e05,
        3.83743716e05,
        3.73052682e05,
        3.64325144e05,
        3.55597605e05,
        3.46870066e05,
        3.38142528e05,
        3.29414989e05,
        3.20687450e05,
        3.11959912e05,
        3.02686261e05,
        2.92725198e05,
        2.82764135e05,
        2.72803072e05,
        2.62842009e05,
        2.52880946e05,
        2.42919883e05,
        2.34039966e05,
        2.29742020e05,
        2.25444073e05,
        2.21146126e05,
        2.16848180e05,
        2.12550233e05,
        2.08252286e05,
        2.03954340e05,
        1.99656393e05,
        1.95358446e05,
        1.91060500e05,
        1.86762553e05,
        1.83491666e05,
        1.80594526e05,
        1.77697385e05,
        1.74800245e05,
        1.71903104e05,
        1.69005964e05,
        1.66108823e05,
        1.63211683e05,
        1.60314542e05,
        1.57417402e05,
        1.54520261e05,
        1.51623121e05,
        1.48725980e05,
        1.45716230e05,
        1.42022358e05,
        1.38328487e05,
        1.34634616e05,
        1.30940744e05,
        1.27246873e05,
        1.23553001e05,
        1.19859130e05,
        1.16165258e05,
        1.12471387e05,
        1.08777515e05,
        1.05083644e05,
        1.01389772e05,
        9.80586797e04,
        9.58204444e04,
        9.35822092e04,
        9.13439740e04,
        8.91057387e04,
        8.68675035e04,
        8.46292682e04,
        8.23910330e04,
        8.01527978e04,
        7.79145625e04,
        7.56763273e04,
        7.34380921e04,
        7.11998568e04,
        6.90501312e04,
        6.70597473e04,
        6.50693634e04,
        6.30789795e04,
        6.10885955e04,
        5.90982116e04,
        5.71078277e04,
        5.51174438e04,
        5.31270599e04,
        5.11366759e04,
        4.91462920e04,
        4.71559081e04,
        4.51655242e04,
        4.37894263e04,
        4.31201107e04,
        4.24507950e04,
        4.17814793e04,
        4.11121637e04,
        4.04428480e04,
        3.97735324e04,
        3.91042167e04,
        3.84349010e04,
        3.77655854e04,
        3.70962697e04,
        3.64269540e04,
        3.57576384e04,
        3.50827527e04,
        3.44037141e04,
        3.37246755e04,
        3.30456369e04,
        3.23665983e04,
        3.16875598e04,
        3.10085212e04,
        3.03294826e04,
        2.96504440e04,
        2.89714054e04,
        2.82923668e04,
        2.76133283e04,
        2.69342897e04,
        2.68937332e04,
        2.71525903e04,
        2.74114474e04,
        2.76703045e04,
        2.79291616e04,
        2.81880187e04,
        2.84468758e04,
        2.87057329e04,
        2.89645900e04,
        2.92234471e04,
        2.94823042e04,
        2.97411613e04,
        3.00000184e04,
        2.95897779e04,
        2.90002202e04,
        2.84106625e04,
        2.78211048e04,
        2.72315471e04,
        2.66419894e04,
        2.60524316e04,
        2.54628739e04,
        2.48733162e04,
        2.42837585e04,
        2.36942008e04,
        2.31046431e04,
        2.25150854e04,
        2.19377042e04,
        2.13617284e04,
        2.07857526e04,
        2.02097768e04,
        1.96338010e04,
        1.90578251e04,
        1.84818493e04,
        1.79058735e04,
        1.73298977e04,
        1.67539219e04,
        1.61779461e04,
        1.56019702e04,
        1.50256153e04,
        1.43637284e04,
        1.37018416e04,
        1.30399547e04,
        1.23780678e04,
        1.17161809e04,
        1.10542940e04,
        1.03924071e04,
        9.73052025e03,
        9.06863337e03,
        8.40674648e03,
        7.74485960e03,
        7.08297272e03,
        6.48802803e03,
        6.42225935e03,
        6.35649066e03,
        6.29072198e03,
        6.22495330e03,
        6.15918462e03,
        6.09341593e03,
        6.02764725e03,
        5.96187857e03,
        5.89610989e03,
        5.83034120e03,
        5.76457252e03,
        5.69880384e03,
        5.55058914e03,
        5.11037391e03,
        4.67015868e03,
        4.22994345e03,
        3.78972822e03,
        3.34951299e03,
        2.90929776e03,
        2.46908253e03,
        2.02886730e03,
        1.58865207e03,
        1.14843684e03,
        7.08221605e02,
        2.68006374e02,
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
