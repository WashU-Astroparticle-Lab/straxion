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
        6.33251843e-05,
        5.20222799e-04,
        9.77120415e-04,
        1.43401803e-03,
        2.88320284e-03,
        4.38753164e-03,
        5.89186043e-03,
        7.39618923e-03,
        8.90051802e-03,
        1.04048468e-02,
        1.19091756e-02,
        1.34135044e-02,
        1.49178332e-02,
        1.64221620e-02,
        1.79264908e-02,
        1.94308196e-02,
        7.54361711e-02,
        2.75133786e-01,
        4.74831402e-01,
        6.74529017e-01,
        8.74226632e-01,
        1.07392425e00,
        1.27107660e00,
        1.25968894e00,
        1.24830128e00,
        1.23691362e00,
        1.22552596e00,
        1.19832656e00,
        1.17091147e00,
        1.15424786e00,
        1.13758425e00,
        1.10796839e00,
        1.05431608e00,
        9.86060640e-01,
        9.39979715e-01,
        9.17236486e-01,
        8.94493257e-01,
        8.71205889e-01,
        8.44477459e-01,
        8.17749028e-01,
        8.01241759e-01,
        7.86001437e-01,
        7.70761115e-01,
        7.55520794e-01,
        7.31817509e-01,
        7.03744621e-01,
        6.76595536e-01,
        6.64155751e-01,
        6.51715965e-01,
        6.39276180e-01,
        6.26836394e-01,
        6.07388285e-01,
        5.87866607e-01,
        5.68344929e-01,
        5.51662445e-01,
        5.36973226e-01,
        5.22284007e-01,
        5.07594788e-01,
        4.92905570e-01,
        4.80298673e-01,
        4.68374888e-01,
        4.56451102e-01,
        4.44527317e-01,
        4.32603532e-01,
        4.22839116e-01,
        4.13076830e-01,
        4.03314545e-01,
        3.93552259e-01,
        3.83789973e-01,
        3.74027688e-01,
        3.64465076e-01,
        3.54918510e-01,
        3.45371944e-01,
        3.35825378e-01,
        3.26278812e-01,
        3.16732246e-01,
        3.07908140e-01,
        3.00704653e-01,
        2.93501166e-01,
        2.86297679e-01,
        2.79094192e-01,
        2.71890705e-01,
        2.64687217e-01,
        2.57483730e-01,
        2.49829496e-01,
        2.41607889e-01,
        2.33386283e-01,
        2.25164676e-01,
        2.16943070e-01,
        2.08721463e-01,
        2.00499857e-01,
        1.93170601e-01,
        1.89623186e-01,
        1.86075770e-01,
        1.82528355e-01,
        1.78980940e-01,
        1.75433525e-01,
        1.71886109e-01,
        1.68338694e-01,
        1.64791279e-01,
        1.61243864e-01,
        1.57696449e-01,
        1.54149033e-01,
        1.51449327e-01,
        1.49058101e-01,
        1.46666876e-01,
        1.44275650e-01,
        1.41884424e-01,
        1.39493199e-01,
        1.37101973e-01,
        1.34710747e-01,
        1.32319522e-01,
        1.29928296e-01,
        1.27537070e-01,
        1.25145845e-01,
        1.22754619e-01,
        1.20270448e-01,
        1.17221621e-01,
        1.14172794e-01,
        1.11123967e-01,
        1.08075140e-01,
        1.05026313e-01,
        1.01977486e-01,
        9.89286592e-02,
        9.58798322e-02,
        9.28310052e-02,
        8.97821782e-02,
        8.67333512e-02,
        8.36845241e-02,
        8.09351256e-02,
        7.90877435e-02,
        7.72403614e-02,
        7.53929793e-02,
        7.35455972e-02,
        7.16982151e-02,
        6.98508330e-02,
        6.80034508e-02,
        6.61560687e-02,
        6.43086866e-02,
        6.24613045e-02,
        6.06139224e-02,
        5.87665403e-02,
        5.69922118e-02,
        5.53493998e-02,
        5.37065879e-02,
        5.20637759e-02,
        5.04209639e-02,
        4.87781520e-02,
        4.71353400e-02,
        4.54925280e-02,
        4.38497161e-02,
        4.22069041e-02,
        4.05640921e-02,
        3.89212802e-02,
        3.72784682e-02,
        3.61426722e-02,
        3.55902362e-02,
        3.50378002e-02,
        3.44853642e-02,
        3.39329281e-02,
        3.33804921e-02,
        3.28280561e-02,
        3.22756201e-02,
        3.17231840e-02,
        3.11707480e-02,
        3.06183120e-02,
        3.00658759e-02,
        2.95134399e-02,
        2.89564065e-02,
        2.83959455e-02,
        2.78354844e-02,
        2.72750233e-02,
        2.67145622e-02,
        2.61541012e-02,
        2.55936401e-02,
        2.50331790e-02,
        2.44727179e-02,
        2.39122569e-02,
        2.33517958e-02,
        2.27913347e-02,
        2.22308736e-02,
        2.21973993e-02,
        2.24110534e-02,
        2.26247074e-02,
        2.28383614e-02,
        2.30520155e-02,
        2.32656695e-02,
        2.34793235e-02,
        2.36929775e-02,
        2.39066316e-02,
        2.41202856e-02,
        2.43339396e-02,
        2.45475936e-02,
        2.47612477e-02,
        2.44226457e-02,
        2.39360398e-02,
        2.34494340e-02,
        2.29628281e-02,
        2.24762223e-02,
        2.19896164e-02,
        2.15030106e-02,
        2.10164047e-02,
        2.05297989e-02,
        2.00431930e-02,
        1.95565872e-02,
        1.90699813e-02,
        1.85833755e-02,
        1.81068198e-02,
        1.76314241e-02,
        1.71560284e-02,
        1.66806327e-02,
        1.62052370e-02,
        1.57298413e-02,
        1.52544456e-02,
        1.47790499e-02,
        1.43036542e-02,
        1.38282585e-02,
        1.33528628e-02,
        1.28774671e-02,
        1.24017585e-02,
        1.18554540e-02,
        1.13091495e-02,
        1.07628450e-02,
        1.02165405e-02,
        9.67023598e-03,
        9.12393148e-03,
        8.57762698e-03,
        8.03132248e-03,
        7.48501798e-03,
        6.93871348e-03,
        6.39240898e-03,
        5.84610448e-03,
        5.35505235e-03,
        5.30076856e-03,
        5.24648478e-03,
        5.19220099e-03,
        5.13791720e-03,
        5.08363341e-03,
        5.02934962e-03,
        4.97506584e-03,
        4.92078205e-03,
        4.86649826e-03,
        4.81221447e-03,
        4.75793068e-03,
        4.70364690e-03,
        4.58131428e-03,
        4.21797189e-03,
        3.85462950e-03,
        3.49128711e-03,
        3.12794472e-03,
        2.76460233e-03,
        2.40125994e-03,
        2.03791755e-03,
        1.67457516e-03,
        1.31123277e-03,
        9.47890383e-04,
        5.84547994e-04,
        2.21205605e-04,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
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
