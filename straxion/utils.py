import bisect
import numpy as np
import pickle
import os

from .constants import TIME_DTYPE, LENGTH_DTYPE, CHANNEL_DTYPE
from .constants import *  # noqa: F401, F403


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


def find_run_for_hit(hit_time, run_ids, run_ends_ns=None):
    """Return the run_id whose time window contains a hit.

    Run IDs are assumed to be Unix-second timestamp strings (the straxion convention,
    e.g. "1756824965"), and ``hit_time`` is in nanoseconds since the Unix epoch
    (the strax ``time`` field convention).

    Args:
        hit_time (int): Hit timestamp in nanoseconds since the Unix epoch.
        run_ids (sequence of str or int): Run identifiers. Each is interpreted as a
            Unix-second start time. Need not be sorted.
        run_ends_ns (sequence of int, optional): Explicit run end times in
            nanoseconds since the Unix epoch, aligned with ``run_ids``. If provided,
            a hit only matches a run when ``start <= hit_time < end``; otherwise
            ``None`` is returned. If ``None``, each run is treated as ending at the
            next run's start (and the last run as open-ended).

    Returns:
        str or None: The matching run_id (as a string, matching strax conventions),
        or ``None`` if the hit does not fall inside any run.

    """
    if len(run_ids) == 0:
        return None

    starts_ns = np.array([int(rid) * 1_000_000_000 for rid in run_ids], dtype=np.int64)
    order = np.argsort(starts_ns)
    starts_sorted = starts_ns[order]
    ids_sorted = [str(run_ids[i]) for i in order]

    idx = bisect.bisect_right(starts_sorted.tolist(), int(hit_time)) - 1
    if idx < 0:
        return None

    if run_ends_ns is not None:
        end_ns = int(np.asarray(run_ends_ns)[order[idx]])
        if int(hit_time) >= end_ns:
            return None

    return ids_sorted[idx]


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
