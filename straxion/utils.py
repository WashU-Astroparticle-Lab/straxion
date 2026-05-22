import bisect
import numpy as np
import pickle
import os

from .constants import (
    TIME_DTYPE,
    LENGTH_DTYPE,
    CHANNEL_DTYPE,
    MIRROR_CHANNELS,
    QUALIPHIDE_BASE_FOLDERS,
    PHOTON_25um_meV,
)
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


def find_sample_for_hit(hit_time, records):
    """Return the sample index in ``records`` corresponding to a hit's start time.

    A hit's ``time`` field is the start time of the hit window in nanoseconds
    since the Unix epoch (the strax convention). All channels share the same
    timing within a chunk, so this finds any record whose ``[time, endtime)``
    window contains ``hit_time`` and returns

        ``sample_index = (hit_time - record["time"]) // record["dt"]``

    which is the same for every channel.

    Args:
        hit_time (int): Hit start time in nanoseconds since the Unix epoch.
        records (np.ndarray): Structured array with fields ``time``, ``endtime``,
            and ``dt``.

    Returns:
        int or None: Sample index ``0 <= sample_index < record["length"]`` of
        the matched record, or ``None`` if no record contains ``hit_time``.

    """
    if len(records) == 0:
        return None

    hit_time = int(hit_time)
    mask = (records["time"] <= hit_time) & (hit_time < records["endtime"])
    matches = np.where(mask)[0]
    if len(matches) == 0:
        return None

    record = records[matches[0]]
    return int((hit_time - int(record["time"])) // int(record["dt"]))


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


def _list_runs_and_configs(base_folder):
    """List run and configuration unix-second timestamps in a QUALIPHIDE folder.

    Files are named ``ts_<fs>kHz-<unix_seconds>.npy`` for DAQ runs and
    ``<name>-<unix_seconds>.npy`` for configuration files.

    Returns:
        tuple: ``(run_timestamps, config_timestamps)`` as sorted ``np.ndarray`` of int.
    """
    import glob as _glob

    run_ts = set()
    config_ts = set()
    for path in _glob.glob(os.path.join(base_folder, "*.npy")):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            tag = int(stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if stem.startswith("ts_"):
            run_ts.add(tag)
        else:
            config_ts.add(tag)
    return np.array(sorted(run_ts)), np.array(sorted(config_ts))


def _find_config_for_run(run_ts, configurations):
    """Return the most recent configuration timestamp strictly before ``run_ts``."""
    earlier = configurations[configurations < int(run_ts)]
    if len(earlier) == 0:
        raise ValueError(f"No configuration timestamp found before run {run_ts}.")
    return int(earlier[-1])


def _wrapped_get_array(
    sr="SR3",
    targets=("hits", "hit_classification"),
    check_available=("hits",),
    config_override=None,
    st=None,
    fs_kHz=38,
    make=False,
    output_folder=None,
    keep_columns=None,
    load_only=False,
    exclude_mirror=False,
    energy_range=None,
    max_count=None,
    pre_defined_runlist=None,
):
    """Build or load strax data for every run in a QUALIPHIDE science run.

    Server-only: expects the QUALIPHIDE data layout under
    ``QUALIPHIDE_BASE_FOLDERS`` on the analysis server.

    Args:
        sr (str): Science run key, one of ``QUALIPHIDE_BASE_FOLDERS`` (e.g. ``"SR3"``).
        targets (tuple or str): strax target(s) passed to ``st.get_array`` / ``st.make``.
        check_available (tuple or str): Target(s) checked by ``st.is_stored`` when
            ``load_only=True``.
        config_override (dict, optional): Per-run config overrides merged on top of
            the auto-derived file paths.
        st (strax.Context, optional): straxion context. Defaults to a fresh
            ``straxion.qualiphide_thz_offline()`` context.
        fs_kHz (int): Sampling rate used to locate DAQ ``ts_<fs>kHz-...`` files.
        make (bool): If True, run ``st.make`` for each run and return ``None``.
        output_folder (str, optional): Storage path; defaults to
            ``<base>/strax_data``.
        keep_columns (list, optional): Forwarded to ``st.get_array``.
        load_only (bool): If True, only load runs already stored, skipping the rest.
            Enables ``energy_range`` / ``max_count`` filtering.
        exclude_mirror (bool): If True, drop ``MIRROR_CHANNELS`` from the final array.
        energy_range (tuple, optional): ``(lo, hi)`` in meV; rows outside are dropped.
            Only applied when ``load_only=True``.
        max_count (int, optional): Stop loading once the concatenated result exceeds
            this many rows. Only applied when ``load_only=True``.
        pre_defined_runlist (sequence, optional): Use this runlist instead of scanning
            the folder. Pass ``None`` to scan and use every run in the folder.

    Returns:
        tuple or None: ``(final_result, loaded_runlist)`` if ``make=False``,
        otherwise ``None``.
    """
    import strax
    import straxion
    from tqdm import tqdm

    if config_override is None:
        config_override = {}
    if st is None:
        st = straxion.qualiphide_thz_offline()

    base_folder = QUALIPHIDE_BASE_FOLDERS[sr]
    storage = (
        output_folder if output_folder is not None else os.path.join(base_folder, "strax_data")
    )
    st.storage = [strax.DataDirectory(storage)]

    scanned_runs, configurations = _list_runs_and_configs(base_folder)
    runlist = scanned_runs if pre_defined_runlist is None else pre_defined_runlist

    results = []
    loaded_runlist = []

    for run in tqdm(runlist):
        config_run = _find_config_for_run(run, configurations)
        configs = {
            "daq_input_dir": os.path.join(base_folder, f"ts_{fs_kHz}kHz-{run}.npy"),
            "iq_finescan_dir": base_folder,
            "iq_finescan_filename": f"iq_fine_z_2dB_below_pcrit-{config_run}.npy",
            "iq_widescan_dir": base_folder,
            "iq_widescan_filename": f"iq_wide_z_2dB_below_pcrit-{config_run}.npy",
            "resonant_frequency_dir": base_folder,
            "resonant_frequency_filename": f"fres_2dB-{config_run}.npy",
        }
        configs.update(config_override)
        run_str = str(run)

        if make:
            st.make(run_str, targets, config=configs, progress_bar=False)
            continue

        if not load_only:
            results.append(
                st.get_array(
                    run_str,
                    targets,
                    config=configs,
                    progress_bar=False,
                    keep_columns=keep_columns,
                )
            )
            continue

        if not st.is_stored(run_str, check_available, config=configs):
            continue
        result = st.get_array(
            run_str,
            targets,
            config=configs,
            progress_bar=False,
            keep_columns=keep_columns,
        )
        if energy_range is not None:
            energy = result["best_aOF"] * PHOTON_25um_meV
            result = result[(energy > energy_range[0]) & (energy < energy_range[1])]
        results.append(result)
        loaded_runlist.append(run)
        if max_count is not None and sum(len(r) for r in results) > max_count:
            break

    print(f"Loaded {len(loaded_runlist)} runs!")
    if make:
        return None

    final_result = np.concatenate(results) if results else np.array([])
    if exclude_mirror and len(final_result):
        keep = ~np.isin(final_result["channel"], MIRROR_CHANNELS)
        final_result = final_result[keep]
    return final_result, loaded_runlist
