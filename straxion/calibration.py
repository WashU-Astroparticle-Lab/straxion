"""Frozen calibration preparation for completed QUALIPHIDE science runs.

Only scans are loaded here, never the waveform files. Epoch membership is explicit
in the manifest; adding acquisitions later requires a new calibration directory.
"""

import hashlib
import io
import json
from pathlib import Path
import re
import warnings

import numpy as np

__all__ = ["prepare_offline_calibration"]

SCHEMA_VERSION = 1


def file_sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_file(path, expected):
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"Calibration file changed: {path}; prepare a new calibration release")
    return actual


def read_dx_map(path, expected_hash="", n_channels=None):
    """Read and validate one complete map, hashing the exact bytes being loaded."""
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_hash and digest != expected_hash:
        raise ValueError(f"Calibration file changed: {path}")
    with np.load(io.BytesIO(content), allow_pickle=False) as data:
        x, y = data["x"], data["y_med"]
    if x.ndim != 2 or x.shape != y.shape or x.shape[0] == 0 or x.shape[1] < 2:
        raise ValueError("Averaged map x and y_med must have matching (channels, grid) shapes")
    if n_channels is not None and len(x) != n_channels:
        raise ValueError(f"Averaged map has {len(x)} channels; expected {n_channels}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Averaged map must be finite for every channel")
    if not np.all(np.diff(x, axis=1) > 0):
        raise ValueError("Averaged map phase grids must be strictly increasing")
    if not np.all((x[:, 0] < 0) & (x[:, -1] > 0)):
        raise ValueError("Averaged map must bracket zero phase for every channel")
    return x, y, digest


def scan_config(input_folder, files):
    base = str(Path(input_folder).resolve())
    return dict(
        iq_finescan_dir=base,
        iq_finescan_filename=files["fine_z"],
        iq_widescan_dir=base,
        iq_widescan_filename=files["wide_z"],
        resonant_frequency_dir=base,
        resonant_frequency_filename=files["fres"],
    )


def calibration_tables(config):
    """Use the very same per-scan calibration and interpolation as DxRecords."""
    from .plugins.records import DxRecords

    plugin = DxRecords()
    plugin.config = dict(config, dx_map_filename="")
    plugin._setup_iq_correction_and_calibration()
    plugin._setup_frequency_interpolation_models()
    return [
        (x.copy(), (f - f0) / f0)
        for x, f, f0 in zip(plugin.interp_x_data, plugin.interp_y_data, plugin.interpolated_freqs)
    ]


def average_maps(tables, n_grid=2000):
    """Pointwise median of normalized scan maps on their common phase interval.

    Matches the reprocessing notebooks' unsmoothed y_med. Percentiles describe
    scan scatter, not uncertainty on the median. Each scan has equal weight.
    """
    if not tables or n_grid < 2 or int(n_grid) != n_grid:
        raise ValueError("Need at least one scan and an integer n_grid >= 2")
    n_grid = int(n_grid)
    n_channels = len(tables[0])
    if not n_channels or any(len(t) != n_channels for t in tables):
        raise ValueError("All scans must have the same nonzero channel count")
    result = {key: np.empty((n_channels, n_grid)) for key in ("x", "y_med", "y16", "y84")}
    for channel in range(n_channels):
        scans = [t[channel] for t in tables]
        for x, y in scans:
            if (
                x.ndim != 1
                or x.shape != y.shape
                or len(x) < 2
                or not np.all(np.isfinite(x))
                or not np.all(np.isfinite(y))
                or not np.all(np.diff(x) > 0)
            ):
                raise ValueError(f"Invalid calibration table for channel {channel}")
        lo, hi = max(x[0] for x, _ in scans), min(x[-1] for x, _ in scans)
        if not lo < 0 < hi:
            raise ValueError(
                f"Scans have no common phase interval bracketing zero: channel {channel}"
            )
        grid = np.linspace(lo, hi, n_grid)
        ys = np.array([np.interp(grid, x, y) for x, y in scans])
        result["x"][channel] = grid
        result["y_med"][channel] = np.median(ys, axis=0)
        result["y16"][channel], result["y84"][channel] = np.percentile(ys, [16, 84], axis=0)
    return result


def _timestamp_files(folder, pattern, regex):
    result = {}
    for path in sorted(folder.glob(pattern)):
        match = re.fullmatch(regex, path.name)
        if match is None:
            continue
        stamp = int(match[1])
        if stamp in result:
            raise ValueError(f"Ambiguous input timestamp {stamp}: {result[stamp]}, {path.name}")
        result[stamp] = path.name
    return result


def prepare_offline_calibration(
    input_folder,
    calibration_dir,
    *,
    epochs=None,
    exclude_runs=(),
    gap_hours=6.0,
    fs=38000,
    n_grid=2000,
    widescan_resolution=1000,
    cable_correction_polyfit_order=5,
):
    """Build immutable epoch maps and return the manifest's absolute Path.

    ``epochs`` optionally maps labels to acquisition timestamps and must cover
    every non-excluded acquisition exactly once. Otherwise split at start-time
    gaps strictly greater than ``gap_hours`` (the historical notebook rule).
    Exclude bad acquisitions explicitly; only scans assigned to retained runs
    contribute. A scan is the latest complete configuration strictly before an
    acquisition. Sparse epochs warn but are never merged automatically.

    The destination must not exist: preparing a replacement always creates a
    separate release, so an interrupted or previous preparation is not overwritten.
    """
    folder = Path(input_folder).resolve()
    destination = Path(calibration_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Calibration destination exists: {destination}; use a new directory")
    if not np.isfinite(gap_hours) or gap_hours <= 0:
        raise ValueError("gap_hours must be positive and finite")
    if not isinstance(fs, int) or fs <= 0 or fs % 1000:
        raise ValueError("fs must be a positive integer multiple of 1000 Hz")
    if not np.isfinite(widescan_resolution) or widescan_resolution <= 0:
        raise ValueError("widescan_resolution must be positive and finite")
    if not isinstance(cable_correction_polyfit_order, int) or cable_correction_polyfit_order < 0:
        raise ValueError("cable_correction_polyfit_order must be a nonnegative integer")
    runs = _timestamp_files(folder, f"ts_{fs // 1000}kHz-*.npy", r"ts_\d+kHz-(\d+)\.npy")
    excluded = {int(r) for r in exclude_runs}
    if excluded - runs.keys():
        raise ValueError(f"Excluded acquisitions not found: {sorted(excluded - runs.keys())}")
    retained = sorted(runs.keys() - excluded)
    if not retained:
        raise ValueError(f"No retained acquisitions in {folder}")
    if epochs is None:
        groups = np.split(retained, np.flatnonzero(np.diff(retained) > gap_hours * 3600) + 1)
        epochs = {f"e{i}": [int(r) for r in group] for i, group in enumerate(groups)}
    else:
        epochs = {str(label): [int(r) for r in members] for label, members in epochs.items()}
    members = [r for group in epochs.values() for r in group]
    if len(members) != len(set(members)) or set(members) != set(retained):
        raise ValueError("Explicit epochs must cover every retained acquisition exactly once")
    if any(not group for group in epochs.values()):
        raise ValueError("Epochs cannot be empty")

    patterns = {
        "fine_z": ("iq_fine_z_*.npy", r"iq_fine_z_.*-(\d+)\.npy"),
        "fine_f": ("iq_fine_f_*.npy", r"iq_fine_f_.*-(\d+)\.npy"),
        "wide_z": ("iq_wide_z_*.npy", r"iq_wide_z_.*-(\d+)\.npy"),
        "wide_f": ("iq_wide_f_*.npy", r"iq_wide_f_.*-(\d+)\.npy"),
        "fres": ("fres_*.npy", r"fres_.*-(\d+)\.npy"),
    }
    files = {key: _timestamp_files(folder, *pattern) for key, pattern in patterns.items()}
    # An incomplete newer scan must not silently select an older calibration.
    scan_times = sorted(set().union(*(set(f) for f in files.values())))
    assignments = {}
    for run in retained:
        previous = [stamp for stamp in scan_times if stamp < run]
        if not previous:
            raise ValueError(f"No scan strictly before acquisition {run}")
        stamp = previous[-1]
        if any(stamp not in f for f in files.values()):
            raise ValueError(f"Incomplete scan {stamp} for acquisition {run}")
        assignments[run] = stamp

    settings = dict(
        widescan_resolution=widescan_resolution,
        cable_correction_polyfit_order=cable_correction_polyfit_order,
    )
    scans, tables = {}, {}
    for stamp in sorted(set(assignments.values())):
        names = {key: f[stamp] for key, f in files.items()}
        for key in ("fine", "wide"):
            if names[f"{key}_z"].replace(f"{key}_z", f"{key}_f") != names[f"{key}_f"]:
                raise ValueError(f"Mismatched {key} scan files at {stamp}")
        hashes = {name: file_sha256(folder / name) for name in names.values()}
        tables[stamp] = calibration_tables(dict(scan_config(folder, names), **settings))
        for name, digest in hashes.items():
            verify_file(folder / name, digest)
        scans[str(stamp)] = dict(files=names, sha256=hashes)

    maps = {}
    for label, group in epochs.items():
        stamps = sorted({assignments[r] for r in group})
        if len(stamps) < 3:
            warnings.warn(
                f"Epoch {label} has only {len(stamps)} scan(s); weak median", stacklevel=2
            )
        maps[label] = (average_maps([tables[s] for s in stamps], n_grid), stamps)

    # Write maps first; the final manifest is the completion marker.
    destination.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema_version=SCHEMA_VERSION,
        algorithm="pointwise-median-v1",
        producer_sha256={
            "calibration.py": file_sha256(Path(__file__)),
            "plugins/records.py": file_sha256(Path(__file__).parent / "plugins" / "records.py"),
        },
        settings=dict(settings, fs=fs, n_grid=n_grid, gap_hours=gap_hours),
        excluded_runs=sorted(excluded),
        scans=scans,
        runs={},
        epochs={},
    )
    for i, (label, (data, stamps)) in enumerate(maps.items()):
        path = destination / f"dx_map_{i}.npz"
        np.savez(path, **data, configs=np.array(stamps))
        read_dx_map(path)
        manifest["epochs"][label] = dict(
            runs=sorted(epochs[label]), scans=stamps, filename=path.name, sha256=file_sha256(path)
        )
        for run in epochs[label]:
            manifest["runs"][str(run)] = dict(
                filename=runs[run], scan=str(assignments[run]), epoch=label
            )
    manifest_path = destination / "manifest.json"
    temporary = destination / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return manifest_path
