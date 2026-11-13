"""Local CRS Zarr acquisitions: explicit inputs, native timing, per-scan calibration.

The caller supplies a completed, immutable group. This module does not schedule
acquisitions or decide when the DAQ has finished writing them.
"""

from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pytz

from .calibration import file_sha256
from .offline import OfflineContext

SCAN_ROLES = ("fine_f", "fine_z", "wide_f", "wide_z", "fres")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def open_acquisition(path):
    """Validate a local Zarr v3 group without allocating its full timestream.

    Missing chunks otherwise read silently as fill values. CRS writes every
    chunk/shard, so require all physical payloads for the arrays we consume.
    This is a consistency check, not an acquisition-completion protocol.
    """
    try:
        import tensorstore as ts
    except ImportError as exc:
        raise ImportError("SR5 requires pip install 'straxion[sr5]'") from exc
    path = Path(path)
    group_metadata = json.loads((path / "zarr.json").read_text())
    if group_metadata.get("zarr_format") != 3 or group_metadata.get("node_type") != "group":
        raise ValueError("Expected a Zarr v3 acquisition group")
    arrays = {}
    for name in ("z", "dt", "counts_to_s21", "fres", "fres_requested"):
        array_path = path / name
        meta = json.loads((array_path / "zarr.json").read_text())
        physical_shape = meta["chunk_grid"]["configuration"]["chunk_shape"]
        encoding = meta["chunk_key_encoding"]
        if encoding["name"] != "default":
            raise ValueError("Expected the CRS default Zarr chunk-key encoding")
        separator = encoding.get("configuration", {}).get("separator", "/")
        grid = [(n + c - 1) // c for n, c in zip(meta["shape"], physical_shape)]
        for index in product(*(range(n) for n in grid)):
            payload = array_path / separator.join(("c", *(str(i) for i in index)))
            if not payload.is_file() or payload.stat().st_size == 0:
                raise ValueError(f"Missing or empty Zarr payload: {payload}")
        arrays[name] = ts.open(
            dict(
                driver="zarr3",
                kvstore=dict(driver="file", path=str(array_path) + "/"),
                fill_missing_data_reads=False,
            ),
            read=True,
            write=False,
        ).result()
    z = arrays["z"]
    if len(z.shape) != 3 or z.shape[0] != 2 or min(z.shape) <= 0:
        raise ValueError("Expected z with shape (2, channels, samples)")
    if z.dtype.numpy_dtype != np.dtype("int32"):
        raise ValueError("Expected CRS int32 I/Q counts")
    n_channels = z.shape[1]
    if arrays["dt"].shape != ():
        raise ValueError("Expected scalar dt")
    dt = float(arrays["dt"].read().result())
    if not np.isfinite(dt) or dt <= 0 or dt * 1e9 < 1:
        raise ValueError("dt must be a finite, positive sample interval >=1 ns")
    metadata = dict(shape=list(z.shape), dt=dt, timestamp=group_metadata["attributes"]["timestamp"])
    for name in ("counts_to_s21", "fres", "fres_requested"):
        values = np.asarray(arrays[name].read().result(), dtype=np.float64)
        if values.shape != (n_channels,) or not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError(f"{name} must contain one finite, positive value per channel")
        metadata[name] = values.tolist()
    # The DAQ writes California wall-clock time without an offset. Reject the
    # ambiguous/nonexistent DST hour rather than silently picking a UTC instant.
    local = datetime.strptime(metadata["timestamp"], "%Y%m%d,%H:%M:%S")
    utc = pytz.timezone("America/Los_Angeles").localize(local, is_dst=None).astimezone(timezone.utc)
    start = int((utc - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()) * 10**9
    metadata.update(start_ns=start, timezone="America/Los_Angeles")
    return arrays, metadata, _digest(metadata)


class SR5Context(OfflineContext):
    """Resolve native metadata before dtype inference and lineage construction."""

    def _resolved_config(self, run_id, plugin):
        cfg = super()._resolved_config(run_id, plugin)
        if self.context_config["calibration_mode"] != "per_scan":
            raise ValueError("SR5 online processing requires per_scan calibration")
        _, metadata, digest = open_acquisition(cfg["sr5_group_path"])
        low, high = cfg["channel_map"][cfg.get("sub_detector", "kids")]
        if low != 0 or high + 1 != metadata["shape"][1]:
            raise ValueError("channel_map must match all Zarr channels, starting at zero")
        owned = dict(fs=1.0 / metadata["dt"], record_length=metadata["shape"][2])
        for key, value in owned.items():
            if key in cfg and cfg[key] != value:
                raise ValueError(f"{key} conflicts with the native Zarr acquisition")
        cfg.update(owned)
        cfg.update(sr5_metadata_sha256=digest, sr5_start_ns=metadata["start_ns"])
        if "sr5_scan_files" in plugin.takes_config:
            files = cfg["sr5_scan_files"]
            if set(files) != set(SCAN_ROLES):
                raise ValueError(f"Provide exactly these scan files: {SCAN_ROLES}")
            cfg["scan_calibration_sha256"] = _digest(
                {role: file_sha256(files[role]) for role in SCAN_ROLES}
            )
            cfg["sr5_fres"] = tuple(metadata["fres"])
            cfg["sr5_fres_requested"] = tuple(metadata["fres_requested"])
        return cfg

    def _set_plugin_config(self, p, run_id, tolerant=True):
        previous = getattr(p, "config", {}).get("sr5_metadata_sha256")
        super()._set_plugin_config(p, run_id, tolerant=tolerant)
        if previous and previous != p.config.get("sr5_metadata_sha256"):
            raise ValueError("Zarr metadata changed after lineage creation")
