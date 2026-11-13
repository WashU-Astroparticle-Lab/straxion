"""SR5 adapters for the existing offline processing chain."""

import numpy as np
import strax
from immutabledict import immutabledict

from .raw_records import QUALIPHIDETHzReader
from .records import DxRecords
from ..sr5 import open_acquisition


@strax.takes_config(
    strax.Option("sr5_group_path", type=str, track=False),
    strax.Option("sr5_metadata_sha256", type=str),
    strax.Option("sr5_start_ns", type=int),
    strax.Option("fs", type=(int, float)),
    strax.Option("record_length", type=int),
)
class SR5RawRecords(QUALIPHIDETHzReader):
    """Read all samples of one CRS group into normalized float32 S21 I/Q.

    Storage chunks are copied in blocks into the final record arrays. They do
    not become separate runs or separate processing chunks.
    """

    __version__ = "0.1.0"
    takes_config = immutabledict(
        {
            key: value
            for key, value in QUALIPHIDETHzReader.takes_config.items()
            if key not in ("daq_input_dir", "fs", "record_length")
        }
    )

    def infer_dtype(self):
        dtype = super().infer_dtype()
        for i, field in enumerate(dtype):
            if isinstance(field[0], tuple) and field[0][1] in ("data_i", "data_q"):
                name = field[0][1]
                dtype[i] = (("Normalized S21 " + name[-1].upper(), name), *field[1:])
        return dtype

    def compute(self):
        group, metadata, digest = open_acquisition(self.config["sr5_group_path"])
        if digest != self.config["sr5_metadata_sha256"]:
            raise ValueError("Zarr metadata changed after lineage creation")
        z = group["z"]
        _, channels, samples = z.shape
        results = np.zeros(channels, dtype=self.dtype)
        start = self.config["sr5_start_ns"]
        end = start + int(samples * metadata["dt"] * 10**9)
        results["time"], results["endtime"] = start, end
        results["dt"], results["length"] = self.dt, samples
        results["channel"] = np.arange(channels)
        scale = np.asarray(metadata["counts_to_s21"])[:, None]
        block_size = z.chunk_layout.read_chunk.shape[-1]
        for left in range(0, samples, block_size):
            right = min(left + block_size, samples)
            block = z[:, :, left:right].read().result()
            results["data_i"][:, left:right] = block[0] * scale
            results["data_q"][:, left:right] = block[1] * scale
        return self.chunk(start=start, end=end, data=results)


@strax.takes_config(
    strax.Option("sr5_scan_files", type=dict, track=False),
    strax.Option("sr5_fres", type=tuple),
    strax.Option("sr5_fres_requested", type=tuple),
)
class SR5DxRecords(DxRecords):
    """Use explicit scan files and the frequencies actually transmitted by CRS."""

    __version__ = "0.1.0"
    takes_config = immutabledict(
        {
            key: value
            for key, value in DxRecords.takes_config.items()
            if key not in (
                "iq_finescan_dir",
                "iq_widescan_dir",
                "resonant_frequency_dir",
                "iq_finescan_filename",
                "iq_widescan_filename",
                "resonant_frequency_filename",
            )
        }
    )

    def _load_correction_files(self):
        arrays = {
            role: np.load(path, allow_pickle=False)
            for role, path in self.config["sr5_scan_files"].items()
        }
        requested = np.asarray(self.config["sr5_fres_requested"])
        if not np.array_equal(arrays["fres"], requested):
            raise ValueError("Scan fres does not match the acquisition's fres_requested")
        self.fres = np.asarray(self.config["sr5_fres"], dtype=np.float64)
        for kind in ("fine", "wide"):
            f, z = arrays[f"{kind}_f"], arrays[f"{kind}_z"]
            if (
                f.ndim != 2
                or f.shape != z.shape
                or f.shape[0] != len(self.fres)
                or not np.all(np.isfinite(f))
                or not np.all(np.isfinite(z))
            ):
                raise ValueError(f"Invalid {kind} scan shape or nonfinite values")
            if not np.all((self.fres >= f.min(axis=1)) & (self.fres <= f.max(axis=1))):
                raise ValueError(f"Actual tone frequencies lie outside the {kind} scan")
            setattr(self, f"{kind}_f", f)
            setattr(self, f"{kind}_z", z)
