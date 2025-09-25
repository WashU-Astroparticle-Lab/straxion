import strax
import numpy as np
from straxion.utils import (
    DATA_DTYPE,
    SECOND_TO_NANOSECOND,
    HIT_WINDOW_LENGTH_LEFT,
    HIT_WINDOW_LENGTH_RIGHT,
    base_waveform_dtype,
)

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        "noise_window_gap",
        default=0,
        track=True,
        type=int,
        help=("Gap between noise window and hit window in unit of samples."),
    ),
    strax.Option(
        "fs",
        default=38_000,
        track=True,
        type=int,
        help="Sampling frequency (assumed the same for all channels) in unit of Hz",
    ),
)
class NoiseBank(strax.Plugin):
    """Use the waveform noise_window_gap before the hit window to estimate the noise condition."""

    __version__ = "0.0.0"
    # Inherited from straxen. Not optimized outside XENONnT.
    rechunk_on_save = False
    compressor = "zstd"
    chunk_target_size_mb = 2000
    rechunk_on_load = True
    chunk_source_size_mb = 100

    depends_on = ["records", "hits"]
    provides = "noises"
    data_kind = "noises"
    save_when = strax.SaveWhen.ALWAYS

    def infer_dtype(self):
        dtype = base_waveform_dtype()
        self.noise_window_length = HIT_WINDOW_LENGTH_LEFT + HIT_WINDOW_LENGTH_RIGHT

        dtype.append(
            (
                (
                    "Noise waveform of df/f after baseline corrections",
                    "data_dx",
                ),
                DATA_DTYPE,
                self.noise_window_length,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Noise waveform of dx=df/f0 further smoothed by moving average, "
                        "aligned at the maximum of the dx=df/f0 waveform."
                    ),
                    "data_dx_moving_average",
                ),
                DATA_DTYPE,
                self.noise_window_length,
            )
        )
        dtype.append(
            (
                (
                    "Noise waveform of dx=df/f0 further smoothed by pulse kernel, "
                    "aligned at the maximum of the dx=df/f0 waveform.",
                    "data_dx_convolved",
                ),
                DATA_DTYPE,
                self.noise_window_length,
            )
        )
        dtype.append(
            (
                (
                    "Hit finding threshold in unit of dx=df/f0 for kernel convolved signal.",
                    "hit_threshold",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    ("Maximum amplitude of the dx noise waveform",),
                    "amplitude",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    "Maximum amplitude of the dx noise waveform smoothed by moving average",
                    "amplitude_moving_average",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    "Maximum amplitude of the dx noise waveform smoothed by pulse kernel",
                    "amplitude_convolved",
                ),
                DATA_DTYPE,
            )
        )
        return dtype

    def setup(self):
        self.noise_window_length = HIT_WINDOW_LENGTH_LEFT + HIT_WINDOW_LENGTH_RIGHT
        self.fs = self.config["fs"]
        self.dt_exact = 1 / self.fs * SECOND_TO_NANOSECOND
        self.gap = self.config["noise_window_gap"]

    def compute(self, records, hits):
        """Extract noise windows before each hit for noise characterization."""
        if len(hits) == 0:
            return np.zeros(0, dtype=self.infer_dtype())

        # Filter valid hits and extract noise windows in one pass
        valid_hits, noise_data = self._extract_valid_noise_windows(records, hits)

        if len(valid_hits) == 0:
            return np.zeros(0, dtype=self.infer_dtype())

        # Create results array with pre-allocated data
        results = np.zeros(len(valid_hits), dtype=self.infer_dtype())

        # Vectorized assignments
        results["time"] = valid_hits["time"] - np.int64(self.noise_window_length * self.dt_exact)
        results["endtime"] = valid_hits["time"]
        results["data_dx"] = noise_data["data_dx"]
        results["data_dx_moving_average"] = noise_data["data_dx_moving_average"]
        results["data_dx_convolved"] = noise_data["data_dx_convolved"]
        results["amplitude"] = noise_data["amplitudes"]
        results["amplitude_moving_average"] = noise_data["amplitudes_moving_average"]
        results["amplitude_convolved"] = noise_data["amplitudes_convolved"]
        results["channel"] = noise_data["channel"]
        results["length"] = noise_data["length"]
        results["hit_threshold"] = noise_data["hit_threshold"]
        results["dt"] = noise_data["dt"]

        # Sort by time
        results = results[np.argsort(results["time"])]

        # Truncate results start time and endtime to record endtime
        results["endtime"] = np.minimum(results["endtime"], records["endtime"][0])
        results["time"] = np.maximum(results["time"], records["time"][0])

        return results

    def _extract_valid_noise_windows(self, records, hits):
        """Extract valid noise windows and their data in one pass."""
        valid_hits = []
        noise_data = {
            "data_dx": [],
            "data_dx_moving_average": [],
            "data_dx_convolved": [],
            "amplitudes": [],
            "amplitudes_moving_average": [],
            "amplitudes_convolved": [],
            "channel": [],
            "length": [],
            "hit_threshold": [],
            "dt": [],
        }

        # Process each channel separately to maintain hit ordering
        for ch in range(len(records)):
            hits_ch = hits[hits["channel"] == ch]
            if len(hits_ch) == 0:
                continue

            hits_ch = hits_ch[np.argsort(hits_ch["time"])]

            # Vectorized calculation of start/end indices for all hits in channel
            start_indices = (
                hits_ch["amplitude_convolved_max_record_i"]
                - HIT_WINDOW_LENGTH_LEFT
                - self.noise_window_length
                - self.gap
            )
            end_indices = (
                hits_ch["amplitude_convolved_max_record_i"] - HIT_WINDOW_LENGTH_LEFT - self.gap
            )

            # Filter valid hits (start_i >= 0)
            valid_mask = start_indices >= 0

            # Check for overlaps with previous hits
            if len(hits_ch) > 1:
                prev_hit_ends = np.full(len(hits_ch), -1)
                prev_hit_ends[1:] = (
                    hits_ch["amplitude_convolved_max_record_i"][:-1] + HIT_WINDOW_LENGTH_RIGHT
                )
                overlap_mask = prev_hit_ends < start_indices
                valid_mask &= overlap_mask

            # Process valid hits
            valid_hits_ch = hits_ch[valid_mask]
            valid_starts = start_indices[valid_mask]
            valid_ends = end_indices[valid_mask]

            for i, (hit, start_i, end_i) in enumerate(zip(valid_hits_ch, valid_starts, valid_ends)):
                # Extract noise window data
                noise_slice = slice(int(start_i), int(end_i))
                data_dx = records[ch]["data_dx"][noise_slice]
                data_dx_ma = records[ch]["data_dx_moving_average"][noise_slice]
                data_dx_conv = records[ch]["data_dx_convolved"][noise_slice]

                # Store hit and data
                valid_hits.append(hit)
                noise_data["data_dx"].append(data_dx)
                noise_data["data_dx_moving_average"].append(data_dx_ma)
                noise_data["data_dx_convolved"].append(data_dx_conv)
                noise_data["amplitudes"].append(np.max(data_dx))
                noise_data["amplitudes_moving_average"].append(np.max(data_dx_ma))
                noise_data["amplitudes_convolved"].append(np.max(data_dx_conv))
                noise_data["channel"].append(ch)
                noise_data["length"].append(end_i - start_i)
                noise_data["hit_threshold"].append(hit["hit_threshold"])
                noise_data["dt"].append(records[ch]["dt"])

        # Convert lists to numpy arrays
        if valid_hits:
            valid_hits = np.array(valid_hits)
            noise_data = {k: np.array(v) for k, v in noise_data.items()}
        else:
            valid_hits = np.zeros(0, dtype=hits.dtype)
            noise_data = {k: np.zeros(0) for k in noise_data.keys()}

        return valid_hits, noise_data
