import strax
import numpy as np
from straxion.utils import (
    DATA_DTYPE,
    TIME_DTYPE,
    SECOND_TO_NANOSECOND,
    base_waveform_dtype,
    N_BASELINE_MONITOR_INTERVAL,
)

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        "record_length",
        default=1_900_000,
        track=False,  # Not tracking record length, but we will have to check if it is as promised
        type=int,
        help=(
            "Number of samples in each dataset."
            "We assumed that each sample is equally spaced in time, with interval 1/fs."
            "It should not go beyond a billion so that numpy can still handle."
        ),
    ),
    strax.Option(
        "fs",
        default=38_000,
        track=True,
        type=int,
        help="Sampling frequency (assumed the same for all channels) in unit of Hz",
    ),
)
class BaselineMonitor(strax.Plugin):
    """Monitor the baseline std of the phase data."""

    __version__ = "0.0.0"
    rechunk_on_save = False
    compressor = "zstd"

    depends_on = "records"
    provides = "baseline_monitor"
    data_kind = "records"
    save_when = strax.SaveWhen.ALWAYS

    def setup(self):
        total_length_ns = self.config["record_length"] / self.config["fs"] * SECOND_TO_NANOSECOND
        self.baseline_monitor_interval = total_length_ns // N_BASELINE_MONITOR_INTERVAL
        # Window size for computing std.
        self.chunk_size = int(self.config["record_length"] / N_BASELINE_MONITOR_INTERVAL)

    def infer_dtype(self):
        dtype = base_waveform_dtype()
        dtype.append(
            (
                (
                    "Baseline monitoring interval in unit of nanoseconds.",
                    "baseline_monitor_interval",
                ),
                TIME_DTYPE,
            ),
        )
        dtype.append(
            (
                ("Baseline standard deviation of the raw data.", "baseline_monitor_std"),
                DATA_DTYPE,
                N_BASELINE_MONITOR_INTERVAL,
            ),
        )
        dtype.append(
            (
                (
                    "Baseline standard deviation of the smoothed data.",
                    "baseline_monitor_std_moving_average",
                ),
                DATA_DTYPE,
                N_BASELINE_MONITOR_INTERVAL,
            ),
        )
        dtype.append(
            (
                (
                    "Baseline standard deviation of the pulse kernel convolved data.",
                    "baseline_monitor_std_convolved",
                ),
                DATA_DTYPE,
                N_BASELINE_MONITOR_INTERVAL,
            ),
        )

        return dtype

    def compute(self, records):
        # Copy common fields from records.
        results = np.zeros(len(records), dtype=self.infer_dtype())
        for field in ["time", "endtime", "length", "dt", "channel"]:
            results[field] = records[field]

        # Assign baseline monitor interval.
        results["baseline_monitor_interval"] = self.baseline_monitor_interval

        # Compute baseline std for every chunk.
        for i in range(N_BASELINE_MONITOR_INTERVAL):
            start_idx = i * self.chunk_size
            end_idx = start_idx + self.chunk_size

            results["baseline_monitor_std"][:, i] = np.std(
                records["data_theta"][:, start_idx:end_idx], axis=1
            )
            results["baseline_monitor_std_moving_average"][:, i] = np.std(
                records["data_theta_moving_average"][:, start_idx:end_idx], axis=1
            )
            results["baseline_monitor_std_convolved"][:, i] = np.std(
                records["data_theta_convolved"][:, start_idx:end_idx], axis=1
            )

        return results
