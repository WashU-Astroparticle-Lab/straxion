import strax
import numpy as np
from straxion.utils import (
    DATA_DTYPE,
    INDEX_DTYPE,
    SECOND_TO_NANOSECOND,
    HIT_WINDOW_LENGTH_LEFT,
    HIT_WINDOW_LENGTH_RIGHT,
    base_waveform_dtype,
)

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        "record_length",
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
        track=True,
        type=int,
        help="Sampling frequency (assumed the same for all channels) in unit of Hz",
    ),
    strax.Option(
        "hit_thresholds_sigma",
        default=[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        track=True,
        type=list,
        help="Threshold for hit finding in units of sigma of standard deviation of the noise.",
    ),
    strax.Option(
        "noisy_channel_signal_std_multipliers",
        default=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        track=True,
        type=list,
        help=(
            "If the signal standard deviation above this threshold times of signal absolute "
            "mean, the signal is considered noisy and the hit threshold is increased."
        ),
    ),
    strax.Option(
        "min_pulse_widths",
        default=[20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
        track=True,
        type=list,
        help=(
            "Minimum pulse width in unit of samples. If the pulse width is below this "
            "threshold, the hit is not considered a new hit."
        ),
    ),
    strax.Option(
        "hit_convolved_inspection_window_length",
        default=60,
        track=True,
        type=int,
        help=(
            "Length of the convolved hit inspection window (to find maximum and minimum) "
            "in unit of samples."
        ),
    ),
    strax.Option(
        "hit_extended_inspection_window_length",
        default=100,
        track=True,
        type=int,
        help=(
            "Length of the extended convolved hit inspection window (to find maximum and minimum) "
            "in unit of samples."
        ),
    ),
    strax.Option(
        "hit_moving_average_inspection_window_length",
        default=40,
        track=True,
        type=int,
        help=(
            "Length of the moving averaged hit inspection window (to find maximum and minimum) "
            "in unit of samples."
        ),
    ),
)
class Hits(strax.Plugin):
    __version__ = "0.0.0"

    # Inherited from straxen. Not optimized outside XENONnT.
    rechunk_on_save = False
    compressor = "zstd"
    chunk_target_size_mb = 2000
    rechunk_on_load = True
    chunk_source_size_mb = 100

    depends_on = ["records"]
    provides = "hits"
    data_kind = "hits"
    save_when = strax.SaveWhen.ALWAYS

    def setup(self):
        self.hit_window_length_left = HIT_WINDOW_LENGTH_LEFT
        self.hit_window_length_right = HIT_WINDOW_LENGTH_RIGHT

        self.record_length = self.config["record_length"]
        self.dt = 1 / self.config["fs"] * SECOND_TO_NANOSECOND

        self.hit_thresholds_sigma = np.array(self.config["hit_thresholds_sigma"])
        self.noisy_channel_signal_std_multipliers = np.array(
            self.config["noisy_channel_signal_std_multipliers"]
        )
        self.hit_ma_inspection_window_length = self.config[
            "hit_moving_average_inspection_window_length"
        ]

    def infer_dtype(self):
        self.hit_waveform_length = HIT_WINDOW_LENGTH_LEFT + HIT_WINDOW_LENGTH_RIGHT

        dtype = base_waveform_dtype()
        dtype.append(
            (
                (
                    (
                        "Width of the hit waveform (length above the hit threshold) "
                        "in unit of samples.",
                    ),
                    "width",
                ),
                INDEX_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Hit waveform of phase angle (theta) only after baseline corrections, "
                        "aligned at the maximum of the moving averaged waveform."
                    ),
                    "data_theta",
                ),
                DATA_DTYPE,
                self.hit_waveform_length,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Hit waveform of phase angle (theta) further smoothed by moving average, "
                        "aligned at the maximum of the moving averaged waveform."
                    ),
                    "data_theta_moving_average",
                ),
                DATA_DTYPE,
                self.hit_waveform_length,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Hit waveform of phase angle (theta) further smoothed by pulse kernel, "
                        "aligned at the maximum of the moving averaged waveform."
                    ),
                    "data_theta_convolved",
                ),
                DATA_DTYPE,
                self.hit_waveform_length,
            )
        )
        dtype.append(
            (
                (
                    "Hit finding threshold determined by signal statistics in unit of rad.",
                    "hit_threshold",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                ("Index of alignment point (the maximum) in the records", "aligned_at_records_i"),
                INDEX_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Maximum amplitude of the hit waveform (within the hit window) "
                        "in unit of rad.",
                    ),
                    "amplitude_max",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Minimum amplitude of the hit waveform (within the hit window) "
                        "in unit of rad.",
                    ),
                    "amplitude_min",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Maximum amplitude of the hit waveform (within the extended hit window) "
                        "in unit of rad.",
                    ),
                    "amplitude_max_ext",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Minimum amplitude of the hit waveform (within the extended hit window) "
                        "in unit of rad.",
                    ),
                    "amplitude_min_ext",
                ),
                DATA_DTYPE,
            )
        )

        return dtype

    @staticmethod
    def calculate_hit_threshold(signal, hit_threshold_sigma, noisy_channel_signal_std_multiplier):
        """Calculate hit threshold based on signal statistics.

        Args:
            signal (np.ndarray): The signal array to analyze.
            hit_threshold_sigma (float): Threshold multiplier in units of sigma.
            noisy_channel_signal_std_multiplier (float): Multiplier to detect noisy channels.

        Returns:
            float: The calculated hit threshold.

        """
        signal_mean = np.mean(signal)
        signal_abs_mean = np.mean(np.abs(signal))
        signal_std = np.std(signal)

        # The naive hit threshold is a multiple of the standard deviation of the signal.
        hit_threshold = signal_mean + hit_threshold_sigma * signal_std

        # If the signal is noisy, the baseline might be too high.
        if signal_std > noisy_channel_signal_std_multiplier * signal_abs_mean:
            # We will use the quiet part of the signal to redefine a lowered hit threshold.
            quiet_mask = signal < hit_threshold
            hit_threshold = signal_mean + hit_threshold_sigma * np.std(signal[quiet_mask])

        return hit_threshold

    def compute(self, records):
        results = []

        for r in records:
            ch = int(r["channel"])
            signal = r["data_theta_convolved"]
            signal_ma = r["data_theta_moving_average"]
            signal_raw = r["data_theta"]
            min_pulse_width = self.config["min_pulse_widths"][ch]

            hit_threshold = self.calculate_hit_threshold(
                signal, self.hit_thresholds_sigma[ch], self.noisy_channel_signal_std_multipliers[ch]
            )

            below_threshold_indices = np.where(signal < hit_threshold)[0]
            if len(below_threshold_indices) == 0:
                continue
            # Find the start of the hits.
            _hits_width = np.diff(below_threshold_indices, prepend=0)
            # Minimum continuous length of waveform above the hit threshold is
            # required to define a hit.
            hit_start_indicies = below_threshold_indices[_hits_width >= min_pulse_width]

            hits = np.zeros(len(hit_start_indicies), dtype=self.infer_dtype())
            hits_width = _hits_width[_hits_width >= min_pulse_width]
            hits["width"] = hits_width

            # Find the maximum and minimum of the hits.
            for i, h_start_i in enumerate(hit_start_indicies):
                hits[i]["hit_threshold"] = hit_threshold
                hits[i]["channel"] = ch
                hits[i]["length"] = self.hit_waveform_length
                hits[i]["dt"] = self.dt

                # Find the maximum and minimum of the hit in the inspection windows.
                hit_inspection_waveform = signal[
                    h_start_i : min(
                        h_start_i + self.config["hit_convolved_inspection_window_length"],
                        self.record_length,
                    )
                ]
                hit_extended_inspection_waveform = signal[
                    h_start_i : min(
                        h_start_i + self.config["hit_extended_inspection_window_length"],
                        self.record_length,
                    )
                ]
                hits[i]["amplitude_max"] = np.max(hit_inspection_waveform)
                hits[i]["amplitude_min"] = np.min(hit_inspection_waveform)
                hits[i]["amplitude_max_ext"] = np.max(hit_extended_inspection_waveform)
                hits[i]["amplitude_min_ext"] = np.min(hit_extended_inspection_waveform)

                # Index of kernel-convolved signal in records.
                hit_max_i = np.argmax(hit_inspection_waveform) + h_start_i

                # Align waveforms of the hits at the maximum of the moving averaged signal.
                # Search the maximum in the moving averaged signal within the inspection window.
                argmax_ma_i = (
                    np.argmax(
                        signal_ma[
                            max(hit_max_i - self.hit_ma_inspection_window_length, 0) : min(
                                hit_max_i + self.hit_ma_inspection_window_length, self.record_length
                            )
                        ]
                    )
                    + hit_max_i
                )

                # For a physical hit, the left window is expected to be noise dominated.
                # While the right window is expected to be signal dominated.
                n_right_valid_samples = min(
                    self.record_length - argmax_ma_i, self.hit_window_length_right
                )
                n_left_valid_samples = min(argmax_ma_i, self.hit_window_length_left)

                hit_wf_start_i = max(argmax_ma_i - self.hit_window_length_left, 0)
                hit_wf_end_i = min(argmax_ma_i + self.hit_window_length_right, self.record_length)
                hits[i]["time"] = r["time"] + hit_wf_start_i * self.dt
                hits[i]["endtime"] = r["time"] + hit_wf_end_i * self.dt
                hits[i]["length"] = hit_wf_end_i - hit_wf_start_i
                hits[i]["aligned_at_records_i"] = argmax_ma_i
                hits[i]["data_theta"][
                    self.hit_window_length_left
                    - n_left_valid_samples : self.hit_window_length_left
                    + n_right_valid_samples
                ] = signal_raw[hit_wf_start_i:hit_wf_end_i]
                hits[i]["data_theta_moving_average"][
                    self.hit_window_length_left
                    - n_left_valid_samples : self.hit_window_length_left
                    + n_right_valid_samples
                ] = signal_ma[hit_wf_start_i:hit_wf_end_i]
                hits[i]["data_theta_convolved"][
                    self.hit_window_length_left
                    - n_left_valid_samples : self.hit_window_length_left
                    + n_right_valid_samples
                ] = signal[hit_wf_start_i:hit_wf_end_i]

            results.append(hits)

        # Sort hits by time.
        results = np.concatenate(results)
        results = results[np.argsort(results["time"])]

        return results
