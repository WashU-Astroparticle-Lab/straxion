import strax
import numpy as np
import warnings
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
        default=5_000_000,
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
        default=50_000,
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
    """Find and characterize hits in processed phase angle data.

    This plugin identifies significant signal excursions (hits) in processed phase angle
    data and extracts their characteristics including amplitude, timing, and waveform
    data. The hit-finding algorithm uses adaptive thresholds based on signal statistics
    and applies various filtering criteria to distinguish real hits from noise.

    Processing workflow:
    1. Calculate adaptive hit thresholds based on signal statistics for each channel.
    2. Identify hit candidates using threshold crossing and minimum width criteria.
    3. Calculate hit characteristics (amplitude, timing, alignment point).
    4. Extract and align hit waveforms for further analysis.

    Provides:
    - hits: Characterized hits with waveform data and timing information.

    """

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
        self.hit_waveform_length = HIT_WINDOW_LENGTH_LEFT + HIT_WINDOW_LENGTH_RIGHT
        self.hit_window_length_left = HIT_WINDOW_LENGTH_LEFT
        self.hit_window_length_right = HIT_WINDOW_LENGTH_RIGHT

        self.hit_thresholds_sigma = np.array(self.config["hit_thresholds_sigma"])
        self.noisy_channel_signal_std_multipliers = np.array(
            self.config["noisy_channel_signal_std_multipliers"]
        )
        self.hit_ma_inspection_window_length = self.config[
            "hit_moving_average_inspection_window_length"
        ]
        self.hit_convolved_inspection_window_length = self.config[
            "hit_convolved_inspection_window_length"
        ]
        self.hit_extended_inspection_window_length = self.config[
            "hit_extended_inspection_window_length"
        ]

        self.record_length = self.config["record_length"]
        self.dt = 1 / self.config["fs"] * SECOND_TO_NANOSECOND

        self._check_hit_parameters()

    def _check_hit_parameters(self):
        """Check for potentially problematic parameters and issue warnings."""
        if self.hit_ma_inspection_window_length > self.hit_waveform_length:
            warnings.warn(
                "The hit-waveform recording window might be too short to save enough information: "
                f"hit_ma_inspection_window_length={self.hit_ma_inspection_window_length} "
                f"is larger than hit_waveform_length={self.hit_waveform_length}."
            )
        if self.hit_convolved_inspection_window_length > self.hit_waveform_length:
            warnings.warn(
                "The hit-waveform recording window might be too short to save enough information: "
                "hit_convolved_inspection_window_length="
                f"{self.hit_convolved_inspection_window_length} "
                f"is larger than hit_waveform_length={self.hit_waveform_length}."
            )
        if self.hit_extended_inspection_window_length > self.hit_waveform_length:
            warnings.warn(
                "The hit-waveform recording window might be too short to save enough information: "
                "hit_extended_inspection_window_length="
                f"{self.hit_extended_inspection_window_length} "
                f"is larger than hit_waveform_length={self.hit_waveform_length}."
            )

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
        """Process records to find and characterize hits.

        Args:
            records: Array of processed records containing signal data.

        Returns:
            np.ndarray: Array of hits with waveform data and characteristics.

        """
        results = []

        for r in records:
            hits = self._process_single_record(r)
            if hits is not None:
                results.append(hits)

        # Sort hits by time.
        results = np.concatenate(results)
        results = results[np.argsort(results["time"])]

        return results

    def _process_single_record(self, record):
        """Process a single record to find hits.

        Args:
            record: Single record containing signal data.

        Returns:
            np.ndarray or None: Array of hits found in the record, or None if no hits.

        """
        ch = int(record["channel"])
        signal = record["data_theta_convolved"]
        signal_ma = record["data_theta_moving_average"]
        signal_raw = record["data_theta"]
        min_pulse_width = self.config["min_pulse_widths"][ch]

        # Calculate hit threshold and find hit candidates
        hit_threshold = self.calculate_hit_threshold(
            signal, self.hit_thresholds_sigma[ch], self.noisy_channel_signal_std_multipliers[ch]
        )

        hit_candidates = self._find_hit_candidates(signal, hit_threshold, min_pulse_width)
        if len(hit_candidates) == 0:
            return None

        # Process each hit candidate
        hits = self._process_hit_candidates(
            hit_candidates, record, signal, signal_ma, signal_raw, hit_threshold, ch
        )

        return hits

    def _find_hit_candidates(self, signal, hit_threshold, min_pulse_width):
        """Find potential hit candidates based on threshold crossing.

        Args:
            signal: The convolved signal array.
            hit_threshold: Threshold value for hit detection.
            min_pulse_width: Minimum width required for a valid hit.

        Returns:
            tuple: (hit_start_indices, hit_widths) for valid hits.

        """
        below_threshold_indices = np.where(signal < hit_threshold)[0]
        if len(below_threshold_indices) == 0:
            return [], []

        # Find the start of the hits
        hits_width = np.diff(below_threshold_indices, prepend=1)

        # Filter by minimum pulse width
        valid_mask = hits_width >= min_pulse_width
        hit_end_indices = below_threshold_indices[valid_mask]
        hit_widths = hits_width[valid_mask]
        hit_start_indices = hit_end_indices - hit_widths

        return hit_start_indices, hit_widths

    def _process_hit_candidates(
        self, hit_candidates, record, signal, signal_ma, signal_raw, hit_threshold, channel
    ):
        """Process hit candidates to extract hit characteristics and waveforms.

        Args:
            hit_candidates: Tuple of (hit_start_indices, hit_widths).
            record: The original record.
            signal: The convolved signal array.
            signal_ma: The moving average signal array.
            signal_raw: The raw signal array.
            hit_threshold: The hit threshold value.
            channel: The channel number.

        Returns:
            np.ndarray: Array of processed hits.

        """
        hit_start_indices, hit_widths = hit_candidates

        hits = np.zeros(len(hit_start_indices), dtype=self.infer_dtype())
        hits["width"] = hit_widths

        for i, h_start_i in enumerate(hit_start_indices):
            self._process_single_hit(
                hits[i], h_start_i, record, signal, signal_ma, signal_raw, hit_threshold, channel
            )

        return hits

    def _process_single_hit(
        self, hit, hit_start_i, record, signal, signal_ma, signal_raw, hit_threshold, channel
    ):
        """Process a single hit to extract its characteristics and waveform.

        Args:
            hit: The hit array element to populate.
            hit_start_i: Start index of the hit.
            record: The original record.
            signal: The convolved signal array.
            signal_ma: The moving average signal array.
            signal_raw: The raw signal array.
            hit_threshold: The hit threshold value.
            channel: The channel number.

        """
        # Set basic hit properties
        hit["hit_threshold"] = hit_threshold
        hit["channel"] = channel
        hit["dt"] = self.dt

        # Calculate amplitude characteristics
        self._calculate_hit_amplitudes(hit, hit_start_i, signal)

        # Find alignment point and extract waveforms
        aligned_index = self._find_alignment_point(hit_start_i, signal, signal_ma)
        hit["aligned_at_records_i"] = aligned_index

        # Extract and align waveforms
        self._extract_hit_waveforms(hit, aligned_index, record, signal_raw, signal_ma, signal)

    def _calculate_hit_amplitudes(self, hit, hit_start_i, signal):
        """Calculate amplitude characteristics for a hit.

        Args:
            hit: The hit array element to populate.
            hit_start_i: Start index of the hit.
            signal: The convolved signal array.

        """
        # Find the maximum and minimum of the hit in the inspection windows
        hit_inspection_waveform = signal[
            hit_start_i : min(
                hit_start_i + self.hit_convolved_inspection_window_length,
                self.record_length,
            )
        ]
        hit_extended_inspection_waveform = signal[
            hit_start_i : min(
                hit_start_i + self.hit_extended_inspection_window_length,
                self.record_length,
            )
        ]

        hit["amplitude_max"] = np.max(hit_inspection_waveform)
        hit["amplitude_min"] = np.min(hit_inspection_waveform)
        hit["amplitude_max_ext"] = np.max(hit_extended_inspection_waveform)
        hit["amplitude_min_ext"] = np.min(hit_extended_inspection_waveform)

    def _find_alignment_point(self, hit_start_i, signal, signal_ma):
        """Find the alignment point for waveform extraction.

        Args:
            hit_start_i: Start index of the hit.
            signal: The convolved signal array.
            signal_ma: The moving average signal array.

        Returns:
            int: Index of the alignment point.

        """
        # Index of kernel-convolved signal in records
        hit_inspection_waveform = signal[
            hit_start_i : min(
                hit_start_i + self.hit_convolved_inspection_window_length,
                self.record_length,
            )
        ]
        hit_max_i = np.argmax(hit_inspection_waveform) + hit_start_i

        # Align waveforms at the maximum of the moving averaged signal
        # Search the maximum in the moving averaged signal within the inspection window
        search_start = max(hit_max_i - self.hit_ma_inspection_window_length, 0)
        search_end = min(hit_max_i + self.hit_ma_inspection_window_length, self.record_length)

        argmax_ma_i = np.argmax(signal_ma[search_start:search_end]) + search_start

        return argmax_ma_i

    def _extract_hit_waveforms(self, hit, aligned_index, record, signal_raw, signal_ma, signal):
        """Extract and align hit waveforms.

        Args:
            hit: The hit array element to populate.
            aligned_index: Index of the alignment point.
            record: The original record.
            signal_raw: The raw signal array.
            signal_ma: The moving average signal array.
            signal: The convolved signal array.

        """
        # Calculate valid sample ranges
        n_right_valid_samples = min(
            self.record_length - aligned_index, self.hit_window_length_right
        )
        n_left_valid_samples = min(aligned_index, self.hit_window_length_left)

        # Calculate waveform extraction boundaries
        hit_wf_start_i = max(aligned_index - self.hit_window_length_left, 0)
        hit_wf_end_i = min(aligned_index + self.hit_window_length_right, self.record_length)

        # Set timing information
        hit["time"] = record["time"] + hit_wf_start_i * self.dt
        hit["endtime"] = record["time"] + hit_wf_end_i * self.dt
        hit["length"] = hit_wf_end_i - hit_wf_start_i

        # Calculate target indices in the hit waveform arrays
        target_start = self.hit_window_length_left - n_left_valid_samples
        target_end = self.hit_window_length_left + n_right_valid_samples

        # Extract waveforms
        hit["data_theta"][target_start:target_end] = signal_raw[hit_wf_start_i:hit_wf_end_i]
        hit["data_theta_moving_average"][target_start:target_end] = signal_ma[
            hit_wf_start_i:hit_wf_end_i
        ]
        hit["data_theta_convolved"][target_start:target_end] = signal[hit_wf_start_i:hit_wf_end_i]
