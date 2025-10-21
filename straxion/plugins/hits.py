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
        "hit_threshold_dx",
        default=None,
        track=True,
        type=float,
        help=(
            "Threshold for hit finding in units of dx=df/f0. "
            "If None, the hit threshold will be calculated based on the signal statistics."
        ),
    ),
    strax.Option(
        "hit_thresholds_sigma",
        default=[3.0 for _ in range(41)],
        track=True,
        type=list,
        help=(
            "Threshold for hit finding in units of sigma of standard deviation of the noise. "
            "If None, the hit threshold will be calculated based on the signal statistics."
        ),
    ),
    strax.Option(
        "hit_min_width",
        default=0.25e-3,
        track=True,
        type=float,
        help="Minimum width for hit finding in units of seconds.",
    ),
    strax.Option(
        "fs",
        default=38_000,
        track=True,
        type=int,
        help="Sampling frequency (assumed the same for all channels) in unit of Hz",
    ),
)
class DxHits(strax.Plugin):
    """Find and characterize hits in dx=df/f0 data.

    The hit-finding algorithm is based on the kernel convolved signal.
    """

    __version__ = "0.0.2"

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

    def infer_dtype(self):
        dtype = base_waveform_dtype()
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
                        "Hit waveform of dx=df/f0 only after baseline corrections, "
                        "aligned at the maximum of the dx=df/f0 waveform."
                    ),
                    "data_dx",
                ),
                DATA_DTYPE,
                self.hit_waveform_length,
            )
        )
        dtype.append(
            (
                (
                    ("Maximum amplitude of the dx hit waveform",),
                    "amplitude",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    "Maximum amplitude of the dx hit waveform further smoothed by moving average",
                    "amplitude_moving_average",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    "Maximum amplitude of the dx hit waveform further smoothed by pulse kernel",
                    "amplitude_convolved",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Record index of the maximum amplitude of the dx hit waveform "
                        "further smoothed by pulse kernel."
                    ),
                    "amplitude_convolved_max_record_i",
                ),
                INDEX_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Record index of the maximum amplitude of the dx hit waveform "
                        "further smoothed by moving average."
                    ),
                    "amplitude_moving_average_max_record_i",
                ),
                INDEX_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    ("Record index of the maximum amplitude of " "the dx hit waveform."),
                    "amplitude_max_record_i",
                ),
                INDEX_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Hit waveform of dx=df/f0 further smoothed by moving average, "
                        "aligned at the maximum of the dx=df/f0 waveform."
                    ),
                    "data_dx_moving_average",
                ),
                DATA_DTYPE,
                self.hit_waveform_length,
            )
        )
        dtype.append(
            (
                (
                    "Hit waveform of dx=df/f0 further smoothed by pulse kernel, "
                    "aligned at the maximum of the dx=df/f0 waveform.",
                    "data_dx_convolved",
                ),
                DATA_DTYPE,
                self.hit_waveform_length,
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

        return dtype

    def setup(self):
        self.hit_waveform_length = HIT_WINDOW_LENGTH_LEFT + HIT_WINDOW_LENGTH_RIGHT
        self.hit_window_length_left = HIT_WINDOW_LENGTH_LEFT
        self.hit_window_length_right = HIT_WINDOW_LENGTH_RIGHT

        self.hit_threshold_dx = self.config["hit_threshold_dx"]
        self.hit_thresholds_sigma = self.config["hit_thresholds_sigma"]
        self.hit_min_width_samples = self.config["hit_min_width"] * self.fs
        self.fs = self.config["fs"]
        self.dt_exact = 1 / self.fs * SECOND_TO_NANOSECOND

    @staticmethod
    def calculate_hit_threshold(signal, hit_threshold_sigma):
        """Calculate hit threshold based on signal statistics.

        Args:
            signal (np.ndarray): The signal array to analyze.
            hit_threshold_sigma (float): Threshold multiplier in units of sigma.

        Returns:
            float: The calculated hit threshold.

        """
        signal_mean = np.mean(signal, axis=1)
        signal_std = np.std(signal, axis=1)

        # The naive hit threshold is a multiple of the standard deviation of the signal.
        hit_threshold = signal_mean + hit_threshold_sigma * signal_std

        return hit_threshold

    def determine_hit_threshold(self, records):
        """Determine the hit threshold based on the provided configuration.
        You can either provide hit_threshold_dx or hit_thresholds_sigma.
        You cannot provide both.

        Returns:
            np.ndarray: Hit threshold for each record.
        """
        if self.hit_threshold_dx is None and self.hit_thresholds_sigma is not None:
            # If hit_thresholds_sigma are single values,
            # we need to convert them to arrays.
            if isinstance(self.hit_thresholds_sigma, float):
                hit_thresholds_sigma = np.full(len(records["channel"]), self.hit_thresholds_sigma)
            else:
                hit_thresholds_sigma = np.array(self.hit_thresholds_sigma)
            # Calculate hit threshold and find hit candidates
            hit_threshold_dx = self.calculate_hit_threshold(
                records["data_dx_convolved"],
                hit_thresholds_sigma[records["channel"]],
            )
        elif self.hit_threshold_dx is not None and self.hit_thresholds_sigma is None:
            # If hit_threshold_dx is a single value, we need to convert it to an array.
            if isinstance(self.hit_threshold_dx, float):
                hit_threshold_dx = np.full(len(records["channel"]), self.hit_threshold_dx)
            else:
                hit_threshold_dx = np.array(self.hit_threshold_dx)
        else:
            raise ValueError(
                "Either hit_threshold_dx or hit_thresholds_sigma "
                "must be provided. You cannot provide both."
            )

        return hit_threshold_dx

    @staticmethod
    def find_hit_candidates(signal, hit_threshold, min_pulse_width):
        """Finds potential hit candidates, correctly handling edge cases.

        Args:
            signal: The signal array.
            hit_threshold: Threshold value for hit detection.
            min_pulse_width: Minimum width required for a valid hit in samples.

        Returns:
            tuple: (hit_start_indices, hit_widths) for valid hits.
        """
        # 1. Create a boolean array where True means the signal is at or above the threshold
        above_threshold = signal >= hit_threshold

        # 2. Pad the array with False at both ends. This is the key to finding
        #    hits that start at index 0 or end at the last index.
        padded_array = np.concatenate(([False], above_threshold, [False]))

        # 3. Find the changes from False to True (starts) and True to False (ends)
        #    A change from 0 to 1 is a start (diff = 1).
        #    A change from 1 to 0 is an end (diff = -1).
        diffs = np.diff(padded_array.astype(int))

        hit_start_indices = np.where(diffs == 1)[0]
        hit_end_indices = np.where(diffs == -1)[0]

        # If no hits are found, return empty lists
        if len(hit_start_indices) == 0:
            return np.array([]), np.array([])

        # 4. Calculate the width of each potential hit
        hit_widths = hit_end_indices - hit_start_indices

        # 5. Filter the hits by the minimum pulse width
        valid_mask = hit_widths >= min_pulse_width

        return hit_start_indices[valid_mask], hit_widths[valid_mask]

    def compute(self, records):
        """Process records to find and characterize hits.

        Args:
            records: Array of records containing signal data.

        Returns:
            np.ndarray: Array of hits with waveform data and characteristics.
        """
        hit_threshold_dx = self.determine_hit_threshold(records)

        results = []

        for record in records:
            hits = self._process_single_record(record, hit_threshold_dx)
            if hits is not None and len(hits) > 0:
                results.append(hits)

        if not results:
            return np.zeros(0, dtype=self.infer_dtype())

        results = np.concatenate(results)

        # Order hits first by time
        results = results[np.argsort(results["time"])]

        # Truncate hits endtime to record endtime
        results["endtime"] = np.minimum(results["endtime"], records["endtime"][0])

        return results

    def _process_single_record(self, record, hit_threshold_dx):
        """Process a single record to find hits.

        Args:
            record: Single record containing signal data.
            hit_threshold_dx: Hit threshold array for each channel.

        Returns:
            np.ndarray or None: Array of hits found in the record, or None if no hits.
        """
        ch = int(record["channel"])
        hit_start_i, hit_widths = self.find_hit_candidates(
            record["data_dx_convolved"], hit_threshold_dx[ch], self.hit_min_width_samples
        )

        if len(hit_start_i) == 0:
            return None

        hits = np.zeros(len(hit_start_i), dtype=self.infer_dtype())
        hits["width"] = hit_widths
        hits["channel"] = record["channel"]
        hits["dt"] = self.dt_exact  # Will be converted to int when saving
        hits["hit_threshold"] = hit_threshold_dx[hits["channel"]]

        for i, start_i in enumerate(hit_start_i):
            self._process_hit(
                hits[i],
                record["data_dx_convolved"],
                record["data_dx_moving_average"],
                record["data_dx"],
                start_i,
                hit_widths[i],
                record["time"],
                previous_hit_end_i=hit_start_i[i - 1] + hit_widths[i - 1] if i > 0 else None,
                next_hit_start_i=hit_start_i[i + 1] if i < len(hit_start_i) - 1 else None,
            )

        return hits

    def _process_hit(
        self,
        hit,
        signal_convolved,
        signal_ma,
        signal_raw,
        start_i,
        width,
        start_time,
        previous_hit_end_i=None,
        next_hit_start_i=None,
    ):
        """Process a single hit candidate.

        Args:
            hit: Hit array to populate
            signal_convolved: Convolved signal array
            signal_ma: Moving average signal array
            signal_raw: Raw signal array
            start_i: Start index of the hit
            width: Width of the hit in samples
            start_time: Start time of the record
            previous_hit_end_i: End index of the previous hit
            next_hit_start_i: Start index of the next hit
        """
        # Extract hit waveform
        above_threshold = signal_convolved[start_i : start_i + width]

        # Find maximum amplitude and its position
        max_i = np.argmax(above_threshold)

        # Align waveform around maximum
        aligned_i = start_i + max_i

        # Handle left boundary, considering previous hit if it exists
        left_boundary = aligned_i - self.hit_window_length_left
        if previous_hit_end_i is not None:
            left_i = max(0, left_boundary, previous_hit_end_i)
        else:
            left_i = max(0, left_boundary)

        # Handle right boundary, considering next hit if it exists
        right_boundary = aligned_i + self.hit_window_length_right
        if next_hit_start_i is not None:
            right_i = min(len(signal_convolved), right_boundary, next_hit_start_i)
        else:
            right_i = min(len(signal_convolved), right_boundary)

        # Calculate valid sample ranges
        n_right_valid_samples = min(right_i - aligned_i, self.hit_window_length_right)
        n_left_valid_samples = min(aligned_i - left_i, self.hit_window_length_left)

        # Calculate target indices in the hit waveform array
        target_start = self.hit_window_length_left - n_left_valid_samples
        target_end = self.hit_window_length_left + n_right_valid_samples

        # Extract waveform
        hit["data_dx_convolved"][target_start:target_end] = signal_convolved[left_i:right_i]
        hit["data_dx_moving_average"][target_start:target_end] = signal_ma[left_i:right_i]
        hit["data_dx"][target_start:target_end] = signal_raw[left_i:right_i]
        hit["amplitude_convolved"] = np.max(signal_convolved[left_i:right_i])
        hit["amplitude_moving_average"] = np.max(signal_ma[left_i:right_i])
        hit["amplitude"] = np.max(signal_raw[left_i:right_i])
        hit["length"] = right_i - left_i
        hit["amplitude_convolved_max_record_i"] = np.int32(
            np.argmax(signal_convolved[left_i:right_i]) + left_i
        )
        hit["amplitude_moving_average_max_record_i"] = np.int32(
            np.argmax(signal_ma[left_i:right_i]) + left_i
        )
        hit["amplitude_max_record_i"] = np.int32(np.argmax(signal_raw[left_i:right_i]) + left_i)

        # Calculate time and endtime
        hit["time"] = np.int64(start_time + np.int64(left_i * self.dt_exact))
        hit["endtime"] = np.int64(start_time + np.int64(right_i * self.dt_exact))


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
    strax.Option(
        "hit_thresholds_sigma",
        default=[3.0 for _ in range(41)],
        track=True,
        type=list,
        help="Threshold for hit finding in units of sigma of standard deviation of the noise.",
    ),
    strax.Option(
        "noisy_channel_signal_std_multipliers",
        default=[2.0 for _ in range(41)],
        track=True,
        type=list,
        help=(
            "If the signal standard deviation above this threshold times of signal absolute "
            "mean, the signal is considered noisy and the hit threshold is increased."
        ),
    ),
    strax.Option(
        "min_pulse_widths",
        default=[20 for _ in range(41)],
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
        self.dt_exact = 1 / self.config["fs"] * SECOND_TO_NANOSECOND

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
                        "Maximum amplitude of the convolved hit waveform (within the "
                        "hit window) in unit of rad.",
                    ),
                    "amplitude_convolved_max",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Minimum amplitude of the convolved hit waveform (within the "
                        "hit window) in unit of rad.",
                    ),
                    "amplitude_convolved_min",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Maximum amplitude of the convolved hit waveform (within the "
                        "extended hit window) in unit of rad.",
                    ),
                    "amplitude_convolved_max_ext",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Minimum amplitude of the convolved hit waveform (within the "
                        "extended hit window) in unit of rad.",
                    ),
                    "amplitude_convolved_min_ext",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Maximum amplitude of the moving averaged hit waveform (within the "
                        "hit window) in unit of rad.",
                    ),
                    "amplitude_ma_max",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Minimum amplitude of the moving averaged hit waveform (within the "
                        "hit window) in unit of rad.",
                    ),
                    "amplitude_ma_min",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Maximum amplitude of the moving averaged hit waveform (within the "
                        "extended hit window) in unit of rad.",
                    ),
                    "amplitude_ma_max_ext",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    (
                        "Minimum amplitude of the moving averaged hit waveform (within the "
                        "extended hit window) in unit of rad.",
                    ),
                    "amplitude_ma_min_ext",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    "Mean of the convolved signal in the record.",
                    "record_convolved_mean",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    "Standard deviation of the convolved signal in the record.",
                    "record_convolved_std",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    "Mean of the moving averaged signal in the record.",
                    "record_ma_mean",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    "Standard deviation of the moving averaged signal in the record.",
                    "record_ma_std",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    "Mean of the raw signal in the record.",
                    "record_raw_mean",
                ),
                DATA_DTYPE,
            )
        )
        dtype.append(
            (
                (
                    "Standard deviation of the raw signal in the record.",
                    "record_raw_std",
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
            if hits is not None and len(hits) > 0:
                results.append(hits)

        # Sort hits by time.
        if not results:
            return np.zeros(0, dtype=self.infer_dtype())

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

        hits["record_raw_mean"] = np.mean(signal_raw)
        hits["record_raw_std"] = np.std(signal_raw)
        hits["record_convolved_std"] = np.std(signal)
        hits["record_convolved_mean"] = np.mean(signal)
        hits["record_ma_mean"] = np.mean(signal_ma)
        hits["record_ma_std"] = np.std(signal_ma)

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
        hit["dt"] = self.dt_exact  # Will be converted to int when saving

        # Calculate amplitude characteristics
        self._calculate_hit_amplitudes(hit, hit_start_i, signal, signal_ma)

        # Find alignment point and extract waveforms
        aligned_index = self._find_alignment_point(hit_start_i, signal, signal_ma)
        hit["aligned_at_records_i"] = aligned_index

        # Extract and align waveforms
        self._extract_hit_waveforms(hit, aligned_index, record, signal_raw, signal_ma, signal)

    def _calculate_hit_amplitudes(self, hit, hit_start_i, signal, signal_ma):
        """Calculate amplitude characteristics for a hit.

        Args:
            hit: The hit array element to populate.
            hit_start_i: Start index of the hit.
            signal: The convolved signal array.
            signal_ma: The moving average signal array.

        """
        hit_end_i = min(
            hit_start_i + self.hit_convolved_inspection_window_length,
            self.record_length,
        )
        hit_extended_end_i = min(
            hit_start_i + self.hit_extended_inspection_window_length,
            self.record_length,
        )
        # Find the maximum and minimum of the hit in the inspection windows
        hit_convolved_inspection_waveform = signal[hit_start_i:hit_end_i]
        hit_convolved_extended_inspection_waveform = signal[hit_start_i:hit_extended_end_i]
        hit_ma_inspection_waveform = signal_ma[hit_start_i:hit_end_i]
        hit_ma_extended_inspection_waveform = signal_ma[hit_start_i:hit_extended_end_i]

        hit["amplitude_convolved_max"] = np.max(hit_convolved_inspection_waveform)
        hit["amplitude_convolved_min"] = np.min(hit_convolved_inspection_waveform)
        hit["amplitude_convolved_max_ext"] = np.max(hit_convolved_extended_inspection_waveform)
        hit["amplitude_convolved_min_ext"] = np.min(hit_convolved_extended_inspection_waveform)
        hit["amplitude_ma_max"] = np.max(hit_ma_inspection_waveform)
        hit["amplitude_ma_min"] = np.min(hit_ma_inspection_waveform)
        hit["amplitude_ma_max_ext"] = np.max(hit_ma_extended_inspection_waveform)
        hit["amplitude_ma_min_ext"] = np.min(hit_ma_extended_inspection_waveform)

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
        hit["time"] = record["time"] + np.int64(hit_wf_start_i * self.dt_exact)
        hit["endtime"] = min(
            record["time"] + np.int64(hit_wf_end_i * self.dt_exact), record["endtime"]
        )
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
