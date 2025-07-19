import strax
import numpy as np
from straxion.utils import (
    DATA_DTYPE,
    SECOND_TO_NANOSECOND,
    base_waveform_dtype,
)
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
import os
import re
import glob

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        "iq_finescan_dir",
        track=False,  # FIXME: Ideally it should be tracked, by correction rather than config.
        type=str,
        help=("Direcotry to fine frequency scan (IQ loop) of resonator (txt, csv or similar)."),
    ),
    strax.Option(
        "fs",
        track=True,
        type=int,
        help="Sampling frequency (assumed the same for all channels) in unit of Hz",
    ),
    strax.Option(
        "pulse_kernel_start_time",
        default=100_000,
        track=True,
        type=int,
        help="Relative start time of the exponential decay in pulse kernel (t0), in unit of ns.",
    ),
    strax.Option(
        "pulse_kernel_decay_time",
        default=300_000,
        track=True,
        type=int,
        help="Decay time of the exponential falling in pulse kernel (tau), in unit of ns.",
    ),
    strax.Option(
        "pulse_kernel_gaussian_smearing_width",
        default=700_000,  # The original Matlab code says 7 samples (with fs = 1E5Hz).
        track=True,
        type=int,
        help=(
            "Gaussian smearing width of the exponentially-modified-gaussian kernel "
            "(sigma), in unit of ns."
        ),
    ),
    strax.Option(
        "moving_average_width",
        default=500_000,  # The original Matlab code says 5 samples (with fs = 1E5Hz).
        track=True,
        type=int,
        help="Moving average width for smoothed reference waveform, in unit of ns.",
    ),
    strax.Option(
        "pulse_kernel_truncation_factor",
        default=5,
        track=True,
        type=float,
        help=(
            "Factor for truncating the pulse kernel to improve performance. "
            "The kernel is truncated after truncation_factor * tau samples, "
            "where the exponential decay becomes negligible (< 0.7% of peak value)."
        ),
    ),
)
class PulseProcessing(strax.Plugin):
    """Process raw IQ data to extract phase information.

    This plugin converts raw I/Q timestreams to phase angles relative to the IQ loop center and
    applies various signal processing techniques including baseline correction, smoothing, and
    pulse kernel convolution. The functionality is adapted from Chris Albert's Matlab codes:
    https://caltechobscosgroup.slack.com/archives/C07SZDKRNF9/p1752010145654029.

    Important assumptions:
    - Fine scan files are available for each channel to calibrate the IQ loop center.
    - The phase angle at the lowest frequency is defined as the minimum angle.
    - All channels use the same pulse kernel parameters for consistency.

    Processing workflow:
    1. Load fine scan data to determine IQ loop center for each channel
    2. Convert I/Q timestreams to phase angles relative to the loop center
    3. Apply baseline correction and signal normalization
    4. Smooth phase data using moving average filter
    5. Convolve with exponentially-modified Gaussian pulse kernel

    Provides:
    - records: Processed phase angle data with baseline correction and signal processing applied.

    """

    __version__ = "0.0.0"
    rechunk_on_save = False
    compressor = "zstd"  # Inherited from straxen. Not optimized outside XENONnT.

    depends_on = "raw_records"
    provides = "records"
    data_kind = provides
    save_when = strax.SaveWhen.ALWAYS

    def infer_dtype(self):
        """Data type for a waveform record."""
        # Use record_length from setup if available, otherwise compute it.
        if hasattr(self, "record_length"):
            record_length = self.record_length
        else:
            # Get record_length from the plugin making raw_records.
            raw_records_dtype = self.deps["raw_records"].dtype_for("raw_records")
            record_length = len(np.zeros(1, raw_records_dtype)[0]["data_i"])

        dtype = base_waveform_dtype(record_length)
        dtype.append(
            (
                (
                    (
                        "Waveform data of phase angle (theta), "
                        "which has been convoled by a pulse kernel."
                    ),
                    "data_theta",
                ),
                DATA_DTYPE,
                record_length,
            )
        )
        dtype.append(
            (
                (
                    "Waveform data of phase angle (theta) further smoothed by moving average",
                    "data_theta_moving_average",
                ),
                DATA_DTYPE,
                record_length,
            )
        )
        dtype.append(
            (
                (
                    "Waveform data of phase angle (theta) further smoothed by pulse kernel",
                    "data_theta_convolved",
                ),
                DATA_DTYPE,
                record_length,
            )
        )
        dtype.append(
            (("Baseline of phase measurement approximated by a constant", "baseline"), DATA_DTYPE)
        )
        dtype.append((("Standard deviation of the baseline", "baseline_std"), DATA_DTYPE))
        return dtype

    def setup(self):
        # Get record_length from the plugin making raw_records.
        raw_records_dtype = self.deps["raw_records"].dtype_for("raw_records")
        self.record_length = len(np.zeros(1, raw_records_dtype)[0]["data_i"])
        self.dt = 1 / self.config["fs"] * SECOND_TO_NANOSECOND

        self.finescan = self.load_finescan_files(self.config["iq_finescan_dir"])
        self.kernel = self.pulse_kernel_emg(
            self.record_length,
            self.config["fs"],
            self.config["pulse_kernel_start_time"],
            self.config["pulse_kernel_decay_time"],
            self.config["pulse_kernel_gaussian_smearing_width"],
            self.config["pulse_kernel_truncation_factor"],
        )

        # Pre-compute moving average kernel
        moving_average_kernel_width = int(self.config["moving_average_width"] / self.dt)
        self.moving_average_kernel = (
            np.ones(moving_average_kernel_width) / moving_average_kernel_width
        )

        # Pre-compute circle fits for each channel to avoid repeated computation.
        self.channel_centers = {}
        for channel in self.finescan.keys():
            finescan = self.finescan[channel]
            finescan_i = finescan[:, 1]
            finescan_q = finescan[:, 2]
            i_center, q_center, _, _ = self.circfit(finescan_i, finescan_q)
            theta_f_min = np.arctan2(finescan_q[0] - q_center, finescan_i[0] - i_center)
            self.channel_centers[int(channel)] = (i_center, q_center, theta_f_min)

    @staticmethod
    def load_finescan_files(directory):
        """Load fine scan files for all channels from a directory.

        Workflow:
        - Expects files named as '*-ch<CHANNEL>.txt' or '*-ch<CHANNEL>.csv', e.g.:
            finescan-kid-2025042808-ch0.txt
            finescan-kid-2025042808-ch1.txt
            finescan-kid-2025042808-ch2.txt
        - Each file should be a text or CSV file with three columns: index, data_i, data_q.
        - The channel number is extracted from the filename.
        - Files are loaded as numpy arrays and returned in a dict indexed by channel number.

        Args:
            directory (str): Path to the directory containing fine scan files.

        Returns:
            dict: Mapping from channel number (int) to numpy.ndarray of fine scan data.

        Raises:
            FileNotFoundError: If the directory or expected files are not found.
            ValueError: If a file does not have at least three columns.
            RuntimeError: If no valid fine scan files are found or a file cannot be loaded.

        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Fine scan directory not found: {directory}")

        pattern_txt = os.path.join(directory, "*-ch*.txt")
        pattern_csv = os.path.join(directory, "*-ch*.csv")
        files = glob.glob(pattern_txt) + glob.glob(pattern_csv)
        if not files:
            raise FileNotFoundError(
                f"No fine scan files found in {directory}. "
                f"Expected files like '*-ch<CHANNEL>.txt' or '*-ch<CHANNEL>.csv'."
            )
        channel_re = re.compile(r"-ch(\d+)\.(txt|csv)$")
        finescan = {}
        for f in files:
            m = channel_re.search(f)
            if not m:
                continue
            channel = int(m.group(1))
            try:
                arr = np.loadtxt(f, delimiter=None)  # Use autodetect delimiter.
            except Exception as e:
                raise RuntimeError(f"Failed to load fine scan file {f}: {e}")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] < 3:
                raise ValueError(
                    f"File {f} does not have at least 3 columns (index, data_i, data_q)"
                )
            finescan[channel] = arr
        if not finescan:
            raise RuntimeError(f"No valid fine scan files found in {directory}.")
        return finescan

    @staticmethod
    def pulse_kernel_emg(ns, fs, t0, tau, sigma, truncation_factor=5):
        """Generate a pulse train with exponential decay and Gaussian smoothing.

        Translated from Chris Albert's Matlab codes:
        https://caltechobscosgroup.slack.com/archives/C07SZDKRNF9/p1752010145654029.

        Args:
            ns (int): Number of samples.
            fs (int): Sampling frequency in unit of Hz.
            t0 (int): Start time of the pulse in unit of ns.
            tau (int): Decay time constant in unit of ns.
            sigma (int): Smearing width constant in unit of ns in unit of ns.
            truncation_factor (float): Factor for truncating the kernel in unit of tau.
                After truncation_factor * tau, the exponential is less than
                exp(-truncation_factor) of its peak value.

        Returns:
            pulse_kernal (np.ndarray): Smoothed pulse train

        """
        dt = int(1 / fs * SECOND_TO_NANOSECOND)

        # Calculate significant length upfront to avoid unnecessary computation
        significant_length = min(ns, int(truncation_factor * tau / dt))

        # Only create time array for needed samples
        t = np.arange(significant_length) * dt

        # Create exponential decay pulse only for significant portion
        mask = t >= t0
        exponential = np.zeros(significant_length)
        exponential[mask] = np.exp(-(t[mask] - t0) / tau)

        # Convert sigma to samples
        sigma_sample = int(sigma / dt)

        # Apply Gaussian smoothing
        pulse_kernal = gaussian_filter1d(exponential, sigma=sigma_sample)

        # No need for truncation since we already computed only the significant portion
        # But we still need to normalize
        kernel_sum = np.sum(pulse_kernal)
        if kernel_sum > 0:  # Avoid division by zero
            pulse_kernal /= kernel_sum

        # Normalize again to make sure the integral is 1.
        kernel_sum = np.sum(pulse_kernal)
        if kernel_sum > 0:  # Avoid division by zero
            pulse_kernal /= kernel_sum

        return pulse_kernal

    @staticmethod
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

    def convert_iq_to_theta(self, data_i, data_q, channel):
        """Convert data_i/data_q timestreams to theta (phase angle) relative to the IQ loop center.

        This should be used for relative theta shifts only, because the theta=0 point is not
        chosen to be aligned to any particular feature of the loop in this script.
        The angle at the lowest frequency has been defined as the lowest.

        Args:
            data_i (np.ndarray): In-phase timestream.
            data_q (np.ndarray): Quadrature timestream.
            channel (int): Channel number.

        Returns:
            np.ndarray: Timestream in angle with respect to IQ loop center, in unit of rad.

        """
        # Use pre-computed circle centers from setup
        i_center, q_center, theta_f_min = self.channel_centers[int(channel)]

        # Compute theta timestream.
        thetas = np.arctan2(data_q - q_center, data_i - i_center)

        # Correct for angle wrapping: Force the angle at lowest frequency to be the minimum.
        mask_jump = thetas < theta_f_min
        thetas[mask_jump] = thetas[mask_jump] + 2 * np.pi

        return thetas

    def compute(self, raw_records):
        """Process raw IQ records to extract phase information and apply signal processing.

        This method processes raw records containing I/Q data by:
        1. Converting I/Q timestreams to phase angles (theta) relative to the IQ loop center
        2. Applying baseline correction and signal normalization
        3. Smoothing the phase data using moving average
        4. Convolving with an exponentially-modified Gaussian pulse kernel

        Performance optimizations:
        - Circle fits are pre-computed in setup() for each channel
        - Moving average kernel is pre-computed in setup()
        - Pulse kernel is pre-computed in setup()

        Args:
            raw_records (np.ndarray): Array of raw records containing I/Q data.

        Returns:
            np.ndarray: Processed records with additional fields:
                - data_theta: Phase angle data with baseline correction
                - data_theta_moving_average: Phase data smoothed by moving average
                - data_theta_convolved: Phase data convolved with pulse kernel
                - baseline: Mean baseline value of the phase measurement
                - baseline_std: Standard deviation of the baseline

        """
        results = np.zeros(len(raw_records), dtype=self.infer_dtype())

        for i, rr in enumerate(raw_records):
            r = results[i]
            r["time"] = rr["time"]
            r["endtime"] = rr["endtime"]
            r["length"] = rr["length"]
            r["dt"] = rr["dt"]
            r["channel"] = rr["channel"]

            # Get phase from IQ timestream.
            _thetas = self.convert_iq_to_theta(rr["data_i"], rr["data_q"], int(rr["channel"]))
            # Flipping thetas to make largest hits positive.
            if np.abs(np.mean(_thetas) - np.min(_thetas)) > np.abs(
                np.mean(_thetas) - np.max(_thetas)
            ):
                _thetas = -_thetas
            # Baselining by mean of the time series of processed thetas.
            r["baseline"] = np.mean(_thetas)
            r["baseline_std"] = np.std(_thetas)
            r["data_theta"] = _thetas - r["baseline"]

            # Moving average (convolve with a boxcar).
            r["data_theta_moving_average"] = np.convolve(
                r["data_theta"],
                self.moving_average_kernel,
                mode="same",
            )

            # Convolve with EMG pulse kernel.
            # Use FFT-based convolution for large kernels (faster than np.convolve).
            if len(self.kernel) > 10000:  # Use FFT for kernels larger than 10k samples.
                _convolved = fftconvolve(r["data_theta"], self.kernel, mode="full")
            else:
                _convolved = np.convolve(r["data_theta"], self.kernel, mode="full")
            r["data_theta_convolved"] = _convolved[-self.record_length :]

        return results
