import strax
import numpy as np
from straxion.utils import (
    DATA_DTYPE,
    SECOND_TO_NANOSECOND,
    base_waveform_dtype,
    circfit,
)
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import os

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        "iq_finescan_dir",
        track=False,
        type=str,
        help=("Direcotry to fine frequency scan (IQ loop) of resonatorm."),
    ),
    strax.Option(
        "iq_widescan_dir",
        track=False,
        type=str,
        help=("Direcotry to wide frequency scan (IQ loop) of resonatorm."),
    ),
    strax.Option(
        "resonant_frequency_dir",
        track=False,
        type=str,
        help=("Direcotry to resonant frequency npy file."),
    ),
    strax.Option(
        "iq_finescan_filename",
        track=True,
        type=str,
        help=(
            "Filename of the fine frequency scan (IQ loop) of resonator npy file, "
            "just the filename. Assumed the filename starts with iq_fine_<f/z>_*-<TIME>",
        ),
    ),
    strax.Option(
        "iq_widescan_filename",
        track=True,
        type=str,
        help=(
            "Filename of the fine frequency scan (IQ loop) of resonator npy file, "
            "just the filename. Assumed the filename starts with iq_wide_<f/z>_*-<TIME>",
        ),
    ),
    strax.Option(
        "resonant_frequency_filename",
        track=True,
        type=str,
        help=(
            "Filename of the resonant frequency npy file, "
            "just the filename, not the path. Assumed the filename starts with fres_*-<TIME>",
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
        "widescan_resolution",
        track=True,
        default=1000.0,
        type=float,
        help=(
            "Wide scan within fr +/- widescan_resolution/2*fr outside the fine scan range will "
            "be used to correct for cable delay",
        ),
    ),
    strax.Option(
        "cable_correction_polyfit_order",
        track=True,
        default=5,
        type=int,
        help="Order of the polynomial fit for cable correction",
    ),
    strax.Option(
        "pulse_kernel_start_time",
        default=200_000,
        track=True,
        type=int,
        help="Relative start time of the exponential decay in pulse kernel (t0), in unit of ns.",
    ),
    strax.Option(
        "pulse_kernel_decay_time",
        default=600_000,
        track=True,
        type=int,
        help="Decay time of the exponential falling in pulse kernel (tau), in unit of ns.",
    ),
    strax.Option(
        "pulse_kernel_gaussian_smearing_width",
        default=28_000,
        track=True,
        type=int,
        help=(
            "Gaussian smearing width of the exponentially-modified-gaussian kernel "
            "(sigma), in unit of ns."
        ),
    ),
    strax.Option(
        "moving_average_width",
        default=100_000,  # The original Matlab code says 5 samples (with fs = 5E4Hz).
        track=True,
        type=int,
        help="Moving average width for smoothed reference waveform, in unit of ns.",
    ),
    strax.Option(
        "pulse_kernel_truncation_factor",
        default=10,
        track=True,
        type=float,
        help=(
            "Factor for truncating the pulse kernel to improve performance. "
            "The kernel is truncated after truncation_factor * tau samples, "
            "where the exponential decay becomes negligible."
        ),
    ),
)
class DxRecords(strax.Plugin):
    __version__ = "0.1.0"
    rechunk_on_save = False
    compressor = "zstd"  # Inherited from straxen. Not optimized outside XENONnT.

    depends_on = "raw_records"
    provides = "records"
    data_kind = "records"
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        """Data type for a waveform record."""
        # Use record_length from setup if available, otherwise compute it.
        if hasattr(self, "record_length"):
            record_length = self.record_length
        else:
            # Get record_length from the plugin making raw_records.
            raw_records_dtype = self.deps["raw_records"].dtype_for("raw_records")
            record_length = len(np.zeros(1, raw_records_dtype)[0]["data_i"])

        dtype = base_waveform_dtype()
        dtype.append(
            (
                (
                    "Waveform data of phase angle after baseline corrections",
                    "data_dtheta",
                ),
                DATA_DTYPE,
                record_length,
            )
        )
        dtype.append(
            (
                (
                    "Waveform data of df/f after baseline corrections",
                    "data_dx",
                ),
                DATA_DTYPE,
                record_length,
            )
        )
        dtype.append(
            (
                (
                    "Waveform data of df/f further smoothed by moving average",
                    "data_dx_moving_average",
                ),
                DATA_DTYPE,
                record_length,
            )
        )
        dtype.append(
            (
                (
                    "Waveform data of df/f further smoothed by pulse kernel",
                    "data_dx_convolved",
                ),
                DATA_DTYPE,
                record_length,
            )
        )

        return np.dtype(dtype)

    def _load_correction_files(self):
        """Load fine and wide scan files, as well as resonant frequency file.

        Assumed the filenames are in the format of iq_fine_z_*-<TIME>.npy, iq_wide_z_*-<TIME>.npy,
        and fres_*-<TIME>.npy, and the corresponding scan frequency are in the format of
        iq_fine_f_*-<TIME>.npy, iq_wide_f_*-<TIME>.npy.
        """
        iq_finescan_filename = self.config["iq_finescan_filename"]
        iq_widescan_filename = self.config["iq_widescan_filename"]
        fr_filename = self.config["resonant_frequency_filename"]

        assert (
            iq_finescan_filename.endswith(".npy")
            and iq_widescan_filename.endswith(".npy")
            and fr_filename.endswith(".npy")
        ), (
            "Filename of the fine frequency scan (IQ loop) of resonator npy file, "
            "the wide frequency scan (IQ loop) of resonator npy file, "
            "and the resonant frequency npy file should end with .npy",
        )

        assert (
            iq_finescan_filename.startswith("iq_fine_z")
            and iq_widescan_filename.startswith("iq_wide_z")
            and fr_filename.startswith("fres_")
        ), (
            "Filename of the fine frequency scan (IQ loop) of resonator npy file should "
            "start with iq_fine_z, and the wide frequency scan (IQ loop) of resonator npy "
            "file should start with iq_wide_z. The resonant frequency npy file should "
            "start with fres_",
        )

        # Check if filenames have the expected format with timestamps
        finescan_parts = iq_finescan_filename.split("-")
        widescan_parts = iq_widescan_filename.split("-")
        fr_parts = fr_filename.split("-")

        # Only check timestamp consistency if all filenames have the expected format
        if len(finescan_parts) >= 2 and len(widescan_parts) >= 2 and len(fr_parts) >= 2:
            assert finescan_parts[1] == widescan_parts[1] == fr_parts[1], (
                "The time of the fine frequency scan (IQ loop) of resonator npy file, "
                "the wide frequency scan (IQ loop) of resonator npy file, "
                "and the resonant frequency npy file should be the same",
            )

        self.fine_z = np.load(os.path.join(self.config["iq_finescan_dir"], iq_finescan_filename))
        self.fine_f = np.load(
            os.path.join(
                self.config["iq_finescan_dir"],
                iq_finescan_filename.replace("iq_fine_z", "iq_fine_f"),
            )
        )
        self.wide_z = np.load(os.path.join(self.config["iq_widescan_dir"], iq_widescan_filename))
        self.wide_f = np.load(
            os.path.join(
                self.config["iq_widescan_dir"],
                iq_widescan_filename.replace("iq_wide_z", "iq_wide_f"),
            )
        )
        self.fres = np.load(os.path.join(self.config["resonant_frequency_dir"], fr_filename))

    @staticmethod
    def iq_gain_correction_model(fine_f, wide_z, wide_f, fres, widescan_resolution, polyfit_order):
        """Use the wide scan data outside fine scan range but within wide scan resolution to derive
        the gain correction model for each channel.

        Args:
            fine_f (np.ndarray): Fine scan data of the imaginary part.
            wide_z (np.ndarray): Wide scan data of the real part.
            wide_f (np.ndarray): Wide scan data of the imaginary part.
            fres (np.ndarray): Resonant frequency.
            widescan_resolution (float): Resolution of the wide scan.
            polyfit_order (int): Order of the polynomial fit.

        Returns:
            i_models (np.ndarray): Gain correction model for the real part,
                as a function of the frequency offset from the resonant frequency.
            q_models (np.ndarray): Gain correction model for the imaginary part,
                as a function of the frequency offset from the resonant frequency.
        """
        i_models = []
        q_models = []
        for ch in range(len(fres)):
            mask_in_wide_resolution = np.abs(wide_f[ch] - fres[ch]) < fres[ch] / widescan_resolution
            mask_calc_corr = mask_in_wide_resolution.copy()
            for ind in range(len(mask_in_wide_resolution)):
                if wide_f[ch, ind] < fine_f[ch, 0] and wide_f[ch, ind] > fine_f[ch, -1]:
                    mask_calc_corr[ind] = 0

            # Check if we have any data points for fitting
            if not np.any(mask_calc_corr):
                # If no data points, create a constant model (zero-order polynomial)
                pfit_i = np.zeros(polyfit_order + 1)
                pfit_q = np.zeros(polyfit_order + 1)
                pfit_i[-1] = 1.0  # Constant term
                pfit_q[-1] = 1.0  # Constant term
            else:
                pfit_i = np.polyfit(
                    wide_f[ch, mask_calc_corr] - fres[ch],
                    np.real(wide_z[ch, mask_calc_corr]),
                    polyfit_order,
                )
                pfit_q = np.polyfit(
                    wide_f[ch, mask_calc_corr] - fres[ch],
                    np.imag(wide_z[ch, mask_calc_corr]),
                    polyfit_order,
                )

            i_model = np.poly1d(pfit_i)
            q_model = np.poly1d(pfit_q)

            i_models.append(i_model)
            q_models.append(q_model)

        return i_models, q_models

    def _setup_iq_correction_and_calibration(self):
        """Setup IQ correction models and calibrate IQ loop centers and phis.

        This method:
        1. Loads fine and wide scan files, as well as resonant frequency file
        2. Creates IQ gain correction models for each channel
        3. Applies corrections to fine scan data
        4. Calculates IQ loop centers using circle fitting
        5. Computes phi values for each channel
        6. Corrects the fine scan data by rotating back the centered IQ by the phi values.
        """
        # Load fine and wide scan files, as well as resonant frequency file.
        self._load_correction_files()

        # Create IQ gain correction models
        self.i_models, self.q_models = self.iq_gain_correction_model(
            self.fine_f,
            self.wide_z,
            self.wide_f,
            self.fres,
            self.config["widescan_resolution"],
            self.config["cable_correction_polyfit_order"],
        )

        # Initialize corrected data arrays.
        self._fine_z_corrected = self.fine_z.copy()
        self.fine_z_corrected = self.fine_z.copy()
        self.iq_centers = np.zeros(len(self.fres), dtype=np.complex128)
        self.phis = np.zeros(len(self.fres))

        # Process each channel.
        for ch in range(len(self.fres)):
            # Apply IQ gain correction
            self._fine_z_corrected[ch] = self.fine_z[ch] / (
                self.i_models[ch](self.fine_f[ch] - self.fres[ch])
                + 1j * self.q_models[ch](self.fine_f[ch] - self.fres[ch])
            )

            # Calculate IQ loop center using circle fitting.
            i_center, q_center, _, _ = circfit(
                self._fine_z_corrected[ch].real, self._fine_z_corrected[ch].imag
            )
            self.iq_centers[ch] = i_center + 1j * q_center

            # Center the centered IQ fine scan data
            fine_z_centered = self._fine_z_corrected[ch] - self.iq_centers[ch]

            # Calculate phi (assume the last point is a good approximation of infinity or zero).
            self.phis[ch] = np.arctan2(fine_z_centered[-1].imag, fine_z_centered[-1].real)

            # Rotate back the centered IQ by the phi values.
            self.fine_z_corrected[ch] = np.mod(
                np.angle(fine_z_centered * np.exp(-1j * self.phis[ch])), 2 * np.pi
            )

    def _setup_frequency_interpolation_models(self):
        """Setup interpolation models that map theta values to frequency offsets.

        This method:
        1. Finds the theta value at the resonant frequency for each channel
        2. Creates interpolation models that map theta differences to frequency offsets
        3. Validates the interpolation models for self-consistency
        """
        # Initialize interpolation data arrays
        self.thetas_at_fres = np.zeros(len(self.fres), dtype=np.complex128)
        self.interpolated_freqs = np.zeros_like(self.fres)
        self.f_interpolation_models = []

        # Create interpolation models for each channel
        for ch in range(len(self.fres)):
            # Find the index closest to resonant frequency
            f0_idx = np.argmin(np.abs(self.fine_f[ch] - self.fres[ch]))

            # Get theta value at resonant frequency
            self.thetas_at_fres[ch] = self.fine_z_corrected[ch][f0_idx]

            # Calculate theta differences from resonant frequency
            dtheta_fine = self.fine_z_corrected[ch] - self.thetas_at_fres[ch]

            # Create interpolation model mapping theta differences to frequencies
            # Use bounds_error=False to handle out-of-bounds values gracefully
            # and fill_value='extrapolate' to extrapolate beyond the range
            self.f_interpolation_models.append(
                interp1d(dtheta_fine, self.fine_f[ch], bounds_error=False, fill_value="extrapolate")
            )

            # Validate interpolation model (should return resonant frequency at theta=0)
            # This is just for self-consistency check
            self.interpolated_freqs[ch] = self.f_interpolation_models[ch](0)

    @staticmethod
    def pulse_kernel(ns, fs, t0, tau, sigma, truncation_factor=5):
        """Generate a pulse train with flipped, truncated exponential decay and Gaussian smoothing.

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

        # Calculate significant length upfront to avoid unnecessary computation.
        significant_length = min(ns, int((truncation_factor * tau + t0) / dt))

        # Only create time array for needed samples.
        t = np.arange(significant_length) * dt

        # Create exponential decay pulse only for significant portion.
        mask = t >= t0
        exponential = np.zeros(significant_length)
        exponential[mask] = np.exp(-(t[mask] - t0) / tau)

        # Convert sigma to samples.
        sigma_sample = int(sigma / dt)

        # Apply Gaussian smoothing.
        pulse_kernal = gaussian_filter1d(exponential, sigma=sigma_sample)

        # No need for truncation since we already computed only the significant portion.
        # But we still need to normalize.
        kernel_sum = np.sum(pulse_kernal)
        if kernel_sum > 0:  # Avoid division by zero.
            pulse_kernal /= kernel_sum

        # Normalize again to make sure the integral is 1.
        kernel_sum = np.sum(pulse_kernal)
        if kernel_sum > 0:  # Avoid division by zero.
            pulse_kernal /= kernel_sum

        # Flip the kernel.
        pulse_kernal = np.flip(pulse_kernal)

        return pulse_kernal

    def setup(self):
        # Get record_length from the plugin making raw_records.
        raw_records_dtype = self.deps["raw_records"].dtype_for("raw_records")
        self.record_length = len(np.zeros(1, raw_records_dtype)[0]["data_i"])
        self.dt_exact = 1 / self.config["fs"] * SECOND_TO_NANOSECOND

        # Setup IQ correction and calibration
        self._setup_iq_correction_and_calibration()

        # Setup frequency interpolation models
        self._setup_frequency_interpolation_models()

        # Pre-compute pulse kernel.
        self.kernel = self.pulse_kernel(
            self.record_length,
            self.config["fs"],
            self.config["pulse_kernel_start_time"],
            self.config["pulse_kernel_decay_time"],
            self.config["pulse_kernel_gaussian_smearing_width"],
            self.config["pulse_kernel_truncation_factor"],
        )

        # Pre-compute moving average kernel.
        moving_average_kernel_width = int(self.config["moving_average_width"] / self.dt_exact)
        self.moving_average_kernel = (
            np.ones(moving_average_kernel_width) / moving_average_kernel_width
        )

    def compute(self, raw_records):
        """Compute the dx=df/f0 for the timestream data."""
        results = np.zeros(len(raw_records), dtype=self.infer_dtype())

        for i, rr in enumerate(raw_records):
            r = results[i]
            r["time"] = rr["time"]
            r["endtime"] = rr["endtime"]
            r["length"] = rr["length"]
            r["dt"] = rr["dt"]
            r["channel"] = rr["channel"]

            data_z = rr["data_i"] + 1j * rr["data_q"]
            # Apply IQ gain correction.
            data_z = data_z / (
                self.i_models[rr["channel"]](0) + 1j * self.q_models[rr["channel"]](0)
            )
            # Center the data and rotate back by the phi value.
            data_z = (data_z - self.iq_centers[rr["channel"]]) * np.exp(
                -1j * self.phis[rr["channel"]]
            )

            # Convert to theta (phase angle) relative to the IQ loop center.
            theta = np.mod(np.angle(data_z), 2 * np.pi)
            dtheta = theta - self.thetas_at_fres[rr["channel"]]
            r["data_dtheta"] = dtheta

            # Interpolate to get the frequency offset.
            r["data_dx"] = (
                self.f_interpolation_models[rr["channel"]](dtheta)
                - self.interpolated_freqs[rr["channel"]]
            ) / self.interpolated_freqs[rr["channel"]]

            # Moving average (convolve with a boxcar).
            r["data_dx_moving_average"] = np.convolve(
                r["data_dx"],
                self.moving_average_kernel,
                mode="same",
            )

            # Convolve with pulse kernel.
            # Use FFT-based convolution for large kernels (faster than np.convolve).
            if len(self.kernel) > 10000:  # Use FFT for kernels larger than 10k samples.
                _convolved = fftconvolve(r["data_dx"], self.kernel, mode="full")
            else:
                _convolved = np.convolve(r["data_dx"], self.kernel, mode="full")
            r["data_dx_convolved"] = _convolved[-self.record_length :]

        return results


@export
@strax.takes_config(
    strax.Option(
        "iq_finescan_dir",
        track=False,
        type=str,
        help=("Direcotry to fine frequency scan (IQ loop) of resonatorm."),
    ),
    strax.Option(
        "iq_finescan_filename",
        track=True,
        type=str,
        help=(
            "Filename of the fine frequency scan (IQ loop) of resonator (txt, csv or similar), "
            "just the filename, not the path. If not provided, the plugin will try to find the file"
            " in the iq_finescan_dir directory.",
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
        "pulse_kernel_start_time",
        default=200_000,
        track=True,
        type=int,
        help="Relative start time of the exponential decay in pulse kernel (t0), in unit of ns.",
    ),
    strax.Option(
        "pulse_kernel_decay_time",
        default=600_000,
        track=True,
        type=int,
        help="Decay time of the exponential falling in pulse kernel (tau), in unit of ns.",
    ),
    strax.Option(
        "pulse_kernel_gaussian_smearing_width",
        default=28_000,
        track=True,
        type=int,
        help=(
            "Gaussian smearing width of the exponentially-modified-gaussian kernel "
            "(sigma), in unit of ns."
        ),
    ),
    strax.Option(
        "moving_average_width",
        default=100_000,  # The original Matlab code says 5 samples (with fs = 5E4Hz).
        track=True,
        type=int,
        help="Moving average width for smoothed reference waveform, in unit of ns.",
    ),
    strax.Option(
        "pulse_kernel_truncation_factor",
        default=10,
        track=True,
        type=float,
        help=(
            "Factor for truncating the pulse kernel to improve performance. "
            "The kernel is truncated after truncation_factor * tau samples, "
            "where the exponential decay becomes negligible."
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

    __version__ = "0.0.1"
    rechunk_on_save = False
    compressor = "zstd"  # Inherited from straxen. Not optimized outside XENONnT.

    depends_on = "raw_records"
    provides = "records"
    data_kind = "records"
    save_when = strax.SaveWhen.EXPLICIT

    def infer_dtype(self):
        """Data type for a waveform record."""
        # Use record_length from setup if available, otherwise compute it.
        if hasattr(self, "record_length"):
            record_length = self.record_length
        else:
            # Get record_length from the plugin making raw_records.
            raw_records_dtype = self.deps["raw_records"].dtype_for("raw_records")
            record_length = len(np.zeros(1, raw_records_dtype)[0]["data_i"])

        dtype = base_waveform_dtype()
        dtype.append(
            (
                (
                    "Waveform data of phase angle (theta) after baseline corrections",
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
        self.dt_exact = 1 / self.config["fs"] * SECOND_TO_NANOSECOND

        self.finescan = self.load_finescan_files(
            os.path.join(self.config["iq_finescan_dir"], self.config["iq_finescan_filename"])
        )
        self.finescan_available_channels = np.shape(self.finescan)[0]

        # Pre-compute pulse kernel.
        self.kernel = self.pulse_kernel(
            self.record_length,
            self.config["fs"],
            self.config["pulse_kernel_start_time"],
            self.config["pulse_kernel_decay_time"],
            self.config["pulse_kernel_gaussian_smearing_width"],
            self.config["pulse_kernel_truncation_factor"],
        )

        # Pre-compute moving average kernel.
        moving_average_kernel_width = int(self.config["moving_average_width"] / self.dt_exact)
        self.moving_average_kernel = (
            np.ones(moving_average_kernel_width) / moving_average_kernel_width
        )

        # Pre-compute circle fits for each channel to avoid repeated computation.
        self.channel_centers = {}
        for channel in np.arange(self.finescan_available_channels):
            finescan = self.finescan[channel]
            finescan_i = finescan.real
            finescan_q = finescan.imag
            i_center, q_center, _, _ = circfit(finescan_i, finescan_q)
            theta_f_min = np.arctan2(finescan_q[0] - q_center, finescan_i[0] - i_center)
            self.channel_centers[int(channel)] = (i_center, q_center, theta_f_min)

    @staticmethod
    def load_finescan_files(directory_or_file):
        """Load fine scan files from directory or file.

        Args:
            directory_or_file: Path to directory containing fine scan files or
                path to a specific file

        Returns:
            numpy array containing the fine scan data

        Raises:
            FileNotFoundError: If directory/file doesn't exist or no files found
        """
        if not os.path.exists(directory_or_file):
            raise FileNotFoundError("Fine scan directory or file not found")

        # Check if it's a file or directory
        if os.path.isfile(directory_or_file):
            # It's a file, load it directly
            return np.load(directory_or_file)
        else:
            # It's a directory, look for .npy files
            npy_files = [f for f in os.listdir(directory_or_file) if f.endswith(".npy")]
            if not npy_files:
                raise FileNotFoundError("No fine scan files found")

            # Load the first .npy file found
            file_path = os.path.join(directory_or_file, npy_files[0])
            return np.load(file_path)

    @staticmethod
    def pulse_kernel(ns, fs, t0, tau, sigma, truncation_factor=5):
        """Generate a pulse train with flipped, truncated exponential decay and Gaussian smoothing.

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

        # Calculate significant length upfront to avoid unnecessary computation.
        significant_length = min(ns, int((truncation_factor * tau + t0) / dt))

        # Only create time array for needed samples.
        t = np.arange(significant_length) * dt

        # Create exponential decay pulse only for significant portion.
        mask = t >= t0
        exponential = np.zeros(significant_length)
        exponential[mask] = np.exp(-(t[mask] - t0) / tau)

        # Convert sigma to samples.
        sigma_sample = int(sigma / dt)

        # Apply Gaussian smoothing.
        pulse_kernal = gaussian_filter1d(exponential, sigma=sigma_sample)

        # No need for truncation since we already computed only the significant portion.
        # But we still need to normalize.
        kernel_sum = np.sum(pulse_kernal)
        if kernel_sum > 0:  # Avoid division by zero.
            pulse_kernal /= kernel_sum

        # Normalize again to make sure the integral is 1.
        kernel_sum = np.sum(pulse_kernal)
        if kernel_sum > 0:  # Avoid division by zero.
            pulse_kernal /= kernel_sum

        # Flip the kernel.
        pulse_kernal = np.flip(pulse_kernal)

        return pulse_kernal

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

            # Convolve with pulse kernel.
            # Use FFT-based convolution for large kernels (faster than np.convolve).
            if len(self.kernel) > 10000:  # Use FFT for kernels larger than 10k samples.
                _convolved = fftconvolve(r["data_theta"], self.kernel, mode="full")
            else:
                _convolved = np.convolve(r["data_theta"], self.kernel, mode="full")
            r["data_theta_convolved"] = _convolved[-self.record_length :]

        return results
