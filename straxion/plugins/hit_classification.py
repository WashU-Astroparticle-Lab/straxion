import strax
import numpy as np
import pickle
import os
from straxion.utils import (
    TIME_DTYPE,
    CHANNEL_DTYPE,
    SECOND_TO_NANOSECOND,
    HIT_WINDOW_LENGTH_LEFT,
    HIT_WINDOW_LENGTH_RIGHT,
    DATA_DTYPE,
    NOISE_PSD_38kHz,
    DEFAULT_TEMPLATE_INTERP_PATH,
)

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        "max_spike_coincidence",
        type=int,
        default=1,
        track=True,
        help=("Maximum number of spikes that can be coincident with a photon candidate hit."),
    ),
    strax.Option(
        "spike_coincidence_window",
        type=float,
        default=0.131e-3,
        track=True,
        help=("Window length for checking spike coincidence, in unit of seconds."),
    ),
    strax.Option(
        "spike_threshold_dx",
        default=None,
        track=True,
        type=float,
        help="Threshold for spike finding in units of dx=df/f0.",
    ),
    strax.Option(
        "spike_thresholds_sigma",
        default=[3.0 for _ in range(41)],
        track=True,
        type=list,
        help=(
            "Threshold for spike finding in units of sigma of standard deviation of the noise. "
            "If None, the spike threshold will be calculated based on the signal statistics."
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
        "symmetric_spike_inspection_window_length",
        type=int,
        default=25,
        track=True,
        help=(
            "Length of the inspection window for identifying symmetric spikes, "
            "in unit of samples."
        ),
    ),
    strax.Option(
        "symmetric_spike_min_slope",
        type=list,
        default=[0.0 for _ in range(41)],
        track=True,
        help=(
            "Minimum rise edge slope of the moving averaged signal for identifying a physical hit "
            "against symmetric spikes, in unit of dx/second."
        ),
    ),
    strax.Option(
        "template_interp_path",
        type=str,
        default=DEFAULT_TEMPLATE_INTERP_PATH,
        track=True,
        help="Path to the saved template interpolation file.",
    ),
    strax.Option(
        "of_shift_range_min",
        type=int,
        default=-50,
        track=True,
        help="Minimum time shift for optimal filter coarse scan (in samples).",
    ),
    strax.Option(
        "of_shift_range_max",
        type=int,
        default=50,
        track=True,
        help="Maximum time shift for optimal filter coarse scan (in samples).",
    ),
    strax.Option(
        "of_shift_step",
        type=int,
        default=1,
        track=True,
        help="Step size for optimal filter coarse scan (in samples).",
    ),
    strax.Option(
        "noise_psd_placeholder",
        type=list,
        default=NOISE_PSD_38kHz,
        track=True,
        help=(
            "Noise power spectral density (PSD) array. " "The same PSD is used for all channels."
        ),
    ),
    strax.Option(
        "of_window_left",
        type=int,
        default=100,
        track=True,
        help=(
            "Left window size for optimal filter in samples. "
            "Window starts at HIT_WINDOW_LENGTH_LEFT - of_window_left."
        ),
    ),
    strax.Option(
        "of_window_right",
        type=int,
        default=300,
        track=True,
        help=(
            "Right window size for optimal filter in samples. "
            "Window ends at HIT_WINDOW_LENGTH_LEFT + of_window_right."
        ),
    ),
)
class DxHitClassification(strax.Plugin):
    """Classify hits into different types based on their coincidence with spikes."""

    __version__ = "0.2.3"

    depends_on = ("hits", "records", "noises")
    provides = "hit_classification"
    data_kind = "hits"
    save_when = strax.SaveWhen.ALWAYS

    def infer_dtype(self):
        base_dtype = [
            (("Start time since unix epoch [ns]", "time"), TIME_DTYPE),
            (("Exclusive end time since unix epoch [ns]", "endtime"), TIME_DTYPE),
            (("Channel number defined by channel_map", "channel"), CHANNEL_DTYPE),
        ]

        hit_id_dtype = [
            (("Is in coincidence with spikes", "is_coincident_with_spikes"), bool),
            (("Is symmetric spike hit", "is_symmetric_spike"), bool),
            (("Is truncated hit", "is_truncated_hit"), bool),
            (("Photon candidate hit", "is_photon_candidate"), bool),
        ]

        hit_feature_dtype = [
            (
                (
                    "Rise edge slope of the hit waveform, in unit of dx/second",
                    "rise_edge_slope",
                ),
                DATA_DTYPE,
            ),
            (
                ("Number of channels with spikes coinciding with the hit", "n_spikes_coinciding"),
                int,
            ),
            (
                ("Best optimal filter amplitude", "best_aOF"),
                DATA_DTYPE,
            ),
            (
                ("Best chi-squared value from optimal filter", "best_chi2"),
                DATA_DTYPE,
            ),
            (
                ("Best time shift in samples for optimal filter", "best_OF_shift"),
                int,
            ),
        ]

        return base_dtype + hit_id_dtype + hit_feature_dtype

    def setup(self):
        self.spike_coincidence_window = int(
            round(self.config["spike_coincidence_window"] * self.config["fs"])
        )
        self.spike_threshold_dx = self.config["spike_threshold_dx"]
        self.ss_min_slope = np.array(self.config["symmetric_spike_min_slope"])
        self.ss_window = self.config["symmetric_spike_inspection_window_length"]
        self.max_spike_coincidence = self.config["max_spike_coincidence"]
        self.dt_exact = 1 / self.config["fs"] * SECOND_TO_NANOSECOND
        self.template_interp_path = self.config["template_interp_path"]
        self.noise_psd = self.config["noise_psd_placeholder"]
        self.of_window_left = self.config["of_window_left"]
        self.of_window_right = self.config["of_window_right"]

        # Validate noise PSD length matches window size
        expected_length = self.of_window_left + self.of_window_right
        if len(self.noise_psd) != expected_length:
            raise ValueError(
                f"Noise PSD length ({len(self.noise_psd)}) does not match "
                f"optimal filter window size ({expected_length}). "
                f"Expected length: of_window_left ({self.of_window_left}) + "
                f"of_window_right ({self.of_window_right}) = {expected_length}"
            )

        # Load interpolation function
        self.At_interp, self.t_max = self.load_interpolation(self.template_interp_path)

    def compute_per_channel_noise_psd(self, noises, n_channels):
        """Compute per-channel noise PSDs from noise windows.

        Args:
            noises (np.ndarray): Array of noise windows.
            n_channels (int): Number of channels.

        Returns:
            dict: Dictionary mapping channel number to PSD array.
                  Returns None for channels with no noise windows.

        """
        channel_noise_psds = {}
        window_size = self.of_window_left + self.of_window_right

        for ch in range(n_channels):
            # Filter noise windows for this channel
            ch_noises = noises[noises["channel"] == ch]

            if len(ch_noises) == 0:
                # No noise windows for this channel
                channel_noise_psds[ch] = None
            else:
                # Extract first window_size samples and compute PSDs
                psds = []
                for noise in ch_noises:
                    noise_window = noise["data_dx"][:window_size]
                    # Compute PSD: |FFT|^2
                    psd = np.abs(np.fft.fft(noise_window)) ** 2
                    psds.append(psd)

                # Average PSDs across all noise windows
                channel_noise_psds[ch] = np.mean(psds, axis=0)

        return channel_noise_psds

    @staticmethod
    def calculate_spike_threshold(signal, spike_threshold_sigma):
        """Calculate spike threshold based on signal statistics.

        Args:
            signal (np.ndarray): The signal array to analyze.
            spike_threshold_sigma (float): Threshold multiplier in units of sigma.

        Returns:
            float: The calculated spike threshold.

        """
        signal_mean = np.mean(signal, axis=1)
        signal_std = np.std(signal, axis=1)

        # The naive spike threshold is a multiple of the standard deviation of the signal.
        spike_threshold = signal_mean + spike_threshold_sigma * signal_std

        return spike_threshold

    @staticmethod
    def load_interpolation(load_path="template_interp.pkl"):
        """
        Load saved interpolation function.

        Parameters:
        -----------
        load_path : str
            Path to saved interpolation function

        Returns:
        --------
        At_interp : interp1d
            Interpolation function
        t_max : float
            Time of maximum value in template (in seconds)
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Interpolation file not found: {load_path}. "
                "Please run build_and_save_interpolation() first."
            )

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        return data["interp"], data["t_max"]

    def modify_template(
        self,
        St,
        dt_seconds,
        tau,
        At_interp=None,
        t_max_seconds=None,
        amplitude=1.0,
        interp_path="template_interp.pkl",
        apply_window=False,
    ):
        """
        Modify template using pre-built interpolation function.

        Parameters:
        -----------
        St : array
            Signal timestream
        dt_seconds : float
            Time step in seconds
        tau : float
            Time shift parameter (in samples)
        At_interp : interp1d, optional
            Pre-built interpolation function.
            If None, loads from file.
        t_max_seconds : float, optional
            Time of maximum value in seconds. If None, loads from file.
        amplitude : float, optional
            Amplitude multiplier to scale the template. Default is 1.0.
        interp_path : str
            Path to saved interpolation file
        apply_window : bool, optional
            If True, apply windowing using of_window_left and of_window_right.
            Default is False for backward compatibility.

        Returns:
        --------
        At_modified : array
            Extended template array with padding, scaled by amplitude.
            If apply_window is True, returns only the windowed portion.
        """
        # Load interpolation function if not provided
        if At_interp is None or t_max_seconds is None:
            At_interp, t_max_seconds = self.load_interpolation(interp_path)

        target_length = len(St)
        max_index = HIT_WINDOW_LENGTH_LEFT
        final_max_index = max_index + tau
        time_new_seconds = np.arange(target_length) * dt_seconds
        time_shift_seconds = time_new_seconds[final_max_index] - t_max_seconds
        timeshifted_seconds = time_new_seconds - time_shift_seconds
        At_modified = At_interp(timeshifted_seconds) * amplitude

        # Apply windowing if requested
        if apply_window:
            window_start = HIT_WINDOW_LENGTH_LEFT - self.of_window_left
            window_end = HIT_WINDOW_LENGTH_LEFT + self.of_window_right
            At_modified = At_modified[window_start:window_end]

        return At_modified

    @staticmethod
    def _optimal_filter(St, Jf, At):
        """
        Calculate optimal filter amplitude and chi-squared score.

        Parameters:
        -----------
        St : array
            Signal timestream (to be filtered)
        Jf : array
            Noise PSD (taken from averaged FFTs of many noise banks)
        At : array
            Template timestream (to be filtered)

        Returns:
        --------
        ahatOF : float
            Optimal filter amplitude scaling factor
        chisq : float
            Chi-squared score
        """

        Sf = np.fft.fft(St)  # FFT of the hit signal
        Af = np.fft.fft(At)  # FFT of the template

        # Calculate optimal filter amplitude
        numer = np.sum(np.real(np.multiply(Sf, np.conjugate(Af))))
        denom = np.sum(np.real(np.multiply(Af, np.conjugate(Af))))
        ahatOF = numer / denom

        chisq = np.sum(np.abs(Sf - ahatOF * Af) ** 2 / Jf) / (len(Sf) - 1)

        return ahatOF, chisq

    def optimal_filter(self, St, dt_seconds, Jf, At_interp=None, t_max_seconds=None):
        """
        Calculate optimal filter with coarse time shift optimization.

        Parameters:
        -----------
        St : array
            Signal timestream (to be filtered)
        dt_seconds : float
            Time step in seconds
        Jf : array
            Noise PSD (taken from averaged FFTs
            of many noise banks)
        At_interp : interp1d, optional
            Pre-built interpolation function. Uses self.At_interp if None.
        t_max_seconds : float, optional
            Time of maximum value in seconds. Uses self.t_max if None.

        Returns:
        --------
        best_aOF : float
            Best optimal filter amplitude
        best_chi2 : float
            Best chi-squared value
        best_OF_shift : int
            Best time shift in samples
        best_At_shifted : array
            Template shifted to best position and scaled by best_aOF
        """
        # Use pre-loaded interpolation function if not provided
        if At_interp is None:
            At_interp = self.At_interp
        if t_max_seconds is None:
            t_max_seconds = self.t_max

        # Apply windowing to signal
        window_start = HIT_WINDOW_LENGTH_LEFT - self.of_window_left
        window_end = HIT_WINDOW_LENGTH_LEFT + self.of_window_right
        St_windowed = St[window_start:window_end]

        # Coarse scan for optimal time shift
        N_shiftOF_arr = np.arange(
            self.config["of_shift_range_min"],
            self.config["of_shift_range_max"],
            self.config["of_shift_step"],
        )
        ahatOF_arr = np.zeros(np.shape(N_shiftOF_arr))
        chi2_arr = np.zeros(np.shape(N_shiftOF_arr))

        # Test different time shifts
        for nn in range(len(N_shiftOF_arr)):
            N_shiftOF = N_shiftOF_arr[nn]
            At_shifted = self.modify_template(
                St,
                dt_seconds,
                N_shiftOF,
                At_interp=At_interp,
                t_max_seconds=t_max_seconds,
                apply_window=True,
            )
            ahatOF_arr[nn], chi2_arr[nn] = self._optimal_filter(St_windowed, Jf=Jf, At=At_shifted)

        # Find best shift
        best_idx = np.argmin(chi2_arr)
        best_chi2 = chi2_arr[best_idx]
        best_aOF = ahatOF_arr[best_idx]
        best_OF_shift = N_shiftOF_arr[best_idx]

        # Generate final shifted template scaled by best amplitude
        best_At_shifted = self.modify_template(
            St,
            dt_seconds,
            best_OF_shift,
            At_interp=At_interp,
            t_max_seconds=t_max_seconds,
            amplitude=best_aOF,
            apply_window=True,
        )

        return best_aOF, best_chi2, best_OF_shift, best_At_shifted

    def determine_spike_threshold(self, records):
        """Determine the spike threshold based on the provided configuration.
        You can either provide hit_threshold_dx or hit_thresholds_sigma.
        You cannot provide both.
        """
        if self.spike_threshold_dx is None and self.spike_thresholds_sigma is not None:
            # If spike_thresholds_sigma are single values,
            # we need to convert them to arrays.
            if isinstance(self.spike_thresholds_sigma, float):
                self.spike_thresholds_sigma = np.full(
                    len(records["channel"]), self.spike_thresholds_sigma
                )
            else:
                self.spike_thresholds_sigma = np.array(self.spike_thresholds_sigma)
            # Calculate spike threshold and find spike candidates
            self.spike_threshold_dx = self.calculate_spike_threshold(
                records["data_dx_convolved"],
                self.spike_thresholds_sigma[records["channel"]],
            )
        elif self.spike_threshold_dx is not None and self.spike_thresholds_sigma is None:
            # If spike_threshold_dx is a single value, we need to convert it to an array.
            if isinstance(self.spike_threshold_dx, float):
                self.spike_threshold_dx = np.full(len(records["channel"]), self.spike_threshold_dx)
            else:
                self.spike_threshold_dx = np.array(self.spike_threshold_dx)
        else:
            raise ValueError(
                "Either spike_threshold_dx or spike_thresholds_sigma "
                "must be provided. You cannot provide both."
            )

    def _get_ss_window(self, hits, window_start_offset):
        """Extract windows from all hits using vectorized operations.

        Args:
            hits: Array of hits containing the data
            window_start_offset: Offset from climax_shift for window starts in samples.

        Returns:
            Array of extracted windows with shape (n_hits, window_length)
        """
        # The inspected window ends at the maximum of the moving averaged signal.
        climax_shift = (
            hits["amplitude_moving_average_max_record_i"] - hits["amplitude_convolved_max_record_i"]
        )

        # Calculate start indices for all hits at once
        start_indices = window_start_offset + climax_shift

        # Extract windows using vectorized operations
        # Create index arrays for all hits
        n_hits = len(hits)
        window_indices = np.arange(self.ss_window)[None, :]  # Shape: (1, ss_window)
        start_indices = start_indices[:, None]  # Shape: (n_hits, 1)

        # Broadcast to get all indices for all hits
        all_indices = start_indices + window_indices  # Shape: (n_hits, ss_window)

        # Use advanced indexing to extract all windows at once
        return hits["data_dx_moving_average"][np.arange(n_hits)[:, None], all_indices]

    def compute_rise_edge_slope(self, hits, hit_classification):
        """Compute the rise edge slope of the moving averaged signal."""

        # Temporary time stamps for the inspected window, in unit of seconds.
        dt_nanoseconds = self.dt_exact
        times_seconds = np.arange(self.ss_window) * dt_nanoseconds / SECOND_TO_NANOSECOND

        inspected_wfs = self._get_ss_window(hits, HIT_WINDOW_LENGTH_LEFT - self.ss_window)
        # Fit a linear model to the inspected window.
        hit_classification["rise_edge_slope"] = np.polyfit(times_seconds, inspected_wfs.T, 1)[0]

    def is_symmetric_spike_hit(self, hits, hit_classification):
        """Identify symmetric spike hits."""
        hit_classification["is_symmetric_spike"] = (
            hit_classification["rise_edge_slope"] < self.ss_min_slope[hits["channel"]]
        )

    def is_truncated_hit(self, hits, hit_classification):
        """Identify truncated hits.

        A hit is considered truncated if its length does not equal
        the expected full window length.
        """
        expected_length = HIT_WINDOW_LENGTH_LEFT + HIT_WINDOW_LENGTH_RIGHT
        hit_classification["is_truncated_hit"] = hits["length"] != expected_length

    def find_spike_coincidence(self, hit_classification, hits, records):
        """Find the spike coincidence of the hit in the convolved signal."""
        spike_coincidence = np.zeros(len(hits))
        for i, hit in enumerate(hits):
            # Get the index of the hit maximum in the record
            hit_climax_i = hit["amplitude_convolved_max_record_i"]

            # Extract windows from all records at once
            inspected_wfs = records["data_dx_convolved"][
                :,
                hit_climax_i
                - self.spike_coincidence_window : hit_climax_i
                + self.spike_coincidence_window,
            ]

            # Count records with spikes above threshold
            spike_coincidence[i] = np.sum(
                np.max(inspected_wfs, axis=1) > self.spike_threshold_dx[records["channel"]]
            )
        hit_classification["n_spikes_coinciding"] = spike_coincidence

    def compute_optimal_filter_parameters(self, hit_classification, hits, channel_noise_psds):
        """Compute optimal filter parameters for all hits.

        Uses hits["data_dx"] as the signal timestream and
        per-channel noise PSDs from channel_noise_psds.
        Falls back to self.noise_psd placeholder if no PSD available for a channel.

        Args:
            hit_classification (np.ndarray): Array to store classification results.
            hits (np.ndarray): Array of hits.
            channel_noise_psds (dict): Dictionary mapping channel to PSD array.

        Note: All optimal filter calculations require dt in seconds.
        """
        # Convert dt from nanoseconds to seconds for optimal filter
        # self.dt_exact is in nanoseconds (= 1/fs * SECOND_TO_NANOSECOND)
        # Optimal filter functions require dt in seconds
        dt_seconds = self.dt_exact / SECOND_TO_NANOSECOND

        # Convert placeholder noise_psd to numpy array if it's a list
        if isinstance(self.noise_psd, list):
            placeholder_psd = np.array(self.noise_psd)
        else:
            placeholder_psd = self.noise_psd

        for i, hit in enumerate(hits):
            # Extract signal timestream from hit
            St = hit["data_dx"]
            ch = hit["channel"]

            # Get channel-specific PSD or use placeholder
            if channel_noise_psds[ch] is None:
                # No noise windows for this channel, use placeholder
                self.log.warning(
                    f"No noise windows found for channel {ch}, " f"using placeholder PSD"
                )
                Jf = placeholder_psd
            else:
                Jf = channel_noise_psds[ch]

            # Compute optimal filter with shift optimization
            # dt_seconds is explicitly in seconds
            best_aOF, best_chi2, best_OF_shift, _ = self.optimal_filter(St, dt_seconds, Jf)

            # Store results
            hit_classification["best_aOF"][i] = best_aOF
            hit_classification["best_chi2"][i] = best_chi2
            hit_classification["best_OF_shift"][i] = best_OF_shift

    def compute(self, hits, records, noises):
        self.determine_spike_threshold(records)

        # Compute per-channel noise PSDs from noise windows
        n_channels = len(records)
        channel_noise_psds = self.compute_per_channel_noise_psd(noises, n_channels)

        hit_classification = np.zeros(len(hits), dtype=self.infer_dtype())
        hit_classification["time"] = hits["time"]
        hit_classification["endtime"] = hits["endtime"]
        hit_classification["channel"] = hits["channel"]

        self.compute_rise_edge_slope(hits, hit_classification)
        self.find_spike_coincidence(hit_classification, hits, records)
        self.is_symmetric_spike_hit(hits, hit_classification)
        self.is_truncated_hit(hits, hit_classification)

        # Compute optimal filter parameters with per-channel PSDs
        self.compute_optimal_filter_parameters(hit_classification, hits, channel_noise_psds)

        hit_classification["is_coincident_with_spikes"] = (
            hit_classification["n_spikes_coinciding"] > self.max_spike_coincidence
        )
        hit_classification["is_photon_candidate"] = ~(
            hit_classification["is_coincident_with_spikes"]
            | hit_classification["is_symmetric_spike"]
            | hit_classification["is_truncated_hit"]
        )

        return hit_classification


@export
@strax.takes_config(
    strax.Option(
        "cr_ma_std_coeff",
        type=list,
        default=[20.0 for _ in range(41)],
        help=(
            "Coefficients applied to the moving averaged signal's "
            "standard deviation for identifying cosmic ray hits."
        ),
    ),
    strax.Option(
        "cr_convolved_std_coeff",
        type=list,
        default=[20.0 for _ in range(41)],
        help=(
            "Coefficients applied to the convolved signal's "
            "standard deviation for identifying cosmic ray hits."
        ),
    ),
    strax.Option(
        "cr_min_ma_amplitude",
        type=list,
        default=[1.0 for _ in range(41)],
        help=(
            "Minimum amplitude of the moving averaged signal's " "for identifying cosmic ray hits."
        ),
    ),
    strax.Option(
        "symmetric_spike_min_slope",
        type=list,
        default=[75.0 for _ in range(41)],
        help=(
            "Minimum slope for identifying a physical hit against symmetric spikes, "
            "in unit of rad/second."
        ),
    ),
    strax.Option(
        "symmetric_spike_inspection_window_length",
        type=int,
        default=25,
        help=(
            "Length of the inspection window for identifying symmetric spikes, "
            "in unit of samples."
        ),
    ),
)
class HitClassification(strax.Plugin):
    """Classify hits into different types based on their characteristics."""

    __version__ = "0.0.0"

    depends_on = "hits"
    provides = "hit_classification"
    data_kind = "hits"
    save_when = strax.SaveWhen.ALWAYS

    def setup(self):
        self.cr_ma_std_coeff = np.array(self.config["cr_ma_std_coeff"])
        self.cr_convolved_std_coeff = np.array(self.config["cr_convolved_std_coeff"])
        self.cr_min_ma_amplitude = np.array(self.config["cr_min_ma_amplitude"])
        self.ss_min_slope = np.array(self.config["symmetric_spike_min_slope"])
        self.ss_window = self.config["symmetric_spike_inspection_window_length"]

    def infer_dtype(self):
        base_dtype = [
            (("Start time since unix epoch [ns]", "time"), TIME_DTYPE),
            (("Exclusive end time since unix epoch [ns]", "endtime"), TIME_DTYPE),
            (("Channel number defined by channel_map", "channel"), CHANNEL_DTYPE),
        ]

        hit_id_dtype = [
            (("Is identified as cosmic ray hit", "is_cr"), bool),
            (("Is identified as symmetric spike hit", "is_symmetric_spike"), bool),
            (("Is unidentified hit", "is_unidentified"), bool),
        ]

        hit_feature_dtype = [
            (
                (
                    "Rise edge slope of the moving averaged signal, in unit of rad/second",
                    "ma_rise_edge_slope",
                ),
                DATA_DTYPE,
            ),
        ]

        return base_dtype + hit_id_dtype + hit_feature_dtype

    def compute_ma_rise_edge_slope(self, hits, hit_classification):
        """Compute the rise edge slope of the moving averaged signal."""
        assert (
            len(np.unique(hits["dt"])) == 1
        ), "The sampling frequency is not constant!? We found {} unique values: {}".format(
            len(np.unique(hits["dt"])), np.unique(hits["dt"])
        )
        # Temporary time stamps for the inspected window, in unit of seconds.
        dt = hits["dt"][0]
        times = np.arange(self.ss_window) * dt / SECOND_TO_NANOSECOND

        # Extract windows from all hits at once (fully vectorized)
        inspected_wfs = hits["data_theta_moving_average"][
            :, HIT_WINDOW_LENGTH_LEFT - self.ss_window : HIT_WINDOW_LENGTH_LEFT
        ]
        # Fit a linear model to the inspected window.
        hit_classification["ma_rise_edge_slope"] = np.polyfit(times, inspected_wfs.T, 1)[0]

    def is_unidentified_hit(self, hits, hit_classification):
        """Identify unidentified hits.

        The hit is identified as an unidentified hit if the amplitude of the hit is
        less than the threshold.

        Args:
            hits (np.ndarray): Hit array.

        """
        hit_classification["is_unidentified"] = (
            hits["amplitude_convolved_max"] < hits["hit_threshold"]
        )

    def is_cr_hit(self, hits, hit_classification):
        """Identify cosmic ray hits.

        The hit is identified as a cosmic ray hit if it satisfies either of the following
        conditions, for both convolved and moving averaged signals:
        1. The amplitude of the hit is greater than the threshold.
        2. The amplitude of the hit is greater than the standard deviation of the signal
        multiplied by the coefficient.

        Args:
            hits (np.ndarray): Hit array.

        Returns:
            np.ndarray: Hit array with `is_cr` field.

        """
        mask_convolved = hits["amplitude_convolved_max"] >= hits["hit_threshold"]
        mask_convolved &= hits["amplitude_convolved_max_ext"] >= (
            hits["record_convolved_std"] * self.cr_convolved_std_coeff[hits["channel"]]
        )
        mask_ma = hits["amplitude_ma_max_ext"] >= self.cr_min_ma_amplitude[hits["channel"]]
        mask_ma |= hits["amplitude_ma_max_ext"] >= (
            hits["record_ma_mean"] + hits["record_ma_std"] * self.cr_ma_std_coeff[hits["channel"]]
        )
        hit_classification["is_cr"] = mask_convolved | mask_ma

    def is_symmetric_spike_hit(self, hits, hit_classification):
        self.compute_ma_rise_edge_slope(hits, hit_classification)
        hit_classification["is_symmetric_spike"] = (
            hit_classification["ma_rise_edge_slope"] < self.ss_min_slope[hits["channel"]]
        )

    def compute(self, hits):
        hit_classification = np.zeros(len(hits), dtype=self.infer_dtype())
        hit_classification["time"] = hits["time"]
        hit_classification["endtime"] = hits["endtime"]
        hit_classification["channel"] = hits["channel"]

        self.is_unidentified_hit(hits, hit_classification)
        self.is_cr_hit(hits, hit_classification)
        self.is_symmetric_spike_hit(hits, hit_classification)

        return hit_classification
