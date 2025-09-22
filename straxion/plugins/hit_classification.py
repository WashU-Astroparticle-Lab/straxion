import strax
import numpy as np
from straxion.utils import (
    TIME_DTYPE,
    CHANNEL_DTYPE,
    SECOND_TO_NANOSECOND,
    HIT_WINDOW_LENGTH_LEFT,
    DATA_DTYPE,
)

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        "max_spike_coincidence",
        type=int,
        default=1,
        help=("Maximum number of spikes that can be coincident with a photon candidate hit."),
    ),
    strax.Option(
        "spike_coincidence_window",
        type=float,
        default=0.131e-3,
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
        "noisy_channel_signal_std_multipliers",
        default=[2.0 for _ in range(41)],
        track=True,
        type=list,
        help=(
            "If the signal standard deviation above this threshold times of signal absolute "
            "mean, the signal is considered noisy and the spike threshold is increased. "
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
        help=(
            "Length of the inspection window for identifying symmetric spikes, "
            "in unit of samples."
        ),
    ),
    strax.Option(
        "symmetric_spike_min_slope",
        type=list,
        default=[0.0 for _ in range(41)],
        help=(
            "Minimum rise edge slope of the moving averaged signal for identifying a physical hit "
            "against symmetric spikes, in unit of dx/second."
        ),
    ),
)
class SpikeCoincidence(strax.Plugin):
    """Classify hits into different types based on their coincidence with spikes."""

    __version__ = "0.1.0"

    depends_on = ("hits", "records")
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

    @staticmethod
    def calculate_spike_threshold(
        signal, spike_threshold_sigma, noisy_channel_signal_std_multiplier
    ):
        """Calculate hit threshold based on signal statistics.
        The algorithm is the same as DxHits.calculate_hit_threshold.

        Args:
            signal (np.ndarray): The signal array to analyze.
            hit_threshold_sigma (float): Threshold multiplier in units of sigma.
            noisy_channel_signal_std_multiplier (float): Multiplier to detect noisy channels.

        Returns:
            float: The calculated hit threshold.

        """
        signal_mean = np.mean(signal, axis=1)
        signal_abs_mean = np.mean(np.abs(signal), axis=1)
        signal_std = np.std(signal, axis=1)

        # The naive hit threshold is a multiple of the standard deviation of the signal.
        spike_threshold = signal_mean + spike_threshold_sigma * signal_std

        # If the signal is noisy, the baseline might be too high.
        for ch in range(len(signal_std)):
            if signal_std[ch] > noisy_channel_signal_std_multiplier[ch] * signal_abs_mean[ch]:
                # We will use the quiet part of the signal to redefine a lowered hit threshold.
                quiet_mask = signal[ch] < spike_threshold[ch]
                spike_threshold[ch] = signal_mean[ch] + spike_threshold_sigma[ch] * np.std(
                    signal[ch][quiet_mask]
                )

        return spike_threshold

    def determine_spike_threshold(self, records):
        """Determine the spike threshold based on the provided configuration.
        You can either provide spike_threshold_dx or
        (hit_thresholds_sigma and noisy_channel_signal_std_multipliers).
        You cannot provide both.
        The algorithm is the same as DxHits.determine_hit_threshold.
        """
        if (
            self.spike_threshold_dx is None
            and self.spike_thresholds_sigma is not None
            and self.noisy_channel_signal_std_multipliers is not None
        ):
            # If hit_thresholds_sigma and noisy_channel_signal_std_multipliers are single values,
            # we need to convert them to arrays.
            if isinstance(self.spike_thresholds_sigma, float):
                self.spike_thresholds_sigma = np.full(
                    len(records["channel"]), self.spike_thresholds_sigma
                )
            else:
                self.spike_thresholds_sigma = np.array(self.spike_thresholds_sigma)
            if isinstance(self.noisy_channel_signal_std_multipliers, float):
                self.noisy_channel_signal_std_multipliers = np.full(
                    len(records["channel"]), self.noisy_channel_signal_std_multipliers
                )
            else:
                self.noisy_channel_signal_std_multipliers = np.array(
                    self.noisy_channel_signal_std_multipliers
                )
            # Calculate spike threshold and find spike candidates
            self.spike_threshold_dx = self.calculate_spike_threshold(
                records["data_dx_convolved"],
                self.spike_thresholds_sigma[records["channel"]],
                self.noisy_channel_signal_std_multipliers[records["channel"]],
            )
        elif (
            self.spike_threshold_dx is not None
            and self.spike_thresholds_sigma is None
            and self.noisy_channel_signal_std_multipliers is None
        ):
            # If spike_threshold_dx is a single value, we need to convert it to an array.
            if isinstance(self.spike_threshold_dx, float):
                self.spike_threshold_dx = np.full(len(records["channel"]), self.spike_threshold_dx)
            else:
                self.spike_threshold_dx = np.array(self.spike_threshold_dx)
        else:
            raise ValueError(
                "Either spike_threshold_dx or spike_thresholds_sigma and "
                "noisy_channel_signal_std_multipliers must be provided. You cannot provide both."
            )

    def compute_rise_edge_slope(self, hits, hit_classification):
        """Compute the rise edge slope of the moving averaged signal."""

        # Temporary time stamps for the inspected window, in unit of seconds.
        dt = self.dt_exact
        times = np.arange(self.ss_window) * dt / SECOND_TO_NANOSECOND

        inspected_wfs = hits["data_dx_moving_average"][
            :, HIT_WINDOW_LENGTH_LEFT - self.ss_window : HIT_WINDOW_LENGTH_LEFT
        ]
        # Fit a linear model to the inspected window.
        hit_classification["rise_edge_slope"] = np.polyfit(times, inspected_wfs.T, 1)[0]

    def is_symmetric_spike_hit(self, hits, hit_classification):
        """Identify symmetric spike hits."""
        hit_classification["is_symmetric_spike"] = (
            hit_classification["rise_edge_slope"] < self.ss_min_slope[hits["channel"]]
        )

    def find_spike_coincidence(self, hit_classification, hits, records):
        """Find the spike coincidence of the hit in the convolved signal."""
        spike_coincidence = np.zeros(len(hits))
        for i, hit in enumerate(hits):
            # Get the index of the hit maximum in the record
            hit_climax_i = hit["amplitude_max_record_i"]

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

    def compute(self, hits, records):
        self.determine_spike_threshold(records)

        hit_classification = np.zeros(len(hits), dtype=self.infer_dtype())
        hit_classification["time"] = hits["time"]
        hit_classification["endtime"] = hits["endtime"]
        hit_classification["channel"] = hits["channel"]

        self.compute_rise_edge_slope(hits, hit_classification)
        self.find_spike_coincidence(hit_classification, hits, records)
        self.is_symmetric_spike_hit(hits, hit_classification)

        hit_classification["is_coincident_with_spikes"] = (
            hit_classification["n_spikes_coinciding"] > self.max_spike_coincidence
        )
        hit_classification["is_photon_candidate"] = ~(
            hit_classification["is_coincident_with_spikes"]
            | hit_classification["is_symmetric_spike"]
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
