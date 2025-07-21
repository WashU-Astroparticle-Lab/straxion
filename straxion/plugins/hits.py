import strax
import numpy as np

export, __all__ = strax.exporter()


@export
@strax.takes_config(
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
        "min_pulse_widths_samples",
        default=[20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
        track=True,
        type=list,
        help=(
            "Minimum pulse width in unit of samples. If the pulse width is below this "
            "threshold, the hit is not considered a new hit."
        ),
    ),
    strax.Option(
        "hit_window_length_left",
        default=100,
        track=True,
        type=int,
        help="Length of the hit window in unit of samples.",
    ),
)
class Hits(strax.Plugin):
    __version__ = "0.0.0"
    rechunk_on_save = False
    compressor = "zstd"  # Inherited from straxen. Not optimized outside XENONnT.

    depends_on = ["records"]
    provides = "hits"
    data_kind = "hits"
    save_when = strax.SaveWhen.ALWAYS

    chunk_target_size_mb = 2000
    rechunk_on_load = True
    chunk_source_size_mb = 100

    def setup(self):
        self.hit_thresholds_sigma = np.array(self.config["hit_thresholds_sigma"])
        self.noisy_channel_signal_std_multipliers = np.array(
            self.config["noisy_channel_signal_std_multipliers"]
        )

    def infer_dtype(self):
        pass

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
        for r in records:
            ch = int(r["channel"])
            signal = r["data_theta_convolved"]
            min_pulse_width = self.config["min_pulse_widths_samples"][ch]

            hit_threshold = self.calculate_hit_threshold(
                signal, self.hit_thresholds_sigma[ch], self.noisy_channel_signal_std_multipliers[ch]
            )

            below_threshold_indices = np.where(signal < hit_threshold)[0]
            if len(below_threshold_indices) == 0:
                continue
            # Find the start of the hits.
            hit_start_indicies = below_threshold_indices[
                np.diff(below_threshold_indices) > min_pulse_width
            ]

            hit_start_indicies
