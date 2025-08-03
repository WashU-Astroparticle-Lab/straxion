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
        "cr_ma_std_coeff",
        type=list,
        default=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        help=(
            "Coefficients applied to the moving averaged signal's "
            "standard deviation for identifying cosmic ray hits."
        ),
    ),
    strax.Option(
        "cr_convolved_std_coeff",
        type=list,
        default=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        help=(
            "Coefficients applied to the convolved signal's "
            "standard deviation for identifying cosmic ray hits."
        ),
    ),
    strax.Option(
        "cr_min_ma_amplitude",
        type=list,
        default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        help=(
            "Minimum amplitude of the moving averaged signal's " "for identifying cosmic ray hits."
        ),
    ),
    strax.Option(
        "symmetric_spike_min_slope",
        type=list,
        default=[75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0],
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
        mask_convolved &= hits["amplitude_convolved_max"] >= (
            hits["record_convolved_std"] * self.cr_convolved_std_coeff[hits["channel"]]
        )
        mask_ma = hits["amplitude_ma_max"] >= self.cr_min_ma_amplitude[hits["channel"]]
        mask_ma &= hits["amplitude_ma_max"] >= (
            hits["record_ma_mean"] + hits["record_ma_std"] * self.cr_ma_std_coeff[hits["channel"]]
        )
        hit_classification["is_cr"] = mask_convolved | mask_ma

    def is_symmetric_spike_hit(self, hits):
        self.compute_ma_rise_edge_slope(hits)
        hits["is_symmetric_spike"] = hits["ma_rise_edge_slope"] > self.ss_min_slope[hits["channel"]]

    def compute(self, hits):
        hit_classification = np.zeros(len(hits), dtype=self.infer_dtype())
        self.is_unidentified_hit(hits, hit_classification)
        self.is_cr_hit(hits, hit_classification)
        self.is_symmetric_spike_hit(hits, hit_classification)
