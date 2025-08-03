import strax
import numpy as np
from straxion.utils import TIME_DTYPE, CHANNEL_DTYPE

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        "cr_ma_std_coeff",
        type=np.float32,
        default=20.0,
        help=(
            "Coefficients applied to the moving averaged signal's "
            "standard deviation for identifying cosmic ray hits."
        ),
    ),
    strax.Option(
        "cr_convolved_std_coeff",
        type=np.float32,
        default=20.0,
        help=(
            "Coefficients applied to the convolved signal's "
            "standard deviation for identifying cosmic ray hits."
        ),
    ),
    strax.Option(
        "cr_min_ma_amplitude",
        type=np.float32,
        default=1.0,
        help=(
            "Minimum amplitude of the moving averaged signal's " "for identifying cosmic ray hits."
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
        self.cr_ma_std_coeff = self.config["cr_ma_std_coeff"]
        self.cr_convolved_std_coeff = self.config["cr_convolved_std_coeff"]
        self.cr_min_ma_amplitude = self.config["cr_min_ma_amplitude"]

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

        return base_dtype + hit_id_dtype

    def is_unidentified_hit(self, hits):
        hits["is_unidentified"] = hits["amplitude_convolved_max"] < hits["hit_threshold"]

    def is_cr_hit(self, hits):
        mask = hits["amplitude_convolved_max"] >= hits["hit_threshold"]
        mask &= hits["amplitude_convolved_max"] >= (
            hits["record_convolved_std"] * self.cr_convolved_std_coeff
        )
        mask |= (hits["amplitude_ma_max"] >= self.cr_min_ma_amplitude) & (
            hits["amplitude_ma_max"]
            >= (hits["record_ma_mean"] + hits["record_ma_std"] * self.cr_ma_std_coeff)
        )
        hits["is_cr"] = mask | hits["is_unidentified"]

    def is_symmetric_spike_hit(self, hits):
        pass

    def compute(self, hits):
        pass
