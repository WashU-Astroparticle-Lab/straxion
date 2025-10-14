import strax
import numpy as np
from straxion.utils import (
    TIME_DTYPE,
    CHANNEL_DTYPE,
    DATA_DTYPE,
    SECOND_TO_NANOSECOND,
)

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        "random_seed",
        default=137,
        track=True,
        type=int,
        help="Seed for random generator used for channel selection.",
    ),
    strax.Option(
        "salt_rate",
        default=0,
        track=True,
        type=(int, float),
        help="Rate of SALT events in unit of Hz.",
    ),
    strax.Option(
        "energy_meV",
        default=50,
        track=True,
        type=(int, float),
        help="Photon energy in unit of meV.",
    ),
)
class Truth(strax.Plugin):
    """Generate ground truth SALT events for simulation.

    This plugin creates simulated SALT (Some Arbitrary Light Transient)
    events at a constant rate by randomly selecting channels. The events
    are uniformly distributed in time and channel according to the
    specified rate.

    Provides:
    - truth: Ground truth SALT events with timing and energy information.

    """

    __version__ = "0.0.0"

    depends_on = ("raw_records",)
    provides = "truth"
    data_kind = "truth"
    save_when = strax.SaveWhen.ALWAYS

    rechunk_on_save = False
    compressor = "zstd"

    def infer_dtype(self):
        """Define the data type for truth events."""
        dtype = [
            (("Start time since unix epoch [ns]", "time"), TIME_DTYPE),
            (("Exclusive end time since unix epoch [ns]", "endtime"), TIME_DTYPE),
            (("True energy of the photon in meV", "energy_true"), DATA_DTYPE),
            (("Channel number where event occurred", "channel"), CHANNEL_DTYPE),
        ]
        return dtype

    def setup(self):
        """Initialize random number generator and time interval."""
        self.rng = np.random.default_rng(self.config["random_seed"])
        # Time interval between events in nanoseconds
        self.dt_salt = int(SECOND_TO_NANOSECOND / self.config["salt_rate"])

    def compute(self, raw_records):
        """Generate truth events based on raw_records time range.

        Args:
            raw_records: Array of raw record data.

        Returns:
            strax.Chunk: Truth events with timing and channel info.

        """
        # Get the time range from raw_records
        time_start = np.min(raw_records["time"])
        time_end = np.max(raw_records["endtime"])

        # Find available channels
        available_channels = np.unique(raw_records["channel"])

        # Calculate number of events based on time range and rate
        time_duration_ns = time_end - time_start
        n_events = int(time_duration_ns / self.dt_salt)

        if n_events == 0:
            # No events to generate, return empty array
            return self.chunk(
                start=time_start,
                end=time_end,
                data=np.zeros(0, dtype=self.infer_dtype()),
                data_type="truth",
            )

        # Initialize results array
        results = np.zeros(n_events, dtype=self.infer_dtype())

        # Generate events at constant time intervals
        for i in range(n_events):
            results["time"][i] = time_start + i * self.dt_salt
            results["endtime"][i] = results["time"][i] + self.dt_salt
            results["energy_true"][i] = self.config["energy_meV"]
            # Randomly select a channel
            results["channel"][i] = self.rng.choice(available_channels)

        # Return as a chunk
        return self.chunk(
            start=time_start,
            end=time_end,
            data=results,
            data_type="truth",
        )
