import strax
import numpy as np
from straxion.utils import (
    TIME_DTYPE,
    CHANNEL_DTYPE,
    DATA_DTYPE,
    SECOND_TO_NANOSECOND,
    PHOTON_25um_meV,
    PHOTON_25um_DX,
    DX_RESOL_OPTIMISTIC,
    DX_RESOL_CONSERVATIVE,
    PULSE_TEMPLATE_LENGTH,
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
    strax.Option(
        "energy_resolution_mode",
        default="optimistic",
        track=True,
        type=str,
        help=(
            "Energy resolution mode: 'optimistic', 'conservative', "
            "or 'none' for no resolution smearing."
        ),
    ),
    strax.Option(
        "fs",
        default=38_000,
        track=True,
        type=int,
        help="Sampling frequency (assumed the same for all channels) in unit of Hz",
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
            (("True dx value in dx units", "dx_true"), DATA_DTYPE),
            (("Channel number where event occurred", "channel"), CHANNEL_DTYPE),
        ]
        return dtype

    def setup(self):
        """Initialize random number generator and time interval."""
        self.rng = np.random.default_rng(self.config["random_seed"])
        # Time interval between events in nanoseconds
        if self.config["salt_rate"] > 0:
            self.dt_salt = int(SECOND_TO_NANOSECOND / self.config["salt_rate"])
        else:
            self.dt_salt = None
        self.dt_exact = 1 / self.config["fs"] * SECOND_TO_NANOSECOND

    @staticmethod
    def sigma_deltax(photon_energy_meV, sigma_deltax_sph):
        """Calculate resolution in dx units."""
        return np.sqrt(photon_energy_meV / PHOTON_25um_meV) * sigma_deltax_sph

    @staticmethod
    def meV_to_dx(photon_energy_meV):
        """Convert photon energy from meV to dx units."""
        return PHOTON_25um_DX * photon_energy_meV / PHOTON_25um_meV

    @staticmethod
    def dx_to_meV(dx):
        """Convert from dx units to meV."""
        return dx / PHOTON_25um_DX * PHOTON_25um_meV

    def sigma_E(self, photon_energy_meV, sigma_deltax_sph):
        """Calculate energy resolution in meV."""
        return self.dx_to_meV(self.sigma_deltax(photon_energy_meV, sigma_deltax_sph))

    @staticmethod
    def gaussian(x, mean, std):
        """Gaussian/Normal distribution probability density function.

        Parameters:
        -----------
        x : float or array-like
            Input value(s) at which to evaluate the Gaussian
        mean : float
            Mean (center) of the distribution
        std : float
            Standard deviation (width) of the distribution

        Returns:
        --------
        float or array
            Probability density at x

        """
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    def f_s(self, E_rec, E_true, mode="optimistic"):
        """Energy resolution function for dark photon signal.

        Parameters:
        -----------
        E_rec : float or array-like
            Reconstructed energy in meV
        E_true : float
            True energy in meV
        mode : str
            Resolution mode: 'optimistic' or 'conservative'

        Returns:
        --------
        float or array
            Probability density at E_rec

        """
        if mode == "optimistic":
            sigma_sph = DX_RESOL_OPTIMISTIC
        elif mode == "conservative":
            sigma_sph = DX_RESOL_CONSERVATIVE
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return self.gaussian(E_rec, E_true, self.sigma_E(E_true, sigma_deltax_sph=sigma_sph))

    def sample_energy(self, E_true, mode):
        """Sample energy from resolution function.

        Parameters:
        -----------
        E_true : float
            True energy in meV
        mode : str
            Resolution mode: 'optimistic', 'conservative', or 'none'

        Returns:
        --------
        float
            Sampled energy in meV

        """
        if mode == "none":
            return E_true

        if mode == "optimistic":
            sigma_sph = DX_RESOL_OPTIMISTIC
        elif mode == "conservative":
            sigma_sph = DX_RESOL_CONSERVATIVE
        else:
            raise ValueError(
                f"Invalid mode: {mode}. " "Must be 'optimistic', 'conservative', or 'none'."
            )

        sigma = self.sigma_E(E_true, sigma_deltax_sph=sigma_sph)
        return self.rng.normal(E_true, sigma)

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
        mode = self.config["energy_resolution_mode"]
        for i in range(n_events):
            results["time"][i] = time_start + i * self.dt_salt
            results["endtime"][i] = results["time"][i] + self.dt_exact * PULSE_TEMPLATE_LENGTH
            # Sample energy from resolution function
            results["energy_true"][i] = self.sample_energy(self.config["energy_meV"], mode)
            # Convert energy to dx units
            results["dx_true"][i] = self.meV_to_dx(results["energy_true"][i])
            # Randomly select a channel
            results["channel"][i] = self.rng.choice(available_channels)

        # Return as a chunk
        return self.chunk(
            start=time_start,
            end=time_end,
            data=results,
            data_type="truth",
        )
