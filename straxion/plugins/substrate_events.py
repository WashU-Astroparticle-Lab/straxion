import strax
import numpy as np
from straxion.constants import (
    TIME_DTYPE,
    CHANNEL_DTYPE,
    DATA_DTYPE,
    INDEX_DTYPE,
    SECOND_TO_NANOSECOND,
)
from straxion.colors import get_channel_position

export, __all__ = strax.exporter()

N_CHANNELS = 41


@export
@strax.takes_config(
    strax.Option(
        "fs",
        default=38_000,
        track=True,
        type=int,
        help="Sampling frequency (assumed the same for all channels) in unit of Hz",
    ),
    strax.Option(
        "substrate_min_spike_coincidence",
        default=5,
        track=True,
        type=int,
        help=(
            "Minimum number of channels with spikes coinciding with a hit "
            "(n_spikes_coinciding) for the hit to be considered substrate-like."
        ),
    ),
    strax.Option(
        "substrate_kappa_min_ms",
        default=0.5,
        track=True,
        type=float,
        help=(
            "Minimum double-sided exponential width kappa (converted from samples "
            "to milliseconds using fs) for a substrate-like hit."
        ),
    ),
    strax.Option(
        "substrate_kappa_max_ms",
        default=3.0,
        track=True,
        type=float,
        help=(
            "Maximum double-sided exponential width kappa (converted from samples "
            "to milliseconds using fs) for a substrate-like hit."
        ),
    ),
    strax.Option(
        "substrate_min_best_chi2",
        default=0.8,
        track=True,
        type=float,
        help=(
            "Minimum optimal-filter best chi-squared for a substrate-like hit. "
            "Substrate hits are template mismatches, so they have LARGE chi2; "
            "this cut removes well-fit (photon-like) waveforms."
        ),
    ),
    strax.Option(
        "substrate_exclude_truncated_hits",
        default=False,
        track=True,
        type=bool,
        help="If True, exclude truncated hits from substrate-like hit candidates.",
    ),
    strax.Option(
        "substrate_coincidence_window_ms",
        default=0.2,
        track=True,
        type=float,
        help=(
            "Maximum time gap between consecutive substrate-like hits for them to "
            "be clustered into the same substrate event, in milliseconds."
        ),
    ),
    strax.Option(
        "substrate_min_hit_coincidence",
        default=5,
        track=True,
        type=int,
        help="Minimum number of clustered substrate-like hits to form a substrate event.",
    ),
    strax.Option(
        "single_photon_dx_amplitude",
        default=1.54e-6,
        track=True,
        type=float,
        help=(
            "Raw dx amplitude corresponding to one single photon of energy "
            "single_photon_energy_mev. Used to convert the per-channel amplitude "
            "pattern into meV."
        ),
    ),
    strax.Option(
        "single_photon_energy_mev",
        default=50.0,
        track=True,
        type=float,
        help=(
            "Photon energy in meV corresponding to single_photon_dx_amplitude. "
            "Used to convert the per-channel amplitude pattern into meV."
        ),
    ),
    strax.Option(
        "of_amplitude_to_mev",
        default=49.64,
        track=True,
        type=float,
        help=(
            "Conversion factor from optimal-filter amplitude (best_aOF, normalized "
            "to the single-photon template) to energy in meV."
        ),
    ),
)
class SubstrateEvents(strax.OverlapWindowPlugin):
    """Cluster substrate-like hits across channels into substrate events.

    Substrate (phonon or radiation-impact) events deposit energy in the silicon
    substrate and are seen simultaneously by many KIDs. This plugin formalizes
    the substrate event finder used in the qualiphide_thz offline analysis
    (lee/single_si_phonon.ipynb, producing substrate_events.npy):

    1. Select substrate-like hits from hit_classification: hits coincident with
       spikes in at least substrate_min_spike_coincidence channels, with kappa
       inside [substrate_kappa_min_ms, substrate_kappa_max_ms], best_chi2 above
       substrate_min_best_chi2 (poor template fit, i.e. NOT photon-like), and
       not flagged as symmetric spike, invalid kappa, or photon candidate.
    2. Cluster the selected hits in time: consecutive hits (across all
       channels) closer than substrate_coincidence_window_ms belong to the same
       cluster; clusters with at least substrate_min_hit_coincidence hits
       become substrate events.
    3. Summarize each event: timing, per-channel amplitude and best_aOF
       patterns (and their meV calibrations), summed energy, the max-energy
       channel and its (x, y) position, and the distance from the max-energy
       position to every channel.

    If a channel contributes more than one hit to a cluster, the hit with the
    largest raw dx amplitude represents that channel in both patterns.

    Provides:
    - substrate_events: One row per substrate event.

    """

    __version__ = "0.0.0"

    depends_on = ("hit_classification",)
    provides = "substrate_events"
    data_kind = "substrate_events"
    save_when = strax.SaveWhen.ALWAYS

    rechunk_on_save = False
    compressor = "zstd"

    def infer_dtype(self):
        base_dtype = [
            (("Start time of the event (first hit) since unix epoch [ns]", "time"), TIME_DTYPE),
            (("Exclusive end time of the event (last hit) [ns]", "endtime"), TIME_DTYPE),
            (
                ("Time of the hit with the largest dx amplitude [ns]", "max_amplitude_time"),
                TIME_DTYPE,
            ),
            (("Number of substrate-like hits clustered into the event", "n_hits"), INDEX_DTYPE),
        ]

        pattern_dtype = [
            (
                ("Per-channel maximum dx amplitude pattern", "pattern_amplitude"),
                DATA_DTYPE,
                (N_CHANNELS,),
            ),
            (
                ("Per-channel best optimal-filter amplitude pattern", "pattern_best_aOF"),
                DATA_DTYPE,
                (N_CHANNELS,),
            ),
            (
                ("Per-channel energy pattern from dx amplitude [meV]", "pattern_energy_mev"),
                DATA_DTYPE,
                (N_CHANNELS,),
            ),
            (
                (
                    "Per-channel energy pattern from optimal-filter amplitude [meV]",
                    "pattern_energy_of_mev",
                ),
                DATA_DTYPE,
                (N_CHANNELS,),
            ),
            (
                ("Summed collected energy from dx amplitudes [meV]", "summed_energy_mev"),
                DATA_DTYPE,
            ),
            (
                (
                    "Summed collected energy from optimal-filter amplitudes [meV]",
                    "summed_energy_of_mev",
                ),
                DATA_DTYPE,
            ),
        ]

        location_dtype = [
            (
                ("Channel with the largest amplitude-based energy", "max_energy_channel"),
                CHANNEL_DTYPE,
            ),
            (
                (
                    "Channel with the largest optimal-filter-based energy",
                    "max_energy_channel_of",
                ),
                CHANNEL_DTYPE,
            ),
            (
                ("(x, y) position of max_energy_channel [mm]", "max_energy_position"),
                DATA_DTYPE,
                (2,),
            ),
            (
                ("(x, y) position of max_energy_channel_of [mm]", "max_energy_position_of"),
                DATA_DTYPE,
                (2,),
            ),
            (
                (
                    "Distance from max_energy_position to every channel [mm]",
                    "all_distances_mm",
                ),
                DATA_DTYPE,
                (N_CHANNELS,),
            ),
        ]

        return base_dtype + pattern_dtype + location_dtype

    def setup(self):
        self.coincidence_window_ns = (
            self.config["substrate_coincidence_window_ms"] / 1e3 * SECOND_TO_NANOSECOND
        )
        self.min_hit_coincidence = self.config["substrate_min_hit_coincidence"]
        # All channel positions, shape (N_CHANNELS, 2), in mm.
        self.channel_positions = get_channel_position(np.arange(N_CHANNELS))

    def get_window_size(self):
        # Clusters extend at most ~min_hit_coincidence * window; a generous
        # multiple of the coincidence window is still tiny compared to chunks.
        return int(100 * self.coincidence_window_ns)

    @staticmethod
    def select_substrate_hits(hits, config):
        """Return the boolean mask of substrate-like hits.

        Args:
            hits (np.ndarray): hit_classification array.
            config (dict): Plugin configuration.

        Returns:
            np.ndarray: Boolean mask of hits passing the substrate-like selection.

        """
        kappa_ms = hits["kappa"] / config["fs"] * 1e3
        mask = (
            (hits["n_spikes_coinciding"] >= config["substrate_min_spike_coincidence"])
            & (kappa_ms > config["substrate_kappa_min_ms"])
            & (kappa_ms < config["substrate_kappa_max_ms"])
            & (hits["best_chi2"] > config["substrate_min_best_chi2"])
            & ~hits["is_symmetric_spike"]
            & ~hits["is_invalid_kappa"]
            & ~hits["is_photon_candidate"]
        )
        if config["substrate_exclude_truncated_hits"]:
            mask &= ~hits["is_truncated_hit"]
        return mask

    @staticmethod
    def find_clusters(times, window_ns, min_coincidence):
        """Assign cluster labels to a sorted time series.

        Consecutive times closer than or equal to window_ns are grouped into
        the same cluster.

        Args:
            times (np.ndarray): Sorted times in ns.
            window_ns (float): Maximum gap between consecutive times inside a cluster.
            min_coincidence (int): Minimum number of times required in a cluster.

        Returns:
            list of np.ndarray: Each array contains the indices of one cluster
                with at least min_coincidence members, in time order.

        """
        times = np.asarray(times)
        if len(times) > 1 and not np.all(np.diff(times) >= 0):
            raise ValueError("times array must be sorted")
        if len(times) == 0:
            return []

        # New cluster starts wherever the gap to the previous time exceeds the window.
        cluster_ids = np.zeros(len(times), dtype=np.int64)
        cluster_ids[1:] = np.cumsum(np.diff(times) > window_ns)

        clusters = []
        for start, end in zip(
            np.searchsorted(cluster_ids, np.arange(cluster_ids[-1] + 1), side="left"),
            np.searchsorted(cluster_ids, np.arange(cluster_ids[-1] + 1), side="right"),
        ):
            if end - start >= min_coincidence:
                clusters.append(np.arange(start, end))
        return clusters

    def build_event(self, event, cluster_hits):
        """Fill one substrate event row from its clustered hits.

        Args:
            event (np.void): One row of the substrate_events array to fill in place.
            cluster_hits (np.ndarray): The substrate-like hits of this cluster.

        """
        event["time"] = cluster_hits["time"][0]
        event["endtime"] = cluster_hits["endtime"].max()
        event["n_hits"] = len(cluster_hits)

        max_hit_index = int(np.argmax(cluster_hits["amplitude"]))
        event["max_amplitude_time"] = cluster_hits["time"][max_hit_index]

        # Per-channel patterns; if a channel fired more than once, the hit with
        # the largest dx amplitude represents it in both patterns.
        order = np.argsort(cluster_hits["amplitude"])
        channels = cluster_hits["channel"][order]
        event["pattern_amplitude"][channels] = cluster_hits["amplitude"][order]
        event["pattern_best_aOF"][channels] = cluster_hits["best_aOF"][order]

        event["pattern_energy_mev"] = (
            event["pattern_amplitude"]
            / self.config["single_photon_dx_amplitude"]
            * self.config["single_photon_energy_mev"]
        )
        event["pattern_energy_of_mev"] = (
            event["pattern_best_aOF"] * self.config["of_amplitude_to_mev"]
        )
        event["summed_energy_mev"] = event["pattern_energy_mev"].sum()
        event["summed_energy_of_mev"] = event["pattern_energy_of_mev"].sum()

        event["max_energy_channel"] = event["pattern_energy_mev"].argmax()
        event["max_energy_channel_of"] = event["pattern_energy_of_mev"].argmax()
        max_energy_position = self.channel_positions[event["max_energy_channel"]]
        event["max_energy_position"] = max_energy_position
        event["max_energy_position_of"] = self.channel_positions[event["max_energy_channel_of"]]
        event["all_distances_mm"] = np.sqrt(
            np.sum((self.channel_positions - max_energy_position) ** 2, axis=1)
        )

    def compute(self, hits):
        substrate_hits = hits[self.select_substrate_hits(hits, self.config)]
        substrate_hits = substrate_hits[np.argsort(substrate_hits["time"], kind="stable")]

        clusters = self.find_clusters(
            substrate_hits["time"],
            self.coincidence_window_ns,
            self.min_hit_coincidence,
        )

        events = np.zeros(len(clusters), dtype=self.infer_dtype())
        for event, cluster_indices in zip(events, clusters):
            self.build_event(event, substrate_hits[cluster_indices])

        return events
