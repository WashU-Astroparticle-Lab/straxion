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
from straxion.substrate_reconstruction import (
    prompt_groups,
    timing_valid_mask,
    apply_veto,
)

export, __all__ = strax.exporter()

N_CHANNELS = 41


@export
@strax.takes_config(
    strax.Option(
        "substrate_time_basis",
        default="ma",
        type=str,
        track=True,
        help="Membership/veto basis: ma (default), start (legacy), raw, or convolved.",
    ),
    strax.Option(
        "substrate_veto_ms",
        default=0.0,
        type=float,
        track=True,
        help="Nonextending veto after last accepted member; zero preserves legacy default.",
    ),
    strax.Option(
        "substrate_seed_coincidence",
        default="hit_peaks",
        type=str,
        track=True,
        help=(
            "hit_peaks (MA detected hits, default) or record_threshold "
            "(explicit legacy comparison)."
        ),
    ),
    strax.Option(
        "substrate_min_peak_channels",
        default=5,
        type=int,
        track=True,
        help="Minimum distinct MA-peak-coincident channels for hit_peaks seeds.",
    ),
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
class SubstrateEvents(strax.Plugin):
    """Cluster substrate-like seeds using MA peak times by default.

    The hit_peaks seed uses actual detected-hit channel multiplicity, kappa,
    chi-square and shape quality; record_threshold is an explicit legacy option.
    Consecutive gaps at or below the configured gap join, including arbitrarily
    long chains. Group membership is retained before a nonextending optional veto.
    No energy cut or new truncation veto is applied.

    Framework intervals enclose original member windows, while separate fields
    describe member peak times. Repeated-channel patterns use the largest raw
    amplitude representative. Scalar inputs are buffered to run EOF so overlapping
    hit windows and peak sorting cannot violate a finite-overlap assumption.
    """

    __version__ = "0.1.1"

    depends_on = ("hit_classification",)
    provides = "substrate_events"
    data_kind = "substrate_events"
    save_when = strax.SaveWhen.ALWAYS

    parallel = False
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

        return (
            base_dtype
            + pattern_dtype
            + location_dtype
            + [
                ("group_id", "S64"),
                ("first_member_time", "i8"),
                ("last_member_time", "i8"),
                ("brightest_member_time", "i8"),
                ("brightest_hit_id", "S64"),
                ("n_channels", "i4"),
                ("n_repeated_channels", "i4"),
                ("max_hits_per_channel", "i4"),
            ]
        )

    def setup(self):
        self.coincidence_window_ns = (
            self.config["substrate_coincidence_window_ms"] / 1e3 * SECOND_TO_NANOSECOND
        )
        if (
            not np.isfinite(self.coincidence_window_ns)
            or self.coincidence_window_ns < 0
            or not np.isfinite(self.config["substrate_veto_ms"])
            or self.config["substrate_veto_ms"] < 0
        ):
            raise ValueError("Gap and veto must be finite and nonnegative")
        self.coincidence_window_ns = int(round(self.coincidence_window_ns))
        self.min_hit_coincidence = self.config["substrate_min_hit_coincidence"]
        # All channel positions, shape (N_CHANNELS, 2), in mm.
        self.channel_positions = get_channel_position(np.arange(N_CHANNELS))

    def iter(self, iters, executor=None):
        """Buffer scalar inputs to run EOF, avoiding any finite chain-span assumption.

        Acquisition keys keep groups/vetoes isolated inside the run. Full-run
        buffering permits arbitrary chunk fragmentation and peak-order reversals.
        """
        chunks = list(iters["hit_classification"])
        if not chunks:
            return
        if any(c.run_id != chunks[0].run_id or c.subruns for c in chunks):
            raise ValueError("Reconstruct ordinary acquisition runs before forming superruns")
        for before, after in zip(chunks[:-1], chunks[1:]):
            if before.end != after.start:
                raise ValueError("Input chunks must cover a contiguous run interval")
        data = np.concatenate([c.data for c in chunks])
        yield self.chunk(start=chunks[0].start, end=chunks[-1].end, data=self.compute(data))

    @staticmethod
    def select_substrate_hits(hits, config):
        """Return the boolean mask of substrate-like hits.

        Args:
            hits (np.ndarray): hit_classification array.
            config (dict): Plugin configuration.

        Returns:
            np.ndarray: Boolean mask of hits passing the substrate-like selection.

        """
        if "sampling_frequency_hz" in hits.dtype.names and np.any(
            hits["sampling_frequency_hz"] != config["fs"]
        ):
            raise ValueError("Hit sampling frequency differs from configured fs")
        kappa_ms = hits["kappa"] / config["fs"] * 1e3
        mode = config.get("substrate_seed_coincidence", "hit_peaks")
        if mode == "record_threshold":
            if "legacy_spike_coincidence_valid" in hits.dtype.names and not np.all(
                hits["legacy_spike_coincidence_valid"]
            ):
                raise ValueError(
                    "Legacy seeds require compute_legacy_spike_coincidence=True upstream"
                )
            coincidence = hits["n_spikes_coinciding"]
        elif mode == "hit_peaks":
            coincidence = hits["hit_coincidence_channels"]
        else:
            raise ValueError("Unknown substrate_seed_coincidence")
        legacy_photon = hits["is_photon_candidate"]
        if mode == "record_threshold" and "legacy_spike_coincidence_valid" in hits.dtype.names:
            legacy_photon = ~(
                hits["is_coincident_with_spikes"]
                | hits["is_symmetric_spike"]
                | hits["is_truncated_hit"]
                | hits["is_invalid_kappa"]
            )
        minimum = (
            config["substrate_min_spike_coincidence"]
            if mode == "record_threshold"
            else config["substrate_min_peak_channels"]
        )
        mask = (
            (coincidence >= minimum)
            & (kappa_ms > config["substrate_kappa_min_ms"])
            & (kappa_ms < config["substrate_kappa_max_ms"])
            & (hits["best_chi2"] > config["substrate_min_best_chi2"])
            & ~hits["is_symmetric_spike"]
            & ~hits["is_invalid_kappa"]
            & (~legacy_photon if mode == "record_threshold" else True)
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
        event["time"] = cluster_hits["time"].min()
        event["endtime"] = cluster_hits["endtime"].max()
        event["n_hits"] = len(cluster_hits)

        max_hit_index = int(np.argmax(cluster_hits["amplitude"]))
        event["max_amplitude_time"] = cluster_hits["time"][max_hit_index]

        # Per-channel patterns; if a channel fired more than once, the hit with
        # the largest dx amplitude represents it in both patterns.
        order = np.argsort(
            cluster_hits["amplitude"],
            kind=("quicksort" if self.config["substrate_time_basis"] == "start" else "stable"),
        )
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

    def reconstruct(self, hits):
        """Return all pre-veto groups, membership links, and a separate veto view."""
        seed_mask = self.select_substrate_hits(hits, self.config)
        basis = self.config["substrate_time_basis"]
        if basis in ("raw", "ma", "convolved"):
            invalid = seed_mask & ~timing_valid_mask(hits, basis)
            if np.any(invalid):
                import logging

                logging.getLogger(__name__).warning(
                    "Excluded %d substrate seeds with invalid %s peak timing",
                    np.count_nonzero(invalid),
                    basis,
                )
            seed_mask = seed_mask & ~invalid
        groups, members = prompt_groups(
            hits,
            seed_mask,
            basis=self.config["substrate_time_basis"],
            gap_ns=self.coincidence_window_ns,
            min_hits=self.min_hit_coincidence,
        )
        veto = apply_veto(groups, int(round(self.config["substrate_veto_ms"] * 1e6)))
        return groups, members, veto

    def compute(self, hits):
        groups, members, veto = self.reconstruct(hits)
        accepted = groups[veto["accepted"]]
        events = np.zeros(len(accepted), dtype=self.infer_dtype())
        membership = {}
        for member in members:
            membership.setdefault(member["group_id"], []).append(member["hit_index"])
        for event, group in zip(events, accepted):
            indices = membership[group["group_id"]]
            self.build_event(event, hits[indices])
            for name in (
                "group_id",
                "brightest_hit_id",
                "n_channels",
                "n_repeated_channels",
                "max_hits_per_channel",
            ):
                event[name] = group[name]
            event["first_member_time"] = group["first_time"]
            event["last_member_time"] = group["last_time"]
            event["brightest_member_time"] = group["brightest_time"]
        # Framework intervals describe original window support, never peak-order position.
        return events[np.argsort(events["time"], kind="stable")]
