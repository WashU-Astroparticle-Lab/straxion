import strax
import numpy as np
from straxion.utils import (
    TIME_DTYPE,
    CHANNEL_DTYPE,
    DATA_DTYPE,
    INDEX_DTYPE,
    NOT_FOUND_INDEX,
    SECOND_TO_NANOSECOND,
    PULSE_TEMPLATE_ARGMAX,
    PULSE_TEMPLATE_LENGTH,
    DEFAULT_TEMPLATE_INTERP_PATH,
    TEMPLATE_INTERP_FOLDER,
    load_interpolation,
)
import os

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        "match_window_ms",
        default=1.5,
        track=True,
        type=(int, float, type(None)),
        help=(
            "Time window around waveform maximum for matching, in "
            "milliseconds. If None, uses full hit/truth time ranges."
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
        "template_interp_path",
        type=str,
        default=DEFAULT_TEMPLATE_INTERP_PATH,
        track=False,
        help="Path to the saved template interpolation file.",
    ),
    strax.Option(
        "template_interp_folder",
        type=str,
        default=TEMPLATE_INTERP_FOLDER,
        track=False,
        help="Folder containing per-channel template interpolation files.",
    ),
)
class Match(strax.Plugin):
    """Match ground truth SALT events with detected hits.

    This plugin correlates ground truth SALT events with detected hits
    based on temporal overlap. For each truth event, it identifies
    corresponding hits in the same channel that overlap in time.

    Matching algorithm:
    1. For each channel, use strax.touching_windows to find hits that
       temporally overlap with truth events.
    2. Categorize each truth event by its destiny:
       - "found": Exactly one hit overlaps with the truth event.
       - "lost": No hit overlaps with the truth event.
       - "split": Multiple hits overlap with the truth event. In this
         case, the hit with minimal distance in amplitude_max_record_i
         compared to the truth's is selected.
    3. For "found" and "split" cases, record hit properties and
       classification results.
    4. For "lost" cases, set hit-related fields to zero/default values.

    Provides:
    - match: Truth events matched with hits and their classifications.

    """

    __version__ = "0.0.4"

    depends_on = ("truth", "hits", "hit_classification")
    provides = "match"
    data_kind = "truth"
    save_when = strax.SaveWhen.ALWAYS

    rechunk_on_save = False
    compressor = "zstd"

    def setup(self):
        """Initialize time conversion factor."""
        self.dt_exact = 1 / self.config["fs"] * SECOND_TO_NANOSECOND
        match_window_ms = self.config["match_window_ms"]
        if match_window_ms is not None:
            self.match_window_ns = match_window_ms * 1_000_000
        else:
            self.match_window_ns = None
        # Calculate pulse template argmax for current sampling frequency
        # We need to recreate the same interpolation process used in records.py
        # to find where the maximum actually occurs in the interpolated template
        At_interp, t_max = load_interpolation(self.config["template_interp_path"])
        dt_seconds = 1.0 / self.config["fs"]
        t_seconds = np.arange(PULSE_TEMPLATE_LENGTH) * dt_seconds
        t_max_target = PULSE_TEMPLATE_ARGMAX * dt_seconds
        time_shift = t_max_target - t_max
        timeshifted_seconds = t_seconds - time_shift
        interpolated_template = At_interp(timeshifted_seconds)
        self.pulse_template_argmax = np.argmax(interpolated_template)

        # Load per-channel templates if folder exists and compute per-channel argmax
        self.pulse_template_argmax_dict = {}
        template_interp_folder = self.config["template_interp_folder"]
        if os.path.isdir(template_interp_folder):
            for file in os.listdir(template_interp_folder):
                if file.endswith(".pkl"):
                    ch = int(file.split("_")[0].split("ch")[1])
                    At_interp_ch, t_max_ch = load_interpolation(
                        os.path.join(template_interp_folder, file)
                    )
                    time_shift_ch = t_max_target - t_max_ch
                    timeshifted_seconds_ch = t_seconds - time_shift_ch
                    interpolated_template_ch = At_interp_ch(timeshifted_seconds_ch)
                    self.pulse_template_argmax_dict[ch] = np.argmax(interpolated_template_ch)

    def infer_dtype(self):
        """Define the data type for match results."""
        dtype = [
            (("Start time since unix epoch [ns]", "time"), TIME_DTYPE),
            (
                ("Exclusive end time since unix epoch [ns]", "endtime"),
                TIME_DTYPE,
            ),
            (("Channel number where event occurred", "channel"), CHANNEL_DTYPE),
            (("True energy of the photon in meV", "energy_true"), DATA_DTYPE),
            (("True dx value in dx units", "dx_true"), DATA_DTYPE),
            (
                (
                    "Destiny of the truth event: found, lost, or split",
                    "destiny",
                ),
                "U5",
            ),
            (
                (
                    ("Index of matched hit in hits array " "(NOT_FOUND_INDEX if not found)"),
                    "hit_index",
                ),
                INDEX_DTYPE,
            ),
            (
                ("Length of matched hit waveform in samples", "length"),
                INDEX_DTYPE,
            ),
            (
                ("Maximum amplitude of the matched hit", "amplitude"),
                DATA_DTYPE,
            ),
            (
                (
                    "Maximum amplitude of matched hit (moving average)",
                    "amplitude_moving_average",
                ),
                DATA_DTYPE,
            ),
            (
                (
                    "Maximum amplitude of matched hit (convolved)",
                    "amplitude_convolved",
                ),
                DATA_DTYPE,
            ),
            (
                ("Hit finding threshold for matched hit", "hit_threshold"),
                DATA_DTYPE,
            ),
            (
                (
                    "Width of matched hit (length above threshold)",
                    "width",
                ),
                INDEX_DTYPE,
            ),
            (
                (
                    "Rise edge slope of matched hit",
                    "rise_edge_slope",
                ),
                DATA_DTYPE,
            ),
            (
                (
                    "Number of spikes coinciding with matched hit",
                    "n_spikes_coinciding",
                ),
                INDEX_DTYPE,
            ),
            (
                (
                    "Is matched hit a photon candidate",
                    "is_photon_candidate",
                ),
                bool,
            ),
            (
                (
                    "Is matched hit a symmetric spike",
                    "is_symmetric_spike",
                ),
                bool,
            ),
            (
                (
                    "Is matched hit coincident with spikes",
                    "is_coincident_with_spikes",
                ),
                bool,
            ),
            (
                (
                    "Best optimal filter amplitude for matched hit",
                    "best_aOF",
                ),
                DATA_DTYPE,
            ),
            (
                (
                    "Best chi-squared value from optimal filter for matched hit",
                    "best_chi2",
                ),
                DATA_DTYPE,
            ),
            (
                (
                    "Best time shift in samples for optimal filter of matched hit",
                    "best_OF_shift",
                ),
                INDEX_DTYPE,
            ),
        ]
        return dtype

    def compute(self, truth, hits):
        """Match truth events with hits based on temporal overlap.

        Args:
            truth: Array of ground truth SALT events.
            hits: Array of detected hits with classification fields merged.

        Returns:
            np.ndarray: Array of match results with same length as truth.

        Note:
            Since both hits and hit_classification have data_kind="hits",
            strax automatically merges their fields into a single array.

        """
        # Initialize results array
        results = np.zeros(len(truth), dtype=self.infer_dtype())

        # Copy truth fields
        results["time"] = truth["time"]
        results["endtime"] = truth["endtime"]
        results["channel"] = truth["channel"]
        results["energy_true"] = truth["energy_true"]
        results["dx_true"] = truth["dx_true"]

        # Initialize hit_index to NOT_FOUND_INDEX (not found)
        results["hit_index"] = NOT_FOUND_INDEX

        # Initialize destiny to "lost"
        results["destiny"] = "lost"

        # Process each channel separately
        for ch in np.unique(truth["channel"]):
            # Get indices for this channel
            ind_hits_ch = np.where(hits["channel"] == ch)[0]
            ind_truth_ch = np.where(truth["channel"] == ch)[0]

            if len(ind_hits_ch) == 0:
                # No hits in this channel, all truths are "lost"
                continue

            hits_ch = hits[ind_hits_ch]
            truth_ch = truth[ind_truth_ch]

            # Optionally restrict time ranges around waveform maxima
            if self.match_window_ns is not None:
                hits_ch, truth_ch = self._restrict_to_maximum_window(hits_ch, truth_ch)

            # Find temporal overlaps
            touching_windows_ch = strax.touching_windows(things=hits_ch, containers=truth_ch)

            # Calculate number of overlapping hits per truth
            n_overlaps = np.diff(touching_windows_ch, axis=1).flatten()

            # Process each truth in this channel
            for i, n_overlap in enumerate(n_overlaps):
                truth_idx = ind_truth_ch[i]

                if n_overlap == 0:
                    # Lost: no hit found
                    results["destiny"][truth_idx] = "lost"
                elif n_overlap == 1:
                    # Found: exactly one hit
                    results["destiny"][truth_idx] = "found"
                    hit_idx = ind_hits_ch[touching_windows_ch[i, 0]]
                    self._fill_hit_info(results[truth_idx], hits[hit_idx], hit_idx)
                else:
                    # Split: multiple hits, choose closest in amplitude_max_record_i
                    results["destiny"][truth_idx] = "split"
                    overlapping_hit_indices = ind_hits_ch[
                        touching_windows_ch[i, 0] : touching_windows_ch[i, 1]
                    ]
                    overlapping_hits = hits[overlapping_hit_indices]

                    # Find hit with minimal distance in amplitude_max_record_i
                    truth_amp_max_i = truth[truth_idx]["amplitude_max_record_i"]
                    distances = np.abs(overlapping_hits["amplitude_max_record_i"] - truth_amp_max_i)
                    min_dist_idx = np.argmin(distances)
                    hit_idx = overlapping_hit_indices[min_dist_idx]

                    self._fill_hit_info(results[truth_idx], hits[hit_idx], hit_idx)

        return results

    def _restrict_to_maximum_window(self, hits_ch, truth_ch):
        """Restrict time ranges to window around waveform maximum.

        Creates views with modified time/endtime fields centered around
        the waveform maximum.

        Args:
            hits_ch: Array of hits for a channel.
            truth_ch: Array of truth events for a channel.

        Returns:
            Tuple of (hits_ch_restricted, truth_ch_restricted) with
            modified time/endtime fields.
        """
        # Make copies to avoid modifying original data
        hits_ch_restricted = hits_ch.copy()
        truth_ch_restricted = truth_ch.copy()

        half_window_ns = self.match_window_ns / 2

        # For truth: The maximum occurs at pulse_template_argmax samples
        # from the start of the pulse template (at the current sampling
        # frequency). Since truth["time"] is the start of the pulse template,
        # we can calculate the maximum time.
        # If the template interpolation file for this channel exists, use it,
        # otherwise use the default one.
        if len(truth_ch) > 0:
            ch = truth_ch["channel"][0]  # All truth in this array have same channel
            if ch in self.pulse_template_argmax_dict.keys():
                pulse_argmax = self.pulse_template_argmax_dict[ch]
            else:
                pulse_argmax = self.pulse_template_argmax
        else:
            pulse_argmax = self.pulse_template_argmax
        truth_max_times = truth_ch["time"] + pulse_argmax * self.dt_exact
        truth_ch_restricted["time"] = np.maximum(
            truth_ch["time"],
            truth_max_times - half_window_ns,
        ).astype(TIME_DTYPE)
        truth_ch_restricted["endtime"] = np.minimum(
            truth_ch["endtime"],
            truth_max_times + half_window_ns,
        ).astype(TIME_DTYPE)

        # For hits: Find the actual maximum position in each hit's waveform
        # Note: hit["time"] corresponds to the start of actual data (left_i),
        # not the start of the padded waveform array. The waveform array may
        # have padding (zeros) at the beginning. We need to find the first
        # non-zero sample to determine the padding offset.
        hit_max_sample_indices = np.argmax(hits_ch["data_dx"], axis=1)
        # Find first non-zero sample index for each hit to account for padding
        # hit["time"] corresponds to when the actual data starts (target_start
        # in waveform array), not index 0
        waveform_data = hits_ch["data_dx"]
        n_hits = len(hits_ch)
        padding_offsets = np.zeros(n_hits, dtype=np.int32)
        for i in range(n_hits):
            non_zero_indices = np.nonzero(waveform_data[i])[0]
            if len(non_zero_indices) > 0:
                padding_offsets[i] = non_zero_indices[0]
        # Adjust max indices: remove padding offset since hit["time"] already
        # accounts for it
        hit_max_sample_indices_adjusted = hit_max_sample_indices - padding_offsets
        hit_max_times = (hits_ch["time"] + hit_max_sample_indices_adjusted * hits_ch["dt"]).astype(
            TIME_DTYPE
        )
        hits_ch_restricted["time"] = np.maximum(
            hits_ch["time"],
            hit_max_times - half_window_ns,
        ).astype(TIME_DTYPE)
        hits_ch_restricted["endtime"] = np.minimum(
            hits_ch["endtime"],
            hit_max_times + half_window_ns,
        ).astype(TIME_DTYPE)

        return hits_ch_restricted, truth_ch_restricted

    def _fill_hit_info(self, result, hit, hit_idx):
        """Fill match result with hit information.

        Args:
            result: Single match result element to populate.
            hit: Single hit element with merged classification fields.
            hit_idx: Index of the hit in the full hits array.

        """
        result["hit_index"] = hit_idx
        result["length"] = hit["length"]
        result["amplitude"] = hit["amplitude"]
        result["amplitude_moving_average"] = hit["amplitude_moving_average"]
        result["amplitude_convolved"] = hit["amplitude_convolved"]
        result["hit_threshold"] = hit["hit_threshold"]
        result["width"] = hit["width"]
        result["rise_edge_slope"] = hit["rise_edge_slope"]
        result["n_spikes_coinciding"] = hit["n_spikes_coinciding"]
        result["is_photon_candidate"] = hit["is_photon_candidate"]
        result["is_symmetric_spike"] = hit["is_symmetric_spike"]
        result["is_coincident_with_spikes"] = hit["is_coincident_with_spikes"]
        result["best_aOF"] = hit["best_aOF"]
        result["best_chi2"] = hit["best_chi2"]
        result["best_OF_shift"] = hit["best_OF_shift"]
