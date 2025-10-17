import strax
import numpy as np
from straxion.utils import (
    TIME_DTYPE,
    CHANNEL_DTYPE,
    DATA_DTYPE,
    INDEX_DTYPE,
    NOT_FOUND_INDEX,
)

export, __all__ = strax.exporter()


@export
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

    __version__ = "0.0.1"

    depends_on = ("truth", "hits", "hit_classification")
    provides = "match"
    data_kind = "truth"
    save_when = strax.SaveWhen.ALWAYS

    rechunk_on_save = False
    compressor = "zstd"

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
