"""Scalar hit contract. Native sample offsets use floor(index * 1e9 / fs).

Support is the half-open native-record interval actually copied to the padded
waveform. Counts exclude the reference sample; valid physical zeros count.
These geometric counts do not certify an uncontaminated tail or a separate pulse.
"""

import hashlib
import numpy as np

TIMING_DTYPE = [
    ("run_id", "S64"),
    ("hit_id", "S64"),
    ("record_time", "i8"),
    ("record_endtime", "i8"),
    ("candidate_record_i", "i8"),
    ("sampling_frequency_hz", "i8"),
    ("alignment_record_i", "i8"),
    ("alignment_time", "i8"),
    ("alignment_valid", "?"),
    ("support_start_record_i", "i8"),
    ("support_stop_record_i", "i8"),
    ("support_start_waveform_i", "i4"),
    ("support_stop_waveform_i", "i4"),
    ("clean_left_alignment", "i4"),
    ("clean_right_alignment", "i4"),
    ("intact25_alignment", "?"),
    ("legacy_nonzero_left_alignment", "i4"),
    ("legacy_nonzero_right_alignment", "i4"),
]
for _basis in ("raw", "ma", "convolved"):
    TIMING_DTYPE += [
        (f"peak_{_basis}_time", "i8"),
        (f"peak_{_basis}_record_i", "i8"),
        (f"peak_{_basis}_valid", "?"),
        (f"peak_{_basis}_at_left_boundary", "?"),
        (f"peak_{_basis}_at_right_boundary", "?"),
        (f"clean_left_{_basis}", "i4"),
        (f"clean_right_{_basis}", "i4"),
    ]

ESTIMATOR_DTYPE = [
    ("noise_window_count", "i4"),
    ("noise_psd_fallback", "?"),
    ("noise_psd_sha256", "S64"),
    ("legacy_spike_coincidence_valid", "?"),
    ("hit_coincidence", "i4"),
    ("hit_coincidence_channels", "i4"),
]


def psd_digest(psd):
    return hashlib.sha256(np.asarray(psd, dtype="<f8").tobytes()).hexdigest().encode()


def validate_records(records):
    """Reject layouts unsupported by the unchanged legacy estimators."""
    if not len(records):
        return
    if (
        not np.array_equal(records["channel"], np.arange(len(records)))
        or np.any(records["time"] != records["time"][0])
        or np.any(records["endtime"] != records["endtime"][0])
    ):
        raise ValueError("Expected one synchronous record per contiguous channel, in channel order")


def fill_timing(hits, record, candidates, aligned, left, right, fs, run_id):
    from straxion.constants import HIT_WINDOW_LENGTH_LEFT

    run = str(run_id).encode()
    if len(run) > 64 or fs <= 0:
        raise ValueError("run_id must fit 64 UTF-8 bytes and fs must be positive")
    origin = int(record["time"])
    hits["run_id"] = run
    hits["record_time"] = origin
    hits["record_endtime"] = record["endtime"]
    hits["candidate_record_i"] = candidates
    hits["sampling_frequency_hz"] = fs
    hits["alignment_record_i"] = aligned
    hits["alignment_time"] = origin + aligned * 1_000_000_000 // fs
    hits["alignment_valid"] = np.isfinite(record["data_dx_convolved"][aligned]) & (
        hits["alignment_time"] < record["endtime"]
    )
    hits["support_start_record_i"] = left
    hits["support_stop_record_i"] = right
    hits["support_start_waveform_i"] = HIT_WINDOW_LENGTH_LEFT - (aligned - left)
    hits["support_stop_waveform_i"] = HIT_WINDOW_LENGTH_LEFT + (right - aligned)
    hits["clean_left_alignment"] = aligned - left
    hits["clean_right_alignment"] = right - aligned - 1
    hits["intact25_alignment"] = (aligned - left >= 25) & (right - aligned - 1 >= 25)
    for i, hit in enumerate(hits):
        identity = f"{run.hex()}:{origin}:{int(record['channel'])}:{int(candidates[i])}"
        hit["hit_id"] = hashlib.sha256(identity.encode()).hexdigest().encode()
        wf = hit["data_dx"]
        for direction, values in (
            ("left", wf[:HIT_WINDOW_LENGTH_LEFT][::-1]),
            ("right", wf[HIT_WINDOW_LENGTH_LEFT + 1 :]),
        ):
            zeros = np.flatnonzero(values == 0)
            hit[f"legacy_nonzero_{direction}_alignment"] = zeros[0] if len(zeros) else len(values)
    for basis, index_field, waveform_field in (
        ("raw", "amplitude_max_record_i", "data_dx"),
        ("ma", "amplitude_moving_average_max_record_i", "data_dx_moving_average"),
        ("convolved", "amplitude_convolved_max_record_i", "data_dx_convolved"),
    ):
        indices = hits[index_field].astype(np.int64)
        hits[f"peak_{basis}_record_i"] = indices
        hits[f"peak_{basis}_time"] = origin + indices * 1_000_000_000 // fs
        hits[f"peak_{basis}_valid"] = [
            left[i] <= index < right[i]
            and origin + int(index) * 1_000_000_000 // fs < int(record["endtime"])
            and np.all(np.isfinite(record[waveform_field][left[i] : right[i]]))
            for i, index in enumerate(indices)
        ]
        hits[f"peak_{basis}_at_left_boundary"] = indices == left
        hits[f"peak_{basis}_at_right_boundary"] = indices == right - 1
        hits[f"clean_left_{basis}"] = indices - left
        hits[f"clean_right_{basis}"] = right - indices - 1
