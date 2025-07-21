import numpy as np

# Common senses.
SECOND_TO_NANOSECOND = 1_000_000_000

# Common dtype constants for waveform records.
TIME_DTYPE = np.int64
LENGTH_DTYPE = np.int64
CHANNEL_DTYPE = np.int16
DATA_DTYPE = np.dtype(">f8")
INDEX_DTYPE = np.int32


def base_waveform_dtype():
    """Return the base dtype list for a waveform record, without the data fields.

    Returns:
        list: List of dtype tuples for the base waveform record fields.

    """
    return [
        (("Start time since unix epoch [ns]", "time"), TIME_DTYPE),
        (("Exclusive end time since unix epoch [ns]", "endtime"), TIME_DTYPE),
        (("Length of the interval in samples", "length"), LENGTH_DTYPE),
        (("Width of one sample [ns]", "dt"), TIME_DTYPE),
        (("Channel number defined by channel_map", "channel"), CHANNEL_DTYPE),
    ]
