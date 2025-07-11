import os
import warnings
from glob import glob
from typing import Tuple
from immutabledict import immutabledict
import strax
import numpy as np

export, __all__ = strax.exporter()


TIME_DTYPE = np.int64
LENGTH_DTYPE = np.int64
CHANNEL_DTYPE = np.int16
DATA_DTYPE = np.dtype(">f8")


@export
@strax.takes_config(
    strax.Option(
        "record_length",
        default=200_000_000,
        track=False,  # Not tracking record length, but we will have to check if it is as promised
        type=int,
        help=(
            "Number of samples in each dataset."
            "We assumed that each sample is equally spaced in time, with interval 1/fs."
            "It should not go beyond a billion so that numpy can still handle."
        ),
    ),
    strax.Option(
        "fs",
        default=100_000,
        track=True,
        type=int,
        help="Sampling frequency (assumed the same for all channels) in unit of Hz",
    ),
    strax.Option(
        "run_start_time",
        type=float,
        track=False,
        default=-1,
        help="Time of start run (in unit of seconds, since unix epoch).",
    ),
    strax.Option(
        "daq_input_dir",
        type=str,
        track=False,
        help="Directory where readers put data. For example: '/my/path/to/timeS66'.",
    ),
    strax.Option(
        "channel_map",
        track=False,
        type=immutabledict,
        infer_type=False,
        help="Immutabledict mapping subdetector to (min, max) channel number.",
    ),
    strax.Option(
        "sub_detector",
        track=True,
        type=str,
        default="kids",
        help="Name of the sub detector of interest (eg. 'kid').",
    ),
)
class DAQReader(strax.Plugin):
    """Read the raw data from citkid. It does nothing beyond reading data into a Python format.
    The functionality was adapted from "eConvertLVBinS.m":
    https://caltechobscosgroup.slack.com/archives/C07SZDKRNF9/p1752010145654029.
    The hyper-parametrization is inspired by straxen.DAQReader:
    https://github.com/XENONnT/straxen/blob/master/straxen/plugins/raw_records/daqreader.py.

    Important assumptions:
    - All channels share the same time stamps, and the clock synchronization is perfect.
    - Assumed no single dataset will be larger than 1 GB.
      Otherwise memory efficiency is too bad.

    Provides:
    - raw_records: the time series of I and Q directly read from raw data.

    """

    __version__ = "0.0.0"  # Should be very careful to bump the version of this plugin!

    # Data structure topology related:
    provides: str = "raw_records"
    data_kind = provides
    depends_on: Tuple = tuple()  # This is the lowest level of strax data in processing.

    # Memory management related:
    rechunk_on_load = False  # Assumed no single dataset will be larger than 1 GB.
    chunk_source_size_mb = strax.DEFAULT_CHUNK_SIZE_MB  # 200 MB; Neglected if no rechunk on load.
    rechunk_on_save = False  # Assumed no chunking at DAQ.
    chunk_target_size_mb = 1000  # Meaningless if rechunk_on_save is False.
    compressor = "lz4"  # Inherited from straxen. Not optimized outside XENONnT.

    def infer_dtype(self):
        """Data type for a waveform raw_record."""
        return [
            (("Start time since unix epoch [us]", "time"), TIME_DTYPE),
            (("Length of the interval in samples", "length"), LENGTH_DTYPE),
            (("Width of one sample [us]", "dt"), TIME_DTYPE),
            (("Channel number defined by channel_map", "channel"), CHANNEL_DTYPE),
            (
                ("Waveform data of I in raw ADC counts", "data_i"),
                DATA_DTYPE,
                self.config["record_length"],
            ),
            (
                ("Waveform data of Q in raw ADC counts", "data_q"),
                DATA_DTYPE,
                self.config["record_length"],
            ),
        ]

    def setup(self):
        self.dt = int(1 / self.config["fs"] * 1e6)  # In unit of us.

    def load_one_channel(self, file_path):
        """Reads and polynomially corrects raw binary DAQ data of a certain readout channel.

        Workflow:
        1. Reads the file header to determine the number of coefficients and IQ channels.
        2. Checks that there are exactly two IQ channels (I and Q).
        3. Reads and reshapes the calibration coefficient matrix.
        4. Reads the raw int16 time-series data and checks its shape.
        5. Applies polynomial calibration to convert raw ADC counts to physical units.
        6. Constructs a time vector based on the sampling interval.
        7. Returns a structured array with fields 'time', 'data_i', and 'data_q'.

        Args:
            file_path (str): Path to the binary DAQ file.

        Returns:
            np.ndarray: Structured array with fields 'time', 'I', and 'Q'.

        """

        def _read_header(f):
            header = np.fromfile(f, dtype=">d", count=2)
            n_coeffs = int(header[0])
            n_iq_channels = int(header[1])
            if n_iq_channels != 2:
                raise ValueError(
                    f"Expected 2 channels (I and Q), but found {n_iq_channels}. "
                    "There should be only two channels: I and Q."
                )
            return n_coeffs

        def _read_coefficients(f, n_coeffs):
            coeffs = np.fromfile(f, dtype=">d", count=n_coeffs * 2)
            coeffs = coeffs.reshape((n_coeffs, 2), order="F")
            offsets = coeffs[1, :]  # DC offsets
            gains = coeffs[2:, :]  # Polynomial coefficients
            return coeffs, offsets, gains

        def _check_n_samples(data_size, record_length):
            if data_size % 2 != 0:
                raise ValueError(
                    f"Data size ({data_size}) is not divisible by 2. "
                    "Data of different channels may have different lengths!"
                )
            n_samples_found = data_size // 2
            if record_length is not None and n_samples_found != record_length:
                raise ValueError(
                    f"The data length is promised to be {record_length} "
                    f"but found to be {n_samples_found}."
                )
            return n_samples_found

        def _read_data(f, record_length=None):
            data_int16 = np.fromfile(f, dtype=">i2")
            n_samples = _check_n_samples(data_int16.size, record_length)
            data_int16 = data_int16.reshape((2, n_samples), order="F")
            return data_int16, n_samples

        def _apply_poly_correction(data_int16, offsets, gains):
            _, n_samples = data_int16.shape
            b = np.tile(offsets[:, np.newaxis], (1, n_samples))
            for k in range(gains.shape[0]):
                exponent = k + 1
                term = gains[k, :][:, np.newaxis] * (data_int16**exponent)
                b += term
            return b

        def _make_time_vector(n_samples):
            return np.arange(n_samples) * self.dt

        with open(file_path, "rb") as f:
            try:
                n_coeffs = _read_header(f)
                _, offsets, gains = _read_coefficients(f, n_coeffs)
                data_int16, n_samples = _read_data(f, self.config["record_length"])
                calibrated = _apply_poly_correction(data_int16, offsets, gains)
                time_vector = _make_time_vector(n_samples)
                dtype = [("time", np.float64), ("data_i", DATA_DTYPE), ("data_q", DATA_DTYPE)]
                structured = np.zeros(n_samples, dtype=dtype)
                structured["time"] = time_vector
                structured["data_i"] = calibrated[0, :]
                structured["data_q"] = calibrated[1, :]
                return structured
            except FileNotFoundError:
                raise ValueError(f"File not found: {file_path}")
            except Exception as e:
                raise ValueError(f"Error reading file {file_path}: {str(e)}")

    def _get_channels(self):
        """Get the sorted list of channels in the input directory.

        Parses filenames to extract channel numbers from files named in the format:
        "<RUN>-ch<CHANNEL>.bin" (e.g., "timeS66-ch0.bin", "timeS66-ch1.bin")

        Returns:
            np.ndarray: Sorted array of channel numbers found in the directory.

        Raises:
            ValueError: If a filename doesn't match the expected format or contains
                non-integer channel numbers.

        """
        file_str_regex = os.path.join(self.config["daq_input_dir"], "*")
        file_list = glob(file_str_regex)

        found_channels = []
        for file_path in file_list:
            filename = os.path.basename(file_path)

            # Parse filename to extract channel number
            if not filename.endswith(".bin"):
                continue  # Skip non-binary files
            try:
                # Extract channel number from "<RUN>-ch<CHANNEL>.bin"
                channel_part = filename.split("-ch")[-1]
                channel_str = channel_part.split(".bin")[0]
                channel_num = int(channel_str)
                found_channels.append(channel_num)
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Invalid filename format: {filename}. "
                    f"Expected format: '<RUN>-ch<CHANNEL>.bin' (e.g., 'timeS66-ch1.bin')"
                ) from e

        if not found_channels:
            raise ValueError(f"No valid channel files found in {self.config['daq_input_dir']}")

        return np.sort(found_channels)

    def _check_channels(self, found_channels):
        """Compare found channels to the channel_map and validate consistency.

        Checks that:
        1. All found channels are within the promised range from channel_map.
        2. No channels are missing from the expected range.
        3. No channels show up twice.

        Args:
            found_channels (np.ndarray): Array of channel numbers found in the directory.

        Raises:
            ValueError: If channels are outside the promised range.
            ValueError: If one channel appears multiple times.

        """
        sub_detector = self.config["sub_detector"]
        promised_channel_range = self.config["channel_map"][sub_detector]
        promised_channel_min = promised_channel_range[0]
        promised_channel_max = promised_channel_range[1]

        # Check if found channels are within the promised range.
        if (
            np.min(found_channels) < promised_channel_min
            or np.max(found_channels) > promised_channel_max
        ):
            raise ValueError(
                f"Found channels {found_channels} are outside the promised range "
                f"[{promised_channel_min}, {promised_channel_max}] "
                f"for sub-detector '{sub_detector}'."
            )

        # Check for missing channels in the expected range.
        expected_channels = set(range(promised_channel_min, promised_channel_max + 1))
        found_channels_set = set(found_channels)
        missing_channels = expected_channels - found_channels_set

        if missing_channels:
            warnings.warn(
                f"Missing channels {sorted(missing_channels)} for sub-detector '{sub_detector}'. "
                f"Expected channels {sorted(expected_channels)}, found {sorted(found_channels)}"
            )

        # Check for multiple occurence
        if len(found_channels) != len(set(found_channels)):
            raise ValueError(
                f"Duplicate channels found in {found_channels}. "
                f"Each channel should appear only once."
            )

        return found_channels

    def get_channel_file(self, channel):
        """Get the file path for a specific channel.

        Constructs the file path based on the assumed naming convention:
        <DAQ_INPUT_DIR>/<RUN>/<RUN>-ch<CHANNEL>.bin

        For example, if daq_input_dir is "/path/to/timeS66" and channel is 1,
        returns "/path/to/timeS66/timeS66-ch1.bin"

        Args:
            channel (int): Channel number to get the file for.

        Returns:
            str: Full path to the channel's binary file.

        Raises:
            ValueError: If channel number is not a non-negative integer.

        """
        if not isinstance(channel, (int, np.integer)) or channel < 0:
            raise ValueError(f"Channel must be a non-negative integer, got {channel}")

        run = os.path.basename(os.path.normpath(self.config["daq_input_dir"]))
        filename = f"{run}-ch{channel}.bin"
        return os.path.join(self.config["daq_input_dir"], filename)

    def compute(self):
        """Process all available channels and return combined raw records.

        Workflow:
        1. Discovers available channels in the input directory.
        2. Validates channels against the channel_map configuration.
        3. Loads and processes each channel's data.
        4. Combines all channel records into a single array.

        Returns:
            np.ndarray: Combined raw records from all channels with dtype from infer_dtype()

        Raises:
            ValueError: If no valid channels are found or if all channel processing fails.

        """
        found_channels = self._get_channels()
        self._check_channels(found_channels)

        results = np.zeros(len(found_channels), dtype=self.infer_dtype())
        for i, ch in enumerate(found_channels):
            # Load and process the channel data.
            file_path = self.get_channel_file(ch)
            channel_data = self.load_one_channel(file_path)

            # Fill in the record data
            r = results[i]
            r["time"] = channel_data["time"][0]
            r["length"] = len(channel_data)
            r["dt"] = self.dt
            r["channel"] = ch
            r["data_i"] = channel_data["data_i"]
            r["data_q"] = channel_data["data_q"]

        return results
