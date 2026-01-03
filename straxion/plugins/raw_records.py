import os
import warnings
from glob import glob
from typing import Tuple
from immutabledict import immutabledict
import strax
import numpy as np
from straxion.utils import (
    DATA_DTYPE,
    SECOND_TO_NANOSECOND,
    base_waveform_dtype,
)

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        "record_length",
        default=1_900_000,
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
        default=38_000,
        track=True,
        type=int,
        help="Sampling frequency (assumed the same for all channels) in unit of Hz",
    ),
    strax.Option(
        "daq_input_dir",
        type=str,
        track=False,
        help="Directory where readers put data. For example: '/path/to/ts_38kHz-1756457052.npy'.",
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
class QUALIPHIDETHzReader(strax.Plugin):
    """DAQ reader for the QUALIPHIDE THz detector.

    Assumed the IQ timestream is already in an npy file, where axis 0 is the channel, axis 1 is the
    time of equal spacing. The time stream is complex.

    Truncate the time stream to the record length.

    """

    __version__ = "0.0.0"  # Should be very careful to bump the version of this plugin!

    # Data structure topology related:
    provides: str = "raw_records"
    data_kind = provides
    depends_on: Tuple = tuple()  # This is the lowest level of strax data in processing.
    save_when = strax.SaveWhen.EXPLICIT

    # Memory management related:
    rechunk_on_load = False  # Assumed no single dataset will be larger than 1 GB.
    chunk_source_size_mb = 200  # 200 MB; Neglected if no rechunk on load.
    rechunk_on_save = False  # Assumed no chunking at DAQ.
    chunk_target_size_mb = 1000  # Meaningless if rechunk_on_save is False.
    compressor = "lz4"  # Inherited from straxen. Not optimized outside XENONnT.

    def infer_dtype(self):
        """Data type for a waveform raw_record."""
        dtype = base_waveform_dtype()
        dtype.append(
            (
                ("Waveform data of I in raw ADC counts", "data_i"),
                DATA_DTYPE,
                self.config["record_length"],
            )
        )
        dtype.append(
            (
                ("Waveform data of Q in raw ADC counts", "data_q"),
                DATA_DTYPE,
                self.config["record_length"],
            )
        )
        return dtype

    def setup(self):
        # Note that this dt is not exact due to the int conversion.
        self.dt = int(1 / self.config["fs"] * SECOND_TO_NANOSECOND)  # In unit of ns.
        self.dt_exact = 1 / self.config["fs"] * SECOND_TO_NANOSECOND

    def source_finished(self):
        """Return whether all chunks the plugin wants to read have been written by DAQ.

        FIXME: We assumed that the DAQ will only produce one chunk for each run!

        """
        return True

    def is_ready(self, chunk_i):
        """Return whether the chunk chunk_i is ready for reading."""
        # We assume there is only one chunk for all runs.
        if chunk_i == 0:
            return True
        # There is no other chunk, so it will never be ready.
        return False

    def load_time_stream(self):
        """Load the time stream from the npy file.

        Returns:
            np.ndarray: The time stream.

        """
        file_path = self.config["daq_input_dir"]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return np.load(file_path)

    def get_run_start_time(self):
        return (
            eval(self.config["daq_input_dir"].split("/")[-1].split("-")[1].split(".")[0])
            * SECOND_TO_NANOSECOND
        )

    def compute(self):
        # Load the time stream.
        time_stream = self.load_time_stream()
        self.run_start_time = self.get_run_start_time()
        found_channels = np.shape(time_stream)[0]

        results = np.zeros(found_channels, dtype=self.infer_dtype())
        results["time"] = self.run_start_time
        results["length"] = self.config["record_length"]
        results["dt"] = self.dt
        results["endtime"] = np.int64(results["time"] + results["length"] * self.dt_exact)
        results["channel"] = np.arange(found_channels)

        if len(time_stream[0]) > self.config["record_length"]:
            results["data_i"] = time_stream.real[:, : self.config["record_length"]]
            results["data_q"] = time_stream.imag[:, : self.config["record_length"]]
        else:
            raise ValueError(
                f"The time stream length is {len(time_stream[0])} "
                f"but the record length is {self.config['record_length']}. "
                f"The time stream length should be at least as long as the record "
                f"length {self.config['record_length']}."
            )

        # We must build a chunk for the lowest data type, as required by strax.
        results = self.chunk(
            start=np.min(results["time"]),
            end=np.max(results["time"]) + np.int64(self.dt_exact * self.config["record_length"]),
            data=results,
            data_type="raw_records",
        )
        return results


@export
@strax.takes_config(
    strax.Option(
        "record_length",
        default=5_000_000,
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
        default=50_000,
        track=True,
        type=int,
        help="Sampling frequency (assumed the same for all channels) in unit of Hz",
    ),
    strax.Option(
        "run_start_time",
        type=float,
        track=False,
        default=900_714_600_000_000_000,  # Placeholder: 1998-07-17 22:30:00 GMT.
        help="Time of start run (in unit of nanoseconds, since unix epoch).",
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
class NX3LikeReader(strax.Plugin):
    """Read the raw data from citkid. It does nothing beyond reading data into a Python format.
    The functionality was adapted from "eConvertLVBinS.m":
    https://caltechobscosgroup.slack.com/archives/C07SZDKRNF9/p1752010145654029.
    The hyper-parametrization is inspired by straxen.NX3LikeReader:
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
    chunk_source_size_mb = 200  # 200 MB; Neglected if no rechunk on load.
    rechunk_on_save = False  # Assumed no chunking at DAQ.
    chunk_target_size_mb = 1000  # Meaningless if rechunk_on_save is False.
    compressor = "lz4"  # Inherited from straxen. Not optimized outside XENONnT.

    def infer_dtype(self):
        """Data type for a waveform raw_record."""
        dtype = base_waveform_dtype()
        dtype.append(
            (
                ("Waveform data of I in raw ADC counts", "data_i"),
                DATA_DTYPE,
                self.config["record_length"],
            )
        )
        dtype.append(
            (
                ("Waveform data of Q in raw ADC counts", "data_q"),
                DATA_DTYPE,
                self.config["record_length"],
            )
        )
        return dtype

    def setup(self):
        # Note that this dt is not exact due to the int conversion.
        self.dt = int(1 / self.config["fs"] * SECOND_TO_NANOSECOND)  # In unit of ns.
        self.dt_exact = 1 / self.config["fs"] * SECOND_TO_NANOSECOND

    @staticmethod
    def load_one_channel(file_path, dt, record_length=None):
        """Reads and polynomially corrects raw binary DAQ data of a
        certain readout channel.

        Workflow:
        1. Reads the file header to determine the number of
           coefficients and IQ channels.
        2. Checks that there are exactly two IQ channels (I and Q).
        3. Reads and reshapes the calibration coefficient matrix.
        4. Reads the raw int16 time-series data and checks its shape.
        5. Applies polynomial calibration to convert raw ADC counts
           to physical units.
        6. Constructs a time vector based on the sampling interval.
        7. Returns a structured array with fields 'time', 'data_i',
           and 'data_q'.

        Args:
            file_path (str): Path to the binary DAQ file.
            dt (float): Sampling time interval.
            record_length (int, optional): Expected number of samples. If None, the number of
            samples will be determined by the data size.

        Returns:
            np.ndarray: Structured array with fields 'time', 'I',
                        and 'Q'.

        """

        def _read_header(f):
            header = np.fromfile(f, dtype=">d", count=2)
            n_coeffs = int(header[0])
            n_iq_channels = int(header[1])
            if n_iq_channels != 2:
                raise ValueError(
                    f"Expected 2 channels (I and Q), but found "
                    f"{n_iq_channels}. There should be only two "
                    f"channels: I and Q."
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
                    "Data of different channels may have different "
                    "lengths!"
                )
            n_samples_found = data_size // 2
            if record_length is not None and n_samples_found != record_length:
                raise ValueError(
                    f"The data length is promised to be "
                    f"{record_length} but found to be "
                    f"{n_samples_found}."
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

        def _make_time_vector(n_samples, dt):
            return np.arange(n_samples) * dt

        with open(file_path, "rb") as f:
            try:
                n_coeffs = _read_header(f)
                _, offsets, gains = _read_coefficients(f, n_coeffs)
                data_int16, n_samples = _read_data(f, record_length)
                calibrated = _apply_poly_correction(data_int16, offsets, gains)
                time_vector = _make_time_vector(n_samples, dt)
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

    def source_finished(self):
        """Return whether all chunks the plugin wants to read have been written by DAQ.

        FIXME: We assumed that the DAQ will only produce one chunk for each run!

        """
        return True

    def is_ready(self, chunk_i):
        """Return whether the chunk chunk_i is ready for reading."""
        # We assume there is only one chunk for all runs.
        if chunk_i == 0:
            return True
        # There is no other chunk, so it will never be ready.
        return False

    def compute(self):
        """Process all available channels and return combined raw records.

        Workflow:
        1. Discovers available channels in the input directory.
        2. Validates channels against the channel_map configuration.
        3. Loads and processes each channel's data.
        4. Combines all channel records into a single array.

        Returns:
            strax.Chunk: Combined raw records from all channels with dtype from infer_dtype()

        Raises:
            ValueError: If no valid channels are found or if all channel processing fails.

        """
        found_channels = self._get_channels()
        self._check_channels(found_channels)

        results = np.zeros(len(found_channels), dtype=self.infer_dtype())
        for i, ch in enumerate(found_channels):
            # Load and process the channel data.
            file_path = self.get_channel_file(ch)
            channel_data = self.load_one_channel(file_path, self.dt, self.config["record_length"])

            # Fill in the record data
            r = results[i]
            r["time"] = channel_data["time"][0] * SECOND_TO_NANOSECOND
            r["length"] = len(channel_data)
            r["dt"] = self.dt
            r["endtime"] = np.int64(r["time"] + r["length"] * self.dt_exact)
            r["channel"] = ch
            r["data_i"] = channel_data["data_i"]
            r["data_q"] = channel_data["data_q"]

        # Shift all time stamps by the run start time. Now all time stamps are since unix epoch.
        results["time"] += self.config["run_start_time"]
        results["endtime"] += self.config["run_start_time"]

        # We must build a chunk for the lowest data type, as required by strax.
        results = self.chunk(
            start=np.min(results["time"]),
            end=np.max(results["time"]) + np.int64(self.dt * self.config["record_length"]),
            data=results,
            data_type="raw_records",
        )

        return results
