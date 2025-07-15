import os
import pytest
import numpy as np
import straxion
from straxion.plugins.raw_records import DAQReader


def test_qualiphide_context_creation():
    """Test that the qualiphide context can be created without errors."""
    st = straxion.qualiphide()
    assert st is not None
    assert hasattr(st, "get_array")


def test_daq_reader_plugin_registration():
    """Test that the DAQReader plugin is properly registered in the context."""
    st = straxion.qualiphide()
    assert "raw_records" in st._plugin_class_registry
    assert st._plugin_class_registry["raw_records"] == DAQReader


def test_daq_reader_dtype_inference():
    """Test that DAQReader can infer the correct data type."""
    st = straxion.qualiphide()
    config = {"daq_input_dir": "abracadabra", "record_length": 5_000_000, "fs": 500_000}
    st.set_config(config)
    plugin = st.get_single_plugin("timeS429", "raw_records")

    dtype = plugin.infer_dtype()
    expected_fields = ["time", "endtime", "length", "dt", "channel", "data_i", "data_q"]

    field_names = [name[1] for name, *_ in dtype]
    for field in expected_fields:
        assert field in field_names


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason="Test data directory not provided via STRAXION_TEST_DATA_DIR environment variable",
)
def test_raw_records_processing():
    """Test the complete raw_records processing pipeline with real data.

    This test requires the STRAXION_TEST_DATA_DIR environment variable to be set to the path
    containing the timeS429 directory with example data.

    """
    test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
    timeS429_dir = os.path.join(test_data_dir, "timeS429")

    if not os.path.exists(timeS429_dir):
        pytest.skip(f"Test data directory {timeS429_dir} does not exist")

    # Create context and process raw_records
    st = straxion.qualiphide()

    config = {"daq_input_dir": timeS429_dir, "record_length": 5_000_000, "fs": 500_000}

    try:
        rr = st.get_array("timeS429", "raw_records", config=config)

        # Basic validation of the output
        assert rr is not None
        assert len(rr) > 0

        # Check that all required fields are present
        required_fields = ["time", "endtime", "length", "dt", "channel", "data_i", "data_q"]
        for field in required_fields:
            assert field in rr.dtype.names

        # Check data types
        assert rr["time"].dtype == np.int64
        assert rr["channel"].dtype == np.int16
        assert rr["data_i"].dtype == np.dtype(">f8")
        assert rr["data_q"].dtype == np.dtype(">f8")

        # Check that all records have the expected length
        expected_length = config["record_length"]
        assert all(rr["length"] == expected_length)

        # Check that all records have the expected dt
        expected_dt = int(1 / config["fs"] * 1_000_000_000)  # Convert to nanoseconds
        assert all(rr["dt"] == expected_dt)

        # Check that channels are within expected range (0-9 based on context config)
        assert all(0 <= rr["channel"]) and all(rr["channel"] <= 9)

        # Check that data arrays have the correct shape
        for record in rr:
            assert record["data_i"].shape == (expected_length,)
            assert record["data_q"].shape == (expected_length,)

        print(
            f"Successfully processed {len(rr)} records "
            f"from {len(np.unique(rr['channel']))} channels"
        )

    except Exception as e:
        pytest.fail(f"Failed to process raw_records: {str(e)}")


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason="Test data directory not provided via STRAXION_TEST_DATA_DIR environment variable",
)
def test_raw_records_data_consistency():
    """Test that the raw_records data is internally consistent."""
    test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
    timeS429_dir = os.path.join(test_data_dir, "timeS429")

    if not os.path.exists(timeS429_dir):
        pytest.skip(f"Test data directory {timeS429_dir} does not exist")

    st = straxion.qualiphide()

    config = {"daq_input_dir": timeS429_dir, "record_length": 5_000_000, "fs": 500_000}

    try:
        rr = st.get_array("timeS429", "raw_records", config=config)

        # Check that endtime is correctly calculated
        for record in rr:
            expected_endtime = record["time"] + record["length"] * record["dt"]
            assert record["endtime"] == expected_endtime

        # Check that time stamps are monotonically increasing within each channel
        for channel in np.unique(rr["channel"]):
            channel_records = rr[rr["channel"] == channel]
            if len(channel_records) > 1:
                times = channel_records["time"]
                assert np.all(
                    np.diff(times) > 0
                ), f"Time stamps not monotonically increasing for channel {channel}"

        # Check that data values are finite
        assert np.all(np.isfinite(rr["data_i"]))
        assert np.all(np.isfinite(rr["data_q"]))

    except Exception as e:
        pytest.fail(f"Failed to validate raw_records consistency: {str(e)}")


def test_daq_reader_missing_data_directory():
    """Test that DAQReader raises appropriate errors when data directory is missing."""
    st = straxion.qualiphide()

    config = {"daq_input_dir": "/nonexistent/path", "record_length": 5_000_000, "fs": 500_000}

    with pytest.raises((ValueError, FileNotFoundError)):
        st.get_array("timeS429", "raw_records", config=config)


def test_daq_reader_invalid_config():
    """Test that DAQReader handles invalid configuration gracefully."""
    st = straxion.qualiphide()

    # Test with invalid record_length
    config = {
        "daq_input_dir": "/nonexistent/path",  # Use a non-existent path
        "record_length": -1,  # Invalid negative value
        "fs": 500_000,
    }

    with pytest.raises(Exception):
        st.get_array("timeS429", "raw_records", config=config)
