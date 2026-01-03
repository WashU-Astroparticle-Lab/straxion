import os
import pytest
import numpy as np
import straxion
import shutil
from straxion.utils import SECOND_TO_NANOSECOND


def test_qualiphide_thz_offline_context_creation():
    """Test that the qualiphide_thz_offline context can be created without errors."""
    st = straxion.qualiphide_thz_offline()
    assert st is not None
    assert hasattr(st, "get_array")


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason="Test data directory not provided via STRAXION_TEST_DATA_DIR environment variable",
)
def test_straxion_test_data_dir_exists_and_not_empty():
    """Test that STRAXION_TEST_DATA_DIR is set, exists, and is not empty."""
    test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
    if test_data_dir:
        assert os.path.exists(test_data_dir) and os.path.isdir(
            test_data_dir
        ), f"STRAXION_TEST_DATA_DIR '{test_data_dir}' does not exist or is not a directory."
        contents = os.listdir(test_data_dir)
        print(f"Contents of STRAXION_TEST_DATA_DIR ({test_data_dir}): {contents}")
        assert len(contents) > 0, f"STRAXION_TEST_DATA_DIR '{test_data_dir}' is empty."
    else:
        pytest.fail("STRAXION_TEST_DATA_DIR is not set.")


def test_qualiphide_thz_offline_plugin_registration():
    """Test that the records plugin is properly registered in the context."""
    st = straxion.qualiphide_thz_offline()
    assert "raw_records" in st._plugin_class_registry


def clean_strax_data():
    """Clean up strax data directory."""
    strax_data_dir = os.path.join(os.getcwd(), "strax_data")
    if os.path.exists(strax_data_dir) and os.path.isdir(strax_data_dir):
        for filename in os.listdir(strax_data_dir):
            file_path = os.path.join(strax_data_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


def _get_test_config(test_data_dir, run_id):
    """Get the test configuration for the given test data directory and run ID."""
    daq_input_dir = os.path.join(test_data_dir, f"ts_38kHz-{run_id}.npy")
    iq_finescan_dir = test_data_dir
    iq_widescan_dir = test_data_dir
    iq_finescan_filename = "iq_fine_z_2dB_below_pcrit-1756824887.npy"
    iq_widescan_filename = "iq_wide_z_2dB_below_pcrit-1756824887.npy"
    resonant_frequency_dir = test_data_dir
    resonant_frequency_filename = "fres_2dB-1756824887.npy"

    configs = {
        "daq_input_dir": daq_input_dir,
        "iq_finescan_dir": iq_finescan_dir,
        "iq_finescan_filename": iq_finescan_filename,
        "iq_widescan_dir": iq_widescan_dir,
        "iq_widescan_filename": iq_widescan_filename,
        "resonant_frequency_dir": resonant_frequency_dir,
        "resonant_frequency_filename": resonant_frequency_filename,
    }
    return configs


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason="Test data directory not provided via STRAXION_TEST_DATA_DIR environment variable",
)
def test_records_processing():
    """Test the complete records processing pipeline with real data.

    This test requires the STRAXION_TEST_DATA_DIR environment variable to be set to the path
    containing the qualiphide_fir_test_data directory with example data.
    """
    test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
    if not test_data_dir:
        pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

    if not os.path.exists(test_data_dir):
        pytest.fail(f"Test data directory {test_data_dir} does not exist")

    # Extract run ID from the test data directory name or use default
    run_id = "1756824965"  # Default run ID based on the example

    # Create context and process records
    st = straxion.qualiphide_thz_offline()
    configs = _get_test_config(test_data_dir, run_id)

    clean_strax_data()
    try:
        records = st.get_array(run_id, "raw_records", config=configs)

        # Basic validation of the output
        assert records is not None
        assert len(records) > 0

        # Check that all required fields are present
        required_fields = ["time", "endtime", "length", "dt", "channel", "data_i", "data_q"]
        for field in required_fields:
            assert field in records.dtype.names

        # Check data types
        assert records["time"].dtype == np.int64
        assert records["channel"].dtype == np.int16
        assert records["data_i"].dtype == np.float32
        assert records["data_q"].dtype == np.float32

        # Check that all records have reasonable lengths
        assert all(records["length"] > 0)
        assert all(records["dt"] > 0)

        # Check that data arrays have the correct shape
        for record in records:
            assert record["data_i"].shape == (record["length"],)
            assert record["data_q"].shape == (record["length"],)

        print(
            f"Successfully processed {len(records)} records "
            f"from {len(np.unique(records['channel']))} channels"
        )

    except Exception as e:
        pytest.fail(f"Failed to process records: {str(e)}")


def _check_endtime_consistency(records, fs=None):
    """Check that endtime is correctly calculated for each record.

    Args:
        records: Array of records to check
        fs: Sampling frequency (Hz). If None, will use dt with tolerance.
    """
    # Calculate dt_exact from fs if provided
    if fs is not None:
        dt_exact = 1 / fs * SECOND_TO_NANOSECOND
    else:
        dt_exact = None

    for record in records:
        if dt_exact is not None:
            expected_endtime = np.int64(record["time"] + record["length"] * dt_exact)
            assert record["endtime"] == expected_endtime, (
                f"Endtime mismatch: got {record['endtime']}, "
                f"expected {expected_endtime} (time={record['time']}, "
                f"length={record['length']}, dt={record['dt']}, "
                f"dt_exact={dt_exact})"
            )
        else:
            # Fallback: use dt but allow small tolerance for rounding
            expected_endtime = record["time"] + record["length"] * record["dt"]
            # Allow tolerance of up to length nanoseconds (accounts for rounding)
            tolerance = record["length"]
            assert abs(record["endtime"] - expected_endtime) <= tolerance, (
                f"Endtime mismatch: got {record['endtime']}, "
                f"expected {expected_endtime}±{tolerance} (time={record['time']}, "
                f"length={record['length']}, dt={record['dt']})"
            )


def _check_monotonic_time(records):
    """Check that time stamps are monotonically increasing within each channel."""
    for channel in np.unique(records["channel"]):
        channel_records = records[records["channel"] == channel]
        if len(channel_records) > 1:
            times = channel_records["time"]
            assert np.all(
                np.diff(times) > 0
            ), f"Time stamps not monotonically increasing for channel {channel}"


def _check_finite_data(records):
    """Check that data values are finite."""
    assert np.all(np.isfinite(records["data_i"])), "Non-finite values found in data_i"
    assert np.all(np.isfinite(records["data_q"])), "Non-finite values found in data_q"


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason="Test data directory not provided via STRAXION_TEST_DATA_DIR environment variable",
)
def test_records_data_consistency():
    """Test that the records data is internally consistent."""
    test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
    if not test_data_dir:
        pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

    if not os.path.exists(test_data_dir):
        pytest.fail(f"Test data directory {test_data_dir} does not exist")

    st = straxion.qualiphide_thz_offline()
    run_id = "1756824965"
    configs = _get_test_config(test_data_dir, run_id)

    clean_strax_data()
    try:
        records = st.get_array(run_id, "raw_records", config=configs)
        # Get fs from context config to calculate dt_exact
        fs = st.config.get("fs", 38000)  # Default to 38000 if not found
        _check_endtime_consistency(records, fs=fs)
        _check_monotonic_time(records)
        _check_finite_data(records)
    except Exception as e:
        pytest.fail(f"Failed to validate records consistency: {str(e)}")


def test_records_missing_data_directory():
    """Test that the records plugin raises appropriate errors when data directory is missing."""
    st = straxion.qualiphide_thz_offline()
    run_id = "1756824965"

    configs = {
        "daq_input_dir": "/nonexistent/path.ts.npy",
        "iq_finescan_dir": "/nonexistent",
        "iq_finescan_filename": "nonexistent.npy",
        "iq_widescan_dir": "/nonexistent",
        "iq_widescan_filename": "nonexistent.npy",
        "resonant_frequency_dir": "/nonexistent",
        "resonant_frequency_filename": "nonexistent.npy",
    }

    clean_strax_data()
    with pytest.raises((ValueError, FileNotFoundError)):
        st.get_array(run_id, "raw_records", config=configs)


def test_records_invalid_config():
    """Test that the records plugin handles invalid configuration gracefully."""
    st = straxion.qualiphide_thz_offline()
    run_id = "1756824965"

    # Test with invalid configuration
    configs = {
        "daq_input_dir": "/nonexistent/path.ts.npy",
        "iq_finescan_dir": "/nonexistent",
        "iq_finescan_filename": "nonexistent.npy",
        "iq_widescan_dir": "/nonexistent",
        "iq_widescan_filename": "nonexistent.npy",
        "resonant_frequency_dir": "/nonexistent",
        "resonant_frequency_filename": "nonexistent.npy",
    }

    clean_strax_data()
    with pytest.raises(Exception):
        st.get_array(run_id, "raw_records", config=configs)


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason="Test data directory not provided via STRAXION_TEST_DATA_DIR environment variable",
)
def test_qualiphide_fir_test_data_files_exist():
    """Test that all expected .npy files exist in the qualiphide_fir_test_data directory."""
    test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
    if not test_data_dir:
        pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

    if not os.path.exists(test_data_dir):
        pytest.fail(f"Test data directory {test_data_dir} does not exist")

    # Expected files based on the new test data structure
    expected_files = [
        "fres_2dB-1756824887.npy",
        "iq_fine_f_2dB_below_pcrit-1756824887.npy",
        "iq_fine_z_2dB_below_pcrit-1756824887.npy",
        "iq_wide_f_2dB_below_pcrit-1756824887.npy",
        "iq_wide_z_2dB_below_pcrit-1756824887.npy",
        "ts_38kHz-1756824965.npy",
    ]

    actual_files = set(os.listdir(test_data_dir))
    missing_files = [f for f in expected_files if f not in actual_files]

    print(f"Expected files: {expected_files}")
    print(f"Actual files: {sorted(actual_files)}")

    assert not missing_files, f"Missing .npy files: {missing_files}"


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason="Test data directory not provided via STRAXION_TEST_DATA_DIR environment variable",
)
def test_npy_files_are_valid():
    """Test that the .npy files can be loaded and contain valid data."""
    test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
    if not test_data_dir:
        pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

    if not os.path.exists(test_data_dir):
        pytest.fail(f"Test data directory {test_data_dir} does not exist")

    # Test loading each .npy file (skip hidden files like ._*)
    npy_files = [
        f for f in os.listdir(test_data_dir) if f.endswith(".npy") and not f.startswith("._")
    ]

    for npy_file in npy_files:
        file_path = os.path.join(test_data_dir, npy_file)
        try:
            data = np.load(file_path)
            assert data is not None, f"Failed to load {npy_file}"
            assert data.size > 0, f"Empty data in {npy_file}"
            print(f"Successfully loaded {npy_file} with shape {data.shape}")
        except Exception as e:
            pytest.fail(f"Failed to load {npy_file}: {str(e)}")
