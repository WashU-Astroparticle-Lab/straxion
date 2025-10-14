import os
import pytest
import numpy as np
import straxion
from straxion.plugins.truth import Truth
from straxion.utils import SECOND_TO_NANOSECOND
import shutil


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


def test_truth_plugin_registration():
    """Test that the Truth plugin is properly registered."""
    st = straxion.qualiphide_thz_online()
    assert "truth" in st._plugin_class_registry
    assert st._plugin_class_registry["truth"] == Truth


def test_truth_dtype_inference():
    """Test that Truth can infer the correct data type."""
    st = straxion.qualiphide_thz_online()
    plugin = st.get_single_plugin("1756824965", "truth")
    dtype = plugin.infer_dtype()

    # Check expected fields
    expected_fields = ["time", "endtime", "energy_true", "channel"]
    field_names = [name for name, *_ in dtype]
    for field in expected_fields:
        assert field in field_names


def test_truth_default_config():
    """Test that Truth plugin has the correct default configuration."""
    st = straxion.qualiphide_thz_online()
    plugin = st.get_single_plugin("1756824965", "truth")

    # Check default values
    assert plugin.config["random_seed"] == 137
    assert plugin.config["salt_rate"] == 100
    assert plugin.config["energy_meV"] == 50


def test_truth_custom_config():
    """Test Truth plugin with custom configuration."""
    st = straxion.qualiphide_thz_online()
    custom_config = {
        "random_seed": 42,
        "salt_rate": 200,
        "energy_meV": 100,
    }
    st.set_config(custom_config)
    plugin = st.get_single_plugin("1756824965", "truth")

    assert plugin.config["random_seed"] == 42
    assert plugin.config["salt_rate"] == 200
    assert plugin.config["energy_meV"] == 100


def test_truth_compute_with_mock_data():
    """Test Truth compute method with mock raw_records data."""
    st = straxion.qualiphide_thz_online()
    plugin = st.get_single_plugin("1756824965", "truth")

    # Create mock raw_records
    n_channels = 5
    time_start = 1000 * SECOND_TO_NANOSECOND
    time_duration = 1 * SECOND_TO_NANOSECOND  # 1 second
    time_end = time_start + time_duration

    mock_raw_records = np.zeros(
        n_channels,
        dtype=[
            ("time", np.int64),
            ("endtime", np.int64),
            ("channel", np.int16),
        ],
    )
    mock_raw_records["time"] = time_start
    mock_raw_records["endtime"] = time_end
    mock_raw_records["channel"] = np.arange(n_channels)

    # Compute truth events
    result = plugin.compute(mock_raw_records)

    # Verify result structure
    assert hasattr(result, "data")
    truth_events = result.data

    # Check that events were generated
    expected_n_events = int(time_duration / (SECOND_TO_NANOSECOND / 100))
    assert len(truth_events) == expected_n_events

    # Check field values
    assert all(truth_events["energy_true"] == 50)
    assert all(np.isin(truth_events["channel"], np.arange(n_channels)))
    assert all(truth_events["time"] >= time_start)
    assert all(truth_events["endtime"] <= time_end)
    assert all(truth_events["endtime"] > truth_events["time"])


def test_truth_empty_time_range():
    """Test Truth with zero time duration (should return empty)."""
    st = straxion.qualiphide_thz_online()
    plugin = st.get_single_plugin("1756824965", "truth")

    # Create mock raw_records with zero duration
    time_start = 1000 * SECOND_TO_NANOSECOND
    mock_raw_records = np.zeros(
        1,
        dtype=[
            ("time", np.int64),
            ("endtime", np.int64),
            ("channel", np.int16),
        ],
    )
    mock_raw_records["time"] = time_start
    mock_raw_records["endtime"] = time_start  # Same as start time
    mock_raw_records["channel"] = 0

    # Compute truth events
    result = plugin.compute(mock_raw_records)

    # Should return empty array
    assert len(result.data) == 0


def test_truth_reproducibility():
    """Test that Truth generates reproducible results with same seed."""
    st = straxion.qualiphide_thz_online()
    st.set_config({"random_seed": 42})

    # Create mock raw_records
    n_channels = 10
    time_start = 1000 * SECOND_TO_NANOSECOND
    time_duration = 0.5 * SECOND_TO_NANOSECOND
    time_end = time_start + time_duration

    mock_raw_records = np.zeros(
        n_channels,
        dtype=[
            ("time", np.int64),
            ("endtime", np.int64),
            ("channel", np.int16),
        ],
    )
    mock_raw_records["time"] = time_start
    mock_raw_records["endtime"] = time_end
    mock_raw_records["channel"] = np.arange(n_channels)

    # Generate truth events twice
    plugin1 = st.get_single_plugin("1756824965", "truth")
    result1 = plugin1.compute(mock_raw_records)

    plugin2 = st.get_single_plugin("1756824965", "truth")
    result2 = plugin2.compute(mock_raw_records)

    # Results should be identical
    assert len(result1.data) == len(result2.data)
    assert np.array_equal(result1.data["channel"], result2.data["channel"])
    assert np.array_equal(result1.data["time"], result2.data["time"])


def test_truth_channel_distribution():
    """Test that Truth distributes events across available channels."""
    st = straxion.qualiphide_thz_online()
    plugin = st.get_single_plugin("1756824965", "truth")

    # Create mock raw_records with multiple channels
    n_channels = 10
    time_start = 1000 * SECOND_TO_NANOSECOND
    time_duration = 10 * SECOND_TO_NANOSECOND  # 10 seconds
    time_end = time_start + time_duration

    mock_raw_records = np.zeros(
        n_channels,
        dtype=[
            ("time", np.int64),
            ("endtime", np.int64),
            ("channel", np.int16),
        ],
    )
    mock_raw_records["time"] = time_start
    mock_raw_records["endtime"] = time_end
    mock_raw_records["channel"] = np.arange(n_channels)

    # Compute truth events
    result = plugin.compute(mock_raw_records)
    truth_events = result.data

    # Check that events span multiple channels
    unique_channels = np.unique(truth_events["channel"])
    assert len(unique_channels) > 1  # Should hit multiple channels

    # All channels should be from available set
    assert all(np.isin(truth_events["channel"], np.arange(n_channels)))


def test_truth_time_intervals():
    """Test that Truth generates events at constant time intervals."""
    st = straxion.qualiphide_thz_online()
    salt_rate = 100  # Hz
    st.set_config({"salt_rate": salt_rate})
    plugin = st.get_single_plugin("1756824965", "truth")

    # Create mock raw_records
    time_start = 1000 * SECOND_TO_NANOSECOND
    time_duration = 1 * SECOND_TO_NANOSECOND
    time_end = time_start + time_duration

    mock_raw_records = np.zeros(
        1,
        dtype=[
            ("time", np.int64),
            ("endtime", np.int64),
            ("channel", np.int16),
        ],
    )
    mock_raw_records["time"] = time_start
    mock_raw_records["endtime"] = time_end
    mock_raw_records["channel"] = 0

    # Compute truth events
    result = plugin.compute(mock_raw_records)
    truth_events = result.data

    # Check time intervals
    expected_dt = SECOND_TO_NANOSECOND / salt_rate
    time_diffs = np.diff(truth_events["time"])

    # All time intervals should be equal (within 1 ns tolerance)
    assert np.allclose(time_diffs, expected_dt, atol=1)


def test_truth_energy_values():
    """Test that Truth assigns correct energy values."""
    st = straxion.qualiphide_thz_online()
    energy_meV = 75
    st.set_config({"energy_meV": energy_meV})
    plugin = st.get_single_plugin("1756824965", "truth")

    # Create mock raw_records
    time_start = 1000 * SECOND_TO_NANOSECOND
    time_duration = 1 * SECOND_TO_NANOSECOND
    time_end = time_start + time_duration

    mock_raw_records = np.zeros(
        1,
        dtype=[
            ("time", np.int64),
            ("endtime", np.int64),
            ("channel", np.int16),
        ],
    )
    mock_raw_records["time"] = time_start
    mock_raw_records["endtime"] = time_end
    mock_raw_records["channel"] = 0

    # Compute truth events
    result = plugin.compute(mock_raw_records)
    truth_events = result.data

    # All events should have the specified energy
    assert all(truth_events["energy_true"] == energy_meV)


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason=("Test data directory not provided via " "STRAXION_TEST_DATA_DIR environment variable"),
)
class TestTruthWithRealData:
    """Test Truth plugin with real data from STRAXION_TEST_DATA_DIR."""

    def _get_test_config(self, test_data_dir, run_id):
        """Get test config for the given test data directory and run ID."""
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

    def test_truth_with_real_raw_records(self):
        """Test Truth plugin with real raw_records data."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_online()
        run_id = "1756824965"
        configs = self._get_test_config(test_data_dir, run_id)

        clean_strax_data()
        try:
            # Get truth events
            truth = st.get_array(run_id, "truth", config=configs)

            # Basic validation
            assert truth is not None
            assert len(truth) > 0

            # Check required fields
            required_fields = ["time", "endtime", "energy_true", "channel"]
            for field in required_fields:
                assert field in truth.dtype.names, f"Required field '{field}' missing from truth"

            # Check data types
            assert truth["time"].dtype == np.int64
            assert truth["endtime"].dtype == np.int64
            assert truth["energy_true"].dtype == np.float32
            assert truth["channel"].dtype == np.int16

            # Check that all truth events have consistent properties
            assert all(truth["endtime"] > truth["time"])
            assert all(truth["energy_true"] == 50)  # Default value
            assert all(truth["channel"] >= 0)

            # Check time ordering
            assert np.all(np.diff(truth["time"]) >= 0), "Time stamps not monotonically increasing"

            print(
                f"Successfully processed {len(truth)} truth events "
                f"across {len(np.unique(truth['channel']))} channels"
            )

        except Exception as e:
            pytest.fail(f"Failed to process truth: {str(e)}")
