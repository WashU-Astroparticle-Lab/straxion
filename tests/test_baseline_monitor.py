import numpy as np
import pytest
import os
import straxion
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
class TestBaselineMonitorOnline:
    """Test the BaselineMonitor plugin with real qualiphide_fir_test_data."""

    def test_qualiphide_thz_online_context_creation(self):
        """Test that the qualiphide_thz_online context can be created without errors."""
        st = straxion.qualiphide_thz_online()
        assert st is not None
        assert hasattr(st, "get_array")

    def test_straxion_test_data_dir_exists_and_not_empty(self):
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

    def test_qualiphide_fir_test_data_files_exist(self):
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

    def test_npy_files_are_valid(self):
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

    def test_baseline_monitor_processing(self):
        """Test the complete baseline monitor processing pipeline with real data.

        This test requires the STRAXION_TEST_DATA_DIR environment variable to be set to the path
        containing the qualiphide_fir_test_data directory with example data.
        """
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")
        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        # Create context and process baseline monitor data
        st = straxion.qualiphide_thz_online()
        run_id = "1756824965"
        configs = _get_test_config(test_data_dir, run_id)

        clean_strax_data()
        try:
            # Get baseline monitor data
            bm = st.get_array(run_id, "baseline_monitor", config=configs)

            # Basic validation of the output
            assert bm is not None
            assert len(bm) > 0

            # Check that all required fields are present
            required_fields = [
                "time",
                "endtime",
                "length",
                "dt",
                "channel",
                "baseline_monitor_interval",
                "baseline_monitor_std",
                "baseline_monitor_std_moving_average",
                "baseline_monitor_std_convolved",
            ]
            for field in required_fields:
                assert (
                    field in bm.dtype.names
                ), f"Required field '{field}' missing from baseline monitor"

            # Check data types
            assert bm["time"].dtype == np.int64
            assert bm["endtime"].dtype == np.int64
            assert bm["length"].dtype == np.int64
            assert bm["dt"].dtype == np.int64
            assert bm["channel"].dtype == np.int16
            assert bm["baseline_monitor_interval"].dtype == np.int64
            assert bm["baseline_monitor_std"].dtype == np.float32
            assert bm["baseline_monitor_std_moving_average"].dtype == np.float32
            assert bm["baseline_monitor_std_convolved"].dtype == np.float32

            # Check that all records have reasonable lengths
            assert all(bm["length"] > 0)
            assert all(bm["dt"] > 0)

            # Check that baseline monitor arrays have the correct shape
            # Should have 100 intervals based on N_BASELINE_MONITOR_INTERVAL
            expected_intervals = 100
            for record in bm:
                assert record["baseline_monitor_std"].shape == (expected_intervals,)
                assert record["baseline_monitor_std_moving_average"].shape == (expected_intervals,)
                assert record["baseline_monitor_std_convolved"].shape == (expected_intervals,)

            # Check that baseline monitor interval is consistent across all records
            # The interval should be calculated based on the record length and number of intervals
            # Allow for small rounding differences
            expected_interval = bm["length"][0] // expected_intervals * bm["dt"][0]
            actual_intervals = bm["baseline_monitor_interval"]
            # Allow for small differences due to rounding
            assert all(np.abs(actual_intervals - expected_interval) <= bm["dt"][0])

            # Check that baseline std values are reasonable (should be positive)
            assert np.all(bm["baseline_monitor_std"] >= 0)
            assert np.all(bm["baseline_monitor_std_moving_average"] >= 0)
            assert np.all(bm["baseline_monitor_std_convolved"] >= 0)

            # Check that baseline std values are not all identical (should have some variation)
            for channel in np.unique(bm["channel"]):
                channel_data = bm[bm["channel"] == channel]
                if len(channel_data) > 1:
                    # Check that std values vary across intervals
                    for record in channel_data:
                        std_values = record["baseline_monitor_std"]
                        std_ma_values = record["baseline_monitor_std_moving_average"]
                        std_conv_values = record["baseline_monitor_std_convolved"]

                        # Should not be all identical
                        assert not np.allclose(
                            std_values, std_values[0]
                        ), f"Channel {channel} std values are constant"
                        assert not np.allclose(
                            std_ma_values, std_ma_values[0]
                        ), f"Channel {channel} moving average std values are constant"
                        assert not np.allclose(
                            std_conv_values, std_conv_values[0]
                        ), f"Channel {channel} convolved std values are constant"

            print(
                f"Successfully processed {len(bm)} baseline monitor records "
                f"from {len(np.unique(bm['channel']))} channels"
            )

        except Exception as e:
            pytest.fail(f"Failed to process baseline monitor data: {str(e)}")

    def test_baseline_monitor_data_consistency(self):
        """Test that the baseline monitor data is internally consistent."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_online()
        run_id = "1756824965"
        configs = _get_test_config(test_data_dir, run_id)

        clean_strax_data()
        try:
            bm = st.get_array(run_id, "baseline_monitor", config=configs)

            # Check endtime consistency
            for record in bm:
                expected_endtime = record["time"] + record["length"] * record["dt"]
                assert record["endtime"] == expected_endtime

            # Check monotonic time within channels
            for channel in np.unique(bm["channel"]):
                channel_records = bm[bm["channel"] == channel]
                if len(channel_records) > 1:
                    times = channel_records["time"]
                    assert np.all(
                        np.diff(times) > 0
                    ), f"Time stamps not monotonically increasing for channel {channel}"

            # Check finite data
            assert np.all(
                np.isfinite(bm["baseline_monitor_std"])
            ), "Non-finite values found in baseline_monitor_std"
            assert np.all(
                np.isfinite(bm["baseline_monitor_std_moving_average"])
            ), "Non-finite values found in baseline_monitor_std_moving_average"
            assert np.all(
                np.isfinite(bm["baseline_monitor_std_convolved"])
            ), "Non-finite values found in baseline_monitor_std_convolved"

        except Exception as e:
            pytest.fail(f"Failed to validate baseline monitor consistency: {str(e)}")

    def test_baseline_monitor_missing_data_directory(self):
        """Test that the baseline monitor plugin raises errors when data directory is missing."""
        st = straxion.qualiphide_thz_online()
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
            st.get_array(run_id, "baseline_monitor", config=configs)

    def test_baseline_monitor_invalid_config(self):
        """Test that the baseline monitor plugin handles invalid configuration gracefully."""
        st = straxion.qualiphide_thz_online()
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
            st.get_array(run_id, "baseline_monitor", config=configs)
