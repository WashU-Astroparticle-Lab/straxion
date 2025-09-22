import os
import pytest
import numpy as np
import straxion
from straxion.plugins.noise_bank import NoiseBank
from straxion.utils import HIT_WINDOW_LENGTH_LEFT, HIT_WINDOW_LENGTH_RIGHT
import shutil


def clean_strax_data():
    """Clean up strax data directory."""
    strax_data_dir = os.path.join(os.getcwd(), "strax_data")
    if os.path.exists(strax_data_dir) and os.path.isdir(strax_data_dir):
        for filename in os.listdir(strax_data_dir):
            file_path = os.path.join(strax_data_dir, filename)
            try:
                if os.path.isfile(file_path) or os.islink(file_path):
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


def test_noise_bank_plugin_registration():
    """Test that the NoiseBank plugin is properly registered in the context."""
    st = straxion.qualiphide_thz_offline()
    assert "noises" in st._plugin_class_registry
    assert st._plugin_class_registry["noises"] == NoiseBank


def test_noise_bank_dtype_inference():
    """Test that NoiseBank can infer the correct data type."""
    st = straxion.qualiphide_thz_offline()
    plugin = st.get_single_plugin("1756824965", "noises")
    dtype = plugin.infer_dtype()
    # List of expected fields (add/remove as needed)
    expected_fields = [
        "time",
        "endtime",
        "length",
        "dt",
        "channel",
        "data_dx",
        "data_dx_moving_average",
        "data_dx_convolved",
        "hit_threshold",
        "amplitude",
        "amplitude_moving_average",
        "amplitude_convolved",
    ]
    field_names = [name[1] for name, *_ in dtype]
    for field in expected_fields:
        assert field in field_names


def test_noise_bank_empty_input():
    """Test that NoiseBank returns empty output for empty input."""
    st = straxion.qualiphide_thz_offline()
    plugin = st.get_single_plugin("1756824965", "noises")
    empty_records = np.array(
        [],
        dtype=[
            ("channel", np.int16),
            ("data_dx", np.float32, 10),
            ("data_dx_moving_average", np.float32, 10),
            ("data_dx_convolved", np.float32, 10),
        ],
    )
    empty_hits = np.array(
        [],
        dtype=[
            ("channel", np.int16),
            ("time", np.int64),
            ("amplitude_convolved_max_record_i", np.int32),
            ("hit_threshold", np.float32),
        ],
    )
    result = plugin.compute(empty_records, empty_hits)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_noise_bank_minimal_valid_input():
    """Test NoiseBank with a minimal valid record that should produce no noises."""
    st = straxion.qualiphide_thz_offline()
    plugin = st.get_single_plugin("1756824965", "noises")
    # Minimal record with all zeros (should not trigger any hits)
    record = np.zeros(
        1,
        dtype=[
            ("channel", np.int16),
            ("data_dx", np.float32, 10),
            ("data_dx_moving_average", np.float32, 10),
            ("data_dx_convolved", np.float32, 10),
            ("time", np.int64),
        ],
    )
    # Minimal hit that would require noise window before it
    hit = np.zeros(
        1,
        dtype=[
            ("channel", np.int16),
            ("time", np.int64),
            ("amplitude_convolved_max_record_i", np.int32),
            ("hit_threshold", np.float32),
        ],
    )
    hit["amplitude_convolved_max_record_i"] = 5  # Position that would need noise window
    result = plugin.compute(record, hit)
    assert isinstance(result, np.ndarray)
    # Should be empty because the noise window would be before the record start
    assert result.size == 0


def test_noise_bank_malformed_input():
    """Test that NoiseBank handles malformed input gracefully."""
    st = straxion.qualiphide_thz_offline()
    plugin = st.get_single_plugin("1756824965", "noises")
    # Missing required fields
    bad_record = np.zeros(1, dtype=[("channel", np.int16)])
    bad_hit = np.zeros(1, dtype=[("channel", np.int16)])
    with pytest.raises(Exception):
        plugin.compute(bad_record, bad_hit)


def test_noise_bank_invalid_config():
    """Test that NoiseBank raises an error with invalid config."""
    st = straxion.qualiphide_thz_offline()
    # Set an invalid config (e.g., negative fs)
    st.set_config({"fs": -1})
    with pytest.raises(Exception):
        st.get_single_plugin("1756824965", "noises")


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason="Test data directory not provided via STRAXION_TEST_DATA_DIR environment variable",
)
class TestNoiseBankWithRealDataOffline:
    """Test noise bank processing with real qualiphide_fir_test_data using offline context."""

    def test_qualiphide_thz_offline_context_creation(self):
        """Test that the qualiphide_thz_offline context can be created without errors."""
        st = straxion.qualiphide_thz_offline()
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

    def test_noise_bank_processing(self):
        """Test the complete noise bank processing pipeline with real data.

        This test requires the STRAXION_TEST_DATA_DIR environment variable to be set to the path
        containing the qualiphide_fir_test_data directory with example data.
        """
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        # Create context and process noises
        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = _get_test_config(test_data_dir, run_id)

        clean_strax_data()
        try:
            noises = st.get_array(run_id, "noises", config=configs)

            # Basic validation of the output
            assert noises is not None
            assert len(noises) >= 0  # Can be empty if no noises found

            # Check that all required fields are present
            required_fields = [
                "time",
                "endtime",
                "length",
                "dt",
                "channel",
                "data_dx",
                "data_dx_moving_average",
                "data_dx_convolved",
                "hit_threshold",
                "amplitude",
                "amplitude_moving_average",
                "amplitude_convolved",
            ]
            for field in required_fields:
                assert field in noises.dtype.names, f"Required field '{field}' missing from noises"

            # Check data types
            assert noises["time"].dtype == np.int64
            assert noises["endtime"].dtype == np.int64
            assert noises["length"].dtype == np.int64
            assert noises["dt"].dtype == np.int64
            assert noises["channel"].dtype == np.int16
            assert noises["data_dx"].dtype == np.float32
            assert noises["data_dx_moving_average"].dtype == np.float32
            assert noises["data_dx_convolved"].dtype == np.float32
            assert noises["hit_threshold"].dtype == np.float32
            assert noises["amplitude"].dtype == np.float32
            assert noises["amplitude_moving_average"].dtype == np.float32
            assert noises["amplitude_convolved"].dtype == np.float32

            # Check that all noises have reasonable lengths and dt
            if len(noises) > 0:
                assert all(noises["length"] > 0)
                assert all(noises["dt"] > 0)

                # Check that channels are within expected range (0-40 based on context config)
                assert all(0 <= noises["channel"]) and all(noises["channel"] <= 40)

                # Check that waveform data has the correct shape
                expected_waveform_length = HIT_WINDOW_LENGTH_LEFT + HIT_WINDOW_LENGTH_RIGHT
                for noise in noises:
                    assert noise["data_dx"].shape == (expected_waveform_length,)
                    assert noise["data_dx_moving_average"].shape == (expected_waveform_length,)
                    assert noise["data_dx_convolved"].shape == (expected_waveform_length,)

                # Check that noise characteristics are reasonable
                for noise in noises:
                    assert noise["amplitude"] >= 0
                    assert noise["amplitude_moving_average"] >= 0
                    assert noise["amplitude_convolved"] >= 0
                    assert noise["hit_threshold"] > 0

            print(
                f"Successfully processed {len(noises)} noise windows "
                f"from {len(np.unique(noises['channel'])) if len(noises) > 0 else 0} channels"
            )

        except Exception as e:
            pytest.fail(f"Failed to process noises: {str(e)}")

    def test_noise_bank_data_consistency(self):
        """Test that the noise bank data is internally consistent."""
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
            noises = st.get_array(run_id, "noises", config=configs)

            if len(noises) > 0:
                # Check endtime consistency
                for noise in noises:
                    expected_endtime = noise["time"] + noise["length"] * noise["dt"]
                    # Allow for small rounding differences in endtime calculation
                    assert (
                        abs(noise["endtime"] - expected_endtime) <= noise["dt"]
                    ), f"Endtime mismatch: {noise['endtime']} vs {expected_endtime}"

                # Check monotonic time
                assert np.all(
                    np.diff(noises["time"]) >= 0
                ), "Time stamps not monotonically increasing"

                # Check finite data
                assert np.all(np.isfinite(noises["data_dx"])), "Non-finite values found in data_dx"
                assert np.all(
                    np.isfinite(noises["data_dx_moving_average"])
                ), "Non-finite values found in data_dx_moving_average"
                assert np.all(
                    np.isfinite(noises["data_dx_convolved"])
                ), "Non-finite values found in data_dx_convolved"

        except Exception as e:
            pytest.fail(f"Failed to validate noise consistency: {str(e)}")

    def test_noise_bank_missing_data_directory(self):
        """Test that the noise bank plugin raises errors when data directory is missing."""
        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"

        configs = {
            "daq_input_dir": "/nonexistent/path.ts.npy",
            "iq_finescan_dir": "/nonexistent",
            "iq_finescan_filename": "iq_fine_z_nonexistent.npy",
            "iq_widescan_dir": "/nonexistent",
            "iq_widescan_filename": "iq_wide_z_nonexistent.npy",
            "resonant_frequency_dir": "/nonexistent",
            "resonant_frequency_filename": "fres_nonexistent.npy",
        }

        clean_strax_data()
        with pytest.raises((ValueError, FileNotFoundError)):
            st.get_array(run_id, "noises", config=configs)

    def test_noise_bank_invalid_config(self):
        """Test that the noise bank plugin handles invalid configuration gracefully."""
        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"

        # Test with invalid configuration
        configs = {
            "daq_input_dir": "/nonexistent/path.ts.npy",
            "iq_finescan_dir": "/nonexistent",
            "iq_finescan_filename": "iq_fine_z_nonexistent.npy",
            "iq_widescan_dir": "/nonexistent",
            "iq_widescan_filename": "iq_wide_z_nonexistent.npy",
            "resonant_frequency_dir": "/nonexistent",
            "resonant_frequency_filename": "fres_nonexistent.npy",
        }

        clean_strax_data()
        with pytest.raises(Exception):
            st.get_array(run_id, "noises", config=configs)


# Note: NoiseBank plugin is only available in the offline context (qualiphide_thz_offline)
# as it depends on DxHits which produces data_dx fields. The online context uses Hits
# which produces data_theta fields. If noise bank functionality is needed for the online
# context, a separate plugin would need to be created.
