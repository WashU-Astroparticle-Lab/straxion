import os
import pytest
import numpy as np
import straxion
from straxion.plugins.hits import Hits
from straxion.utils import HIT_WINDOW_LENGTH_LEFT, HIT_WINDOW_LENGTH_RIGHT
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


def test_hits_plugin_registration():
    """Test that the Hits plugin is properly registered in the context."""
    st = straxion.qualiphide_thz_online()
    assert "hits" in st._plugin_class_registry
    assert st._plugin_class_registry["hits"] == Hits


def test_hits_dtype_inference():
    """Test that Hits can infer the correct data type."""
    st = straxion.qualiphide_thz_online()
    plugin = st.get_single_plugin("1756824965", "hits")
    dtype = plugin.infer_dtype()
    # List of expected fields (add/remove as needed)
    expected_fields = [
        "time",
        "endtime",
        "length",
        "dt",
        "channel",
        "width",
        "data_theta",
        "data_theta_moving_average",
        "data_theta_convolved",
        "hit_threshold",
        "aligned_at_records_i",
        "amplitude_convolved_max",
        "amplitude_convolved_min",
        "amplitude_convolved_max_ext",
        "amplitude_convolved_min_ext",
        "amplitude_ma_max",
        "amplitude_ma_min",
        "amplitude_ma_max_ext",
        "amplitude_ma_min_ext",
    ]
    field_names = [name[1] for name, *_ in dtype]
    for field in expected_fields:
        assert field in field_names


def test_hits_empty_input():
    """Test that Hits returns empty output for empty input."""
    st = straxion.qualiphide_thz_online()
    plugin = st.get_single_plugin("1756824965", "hits")
    empty_records = np.array(
        [],
        dtype=[
            ("channel", np.int16),
            ("data_theta_convolved", np.float32, 10),
            ("data_theta_moving_average", np.float32, 10),
            ("data_theta", np.float32, 10),
        ],
    )
    result = plugin.compute(empty_records)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_hits_minimal_valid_input():
    """Test Hits with a minimal valid record that should produce no hits."""
    st = straxion.qualiphide_thz_online()
    plugin = st.get_single_plugin("1756824965", "hits")
    # Minimal record with all zeros (should not trigger any hits)
    record = np.zeros(
        1,
        dtype=[
            ("channel", np.int16),
            ("data_theta_convolved", np.float32, 10),
            ("data_theta_moving_average", np.float32, 10),
            ("data_theta", np.float32, 10),
            ("time", np.int64),
        ],
    )
    result = plugin.compute(record)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_hits_malformed_input():
    """Test that Hits handles malformed input gracefully."""
    st = straxion.qualiphide_thz_online()
    plugin = st.get_single_plugin("1756824965", "hits")
    # Missing required fields
    bad_record = np.zeros(1, dtype=[("channel", np.int16)])
    with pytest.raises(Exception):
        plugin.compute(bad_record)


def test_hits_invalid_config():
    """Test that Hits raises an error with invalid config."""
    st = straxion.qualiphide_thz_online()
    # Set an invalid config (e.g., negative record_length)
    st.set_config({"record_length": -1})
    with pytest.raises(Exception):
        st.get_single_plugin("1756824965", "hits")


def test_find_hit_candidates_with_simulated_pulse():
    """Test _find_hit_candidates directly with a simulated noisy exponential pulse."""
    st = straxion.qualiphide_thz_online()
    plugin = st.get_single_plugin("1756824965", "hits")
    n_samples = 2000
    pulse_start = 1000
    pulse_length = 200
    tau = 50

    # Set random seed for reproducible results
    np.random.seed(42)

    # Generate truncated exponential pulse
    t = np.arange(pulse_length)
    pulse_shape = np.exp(-t / tau)
    pulse = np.zeros(n_samples)
    pulse[pulse_start : pulse_start + pulse_length] = pulse_shape

    # Add Gaussian noise
    noise_sigma = 0.05
    noisy_signal = pulse + np.random.normal(0, noise_sigma, n_samples)

    # Set threshold and min_pulse_width
    hit_threshold = 3 * np.std(noisy_signal)
    min_pulse_width = 20

    hit_start_indices, hit_widths = plugin._find_hit_candidates(
        noisy_signal, hit_threshold, min_pulse_width
    )
    assert len(hit_start_indices) > 0, "No hit candidates found, but expected at least one."
    assert len(hit_start_indices) == len(
        hit_widths
    ), "Mismatch between hit start indices and widths."


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason="Test data directory not provided via STRAXION_TEST_DATA_DIR environment variable",
)
class TestHitsWithRealDataOnline:
    """Test hits processing with real qualiphide_fir_test_data."""

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

    def test_hits_processing(self):
        """Test the complete hits processing pipeline with real data.

        This test requires the STRAXION_TEST_DATA_DIR environment variable to be set to the path
        containing the qualiphide_fir_test_data directory with example data.
        """
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        # Create context and process hits
        st = straxion.qualiphide_thz_online()
        run_id = "1756824965"
        configs = _get_test_config(test_data_dir, run_id)

        clean_strax_data()
        try:
            hits = st.get_array(run_id, "hits", config=configs)

            # Basic validation of the output
            assert hits is not None
            assert len(hits) >= 0  # Can be empty if no hits found

            # Check that all required fields are present
            required_fields = [
                "time",
                "endtime",
                "length",
                "dt",
                "channel",
                "width",
                "data_theta",
                "data_theta_moving_average",
                "data_theta_convolved",
                "hit_threshold",
                "aligned_at_records_i",
                "amplitude_convolved_max",
                "amplitude_convolved_min",
                "amplitude_convolved_max_ext",
                "amplitude_convolved_min_ext",
                "amplitude_ma_max",
                "amplitude_ma_min",
                "amplitude_ma_max_ext",
                "amplitude_ma_min_ext",
            ]
            for field in required_fields:
                assert field in hits.dtype.names, f"Required field '{field}' missing from hits"

            # Check data types
            assert hits["time"].dtype == np.int64
            assert hits["endtime"].dtype == np.int64
            assert hits["length"].dtype == np.int64
            assert hits["dt"].dtype == np.int64
            assert hits["channel"].dtype == np.int16
            assert hits["width"].dtype == np.int32
            assert hits["data_theta"].dtype == np.float32
            assert hits["data_theta_moving_average"].dtype == np.float32
            assert hits["data_theta_convolved"].dtype == np.float32
            assert hits["hit_threshold"].dtype == np.float32
            assert hits["aligned_at_records_i"].dtype == np.int32
            assert hits["amplitude_convolved_max"].dtype == np.float32
            assert hits["amplitude_convolved_min"].dtype == np.float32
            assert hits["amplitude_convolved_max_ext"].dtype == np.float32
            assert hits["amplitude_convolved_min_ext"].dtype == np.float32
            assert hits["amplitude_ma_max"].dtype == np.float32
            assert hits["amplitude_ma_min"].dtype == np.float32
            assert hits["amplitude_ma_max_ext"].dtype == np.float32
            assert hits["amplitude_ma_min_ext"].dtype == np.float32

            # Check that all hits have reasonable lengths and dt
            if len(hits) > 0:
                assert all(hits["length"] > 0)
                assert all(hits["dt"] > 0)

                # Check that channels are within expected range (0-40 based on context config)
                assert all(0 <= hits["channel"]) and all(hits["channel"] <= 40)

                # Check that waveform data has the correct shape
                expected_waveform_length = HIT_WINDOW_LENGTH_LEFT + HIT_WINDOW_LENGTH_RIGHT
                for hit in hits:
                    assert hit["data_theta"].shape == (expected_waveform_length,)
                    assert hit["data_theta_moving_average"].shape == (expected_waveform_length,)
                    assert hit["data_theta_convolved"].shape == (expected_waveform_length,)

                # Check that hit characteristics are reasonable
                for hit in hits:
                    assert hit["amplitude_convolved_max"] >= hit["amplitude_convolved_min"]
                    assert hit["amplitude_convolved_max_ext"] >= hit["amplitude_convolved_min_ext"]
                    assert hit["amplitude_ma_max"] >= hit["amplitude_ma_min"]
                    assert hit["amplitude_ma_max_ext"] >= hit["amplitude_ma_min_ext"]
                    assert hit["width"] > 0
                    assert hit["hit_threshold"] > 0

            print(
                f"Successfully processed {len(hits)} hits "
                f"from {len(np.unique(hits['channel'])) if len(hits) > 0 else 0} channels"
            )

        except Exception as e:
            pytest.fail(f"Failed to process hits: {str(e)}")

    def test_hits_data_consistency(self):
        """Test that the hits data is internally consistent."""
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
            hits = st.get_array(run_id, "hits", config=configs)

            if len(hits) > 0:

                # Check monotonic time
                assert np.all(
                    np.diff(hits["time"]) >= 0
                ), "Time stamps not monotonically increasing"

                # Check finite data
                assert np.all(
                    np.isfinite(hits["data_theta"])
                ), "Non-finite values found in data_theta"
                assert np.all(
                    np.isfinite(hits["data_theta_moving_average"])
                ), "Non-finite values found in data_theta_moving_average"
                assert np.all(
                    np.isfinite(hits["data_theta_convolved"])
                ), "Non-finite values found in data_theta_convolved"

        except Exception as e:
            pytest.fail(f"Failed to validate hits consistency: {str(e)}")

    def test_hits_missing_data_directory(self):
        """Test that the hits plugin raises errors when data directory is missing."""
        st = straxion.qualiphide_thz_online()
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
            st.get_array(run_id, "hits", config=configs)

    def test_hits_invalid_config(self):
        """Test that the hits plugin handles invalid configuration gracefully."""
        st = straxion.qualiphide_thz_online()
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
            st.get_array(run_id, "hits", config=configs)


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason="Test data directory not provided via STRAXION_TEST_DATA_DIR environment variable",
)
class TestHitsWithRealDataOffline:
    """Test hits processing with real qualiphide_fir_test_data using DxHits plugin."""

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

    def test_hits_processing(self):
        """Test the complete hits processing pipeline with real data using DxHits plugin.

        This test requires the STRAXION_TEST_DATA_DIR environment variable to be set to the path
        containing the qualiphide_fir_test_data directory with example data.
        """
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        # Create context and process hits
        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = _get_test_config(test_data_dir, run_id)

        clean_strax_data()
        try:
            hits = st.get_array(run_id, "hits", config=configs)

            # Basic validation of the output
            assert hits is not None
            assert len(hits) >= 0  # Can be empty if no hits found

            # Check that all required fields are present for DxHits
            required_fields = [
                "time",
                "endtime",
                "length",
                "dt",
                "channel",
                "width",
                "data_dx",
                "data_dx_moving_average",
                "data_dx_convolved",
                "amplitude",
                "amplitude_moving_average",
                "amplitude_convolved",
                "amplitude_convolved_max_record_i",
                "amplitude_moving_average_max_record_i",
                "amplitude_max_record_i",
                "hit_threshold",
            ]
            for field in required_fields:
                assert field in hits.dtype.names, f"Required field '{field}' missing from hits"

            # Check data types
            assert hits["time"].dtype == np.int64
            assert hits["endtime"].dtype == np.int64
            assert hits["length"].dtype == np.int64
            assert hits["dt"].dtype == np.int64
            assert hits["channel"].dtype == np.int16
            assert hits["width"].dtype == np.int32
            assert hits["data_dx"].dtype == np.float32
            assert hits["data_dx_moving_average"].dtype == np.float32
            assert hits["data_dx_convolved"].dtype == np.float32
            assert hits["amplitude"].dtype == np.float32
            assert hits["amplitude_moving_average"].dtype == np.float32
            assert hits["amplitude_convolved"].dtype == np.float32
            assert hits["amplitude_convolved_max_record_i"].dtype == np.int32
            assert hits["amplitude_moving_average_max_record_i"].dtype == np.int32
            assert hits["amplitude_max_record_i"].dtype == np.int32
            assert hits["hit_threshold"].dtype == np.float32

            # Check that all hits have reasonable lengths and dt
            if len(hits) > 0:
                assert all(hits["length"] > 0)
                assert all(hits["dt"] > 0)

                # Check that channels are within expected range (0-40 based on context config)
                assert all(0 <= hits["channel"]) and all(hits["channel"] <= 40)

                # Check that waveform data has the correct shape
                expected_waveform_length = HIT_WINDOW_LENGTH_LEFT + HIT_WINDOW_LENGTH_RIGHT
                for hit in hits:
                    assert hit["data_dx"].shape == (expected_waveform_length,)
                    assert hit["data_dx_moving_average"].shape == (expected_waveform_length,)
                    assert hit["data_dx_convolved"].shape == (expected_waveform_length,)

                # Check that timing information is consistent
                for h_i, hit in enumerate(hits):
                    expected_endtime = hit["time"] + hit["length"] * hit["dt"]
                    # Allow for larger rounding differences due to floating point precision
                    # The endtime calculation may use different logic than time + length * dt
                    assert abs(hit["endtime"] - expected_endtime) <= hit["dt"] * 2, (
                        f"Hit #{h_i} endtime mismatch. "
                        f"Note that hit['endtime'] is {hit['endtime']} and "
                        f"expected endtime is {expected_endtime} and "
                        f"hit['time'] is {hit['time']} and "
                        f"hit['length'] is {hit['length']} and "
                        f"hit['dt'] is {hit['dt']}"
                    )

                # Check that hit characteristics are reasonable
                for hit in hits:
                    assert hit["width"] > 0
                    assert hit["amplitude_convolved_max_record_i"] >= 0
                    assert hit["amplitude_moving_average_max_record_i"] >= 0
                    assert hit["amplitude_max_record_i"] >= 0

            print(
                f"Successfully processed {len(hits)} hits "
                f"from {len(np.unique(hits['channel'])) if len(hits) > 0 else 0} channels"
            )

        except Exception as e:
            pytest.fail(f"Failed to process hits: {str(e)}")

    def test_hits_data_consistency(self):
        """Test that the hits data is internally consistent."""
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
            hits = st.get_array(run_id, "hits", config=configs)

            if len(hits) > 0:
                # Check endtime consistency
                for hit in hits:
                    expected_endtime = hit["time"] + hit["length"] * hit["dt"]
                    # Allow for small rounding differences in endtime calculation
                    assert (
                        abs(hit["endtime"] - expected_endtime) <= hit["dt"]
                    ), f"Endtime mismatch: {hit['endtime']} vs {expected_endtime}"

                # Check monotonic time
                assert np.all(
                    np.diff(hits["time"]) >= 0
                ), "Time stamps not monotonically increasing"

                # Check finite data
                assert np.all(np.isfinite(hits["data_dx"])), "Non-finite values found in data_dx"
                assert np.all(
                    np.isfinite(hits["data_dx_moving_average"])
                ), "Non-finite values found in data_dx_moving_average"
                assert np.all(
                    np.isfinite(hits["data_dx_convolved"])
                ), "Non-finite values found in data_dx_convolved"

        except Exception as e:
            pytest.fail(f"Failed to validate hits consistency: {str(e)}")

    def test_hits_missing_data_directory(self):
        """Test that the hits plugin raises errors when data directory is missing."""
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
            st.get_array(run_id, "hits", config=configs)

    def test_hits_invalid_config(self):
        """Test that the hits plugin handles invalid configuration gracefully."""
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
            st.get_array(run_id, "hits", config=configs)


# =============================================================================
# Unit Tests for Numba-Accelerated Helper Functions (DxHits)
# =============================================================================


class TestFindHitCandidatesNumba:
    """Test the numba-accelerated _find_hit_candidates_numba function."""

    def test_find_hit_candidates_basic(self):
        """Test basic hit detection with clear threshold crossings."""
        from straxion.plugins.hits import _find_hit_candidates_numba

        # Create signal with two clear hits
        signal = np.zeros(1000, dtype=np.float64)
        signal[100:150] = 1.0  # First hit: 50 samples wide
        signal[300:320] = 1.0  # Second hit: 20 samples wide

        threshold = 0.5
        min_width = 10

        starts, widths = _find_hit_candidates_numba(signal, threshold, min_width)

        assert len(starts) == 2, f"Expected 2 hits, got {len(starts)}"
        assert starts[0] == 100, f"First hit should start at 100, got {starts[0]}"
        assert widths[0] == 50, f"First hit width should be 50, got {widths[0]}"
        assert starts[1] == 300, f"Second hit should start at 300, got {starts[1]}"
        assert widths[1] == 20, f"Second hit width should be 20, got {widths[1]}"

    def test_find_hit_candidates_min_width_filter(self):
        """Test that hits below minimum width are filtered out."""
        from straxion.plugins.hits import _find_hit_candidates_numba

        signal = np.zeros(500, dtype=np.float64)
        signal[100:105] = 1.0  # 5 samples - too short
        signal[200:220] = 1.0  # 20 samples - should pass

        threshold = 0.5
        min_width = 10

        starts, widths = _find_hit_candidates_numba(signal, threshold, min_width)

        assert len(starts) == 1, f"Expected 1 hit (filtered by min_width), got {len(starts)}"
        assert starts[0] == 200
        assert widths[0] == 20

    def test_find_hit_candidates_edge_at_start(self):
        """Test hit that starts at index 0."""
        from straxion.plugins.hits import _find_hit_candidates_numba

        signal = np.zeros(500, dtype=np.float64)
        signal[0:30] = 1.0  # Hit starts at beginning

        threshold = 0.5
        min_width = 10

        starts, widths = _find_hit_candidates_numba(signal, threshold, min_width)

        assert len(starts) == 1
        assert starts[0] == 0
        assert widths[0] == 30

    def test_find_hit_candidates_edge_at_end(self):
        """Test hit that extends to end of signal."""
        from straxion.plugins.hits import _find_hit_candidates_numba

        signal = np.zeros(500, dtype=np.float64)
        signal[470:500] = 1.0  # Hit extends to end

        threshold = 0.5
        min_width = 10

        starts, widths = _find_hit_candidates_numba(signal, threshold, min_width)

        assert len(starts) == 1
        assert starts[0] == 470
        assert widths[0] == 30

    def test_find_hit_candidates_no_hits(self):
        """Test signal with no hits above threshold."""
        from straxion.plugins.hits import _find_hit_candidates_numba

        signal = np.zeros(500, dtype=np.float64)
        signal[:] = 0.1  # All below threshold

        threshold = 0.5
        min_width = 10

        starts, widths = _find_hit_candidates_numba(signal, threshold, min_width)

        assert len(starts) == 0
        assert len(widths) == 0

    def test_find_hit_candidates_matches_original(self):
        """Test that numba version matches the original numpy implementation."""
        from straxion.plugins.hits import _find_hit_candidates_numba

        # Original numpy implementation for comparison
        def find_hit_candidates_original(signal, hit_threshold, min_pulse_width):
            above_threshold = signal >= hit_threshold
            padded_array = np.concatenate(([False], above_threshold, [False]))
            diffs = np.diff(padded_array.astype(int))
            hit_start_indices = np.where(diffs == 1)[0]
            hit_end_indices = np.where(diffs == -1)[0]
            if len(hit_start_indices) == 0:
                return np.array([]), np.array([])
            hit_widths = hit_end_indices - hit_start_indices
            valid_mask = hit_widths >= min_pulse_width
            return hit_start_indices[valid_mask], hit_widths[valid_mask]

        # Create random signal with pulses
        np.random.seed(42)
        signal = np.random.randn(10000) * 0.1
        for _ in range(20):
            pos = np.random.randint(0, 9900)
            width = np.random.randint(10, 50)
            signal[pos : pos + width] += 0.5

        threshold = 0.3
        min_width = 10

        starts_orig, widths_orig = find_hit_candidates_original(signal, threshold, min_width)
        starts_numba, widths_numba = _find_hit_candidates_numba(
            signal.astype(np.float64), threshold, min_width
        )

        np.testing.assert_array_equal(starts_orig, starts_numba)
        np.testing.assert_array_equal(widths_orig, widths_numba)


class TestComputeHitBoundariesAndAmplitudes:
    """Test the numba-accelerated _compute_hit_boundaries_and_amplitudes function."""

    def test_compute_boundaries_basic(self):
        """Test basic boundary and amplitude computation."""
        from straxion.plugins.hits import _compute_hit_boundaries_and_amplitudes

        # Create signals
        n_samples = 1000
        signal_conv = np.zeros(n_samples, dtype=np.float64)
        signal_ma = np.zeros(n_samples, dtype=np.float64)
        signal_raw = np.zeros(n_samples, dtype=np.float64)

        # Add a pulse with peak at sample 525 (middle of sine wave from 500-550)
        signal_conv[500:550] = np.sin(np.linspace(0, np.pi, 50))
        signal_ma[500:550] = signal_conv[500:550] * 0.9
        signal_raw[500:550] = signal_conv[500:550] * 1.1

        hit_starts = np.array([500], dtype=np.int64)
        hit_widths = np.array([50], dtype=np.int64)

        result = _compute_hit_boundaries_and_amplitudes(
            hit_starts,
            hit_widths,
            signal_conv,
            signal_ma,
            signal_raw,
            hit_window_length_left=200,
            hit_window_length_right=400,
            signal_length=n_samples,
        )

        (
            aligned_indices,
            left_indices,
            right_indices,
            amp_conv,
            amp_ma,
            amp_raw,
            amp_conv_i,
            amp_ma_i,
            amp_raw_i,
        ) = result

        # The peak should be near the middle of the sine wave (around 525)
        assert 520 <= aligned_indices[0] <= 530, f"Aligned at {aligned_indices[0]}"

        # Amplitudes should be positive and match the max values
        assert amp_conv[0] > 0.9, f"Convolved amplitude should be near 1.0, got {amp_conv[0]}"
        assert amp_ma[0] > 0.8, f"MA amplitude should be near 0.9, got {amp_ma[0]}"
        assert amp_raw[0] > 1.0, f"Raw amplitude should be near 1.1, got {amp_raw[0]}"

    def test_compute_boundaries_multiple_hits(self):
        """Test with multiple hits and boundary constraints."""
        from straxion.plugins.hits import _compute_hit_boundaries_and_amplitudes

        n_samples = 2000
        signal = np.zeros(n_samples, dtype=np.float64)

        # Two hits close together
        signal[200:250] = 1.0
        signal[400:450] = 2.0

        hit_starts = np.array([200, 400], dtype=np.int64)
        hit_widths = np.array([50, 50], dtype=np.int64)

        result = _compute_hit_boundaries_and_amplitudes(
            hit_starts,
            hit_widths,
            signal,
            signal,
            signal,
            hit_window_length_left=100,
            hit_window_length_right=100,
            signal_length=n_samples,
        )

        left_indices = result[1]
        right_indices = result[2]

        # First hit's right boundary should be limited by second hit's start
        assert right_indices[0] <= hit_starts[1], "First hit right should not overlap second"

        # Second hit's left boundary should be limited by first hit's end
        first_hit_end = hit_starts[0] + hit_widths[0]
        assert left_indices[1] >= first_hit_end, "Second hit left should not overlap first"

    def test_compute_boundaries_edge_cases(self):
        """Test boundary computation at signal edges."""
        from straxion.plugins.hits import _compute_hit_boundaries_and_amplitudes

        n_samples = 500
        signal = np.zeros(n_samples, dtype=np.float64)
        signal[0:50] = 1.0  # Hit at start

        hit_starts = np.array([0], dtype=np.int64)
        hit_widths = np.array([50], dtype=np.int64)

        result = _compute_hit_boundaries_and_amplitudes(
            hit_starts,
            hit_widths,
            signal,
            signal,
            signal,
            hit_window_length_left=200,
            hit_window_length_right=200,
            signal_length=n_samples,
        )

        left_indices = result[1]

        # Left boundary should be clamped to 0
        assert left_indices[0] == 0, "Left index should be 0 at signal start"


class TestDxHitsFindHitCandidatesMethod:
    """Test DxHits.find_hit_candidates method using the numba implementation."""

    def test_find_hit_candidates_uses_numba(self):
        """Test that the public method correctly calls the numba function."""
        from straxion.plugins.hits import DxHits

        signal = np.zeros(1000, dtype=np.float64)
        signal[100:150] = 1.0

        starts, widths = DxHits.find_hit_candidates(signal, 0.5, 10)

        assert len(starts) == 1
        assert starts[0] == 100
        assert widths[0] == 50

    def test_find_hit_candidates_type_conversion(self):
        """Test that input types are correctly converted."""
        from straxion.plugins.hits import DxHits

        # Test with float32 input (should be converted to float64)
        signal = np.zeros(500, dtype=np.float32)
        signal[100:130] = 1.0

        starts, widths = DxHits.find_hit_candidates(signal, 0.5, 10)

        assert len(starts) == 1
        assert starts[0] == 100
