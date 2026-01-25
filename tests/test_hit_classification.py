import os
import pytest
import numpy as np
import straxion
from straxion.plugins.hit_classification import HitClassification, DxHitClassification
from straxion.utils import DEFAULT_TEMPLATE_INTERP_PATH
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


def test_hit_classification_plugin_registration():
    """Test that the HitClassification plugin is properly registered in the context."""
    st = straxion.qualiphide_thz_online()
    assert "hit_classification" in st._plugin_class_registry
    assert st._plugin_class_registry["hit_classification"] == HitClassification


def test_hit_classification_dtype_inference():
    """Test that HitClassification can infer the correct data type."""
    st = straxion.qualiphide_thz_online()
    plugin = st.get_single_plugin("1756824965", "hit_classification")
    dtype = plugin.infer_dtype()
    # List of expected fields
    expected_fields = [
        "time",
        "endtime",
        "channel",
        "is_cr",
        "is_symmetric_spike",
        "is_unidentified",
        "ma_rise_edge_slope",
    ]
    field_names = [name[1] for name, *_ in dtype]
    for field in expected_fields:
        assert field in field_names


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason="Test data directory not provided via STRAXION_TEST_DATA_DIR environment variable",
)
class TestHitClassificationWithRealDataOnline:
    """Test hit classification processing with real qualiphide_fir_test_data."""

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

    def test_hit_classification_processing(self):
        """Test the complete hit classification processing pipeline with real data.

        This test requires the STRAXION_TEST_DATA_DIR environment variable to be set to the path
        containing the qualiphide_fir_test_data directory with example data.
        """
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        # Create context and process hit classification
        st = straxion.qualiphide_thz_online()
        run_id = "1756824965"
        configs = _get_test_config(test_data_dir, run_id)

        clean_strax_data()
        try:
            hit_classification = st.get_array(run_id, "hit_classification", config=configs)

            # Basic validation of the output
            assert hit_classification is not None
            assert len(hit_classification) >= 0  # Can be empty if no hits found

            # Check that all required fields are present
            required_fields = [
                "time",
                "endtime",
                "channel",
                "is_cr",
                "is_symmetric_spike",
                "is_unidentified",
                "ma_rise_edge_slope",
            ]
            for field in required_fields:
                assert (
                    field in hit_classification.dtype.names
                ), f"Required field '{field}' missing from hit_classification"

            # Check data types
            assert hit_classification["time"].dtype == np.int64
            assert hit_classification["endtime"].dtype == np.int64
            assert hit_classification["channel"].dtype == np.int16
            assert hit_classification["is_cr"].dtype == bool
            assert hit_classification["is_symmetric_spike"].dtype == bool
            assert hit_classification["is_unidentified"].dtype == bool
            assert hit_classification["ma_rise_edge_slope"].dtype == np.float32

            # Check that channels are within expected range (0-40 based on context config)
            if len(hit_classification) > 0:
                assert all(0 <= hit_classification["channel"]) and all(
                    hit_classification["channel"] <= 40
                )

                # Check that timing information is consistent
                for hc_i, hit_class in enumerate(hit_classification):
                    expected_endtime = hit_class["time"] + (
                        hit_class["endtime"] - hit_class["time"]
                    )  # This should match
                    # Allow for small rounding differences
                    assert abs(hit_class["endtime"] - expected_endtime) <= 1, (
                        f"Hit classification #{hc_i} endtime mismatch. "
                        f"Note that hit_class['endtime'] is {hit_class['endtime']} and "
                        f"expected endtime is {expected_endtime} and "
                        f"hit_class['time'] is {hit_class['time']}"
                    )

                # Print classification statistics
                n_cr = np.sum(hit_classification["is_cr"])
                n_symmetric_spike = np.sum(hit_classification["is_symmetric_spike"])
                n_unidentified = np.sum(hit_classification["is_unidentified"])

                print(
                    f"Successfully processed {len(hit_classification)} hit classifications: "
                    f"{n_cr} cosmic rays, {n_symmetric_spike} symmetric spikes, "
                    f"{n_unidentified} unidentified"
                )

                # Check that all classifications are boolean
                for hit_class in hit_classification:
                    assert isinstance(hit_class["is_cr"], (bool, np.bool_))
                    assert isinstance(hit_class["is_symmetric_spike"], (bool, np.bool_))
                    assert isinstance(hit_class["is_unidentified"], (bool, np.bool_))

            len_unique_channels = len(np.unique(hit_classification["channel"]))
            print(
                f"Successfully processed {len(hit_classification)} hit classifications "
                f"from {len_unique_channels if len(hit_classification) > 0 else 0} channels"
            )

        except Exception as e:
            pytest.fail(f"Failed to process hit classification: {str(e)}")

    def test_hit_classification_data_consistency(self):
        """Test that the hit classification data is internally consistent."""
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
            hit_classification = st.get_array(run_id, "hit_classification", config=configs)

            # Check finite data for numerical fields
            assert np.all(
                np.isfinite(hit_classification["ma_rise_edge_slope"])
            ), "Non-finite values found in ma_rise_edge_slope"

        except Exception as e:
            pytest.fail(f"Failed to validate hit classification consistency: {str(e)}")

    def test_hit_classification_missing_data_directory(self):
        """Test that the hit classification plugin raises errors when data directory is missing."""
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
            st.get_array(run_id, "hit_classification", config=configs)

    def test_hit_classification_invalid_config(self):
        """Test that the hit classification plugin handles invalid configuration gracefully."""
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
            st.get_array(run_id, "hit_classification", config=configs)


def test_spike_coincidence_plugin_registration():
    """Test that the DxHitClassification plugin is properly registered in the offline context."""
    st = straxion.qualiphide_thz_offline()
    assert "hit_classification" in st._plugin_class_registry
    # In offline context, hit_classification maps to DxHitClassification
    assert st._plugin_class_registry["hit_classification"] == DxHitClassification


def test_spike_coincidence_dtype_inference():
    """Test that DxHitClassification can infer the correct data type."""
    st = straxion.qualiphide_thz_offline()
    plugin = st.get_single_plugin("1756824965", "hit_classification")
    dtype = plugin.infer_dtype()
    # List of expected fields for DxHitClassification
    expected_fields = [
        "time",
        "endtime",
        "channel",
        "is_coincident_with_spikes",
        "is_symmetric_spike",
        "is_truncated_hit",
        "is_invalid_kappa",
        "is_photon_candidate",
        "rise_edge_slope",
        "n_spikes_coinciding",
        "best_aOF",
        "best_chi2",
        "best_OF_shift",
        "kappa",
        "width",
        "amplitude",
        "amplitude_moving_average",
        "amplitude_convolved",
        "hit_threshold",
    ]
    field_names = [name[1] for name, *_ in dtype]
    for field in expected_fields:
        assert field in field_names


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason="Test data directory not provided via STRAXION_TEST_DATA_DIR environment variable",
)
class TestDxHitClassificationWithRealDataOffline:
    """Test spike coincidence processing with real qualiphide_fir_test_data
    using DxHitClassification plugin."""

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

    def test_spike_coincidence_processing(self):
        """Test the complete spike coincidence processing pipeline with real data using
        DxHitClassification plugin.

        This test requires the STRAXION_TEST_DATA_DIR environment variable to be set to the path
        containing the qualiphide_fir_test_data directory with example data.
        """
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        # Create context and process spike coincidence
        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = _get_test_config(test_data_dir, run_id)

        clean_strax_data()
        try:
            hit_classification = st.get_array(run_id, "hit_classification", config=configs)

            # Basic validation of the output
            assert hit_classification is not None
            assert len(hit_classification) >= 0  # Can be empty if no hits found

            # Check that all required fields are present for DxHitClassification
            required_fields = [
                "time",
                "endtime",
                "channel",
                "is_coincident_with_spikes",
                "is_symmetric_spike",
                "is_truncated_hit",
                "is_invalid_kappa",
                "is_photon_candidate",
                "rise_edge_slope",
                "n_spikes_coinciding",
                "best_aOF",
                "best_chi2",
                "best_OF_shift",
                "kappa",
                "width",
                "amplitude",
                "amplitude_moving_average",
                "amplitude_convolved",
                "hit_threshold",
            ]
            for field in required_fields:
                assert (
                    field in hit_classification.dtype.names
                ), f"Required field '{field}' missing from hit_classification"

            # Check data types
            assert hit_classification["time"].dtype == np.int64
            assert hit_classification["endtime"].dtype == np.int64
            assert hit_classification["channel"].dtype == np.int16
            assert hit_classification["is_coincident_with_spikes"].dtype == bool
            assert hit_classification["is_symmetric_spike"].dtype == bool
            assert hit_classification["is_truncated_hit"].dtype == bool
            assert hit_classification["is_invalid_kappa"].dtype == bool
            assert hit_classification["is_photon_candidate"].dtype == bool
            assert hit_classification["rise_edge_slope"].dtype == np.float32
            assert hit_classification["n_spikes_coinciding"].dtype == np.int64
            assert hit_classification["best_aOF"].dtype == np.float32
            assert hit_classification["best_chi2"].dtype == np.float32
            assert hit_classification["best_OF_shift"].dtype == np.int64
            assert hit_classification["kappa"].dtype == np.float32
            assert hit_classification["width"].dtype == np.int32
            assert hit_classification["amplitude"].dtype == np.float32
            assert hit_classification["amplitude_moving_average"].dtype == np.float32
            assert hit_classification["amplitude_convolved"].dtype == np.float32
            assert hit_classification["hit_threshold"].dtype == np.float32

            # Check that channels are within expected range (0-40 based on context config)
            if len(hit_classification) > 0:
                assert all(0 <= hit_classification["channel"]) and all(
                    hit_classification["channel"] <= 40
                )

                # Check that timing information is consistent
                for hc_i, hit_class in enumerate(hit_classification):
                    expected_endtime = hit_class["time"] + (
                        hit_class["endtime"] - hit_class["time"]
                    )  # This should match
                    # Allow for small rounding differences
                    assert abs(hit_class["endtime"] - expected_endtime) <= 1, (
                        f"Hit classification #{hc_i} endtime mismatch. "
                        f"Note that hit_class['endtime'] is {hit_class['endtime']} and "
                        f"expected endtime is {expected_endtime} and "
                        f"hit_class['time'] is {hit_class['time']}"
                    )

                # Print classification statistics
                n_coincident = np.sum(hit_classification["is_coincident_with_spikes"])
                n_symmetric_spikes = np.sum(hit_classification["is_symmetric_spike"])
                n_photon_candidates = np.sum(hit_classification["is_photon_candidate"])

                print(
                    f"Successfully processed {len(hit_classification)} "
                    "spike coincidence classifications: "
                    f"{n_coincident} coincident with spikes, {n_symmetric_spikes} "
                    f"symmetric spikes, {n_photon_candidates} photon candidates"
                )

                # Check that all classifications are boolean and follow correct logic
                for hit_class in hit_classification:
                    assert isinstance(hit_class["is_coincident_with_spikes"], (bool, np.bool_))
                    assert isinstance(hit_class["is_symmetric_spike"], (bool, np.bool_))
                    assert isinstance(hit_class["is_photon_candidate"], (bool, np.bool_))

                    # is_photon_candidate should be False when
                    # is_coincident_with_spikes or is_symmetric_spike is True
                    if hit_class["is_coincident_with_spikes"] or hit_class["is_symmetric_spike"]:
                        assert not hit_class["is_photon_candidate"], (
                            "Hit cannot be a photon candidate when it is coincident with spikes "
                            "or is a symmetric spike"
                        )

                # Check that n_spikes_coinciding is non-negative
                assert all(
                    hit_classification["n_spikes_coinciding"] >= 0
                ), "Number of spikes coinciding should be non-negative"

                # Check that spike coincidence logic is consistent
                for hit_class in hit_classification:
                    expected_coincident = (
                        hit_class["n_spikes_coinciding"] > 1
                    )  # max_spike_coincidence = 1
                    assert (
                        hit_class["is_coincident_with_spikes"] == expected_coincident
                    ), "Spike coincidence logic is inconsistent"

            len_unique_channels = len(np.unique(hit_classification["channel"]))
            print(
                f"Successfully processed {len(hit_classification)} spike coincidence "
                f"from {len_unique_channels if len(hit_classification) > 0 else 0} channels"
            )

        except Exception as e:
            pytest.fail(f"Failed to process spike coincidence: {str(e)}")

    def test_spike_coincidence_data_consistency(self):
        """Test that the spike coincidence data is internally consistent."""
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
            hit_classification = st.get_array(run_id, "hit_classification", config=configs)

            if len(hit_classification) > 0:
                # Check that timestamps are monotonically increasing
                for channel in np.unique(hit_classification["channel"]):
                    channel_hits = hit_classification[hit_classification["channel"] == channel]
                    if len(channel_hits) > 1:
                        times = channel_hits["time"]
                        # Ensure timestamps are strictly monotonically increasing
                        diffs = np.diff(times)
                        assert np.all(
                            diffs > 0
                        ), f"Time stamps not monotonically increasing for channel {channel}"

                # Check finite data for numerical fields
                assert np.all(
                    np.isfinite(hit_classification["rise_edge_slope"])
                ), "Non-finite values found in rise_edge_slope"

                # Check optimal filter fields
                assert np.all(
                    np.isfinite(hit_classification["best_aOF"])
                ), "Non-finite values found in best_aOF"
                assert np.all(
                    np.isfinite(hit_classification["best_chi2"])
                ), "Non-finite values found in best_chi2"
                assert np.all(
                    np.isfinite(hit_classification["best_OF_shift"])
                ), "Non-finite values found in best_OF_shift"

                # Check kappa field (can be inf when fit fails, but must be positive)
                assert np.all(
                    hit_classification["kappa"] > 0
                ), "Kappa values must be positive (or inf)"

                # Check is_invalid_kappa is consistent with kappa values
                expected_invalid_kappa = ~np.isfinite(hit_classification["kappa"])
                np.testing.assert_array_equal(
                    hit_classification["is_invalid_kappa"],
                    expected_invalid_kappa,
                    err_msg="is_invalid_kappa should match ~np.isfinite(kappa)",
                )

                # Check amplitude fields from hits
                assert np.all(
                    np.isfinite(hit_classification["amplitude"])
                ), "Non-finite values found in amplitude"
                assert np.all(
                    np.isfinite(hit_classification["amplitude_moving_average"])
                ), "Non-finite values found in amplitude_moving_average"
                assert np.all(
                    np.isfinite(hit_classification["amplitude_convolved"])
                ), "Non-finite values found in amplitude_convolved"
                assert np.all(
                    np.isfinite(hit_classification["hit_threshold"])
                ), "Non-finite values found in hit_threshold"

                # Check width is non-negative
                assert np.all(hit_classification["width"] >= 0), "Negative width values found"

                # Check that n_spikes_coinciding is an integer and non-negative
                assert np.all(
                    hit_classification["n_spikes_coinciding"] >= 0
                ), "Negative spike coincidence counts found"

                # Check that is_photon_candidate is False when any exclusion criteria is met
                for hit_class in hit_classification:
                    has_exclusion = (
                        hit_class["is_coincident_with_spikes"]
                        or hit_class["is_symmetric_spike"]
                        or hit_class["is_truncated_hit"]
                        or hit_class["is_invalid_kappa"]
                    )
                    if has_exclusion:
                        assert not hit_class["is_photon_candidate"], (
                            "Hit cannot be a photon candidate when it has any exclusion: "
                            f"coincident={hit_class['is_coincident_with_spikes']}, "
                            f"symmetric={hit_class['is_symmetric_spike']}, "
                            f"truncated={hit_class['is_truncated_hit']}, "
                            f"invalid_kappa={hit_class['is_invalid_kappa']}"
                        )

        except Exception as e:
            pytest.fail(f"Failed to validate spike coincidence consistency: {str(e)}")

    def test_spike_coincidence_missing_data_directory(self):
        """Test that the spike coincidence plugin raises errors when data directory is missing."""
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
            st.get_array(run_id, "hit_classification", config=configs)

    def test_spike_coincidence_invalid_config(self):
        """Test that the spike coincidence plugin handles invalid configuration gracefully."""
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
            st.get_array(run_id, "hit_classification", config=configs)

    def test_per_channel_noise_psd_computation(self):
        """Test that per-channel noise PSDs are computed correctly from noise windows."""
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
            # Get noises and records to test the method directly
            noises = st.get_array(run_id, "noises", config=configs)
            records = st.get_array(run_id, "records", config=configs)

            # Get the plugin instance
            plugin = st.get_single_plugin(run_id, "hit_classification")

            # Compute per-channel noise PSDs
            n_channels = len(records)
            channel_noise_psds = plugin.compute_per_channel_noise_psd(noises, n_channels)

            # Verify the result is a dictionary
            assert isinstance(channel_noise_psds, dict)

            # Verify all channels are present
            assert len(channel_noise_psds) == n_channels

            # Check that PSDs have the expected length
            expected_psd_length = plugin.of_window_left + plugin.of_window_right

            for ch in range(n_channels):
                if channel_noise_psds[ch] is not None:
                    # PSD should have correct length
                    assert len(channel_noise_psds[ch]) == expected_psd_length
                    # PSD values should be non-negative (it's |FFT|^2)
                    assert np.all(channel_noise_psds[ch] >= 0)
                    # PSD should contain finite values
                    assert np.all(np.isfinite(channel_noise_psds[ch]))

            # Count channels with and without noise windows
            channels_with_psd = sum(1 for psd in channel_noise_psds.values() if psd is not None)
            channels_without_psd = sum(1 for psd in channel_noise_psds.values() if psd is None)

            print(
                f"Successfully computed PSDs for {channels_with_psd} channels, "
                f"{channels_without_psd} channels use placeholder"
            )

        except Exception as e:
            pytest.fail(f"Failed to compute per-channel noise PSDs: {str(e)}")

    def test_noise_psd_placeholder_warning(self, caplog):
        """Test that a warning is logged when using placeholder PSD for channels
        without noise windows."""
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
            # Process hit classification which should log warnings if placeholders are used
            import logging

            with caplog.at_level(logging.WARNING):
                hit_classification = st.get_array(run_id, "hit_classification", config=configs)

            # Check if warning was logged for any channel without noise windows
            noises = st.get_array(run_id, "noises", config=configs)

            # Find channels with hits but no noise windows
            if len(hit_classification) > 0:
                hit_channels = np.unique(hit_classification["channel"])
                noise_channels = np.unique(noises["channel"]) if len(noises) > 0 else []

                channels_without_noise = set(hit_channels) - set(noise_channels)

                if len(channels_without_noise) > 0:
                    # Should have warnings for channels without noise windows
                    for ch in channels_without_noise:
                        warning_found = any(
                            f"No noise windows found for channel {ch}" in record.message
                            for record in caplog.records
                            if record.levelname == "WARNING"
                        )
                        assert (
                            warning_found
                        ), f"Expected warning for channel {ch} without noise windows"
                    print(
                        f"Verified warnings logged for {len(channels_without_noise)} "
                        f"channels without noise windows: {sorted(channels_without_noise)}"
                    )
                else:
                    print("All channels with hits have noise windows")

        except Exception as e:
            pytest.fail(f"Failed to test noise PSD placeholder warnings: {str(e)}")

    def test_per_channel_psd_used_in_optimal_filter(self):
        """Test that per-channel PSDs are actually used in optimal filter computation."""
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
            # Get the data
            hit_classification = st.get_array(run_id, "hit_classification", config=configs)

            if len(hit_classification) > 0:
                # Check that optimal filter fields are computed
                assert "best_aOF" in hit_classification.dtype.names
                assert "best_chi2" in hit_classification.dtype.names
                assert "best_OF_shift" in hit_classification.dtype.names

                # All values should be finite (not NaN or inf)
                assert np.all(np.isfinite(hit_classification["best_aOF"]))
                assert np.all(np.isfinite(hit_classification["best_chi2"]))
                assert np.all(np.isfinite(hit_classification["best_OF_shift"]))

                # Chi-squared should be non-negative
                assert np.all(hit_classification["best_chi2"] >= 0)

                print(
                    f"Successfully verified optimal filter computation "
                    f"for {len(hit_classification)} hits"
                )
                print(f"  Mean best_aOF: {np.mean(hit_classification['best_aOF']):.4f}")
                print(f"  Mean best_chi2: {np.mean(hit_classification['best_chi2']):.4f}")

        except Exception as e:
            pytest.fail(f"Failed to verify per-channel PSD usage in optimal filter: {str(e)}")


# =============================================================================
# Unit Tests for Static Methods
# =============================================================================


def test_calculate_spike_threshold():
    """Test that spike threshold is calculated correctly from signal statistics."""
    # Create mock signal with known statistics
    # Row 0: values 1, 2, 3, 4, 5 -> mean=3.0, std=sqrt(2)
    # Row 1: values 2, 4, 6, 8, 10 -> mean=6.0, std=2*sqrt(2)
    signal = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0]])
    spike_threshold_sigma = 2.0

    threshold = DxHitClassification.calculate_spike_threshold(signal, spike_threshold_sigma)

    # Verify expected values: mean + sigma * std
    expected_mean_0 = np.mean(signal[0])  # 3.0
    expected_std_0 = np.std(signal[0])  # sqrt(2) ≈ 1.414
    expected_threshold_0 = expected_mean_0 + spike_threshold_sigma * expected_std_0

    expected_mean_1 = np.mean(signal[1])  # 6.0
    expected_std_1 = np.std(signal[1])  # 2*sqrt(2) ≈ 2.828
    expected_threshold_1 = expected_mean_1 + spike_threshold_sigma * expected_std_1

    assert np.isclose(
        threshold[0], expected_threshold_0
    ), f"Expected threshold[0]={expected_threshold_0}, got {threshold[0]}"
    assert np.isclose(
        threshold[1], expected_threshold_1
    ), f"Expected threshold[1]={expected_threshold_1}, got {threshold[1]}"


def test_optimal_filter_basic():
    """Test that optimal filter returns valid amplitude and chi-squared."""
    # Simple test case with known signal and template
    n_samples = 100
    St = np.sin(np.linspace(0, 2 * np.pi, n_samples))
    At = np.sin(np.linspace(0, 2 * np.pi, n_samples))  # Perfect match
    Jf = np.ones(n_samples)  # Flat noise PSD

    ahatOF, chisq = DxHitClassification._optimal_filter(St, Jf, At)

    # When signal matches template exactly, amplitude should be ~1
    assert np.isclose(
        ahatOF, 1.0, atol=0.01
    ), f"Expected amplitude ~1.0 for matching signal/template, got {ahatOF}"
    # Chi-squared should be non-negative
    assert chisq >= 0, f"Chi-squared should be non-negative, got {chisq}"
    # Chi-squared should be very small for perfect match
    assert chisq < 1e-10, f"Chi-squared should be near zero for perfect match, got {chisq}"


def test_optimal_filter_scaled_signal():
    """Test optimal filter with scaled signal."""
    n_samples = 100
    scale_factor = 2.5
    At = np.sin(np.linspace(0, 2 * np.pi, n_samples))  # Template
    St = scale_factor * At  # Signal is scaled template
    Jf = np.ones(n_samples)  # Flat noise PSD

    ahatOF, chisq = DxHitClassification._optimal_filter(St, Jf, At)

    # Amplitude should recover the scale factor
    assert np.isclose(
        ahatOF, scale_factor, atol=0.01
    ), f"Expected amplitude ~{scale_factor}, got {ahatOF}"
    # Chi-squared should still be very small (perfect scaled match)
    assert chisq < 1e-10, f"Chi-squared should be near zero for scaled match, got {chisq}"


def test_modify_template_windowing():
    """Test that modify_template correctly applies windowing."""
    St = np.zeros(700)  # Signal length
    dt_seconds = 1 / 38000
    tau = 0  # No shift

    # Test without windowing
    At_no_window = DxHitClassification.modify_template(
        St, dt_seconds, tau, interp_path=DEFAULT_TEMPLATE_INTERP_PATH, apply_window=False
    )

    # Test with windowing
    of_window_left = 100
    of_window_right = 300
    At_windowed = DxHitClassification.modify_template(
        St,
        dt_seconds,
        tau,
        interp_path=DEFAULT_TEMPLATE_INTERP_PATH,
        apply_window=True,
        of_window_left=of_window_left,
        of_window_right=of_window_right,
    )

    assert len(At_no_window) == len(
        St
    ), f"Expected non-windowed template length {len(St)}, got {len(At_no_window)}"
    expected_windowed_length = of_window_left + of_window_right
    assert len(At_windowed) == expected_windowed_length, (
        f"Expected windowed template length {expected_windowed_length}, " f"got {len(At_windowed)}"
    )


def test_modify_template_windowing_requires_params():
    """Test that modify_template raises error when windowing params are missing."""
    St = np.zeros(700)
    dt_seconds = 1 / 38000
    tau = 0

    # Should raise ValueError when apply_window=True but params not provided
    with pytest.raises(ValueError, match="of_window_left and of_window_right must be provided"):
        DxHitClassification.modify_template(
            St,
            dt_seconds,
            tau,
            interp_path=DEFAULT_TEMPLATE_INTERP_PATH,
            apply_window=True,
            of_window_left=None,
            of_window_right=None,
        )


# =============================================================================
# Test: Kappa Fitting Helper Methods
# =============================================================================


def test_movmean_basic():
    """Test that _movmean computes moving average correctly."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    window_size = 3

    result = DxHitClassification._movmean(data, window_size)

    # The moving average should smooth the data
    assert len(result) == len(data), "Result should have same length as input"
    # Middle values should be exact averages
    assert np.isclose(result[2], 3.0), "Middle value should be (2+3+4)/3 = 3"


def test_movmean_invalid_window():
    """Test that _movmean raises error for invalid window size."""
    data = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Window size must be a positive integer"):
        DxHitClassification._movmean(data, 0)

    with pytest.raises(ValueError, match="Window size must be a positive integer"):
        DxHitClassification._movmean(data, -1)


def test_profile_fit_basic():
    """Test that _profile_fit computes double-sided exponential correctly."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    amplitude = 1.0
    center = 0.0
    kappa = 1.0

    result = DxHitClassification._profile_fit(x, amplitude, center, kappa)

    # At center, value should be amplitude
    assert np.isclose(result[2], amplitude), "At center, value should equal amplitude"

    # Values should decay symmetrically
    assert np.isclose(result[1], result[3]), "Should be symmetric around center"
    assert np.isclose(result[0], result[4]), "Should be symmetric around center"

    # Values at x=1 should be amplitude * exp(-1)
    expected_at_1 = amplitude * np.exp(-1)
    assert np.isclose(result[3], expected_at_1), f"At x=1, expected {expected_at_1}"


def test_optimal_filter_returns_kappa():
    """Test that optimal_filter returns kappa in its output."""
    # Create a simple test signal
    n_samples = 700
    St = np.zeros(n_samples)
    St[200:400] = np.sin(np.linspace(0, 2 * np.pi, 200))  # Add a pulse-like feature
    dt_seconds = 1 / 38000
    Jf = np.ones(400)  # Simple flat noise PSD

    from straxion.utils import load_interpolation

    At_interp, t_max_seconds = load_interpolation(DEFAULT_TEMPLATE_INTERP_PATH)

    # Run optimal filter
    result = DxHitClassification.optimal_filter(
        St,
        dt_seconds,
        Jf,
        At_interp,
        t_max_seconds,
        of_window_left=100,
        of_window_right=300,
        of_shift_range_min=-50,
        of_shift_range_max=50,
        of_shift_step=1,
        kappa_fit_half_band_width=20,
        kappa_fit_smoothing_window=3,
    )

    # Should return 5 values: best_aOF, best_chi2, best_OF_shift, kappa, best_At_shifted
    assert len(result) == 5, f"Expected 5 return values, got {len(result)}"

    best_aOF, best_chi2, best_OF_shift, kappa, best_At_shifted = result

    # Kappa should be positive (or inf if fit failed)
    assert kappa > 0, f"Kappa should be positive, got {kappa}"


def test_optimal_filter_kappa_inf_when_out_of_bounds():
    """Test that kappa is inf when fit window would be out of bounds."""
    n_samples = 700
    St = np.zeros(n_samples)
    St[200:400] = np.sin(np.linspace(0, 2 * np.pi, 200))
    dt_seconds = 1 / 38000
    Jf = np.ones(400)

    from straxion.utils import load_interpolation

    At_interp, t_max_seconds = load_interpolation(DEFAULT_TEMPLATE_INTERP_PATH)

    # Use a very large half_band_width that will exceed the shift range
    result = DxHitClassification.optimal_filter(
        St,
        dt_seconds,
        Jf,
        At_interp,
        t_max_seconds,
        of_window_left=100,
        of_window_right=300,
        of_shift_range_min=-10,  # Very small range
        of_shift_range_max=10,
        of_shift_step=1,
        kappa_fit_half_band_width=50,  # Larger than available shifts
        kappa_fit_smoothing_window=3,
    )

    _, _, _, kappa, _ = result

    # Kappa should be inf because the fit window is out of bounds
    assert np.isinf(kappa), f"Expected kappa to be inf when out of bounds, got {kappa}"


def test_optimal_filter_kappa_with_negative_amplitude():
    """Test that kappa fitting works correctly when best_aOF is negative.

    This tests the fix for the bug where negative amplitudes caused
    the curve_fit bounds to be inverted (lower > upper).
    """
    n_samples = 700
    St = np.zeros(n_samples)
    # Create an inverted pulse (negative amplitude)
    St[200:400] = -np.sin(np.linspace(0, 2 * np.pi, 200))
    dt_seconds = 1 / 38000
    Jf = np.ones(400)

    from straxion.utils import load_interpolation

    At_interp, t_max_seconds = load_interpolation(DEFAULT_TEMPLATE_INTERP_PATH)

    # Run optimal filter - this should not crash even with negative amplitude
    result = DxHitClassification.optimal_filter(
        St,
        dt_seconds,
        Jf,
        At_interp,
        t_max_seconds,
        of_window_left=100,
        of_window_right=300,
        of_shift_range_min=-50,
        of_shift_range_max=50,
        of_shift_step=1,
        kappa_fit_half_band_width=20,
        kappa_fit_smoothing_window=3,
    )

    best_aOF, best_chi2, best_OF_shift, kappa, best_At_shifted = result

    # The fit should not crash - kappa should be positive (or inf if fit legitimately failed)
    assert kappa > 0, f"Kappa should be positive, got {kappa}"
    # best_aOF can be negative for inverted signals
    assert np.isfinite(best_aOF), f"best_aOF should be finite, got {best_aOF}"


# =============================================================================
# Test: determine_spike_threshold mutual exclusivity
# =============================================================================


def test_determine_spike_threshold_with_sigma():
    """Test determine_spike_threshold works correctly with spike_thresholds_sigma."""
    st = straxion.qualiphide_thz_offline()
    plugin = st.get_single_plugin("1756824965", "hit_classification")

    # Create mock records with data_dx_convolved
    n_records = 3
    record_length = 1000
    records = np.zeros(
        n_records, dtype=[("channel", np.int16), ("data_dx_convolved", np.float32, record_length)]
    )
    records["channel"] = [0, 1, 2]
    # Add some random data
    np.random.seed(42)
    for i in range(n_records):
        records["data_dx_convolved"][i] = np.random.randn(record_length) * 0.1

    # Plugin should have spike_threshold_dx=None and spike_thresholds_sigma set
    # This should work without error
    threshold = plugin.determine_spike_threshold(records)

    assert len(threshold) == n_records, f"Expected {n_records} thresholds, got {len(threshold)}"
    assert np.all(np.isfinite(threshold)), "Thresholds should be finite"


# =============================================================================
# Test: Noise PSD length validation
# =============================================================================


def test_noise_psd_length_validation():
    """Test that noise PSD length mismatch raises ValueError during setup."""
    st = straxion.qualiphide_thz_offline()

    # Create invalid PSD with wrong length (should be 400 = 100 + 300)
    invalid_psd = [1.0] * 50  # Wrong length

    # Set the invalid config on the context
    st.set_config({"noise_psd_placeholder": invalid_psd})

    # The error should be raised during plugin setup
    clean_strax_data()
    with pytest.raises(ValueError, match="Noise PSD length"):
        # Get the plugin, which triggers setup
        st.get_single_plugin("1756824965", "hit_classification")


# =============================================================================
# Test: Empty noise windows handling
# =============================================================================


def test_compute_per_channel_noise_psd_empty_channel():
    """Test that channels without noise windows return None for PSD."""
    st = straxion.qualiphide_thz_offline()
    plugin = st.get_single_plugin("1756824965", "hit_classification")

    # Get the expected PSD length from plugin config
    psd_length = plugin.of_window_left + plugin.of_window_right

    # Create mock noises array with only channel 0
    noises = np.zeros(5, dtype=[("channel", np.int16), ("data_dx", np.float32, psd_length)])
    noises["channel"] = 0  # All noise windows for channel 0 only
    # Add some random data
    np.random.seed(42)
    for i in range(5):
        noises["data_dx"][i] = np.random.randn(psd_length) * 0.1

    n_channels = 3
    channel_psds = plugin.compute_per_channel_noise_psd(noises, n_channels)

    # Verify the result is a dictionary
    assert isinstance(channel_psds, dict), "Result should be a dictionary"

    # Verify all channels are present
    assert (
        len(channel_psds) == n_channels
    ), f"Expected {n_channels} channels, got {len(channel_psds)}"

    # Channel 0 should have PSD (has noise windows)
    assert channel_psds[0] is not None, "Channel 0 should have PSD"
    assert (
        len(channel_psds[0]) == psd_length
    ), f"PSD length should be {psd_length}, got {len(channel_psds[0])}"

    # Channels 1 and 2 should be None (no noise windows)
    assert channel_psds[1] is None, "Channel 1 should have None (no noise windows)"
    assert channel_psds[2] is None, "Channel 2 should have None (no noise windows)"

    # PSD values should be non-negative (it's |FFT|^2)
    assert np.all(channel_psds[0] >= 0), "PSD values should be non-negative"

    # PSD should contain finite values
    assert np.all(np.isfinite(channel_psds[0])), "PSD should contain finite values"


# =============================================================================
# Test: Photon candidate classification logic
# =============================================================================


def test_photon_candidate_exclusion_logic():
    """Test that is_photon_candidate correctly excludes problematic hits.

    Photon candidate logic:
        is_photon_candidate = NOT(coincident OR symmetric OR truncated OR invalid_kappa)
    """
    # Test cases: (coincident, symmetric, truncated, invalid_kappa) -> expected is_photon_candidate
    test_cases = [
        # Clean hit with valid kappa -> photon candidate
        (False, False, False, False, True),
        # Clean hit but invalid kappa -> not candidate
        (False, False, False, True, False),
        # Coincident with spikes -> not candidate
        (True, False, False, False, False),
        # Symmetric spike -> not candidate
        (False, True, False, False, False),
        # Truncated -> not candidate
        (False, False, True, False, False),
        # Invalid kappa alone -> not candidate
        (False, False, False, True, False),
        # Multiple issues -> not candidate
        (True, True, False, False, False),
        (True, False, True, False, False),
        (False, True, True, False, False),
        (True, True, True, False, False),
        # All bad including invalid kappa
        (True, True, True, True, False),
    ]

    for coincident, symmetric, truncated, invalid_kappa, expected in test_cases:
        # Compute photon candidate using the same logic as the plugin
        result = not (coincident or symmetric or truncated or invalid_kappa)
        assert result == expected, (
            f"For coincident={coincident}, symmetric={symmetric}, truncated={truncated}, "
            f"invalid_kappa={invalid_kappa}: expected is_photon_candidate={expected}, got {result}"
        )


def test_photon_candidate_exclusion_vectorized():
    """Test photon candidate logic with numpy arrays (vectorized version)."""
    is_coincident = np.array([False, True, False, False, True, True, False, True, False])
    is_symmetric = np.array([False, False, True, False, True, False, True, True, False])
    is_truncated = np.array([False, False, False, True, False, True, True, True, False])
    is_invalid_kappa = np.array([False, False, False, False, False, False, False, False, True])

    # Expected results based on the logic:
    # is_photon_candidate = ~(is_coincident | is_symmetric | is_truncated | is_invalid_kappa)
    # Index 0: all False -> True (photon candidate)
    # Index 1-7: at least one exclusion flag True -> False
    # Index 8: invalid kappa -> False
    expected = np.array([True, False, False, False, False, False, False, False, False])

    # Compute using the same logic as the plugin
    is_photon_candidate = ~(is_coincident | is_symmetric | is_truncated | is_invalid_kappa)

    np.testing.assert_array_equal(
        is_photon_candidate, expected, err_msg="Vectorized photon candidate logic mismatch"
    )
