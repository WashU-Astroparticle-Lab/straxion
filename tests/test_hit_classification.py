import os
import pytest
import numpy as np
import straxion
from straxion.plugins.hit_classification import HitClassification, SpikeCoincidence
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
    """Test that the SpikeCoincidence plugin is properly registered in the offline context."""
    st = straxion.qualiphide_thz_offline()
    assert "hit_classification" in st._plugin_class_registry
    # In offline context, hit_classification maps to SpikeCoincidence
    assert st._plugin_class_registry["hit_classification"] == SpikeCoincidence


def test_spike_coincidence_dtype_inference():
    """Test that SpikeCoincidence can infer the correct data type."""
    st = straxion.qualiphide_thz_offline()
    plugin = st.get_single_plugin("1756824965", "hit_classification")
    dtype = plugin.infer_dtype()
    # List of expected fields for SpikeCoincidence
    expected_fields = [
        "time",
        "endtime",
        "channel",
        "is_coincident_with_spikes",
        "is_photon_candidate",
        "rise_edge_slope",
        "n_spikes_coinciding",
        "best_aOF",
        "best_chi2",
        "best_OF_shift",
    ]
    field_names = [name[1] for name, *_ in dtype]
    for field in expected_fields:
        assert field in field_names


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason="Test data directory not provided via STRAXION_TEST_DATA_DIR environment variable",
)
class TestSpikeCoincidenceWithRealDataOffline:
    """Test spike coincidence processing with real qualiphide_fir_test_data
    using SpikeCoincidence plugin."""

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
        SpikeCoincidence plugin.

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

            # Check that all required fields are present for SpikeCoincidence
            required_fields = [
                "time",
                "endtime",
                "channel",
                "is_coincident_with_spikes",
                "is_symmetric_spike",
                "is_photon_candidate",
                "rise_edge_slope",
                "n_spikes_coinciding",
                "best_aOF",
                "best_chi2",
                "best_OF_shift",
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
            assert hit_classification["is_photon_candidate"].dtype == bool
            assert hit_classification["rise_edge_slope"].dtype == np.float32
            assert hit_classification["n_spikes_coinciding"].dtype == np.int64
            assert hit_classification["best_aOF"].dtype == np.float32
            assert hit_classification["best_chi2"].dtype == np.float32
            assert hit_classification["best_OF_shift"].dtype == np.int64

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

                # Check that n_spikes_coinciding is an integer and non-negative
                assert np.all(
                    hit_classification["n_spikes_coinciding"] >= 0
                ), "Negative spike coincidence counts found"

                # Check that is_photon_candidate is False when
                # is_coincident_with_spikes or is_symmetric_spike is True
                for hit_class in hit_classification:
                    if hit_class["is_coincident_with_spikes"] or hit_class["is_symmetric_spike"]:
                        assert not hit_class["is_photon_candidate"], (
                            "Hit cannot be a photon candidate when it is coincident with spikes "
                            "or is a symmetric spike"
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
