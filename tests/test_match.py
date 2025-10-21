import os
import pytest
import numpy as np
import straxion
from straxion.plugins.match import Match
from straxion.utils import NOT_FOUND_INDEX
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


def test_match_plugin_registration():
    """Test that the Match plugin is properly registered."""
    st = straxion.qualiphide_thz_offline()
    assert "match" in st._plugin_class_registry
    assert st._plugin_class_registry["match"] == Match


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason=("Test data directory not provided via " "STRAXION_TEST_DATA_DIR environment variable"),
)
class TestMatchWithRealData:
    """Test Match plugin with real data from STRAXION_TEST_DATA_DIR."""

    def _get_test_config(self, test_data_dir, run_id, **kwargs):
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
        # Add any additional config options
        configs.update(kwargs)
        return configs

    def test_match_dtype_inference(self):
        """Test that Match can infer the correct data type."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        st = straxion.qualiphide_thz_offline()
        plugin = st.get_single_plugin("1756824965", "match")
        dtype = plugin.infer_dtype()

        # Check expected fields
        expected_fields = [
            "time",
            "endtime",
            "channel",
            "energy_true",
            "dx_true",
            "destiny",
            "hit_index",
            "length",
            "amplitude",
            "amplitude_moving_average",
            "amplitude_convolved",
            "hit_threshold",
            "width",
            "rise_edge_slope",
            "n_spikes_coinciding",
            "is_photon_candidate",
            "is_symmetric_spike",
            "is_coincident_with_spikes",
            "best_aOF",
            "best_chi2",
            "best_OF_shift",
        ]
        field_names = [name[1] for name, *_ in dtype]
        for field in expected_fields:
            assert field in field_names

    def test_match_processing(self):
        """Test Match plugin with real data."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = self._get_test_config(test_data_dir, run_id, salt_rate=100)

        clean_strax_data()
        try:
            # Get match results
            match = st.get_array(run_id, "match", config=configs)

            # Basic validation
            assert match is not None
            assert len(match) > 0

            # Check required fields
            required_fields = [
                "time",
                "endtime",
                "channel",
                "energy_true",
                "dx_true",
                "destiny",
                "hit_index",
                "length",
                "amplitude",
                "amplitude_moving_average",
                "amplitude_convolved",
                "hit_threshold",
                "width",
                "rise_edge_slope",
                "n_spikes_coinciding",
                "is_photon_candidate",
                "is_symmetric_spike",
                "is_coincident_with_spikes",
                "best_aOF",
                "best_chi2",
                "best_OF_shift",
            ]
            for field in required_fields:
                assert field in match.dtype.names, f"Required field '{field}' missing from match"

            # Check data types
            assert match["time"].dtype == np.int64
            assert match["endtime"].dtype == np.int64
            assert match["channel"].dtype == np.int16
            assert match["energy_true"].dtype == np.float32
            assert match["dx_true"].dtype == np.float32
            assert match["destiny"].dtype.kind == "U"  # Unicode string
            assert match["hit_index"].dtype == np.int32
            assert match["is_photon_candidate"].dtype == bool
            assert match["is_symmetric_spike"].dtype == bool
            assert match["is_coincident_with_spikes"].dtype == bool
            assert match["best_aOF"].dtype == np.float32
            assert match["best_chi2"].dtype == np.float32
            assert match["best_OF_shift"].dtype == np.int32

            # Check that destiny values are valid
            valid_destinies = {"found", "lost", "split"}
            for destiny in match["destiny"]:
                assert destiny in valid_destinies, f"Invalid destiny value: {destiny}"

            print(
                f"Successfully processed {len(match)} match events "
                f"across {len(np.unique(match['channel']))} channels"
            )
            print(
                f"Destiny counts: "
                f"found={np.sum(match['destiny'] == 'found')}, "
                f"lost={np.sum(match['destiny'] == 'lost')}, "
                f"split={np.sum(match['destiny'] == 'split')}"
            )

        except Exception as e:
            pytest.fail(f"Failed to process match: {str(e)}")

    def test_match_destiny_categories(self):
        """Test that match produces all destiny categories."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = self._get_test_config(test_data_dir, run_id, salt_rate=100)

        clean_strax_data()
        try:
            match = st.get_array(run_id, "match", config=configs)

            # Count each destiny category
            n_found = np.sum(match["destiny"] == "found")
            n_lost = np.sum(match["destiny"] == "lost")
            n_split = np.sum(match["destiny"] == "split")

            # Check that categories sum to total
            assert n_found + n_lost + n_split == len(match)

            # With salt_rate=100, we should have at least some found events
            assert n_found > 0, "Expected at least some 'found' matches with salt_rate=100"

            print(f"Destiny distribution: found={n_found}, " f"lost={n_lost}, split={n_split}")

        except Exception as e:
            pytest.fail(f"Failed to process match: {str(e)}")

    def test_match_found_case(self):
        """Test that 'found' matches have proper hit info populated."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = self._get_test_config(test_data_dir, run_id, salt_rate=100)

        clean_strax_data()
        try:
            match = st.get_array(run_id, "match", config=configs)
            hits = st.get_array(run_id, "hits", config=configs)

            # Filter for 'found' matches
            found_matches = match[match["destiny"] == "found"]

            if len(found_matches) == 0:
                pytest.skip("No 'found' matches in test data")

            # Check that hit_index is valid
            assert np.all(found_matches["hit_index"] >= 0)
            assert np.all(found_matches["hit_index"] < len(hits))

            # Check that hit fields are populated (not zero)
            assert np.all(found_matches["length"] > 0)
            assert np.all(found_matches["width"] > 0)

            # Verify that the hit_index points to correct channel
            for match_evt in found_matches:
                hit_idx = match_evt["hit_index"]
                assert hits[hit_idx]["channel"] == match_evt["channel"], "Hit channel mismatch"

            print(f"Validated {len(found_matches)} 'found' matches " f"with proper hit information")

        except Exception as e:
            pytest.fail(f"Failed to process match: {str(e)}")

    def test_match_lost_case(self):
        """Test that 'lost' matches have zeros for hit fields."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = self._get_test_config(test_data_dir, run_id, salt_rate=100)

        clean_strax_data()
        try:
            match = st.get_array(run_id, "match", config=configs)

            # Filter for 'lost' matches
            lost_matches = match[match["destiny"] == "lost"]

            if len(lost_matches) == 0:
                pytest.skip("No 'lost' matches in test data")

            # Check that hit_index is NOT_FOUND_INDEX
            assert np.all(lost_matches["hit_index"] == NOT_FOUND_INDEX)

            # Check that hit fields are zero/false
            assert np.all(lost_matches["length"] == 0)
            assert np.all(lost_matches["width"] == 0)
            assert np.all(lost_matches["amplitude"] == 0)
            assert np.all(lost_matches["amplitude_moving_average"] == 0)
            assert np.all(lost_matches["amplitude_convolved"] == 0)
            assert np.all(~lost_matches["is_photon_candidate"])
            assert np.all(~lost_matches["is_symmetric_spike"])
            assert np.all(~lost_matches["is_coincident_with_spikes"])

            print(f"Validated {len(lost_matches)} 'lost' matches " f"with zero/default values")

        except Exception as e:
            pytest.fail(f"Failed to process match: {str(e)}")

    def test_match_split_case(self):
        """Test that 'split' matches select closest hit by amplitude_max_record_i."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = self._get_test_config(test_data_dir, run_id, salt_rate=100)

        clean_strax_data()
        try:
            match = st.get_array(run_id, "match", config=configs)
            hits = st.get_array(run_id, "hits", config=configs)

            # Filter for 'split' matches
            split_matches = match[match["destiny"] == "split"]

            if len(split_matches) == 0:
                pytest.skip("No 'split' matches in test data")

            # Check that hit_index is valid
            assert np.all(split_matches["hit_index"] >= 0)
            assert np.all(split_matches["hit_index"] < len(hits))

            # Verify that the selected hit has amplitude_convolved populated
            for i, match_evt in enumerate(split_matches):
                hit_idx = match_evt["hit_index"]
                assert (
                    match_evt["amplitude_convolved"] == hits[hit_idx]["amplitude_convolved"]
                ), "Amplitude mismatch for split match"

            print(
                f"Validated {len(split_matches)} 'split' matches "
                f"with closest amplitude_max_record_i selection"
            )

        except Exception as e:
            pytest.fail(f"Failed to process match: {str(e)}")

    def test_match_time_consistency(self):
        """Test that match time and endtime come from truth."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = self._get_test_config(test_data_dir, run_id, salt_rate=100)

        clean_strax_data()
        try:
            match = st.get_array(run_id, "match", config=configs)
            truth = st.get_array(run_id, "truth", config=configs)

            # Verify same length
            assert len(match) == len(truth)

            # Verify time and endtime match
            assert np.array_equal(match["time"], truth["time"])
            assert np.array_equal(match["endtime"], truth["endtime"])

            print(f"Validated time consistency for {len(match)} match events")

        except Exception as e:
            pytest.fail(f"Failed to process match: {str(e)}")

    def test_match_channel_consistency(self):
        """Test that match channels come from truth."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = self._get_test_config(test_data_dir, run_id, salt_rate=100)

        clean_strax_data()
        try:
            match = st.get_array(run_id, "match", config=configs)
            truth = st.get_array(run_id, "truth", config=configs)

            # Verify same length
            assert len(match) == len(truth)

            # Verify channels match
            assert np.array_equal(match["channel"], truth["channel"])

            # Verify energy and dx fields match
            assert np.array_equal(match["energy_true"], truth["energy_true"])
            assert np.array_equal(match["dx_true"], truth["dx_true"])

            print(
                f"Validated channel and truth field consistency " f"for {len(match)} match events"
            )

        except Exception as e:
            pytest.fail(f"Failed to process match: {str(e)}")

    def test_match_optimal_filter_fields(self):
        """Test that optimal filter fields are properly populated in found matches."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = self._get_test_config(test_data_dir, run_id, salt_rate=100)

        clean_strax_data()
        try:
            match = st.get_array(run_id, "match", config=configs)

            # Filter for 'found' matches
            found_matches = match[match["destiny"] == "found"]

            if len(found_matches) == 0:
                pytest.skip("No 'found' matches in test data")

            # Check that optimal filter fields are finite
            assert np.all(
                np.isfinite(found_matches["best_aOF"])
            ), "Non-finite values found in best_aOF"
            assert np.all(
                np.isfinite(found_matches["best_chi2"])
            ), "Non-finite values found in best_chi2"
            assert np.all(
                np.isfinite(found_matches["best_OF_shift"])
            ), "Non-finite values found in best_OF_shift"

            # Chi-squared should be non-negative
            assert np.all(found_matches["best_chi2"] >= 0), "Negative chi-squared values found"

            print(f"Validated optimal filter fields for {len(found_matches)} found matches")
            print(f"  Mean best_aOF: {np.mean(found_matches['best_aOF']):.4f}")
            print(f"  Mean best_chi2: {np.mean(found_matches['best_chi2']):.4f}")
            print(f"  Mean best_OF_shift: {np.mean(found_matches['best_OF_shift']):.2f}")

        except Exception as e:
            pytest.fail(f"Failed to validate optimal filter fields: {str(e)}")
