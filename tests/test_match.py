import os
import pytest
import numpy as np
import straxion
from straxion.plugins.match import Match
from straxion.utils import (
    NOT_FOUND_INDEX,
    PULSE_TEMPLATE_LENGTH,
    DEFAULT_TEMPLATE_INTERP_PATH,
    TEMPLATE_INTERP_FOLDER,
    TIME_DTYPE,
    DATA_DTYPE,
    HIT_WINDOW_LENGTH_LEFT,
    HIT_WINDOW_LENGTH_RIGHT,
    PULSE_TEMPLATE_ARGMAX,
)
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
            "is_truncated_hit",
            "is_invalid_kappa",
            "best_aOF",
            "best_chi2",
            "best_OF_shift",
            "kappa",
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
                "is_truncated_hit",
                "is_invalid_kappa",
                "best_aOF",
                "best_chi2",
                "best_OF_shift",
                "kappa",
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
            assert match["is_truncated_hit"].dtype == bool
            assert match["is_invalid_kappa"].dtype == bool
            assert match["best_aOF"].dtype == np.float32
            assert match["best_chi2"].dtype == np.float32
            assert match["best_OF_shift"].dtype == np.int32
            assert match["kappa"].dtype == np.float32

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
            assert np.all(~lost_matches["is_truncated_hit"])
            assert np.all(~lost_matches["is_invalid_kappa"])
            assert np.all(lost_matches["kappa"] == 0)

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

    def test_match_window_ms_configuration(self):
        """Test that match_window_ms option works correctly."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = self._get_test_config(test_data_dir, run_id, salt_rate=100, match_window_ms=None)

        clean_strax_data()
        try:
            # Test with match_window_ms=None (full time range)
            match_no_window = st.get_array(run_id, "match", config=configs)

            # Test with match_window_ms=2 (restricted window)
            configs_with_window = self._get_test_config(
                test_data_dir, run_id, salt_rate=100, match_window_ms=2
            )
            match_with_window = st.get_array(run_id, "match", config=configs_with_window)

            # Both should produce results
            assert match_no_window is not None
            assert match_with_window is not None
            assert len(match_no_window) > 0
            assert len(match_with_window) > 0

            # With restricted window, we might have fewer matches
            # (some matches might be excluded due to window restriction)
            # But the window should not produce more matches
            assert len(match_with_window) <= len(match_no_window)

            # Test that window restriction still produces valid results
            valid_destinies = {"found", "lost", "split"}
            for destiny in match_with_window["destiny"]:
                assert destiny in valid_destinies, f"Invalid destiny value: {destiny}"

            print(
                f"match_window_ms test: "
                f"no_window={len(match_no_window)}, "
                f"with_window={len(match_with_window)}"
            )

        except Exception as e:
            pytest.fail(f"Failed to test match_window_ms: {str(e)}")


class TestMatchRestrictWindow:
    """Test the _restrict_to_maximum_window method of Match plugin."""

    def setup_method(self):
        """Set up test data and create a Match instance."""
        from straxion.utils import DEFAULT_TEMPLATE_INTERP_PATH, TEMPLATE_INTERP_FOLDER

        self.match = Match()
        self.match.config = {
            "fs": 38000,
            "match_window_ms": 1.5,
            "template_interp_path": DEFAULT_TEMPLATE_INTERP_PATH,
            "template_interp_folder": TEMPLATE_INTERP_FOLDER,
        }
        self.match.setup()

        # Store constants for tests
        self.HIT_WINDOW_LENGTH_LEFT = HIT_WINDOW_LENGTH_LEFT
        self.HIT_WINDOW_LENGTH_RIGHT = HIT_WINDOW_LENGTH_RIGHT
        self.hit_waveform_length = HIT_WINDOW_LENGTH_LEFT + HIT_WINDOW_LENGTH_RIGHT

    def test_pulse_template_argmax_calculation(self):
        """Test that pulse_template_argmax is calculated correctly."""
        # Verify pulse_template_argmax is set
        assert hasattr(self.match, "pulse_template_argmax")
        assert isinstance(self.match.pulse_template_argmax, (int, np.integer))
        assert 0 <= self.match.pulse_template_argmax < PULSE_TEMPLATE_LENGTH

    def test_pulse_template_argmax_different_fs(self):
        """Test that pulse_template_argmax is preserved at PULSE_TEMPLATE_ARGMAX.

        The interpolation process preserves the maximum at the same sample index
        (PULSE_TEMPLATE_ARGMAX) regardless of sampling frequency, because it
        aligns the maximum at the target time which corresponds to that index.
        """
        from straxion.utils import DEFAULT_TEMPLATE_INTERP_PATH, TEMPLATE_INTERP_FOLDER

        # Test at 38kHz (default)
        match_38k = Match()
        match_38k.config = {
            "fs": 38000,
            "match_window_ms": 1.5,
            "template_interp_path": DEFAULT_TEMPLATE_INTERP_PATH,
            "template_interp_folder": TEMPLATE_INTERP_FOLDER,
        }
        match_38k.setup()

        # Test at different frequency (e.g., 50kHz)
        match_50k = Match()
        match_50k.config = {
            "fs": 50000,
            "match_window_ms": 1.5,
            "template_interp_path": DEFAULT_TEMPLATE_INTERP_PATH,
            "template_interp_folder": TEMPLATE_INTERP_FOLDER,
        }
        match_50k.setup()

        # The interpolation process preserves the maximum at PULSE_TEMPLATE_ARGMAX
        # regardless of sampling frequency, because it aligns the maximum at the
        # target time which corresponds to that index
        assert match_38k.pulse_template_argmax == PULSE_TEMPLATE_ARGMAX
        assert match_50k.pulse_template_argmax == PULSE_TEMPLATE_ARGMAX

    def test_restrict_window_truth_basic(self):
        """Test truth window restriction with basic case."""

        # Create simple truth data
        truth_ch = np.zeros(
            2,
            dtype=[
                ("time", TIME_DTYPE),
                ("endtime", TIME_DTYPE),
                ("channel", np.int16),
            ],
        )
        truth_ch["time"] = [1000, 5000]
        truth_ch["endtime"] = [
            1000 + PULSE_TEMPLATE_LENGTH * self.match.dt_exact,
            5000 + PULSE_TEMPLATE_LENGTH * self.match.dt_exact,
        ]
        truth_ch["channel"] = 0

        # Create dummy hits with proper dtype (not used in truth calculation)
        # Empty array but with correct structure
        hits_ch = np.zeros(
            0,
            dtype=[
                ("time", TIME_DTYPE),
                ("endtime", TIME_DTYPE),
                ("dt", TIME_DTYPE),
                ("data_dx", DATA_DTYPE, self.hit_waveform_length),
            ],
        )

        hits_restricted, truth_restricted = self.match._restrict_to_maximum_window(
            hits_ch, truth_ch
        )

        # Check that windows are centered around maximum
        for i in range(len(truth_ch)):
            expected_max_time = (
                truth_ch["time"][i] + self.match.pulse_template_argmax * self.match.dt_exact
            )
            half_window = self.match.match_window_ns / 2
            expected_start = expected_max_time - half_window
            expected_end = expected_max_time + half_window

            # Should be clipped to original time/endtime bounds
            assert truth_restricted["time"][i] >= truth_ch["time"][i]
            assert truth_restricted["time"][i] <= expected_start or np.isclose(
                truth_restricted["time"][i], expected_start
            )
            assert truth_restricted["endtime"][i] <= truth_ch["endtime"][i]
            assert truth_restricted["endtime"][i] >= expected_end or np.isclose(
                truth_restricted["endtime"][i], expected_end
            )

    def test_restrict_window_hit_no_padding(self):
        """Test hit window restriction with no padding."""

        # Create hit with no padding (data starts at index 0)
        dt_ns = int(self.match.dt_exact)
        hit_time = 10000
        max_sample_idx = 200  # Maximum at sample 200 (typical alignment point)

        # Create waveform: max at index 200, no padding (data starts at index 0)
        waveform = np.zeros(self.hit_waveform_length, dtype=DATA_DTYPE)
        waveform[0] = 0.1  # Small value at start to indicate no padding
        waveform[max_sample_idx] = 10.0  # Maximum value
        waveform[max_sample_idx - 1] = 9.0
        waveform[max_sample_idx + 1] = 8.0

        hits_ch = np.zeros(
            1,
            dtype=[
                ("time", TIME_DTYPE),
                ("endtime", TIME_DTYPE),
                ("dt", TIME_DTYPE),
                ("data_dx", DATA_DTYPE, self.hit_waveform_length),
            ],
        )
        hits_ch["time"] = [hit_time]
        hits_ch["dt"] = [dt_ns]
        hits_ch["endtime"] = [hit_time + self.hit_waveform_length * dt_ns]
        hits_ch["data_dx"][0] = waveform

        # Create dummy truth (not used but needs channel field)
        truth_ch = np.zeros(
            0,
            dtype=[
                ("time", TIME_DTYPE),
                ("endtime", TIME_DTYPE),
                ("channel", np.int16),
            ],
        )

        hits_restricted, truth_restricted = self.match._restrict_to_maximum_window(
            hits_ch, truth_ch
        )

        # Expected maximum time: hit_time + max_sample_idx * dt
        # (no padding offset since data starts at index 0)
        expected_max_time = hit_time + max_sample_idx * dt_ns
        half_window = self.match.match_window_ns / 2
        expected_start = expected_max_time - half_window
        expected_end = expected_max_time + half_window

        # The window should be centered around expected_max_time, but clipped to bounds
        assert hits_restricted["time"][0] >= hits_ch["time"][0]
        assert hits_restricted["time"][0] <= expected_start or np.isclose(
            hits_restricted["time"][0], expected_start, rtol=1e-3
        )
        assert hits_restricted["endtime"][0] <= hits_ch["endtime"][0]
        # The endtime should be at least expected_end (if not clipped)
        # or the hit endtime (if clipped)
        assert hits_restricted["endtime"][0] >= min(
            expected_end, hits_ch["endtime"][0]
        ) or np.isclose(
            hits_restricted["endtime"][0],
            min(expected_end, hits_ch["endtime"][0]),
            rtol=1e-3,
        )

    def test_restrict_window_hit_with_padding(self):
        """Test hit window restriction with padding at the beginning."""

        # Create hit with padding (zeros at start)
        dt_ns = int(self.match.dt_exact)
        hit_time = 10000
        padding_offset = 50  # 50 samples of padding
        max_sample_idx_in_waveform = 250  # Maximum at sample 250 in waveform array
        max_sample_idx_actual = max_sample_idx_in_waveform - padding_offset  # Actual offset

        # Create waveform: padding at start, max at index 250
        waveform = np.zeros(self.hit_waveform_length, dtype=DATA_DTYPE)
        waveform[padding_offset : padding_offset + 100] = 1.0  # Some data
        waveform[max_sample_idx_in_waveform] = 10.0  # Maximum value

        hits_ch = np.zeros(
            1,
            dtype=[
                ("time", TIME_DTYPE),
                ("endtime", TIME_DTYPE),
                ("dt", TIME_DTYPE),
                ("data_dx", DATA_DTYPE, self.hit_waveform_length),
            ],
        )
        hits_ch["time"] = [hit_time]
        hits_ch["dt"] = [dt_ns]
        hits_ch["endtime"] = [hit_time + (self.hit_waveform_length - padding_offset) * dt_ns]
        hits_ch["data_dx"][0] = waveform

        # Create dummy truth (not used but needs channel field)
        truth_ch = np.zeros(
            0,
            dtype=[
                ("time", TIME_DTYPE),
                ("endtime", TIME_DTYPE),
                ("channel", np.int16),
            ],
        )

        hits_restricted, truth_restricted = self.match._restrict_to_maximum_window(
            hits_ch, truth_ch
        )

        # Expected maximum time: hit_time + (max_sample_idx_in_waveform - padding_offset) * dt
        expected_max_time = hit_time + max_sample_idx_actual * dt_ns
        half_window = self.match.match_window_ns / 2

        # The window should be centered around the expected maximum
        assert hits_restricted["time"][0] >= hits_ch["time"][0]
        assert hits_restricted["endtime"][0] <= hits_ch["endtime"][0]

        # Check that the window is approximately centered (allow for clipping at boundaries)
        window_center = (hits_restricted["time"][0] + hits_restricted["endtime"][0]) / 2
        assert abs(window_center - expected_max_time) < half_window

    def test_restrict_window_boundaries(self):
        """Test that restricted windows don't exceed original boundaries."""

        # Create truth with sufficient length to accommodate the window
        # Need at least pulse_template_argmax samples + half window on each side
        truth_ch = np.zeros(
            1,
            dtype=[
                ("time", TIME_DTYPE),
                ("endtime", TIME_DTYPE),
                ("channel", np.int16),
            ],
        )
        truth_ch["time"] = [1000]
        truth_ch["channel"] = 0
        # Ensure endtime is long enough for the template and window
        min_length = (
            self.match.pulse_template_argmax * self.match.dt_exact + self.match.match_window_ns
        )
        truth_ch["endtime"] = [1000 + int(min_length) + 1000000]  # Add some extra margin

        # Create dummy hits with proper dtype (not used in truth calculation)
        hits_ch = np.zeros(
            0,
            dtype=[
                ("time", TIME_DTYPE),
                ("endtime", TIME_DTYPE),
                ("dt", TIME_DTYPE),
                ("data_dx", DATA_DTYPE, self.hit_waveform_length),
            ],
        )

        hits_restricted, truth_restricted = self.match._restrict_to_maximum_window(
            hits_ch, truth_ch
        )

        # Restricted window should not exceed original boundaries
        assert truth_restricted["time"][0] >= truth_ch["time"][0]
        assert truth_restricted["endtime"][0] <= truth_ch["endtime"][0]
        assert truth_restricted["time"][0] < truth_restricted["endtime"][0]

    def test_restrict_window_match_window_ns_none(self):
        """Test that _restrict_to_maximum_window is not called when match_window_ns is None."""
        # This test verifies the calling code, not the method itself
        # The method should still work if called directly (it's just not called)
        match_no_window = Match()
        match_no_window.config = {
            "fs": 38000,
            "match_window_ms": None,
            "template_interp_path": DEFAULT_TEMPLATE_INTERP_PATH,
            "template_interp_folder": TEMPLATE_INTERP_FOLDER,
        }
        match_no_window.setup()

        assert match_no_window.match_window_ns is None
