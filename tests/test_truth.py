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
    st = straxion.qualiphide_thz_offline()
    assert "truth" in st._plugin_class_registry
    assert st._plugin_class_registry["truth"] == Truth


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason=("Test data directory not provided via " "STRAXION_TEST_DATA_DIR environment variable"),
)
class TestTruthWithRealData:
    """Test Truth plugin with real data from STRAXION_TEST_DATA_DIR."""

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

    def test_truth_dtype_inference(self):
        """Test that Truth can infer the correct data type."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        st = straxion.qualiphide_thz_offline()
        plugin = st.get_single_plugin("1756824965", "truth")
        dtype = plugin.infer_dtype()

        # Check expected fields
        expected_fields = ["time", "endtime", "energy_true", "dx_true", "channel"]
        field_names = [name for name, *_ in dtype]
        for field in expected_fields:
            assert field in field_names

    def test_truth_default_config(self):
        """Test that Truth plugin has correct default configuration."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        st = straxion.qualiphide_thz_offline()
        plugin = st.get_single_plugin("1756824965", "truth")

        # Check default values
        assert plugin.config["random_seed"] == 137
        assert plugin.config["salt_rate"] == 0
        assert plugin.config["energy_meV"] == 50
        assert plugin.config["energy_resolution_mode"] == "optimistic"

    def test_truth_custom_config(self):
        """Test Truth plugin with custom configuration."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        st = straxion.qualiphide_thz_offline()
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

    def test_truth_zero_rate_default(self):
        """Test that default zero rate produces no events."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = self._get_test_config(test_data_dir, run_id, salt_rate=0)

        clean_strax_data()
        try:
            truth = st.get_array(run_id, "truth", config=configs)
            # Should return empty array with zero rate
            assert len(truth) == 0
        except Exception as e:
            pytest.fail(f"Failed to process truth: {str(e)}")

    def test_truth_with_real_raw_records(self):
        """Test Truth plugin with real raw_records data."""
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
            # Get truth events
            truth = st.get_array(run_id, "truth", config=configs)

            # Basic validation
            assert truth is not None
            assert len(truth) > 0

            # Check required fields
            required_fields = ["time", "endtime", "energy_true", "dx_true", "channel"]
            for field in required_fields:
                assert field in truth.dtype.names, f"Required field '{field}' missing from truth"

            # Check data types
            assert truth["time"].dtype == np.int64
            assert truth["endtime"].dtype == np.int64
            assert truth["energy_true"].dtype == np.float32
            assert truth["dx_true"].dtype == np.float32
            assert truth["channel"].dtype == np.int16

            # Check that all truth events have consistent properties
            assert all(truth["endtime"] > truth["time"])
            # With energy resolution, check energies are near expected value
            assert np.abs(np.mean(truth["energy_true"]) - 50) < 10
            assert all(truth["channel"] >= 0)

            # Check that dx_true is positive and consistent with energy_true
            assert all(truth["dx_true"] > 0)

            # Check time ordering
            assert np.all(np.diff(truth["time"]) >= 0), "Time stamps not monotonically increasing"

            print(
                f"Successfully processed {len(truth)} truth events "
                f"across {len(np.unique(truth['channel']))} channels"
            )

        except Exception as e:
            pytest.fail(f"Failed to process truth: {str(e)}")

    def test_truth_reproducibility(self):
        """Test that Truth generates reproducible results with same seed."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        run_id = "1756824965"
        configs = self._get_test_config(test_data_dir, run_id, random_seed=42, salt_rate=100)

        # Generate truth events twice with same seed
        clean_strax_data()
        st1 = straxion.qualiphide_thz_offline()
        truth1 = st1.get_array(run_id, "truth", config=configs)

        clean_strax_data()
        st2 = straxion.qualiphide_thz_offline()
        truth2 = st2.get_array(run_id, "truth", config=configs)

        # Results should be identical
        assert len(truth1) == len(truth2)
        assert np.array_equal(truth1["channel"], truth2["channel"])
        assert np.array_equal(truth1["time"], truth2["time"])
        assert np.array_equal(truth1["energy_true"], truth2["energy_true"])

    def test_truth_channel_distribution(self):
        """Test that Truth distributes events across available channels."""
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
            truth = st.get_array(run_id, "truth", config=configs)

            # Check that events span multiple channels
            unique_channels = np.unique(truth["channel"])
            assert len(unique_channels) > 1  # Should hit multiple channels

            # Get raw_records to check available channels
            raw_records = st.get_array(run_id, "raw_records", config=configs)
            available_channels = np.unique(raw_records["channel"])

            # All truth channels should be from available set
            assert all(np.isin(truth["channel"], available_channels))

        except Exception as e:
            pytest.fail(f"Failed to process truth: {str(e)}")

    def test_truth_time_intervals(self):
        """Test that Truth generates events at constant time intervals."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        salt_rate = 100  # Hz
        configs = self._get_test_config(test_data_dir, run_id, salt_rate=salt_rate)

        clean_strax_data()
        try:
            truth = st.get_array(run_id, "truth", config=configs)

            # Check time intervals
            expected_dt = SECOND_TO_NANOSECOND / salt_rate
            time_diffs = np.diff(truth["time"])

            # All time intervals should be equal (within 1 ns tolerance)
            assert np.allclose(time_diffs, expected_dt, atol=1)

        except Exception as e:
            pytest.fail(f"Failed to process truth: {str(e)}")

    def test_truth_energy_values(self):
        """Test that Truth assigns correct energy values."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        energy_meV = 75
        configs = self._get_test_config(test_data_dir, run_id, energy_meV=energy_meV, salt_rate=100)

        clean_strax_data()
        try:
            truth = st.get_array(run_id, "truth", config=configs)

            # With energy resolution, energies are sampled from Gaussian
            # Check they're centered around the expected value
            assert np.abs(np.mean(truth["energy_true"]) - energy_meV) < 5

        except Exception as e:
            pytest.fail(f"Failed to process truth: {str(e)}")

    def test_truth_energy_resolution_none(self):
        """Test Truth with no energy resolution smearing."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = self._get_test_config(
            test_data_dir, run_id, energy_resolution_mode="none", salt_rate=100
        )

        clean_strax_data()
        try:
            truth = st.get_array(run_id, "truth", config=configs)

            # With mode="none", all energies should be exactly equal
            assert all(truth["energy_true"] == 50)

        except Exception as e:
            pytest.fail(f"Failed to process truth: {str(e)}")

    def test_truth_energy_resolution_modes(self):
        """Test Truth with different energy resolution modes."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        run_id = "1756824965"
        energy_meV = 50

        # Test optimistic mode
        clean_strax_data()
        st_opt = straxion.qualiphide_thz_offline()
        configs_opt = self._get_test_config(
            test_data_dir, run_id, energy_resolution_mode="optimistic", salt_rate=100
        )
        truth_opt = st_opt.get_array(run_id, "truth", config=configs_opt)
        energies_opt = truth_opt["energy_true"]

        # Test conservative mode
        clean_strax_data()
        st_cons = straxion.qualiphide_thz_offline()
        configs_cons = self._get_test_config(
            test_data_dir, run_id, energy_resolution_mode="conservative", salt_rate=100
        )
        truth_cons = st_cons.get_array(run_id, "truth", config=configs_cons)
        energies_cons = truth_cons["energy_true"]

        # Conservative should have larger spread than optimistic
        assert np.std(energies_cons) > np.std(energies_opt)

        # Both should be centered near the true energy
        assert np.abs(np.mean(energies_opt) - energy_meV) < 5
        assert np.abs(np.mean(energies_cons) - energy_meV) < 5

    def test_truth_invalid_resolution_mode(self):
        """Test that invalid resolution mode raises error."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = self._get_test_config(
            test_data_dir, run_id, energy_resolution_mode="invalid", salt_rate=100
        )

        clean_strax_data()
        # Should raise ValueError for invalid mode
        with pytest.raises(ValueError):
            st.get_array(run_id, "truth", config=configs)

    def test_truth_dx_true_calculation(self):
        """Test that dx_true is correctly calculated from energy_true."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = self._get_test_config(
            test_data_dir, run_id, energy_resolution_mode="none", salt_rate=100
        )

        clean_strax_data()
        try:
            truth = st.get_array(run_id, "truth", config=configs)
            plugin = st.get_single_plugin(run_id, "truth")

            # With mode="none", all energies should be exactly 50 meV
            # And dx_true should be meV_to_dx(50)
            expected_dx = plugin.meV_to_dx(50)
            assert all(truth["energy_true"] == 50)
            assert all(truth["dx_true"] == expected_dx)

            # Verify the conversion is correct
            for event in truth:
                calculated_dx = plugin.meV_to_dx(event["energy_true"])
                assert np.isclose(event["dx_true"], calculated_dx)

        except Exception as e:
            pytest.fail(f"Failed to process truth: {str(e)}")
