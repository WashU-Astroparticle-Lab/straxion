import numpy as np
import pytest
import os
import tempfile
import shutil
from straxion.plugins.records import DxRecords
from straxion.constants import DEFAULT_TEMPLATE_INTERP_PATH, TEMPLATE_INTERP_FOLDER


class TestDxRecordsAveragedDxMap:
    """Test the optional epoch-averaged theta->frequency map of DxRecords."""

    def setup_method(self):
        """Set up synthetic scan files, a DxRecords instance, and its mocked config."""
        self.temp_dir = tempfile.mkdtemp()
        self.map_dir = tempfile.mkdtemp()
        self.dx_records = DxRecords()

        # Mock the config (dx_map_filename empty = standard per-scan behaviour)
        self.dx_records.config = {
            "iq_finescan_dir": self.temp_dir,
            "iq_widescan_filename": "iq_wide_z_test-1234567890.npy",
            "iq_finescan_filename": "iq_fine_z_test-1234567890.npy",
            "resonant_frequency_filename": "fres_test-1234567890.npy",
            "resonant_frequency_dir": self.temp_dir,
            "iq_widescan_dir": self.temp_dir,
            "widescan_resolution": 1000.0,
            "cable_correction_polyfit_order": 3,
            "fs": 38000,
            "pulse_kernel_start_time": 200000,
            "pulse_kernel_decay_time": 600000,
            "pulse_kernel_gaussian_smearing_width": 28000,
            "pulse_kernel_truncation_factor": 10,
            "moving_average_width": 100000,
            "pca_n_components": 4,
            "template_interp_path": DEFAULT_TEMPLATE_INTERP_PATH,
            "template_interp_folder": TEMPLATE_INTERP_FOLDER,
            "dx_map_dir": self.map_dir,
            "dx_map_filename": "",
        }

        # Create test data files
        rng = np.random.default_rng(1234)
        self.n_channels = 3
        self.n_fine_points = 10
        self.n_wide_points = 20

        fine_z_data = rng.random((self.n_channels, self.n_fine_points)) + 1j * rng.random(
            (self.n_channels, self.n_fine_points)
        )
        fine_f_data = rng.random((self.n_channels, self.n_fine_points)) * 1000 + 1000
        wide_z_data = rng.random((self.n_channels, self.n_wide_points)) + 1j * rng.random(
            (self.n_channels, self.n_wide_points)
        )
        wide_f_data = rng.random((self.n_channels, self.n_wide_points)) * 2000 + 500
        fres_data = np.array([1500.0, 1600.0, 1700.0])

        # Save test files
        np.save(os.path.join(self.temp_dir, "iq_fine_z_test-1234567890.npy"), fine_z_data)
        np.save(os.path.join(self.temp_dir, "iq_fine_f_test-1234567890.npy"), fine_f_data)
        np.save(os.path.join(self.temp_dir, "iq_wide_z_test-1234567890.npy"), wide_z_data)
        np.save(os.path.join(self.temp_dir, "iq_wide_f_test-1234567890.npy"), wide_f_data)
        np.save(os.path.join(self.temp_dir, "fres_test-1234567890.npy"), fres_data)

        # Reference per-scan tables (no map)
        self.dx_records._setup_iq_correction_and_calibration()
        self.dx_records._setup_frequency_interpolation_models()
        self.ref_x = [x.copy() for x in self.dx_records.interp_x_data]
        self.ref_y = [y.copy() for y in self.dx_records.interp_y_data]
        self.ref_f0 = self.dx_records.interpolated_freqs.copy()
        self.ref_thetas = self.dx_records.thetas_at_fres.copy()

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.map_dir)

    def _fractional_map_from_per_scan_tables(self):
        """Build (x, y_med) exactly equal to the per-scan tables' own fractional maps."""
        x = np.stack(self.ref_x)
        y_med = np.stack([(y - f0) / f0 for y, f0 in zip(self.ref_y, self.ref_f0)])
        return x, y_med

    def _save_map(self, filename, **arrays):
        np.savez(os.path.join(self.map_dir, filename), **arrays)

    def _rerun_setup(self):
        self.dx_records._setup_iq_correction_and_calibration()
        self.dx_records._setup_frequency_interpolation_models()

    def test_no_map_is_default(self):
        """Empty dx_map_filename leaves everything untouched and flags no channel."""
        assert hasattr(self.dx_records, "avg_map_used")
        assert self.dx_records.avg_map_used.dtype == bool
        assert self.dx_records.avg_map_used.shape == (self.n_channels,)
        assert not np.any(self.dx_records.avg_map_used)

        # Re-run without the dx_map keys at all (legacy configs) -> identical tables
        self.dx_records.config.pop("dx_map_filename")
        self.dx_records.config.pop("dx_map_dir")
        self._rerun_setup()
        for ch in range(self.n_channels):
            np.testing.assert_array_equal(self.dx_records.interp_x_data[ch], self.ref_x[ch])
            np.testing.assert_array_equal(self.dx_records.interp_y_data[ch], self.ref_y[ch])
        np.testing.assert_array_equal(self.dx_records.interpolated_freqs, self.ref_f0)
        assert not np.any(self.dx_records.avg_map_used)

    def test_self_consistent_map_reproduces_per_scan_tables(self):
        """A map equal to the scan's own fractional map gives f0*(1+y_med) and f(0) == f0."""
        x, y_med = self._fractional_map_from_per_scan_tables()
        # Extra diagnostic keys must be ignored
        self._save_map("dx_map_test.npz", x=x, y_med=y_med, y16=y_med, y84=y_med, S=np.ones(3))
        self.dx_records.config["dx_map_filename"] = "dx_map_test.npz"
        self._rerun_setup()

        assert np.all(self.dx_records.avg_map_used)
        for ch in range(self.n_channels):
            f0 = self.ref_f0[ch]
            np.testing.assert_array_equal(self.dx_records.interp_x_data[ch], x[ch])
            np.testing.assert_allclose(
                self.dx_records.interp_y_data[ch], f0 * (1.0 + y_med[ch]), rtol=0, atol=0
            )
            # The map reproduces the per-scan table up to floating-point round-off
            np.testing.assert_allclose(
                self.dx_records.interp_y_data[ch], self.ref_y[ch], rtol=1e-12
            )
            assert self.dx_records.interp_x_data[ch].dtype == np.float64
            assert self.dx_records.interp_y_data[ch].dtype == np.float64
            assert self.dx_records.interp_x_data[ch].flags["C_CONTIGUOUS"]
            assert self.dx_records.interp_y_data[ch].flags["C_CONTIGUOUS"]
            # Operating point unchanged: f(dtheta=0) == f0 to 1e-9
            assert abs(self.dx_records.interpolated_freqs[ch] - f0) < 1e-9 * abs(f0)
        # Per-scan quantities are untouched
        np.testing.assert_array_equal(self.dx_records.thetas_at_fres, self.ref_thetas)

    def test_map_rescales_with_scan_f0(self):
        """A pure-slope map is scaled by this scan's f0 for each channel."""
        n_grid = 21
        x = np.tile(np.linspace(-1.0, 1.0, n_grid), (self.n_channels, 1))
        slope = -5e-6
        y_med = slope * x
        self._save_map("dx_map_slope.npz", x=x, y_med=y_med)
        self.dx_records.config["dx_map_filename"] = "dx_map_slope.npz"
        self._rerun_setup()

        assert np.all(self.dx_records.avg_map_used)
        for ch in range(self.n_channels):
            f0 = self.ref_f0[ch]
            np.testing.assert_array_equal(self.dx_records.interp_x_data[ch], x[ch])
            np.testing.assert_allclose(
                self.dx_records.interp_y_data[ch], f0 * (1.0 + slope * x[ch]), rtol=1e-15
            )
            assert abs(self.dx_records.interpolated_freqs[ch] - f0) < 1e-9 * abs(f0)

    def test_nan_channel_and_missing_channel_keep_per_scan_table(self):
        """Channels with non-finite y_med or absent from the map keep their per-scan table."""
        x, y_med = self._fractional_map_from_per_scan_tables()
        # Channel 1: NaN in the map. Channel 2: not present in the map (only 2 rows).
        y_med[1, 3] = np.nan
        self._save_map("dx_map_partial.npz", x=x[:2], y_med=y_med[:2])
        self.dx_records.config["dx_map_filename"] = "dx_map_partial.npz"
        with pytest.warns(UserWarning, match="applied to 1 of 3 channels"):
            self._rerun_setup()

        np.testing.assert_array_equal(self.dx_records.avg_map_used, [True, False, False])
        for ch in (1, 2):
            np.testing.assert_array_equal(self.dx_records.interp_x_data[ch], self.ref_x[ch])
            np.testing.assert_array_equal(self.dx_records.interp_y_data[ch], self.ref_y[ch])
            assert self.dx_records.interpolated_freqs[ch] == self.ref_f0[ch]
        np.testing.assert_array_equal(self.dx_records.interp_x_data[0], x[0])
        np.testing.assert_allclose(self.dx_records.interp_y_data[0], self.ref_y[0], rtol=1e-12)

    def test_missing_file_raises(self):
        """A non-existent map file raises FileNotFoundError."""
        self.dx_records.config["dx_map_filename"] = "does_not_exist.npz"
        with pytest.raises(FileNotFoundError, match="Averaged dx map file not found"):
            self._rerun_setup()

    def test_missing_keys_raise(self):
        """A map without the x / y_med keys raises KeyError."""
        x, y_med = self._fractional_map_from_per_scan_tables()
        self._save_map("dx_map_badkeys.npz", x=x, y_median=y_med)
        self.dx_records.config["dx_map_filename"] = "dx_map_badkeys.npz"
        with pytest.raises(KeyError, match="y_med"):
            self._rerun_setup()

    def test_option_registration_and_version(self):
        """The new options are registered with the intended tracking and the version is bumped."""
        options = DxRecords.takes_config
        assert options["dx_map_dir"].track is False
        assert options["dx_map_dir"].default == ""
        assert options["dx_map_filename"].track is True
        assert options["dx_map_filename"].default == ""
        assert DxRecords.__version__ == "0.5.0"
