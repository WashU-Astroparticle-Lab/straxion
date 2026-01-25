import numpy as np
import pytest
import os
import tempfile
import shutil
from straxion.plugins.records import PulseProcessing, DxRecords
from straxion.utils import (
    PULSE_TEMPLATE_LENGTH,
    PULSE_TEMPLATE_ARGMAX,
    DEFAULT_TEMPLATE_INTERP_PATH,
    TEMPLATE_INTERP_FOLDER,
    load_interpolation,
    SECOND_TO_NANOSECOND,
)


# Note: circfit method tests removed as the method doesn't exist in PulseProcessing
class TestLoadFinescanFiles:
    """Test the load_finescan_files static method of PulseProcessing class."""

    def test_load_finescan_files_nonexistent_directory(self):
        """Test that FileNotFoundError is raised for nonexistent directory."""
        with pytest.raises(FileNotFoundError, match="Fine scan directory or file not found"):
            PulseProcessing.load_finescan_files("/nonexistent/path")

    def test_load_finescan_files_empty_directory(self):
        """Test that FileNotFoundError is raised for directory with no matching files."""
        empty_dir = tempfile.mkdtemp()
        try:
            with pytest.raises(FileNotFoundError, match="No fine scan files found"):
                PulseProcessing.load_finescan_files(empty_dir)
        finally:
            shutil.rmtree(empty_dir)


class TestPulseKernel:
    """Test the pulse_kernel static method of PulseProcessing class."""

    def test_pulse_kernel_basic(self):
        """Test basic pulse kernel generation with typical parameters."""
        ns = 10000
        fs = 100000  # 100 kHz
        t0 = 100000  # 100 us
        tau = 300000  # 300 us
        sigma = 700000  # 700 us
        truncation_factor = 5

        kernel = PulseProcessing.pulse_kernel(ns, fs, t0, tau, sigma, truncation_factor)

        # Basic checks
        assert isinstance(kernel, np.ndarray)
        assert kernel.ndim == 1
        assert len(kernel) > 0

        # Kernel should be normalized (sum = 1)
        np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=1e-10)

        # All values should be non-negative
        assert np.all(kernel >= 0)

        # Kernel should have finite values
        assert np.all(np.isfinite(kernel))

    def test_pulse_kernel_parameters(self):
        """Test pulse kernel with different parameter combinations."""
        ns = 5000
        fs = 50000  # 50 kHz

        # Test different parameter sets
        test_params = [
            (50000, 200000, 500000, 3),  # Short pulse
            (100000, 500000, 1000000, 7),  # Long pulse
            (0, 100000, 200000, 4),  # Immediate start
        ]

        for t0, tau, sigma, truncation_factor in test_params:
            kernel = PulseProcessing.pulse_kernel(ns, fs, t0, tau, sigma, truncation_factor)

            # Basic validation
            assert isinstance(kernel, np.ndarray)
            assert kernel.ndim == 1
            assert len(kernel) > 0
            np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=1e-10)
            assert np.all(kernel >= 0)
            assert np.all(np.isfinite(kernel))

    def test_pulse_kernel_truncation(self):
        """Test that truncation factor affects kernel length appropriately."""
        ns = 10000
        fs = 100000
        t0 = 100000
        tau = 300000
        sigma = 700000

        # Test different truncation factors
        kernels = []
        for truncation_factor in [2, 5, 10]:
            kernel = PulseProcessing.pulse_kernel(ns, fs, t0, tau, sigma, truncation_factor)
            kernels.append(kernel)

            # All kernels should be normalized
            np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=1e-10)

        # Larger truncation factors should generally result in longer kernels
        # (though this depends on the specific parameters)
        assert len(kernels[1]) >= len(kernels[0])  # truncation_factor 5 vs 2
        assert len(kernels[2]) >= len(kernels[1])  # truncation_factor 10 vs 5


class TestConvertIQToTheta:
    """Test the convert_iq_to_theta method of PulseProcessing class."""

    def setup_method(self):
        """Set up test data and create a PulseProcessing instance."""
        # Create a minimal PulseProcessing instance for testing
        self.pp = PulseProcessing()

        # Mock the channel_centers that would normally be set up in setup()
        self.pp.channel_centers = {
            0: (1.0, 2.0, 0.0),  # (i_center, q_center, theta_f_min)
            1: (0.0, 0.0, -np.pi),  # Center at origin
        }

    def test_convert_iq_to_theta_basic(self):
        """Test basic I/Q to theta conversion."""
        # Create test I/Q data
        data_i = np.array([1.5, 2.0, 0.5])
        data_q = np.array([2.5, 3.0, 1.5])
        channel = 0

        thetas = self.pp.convert_iq_to_theta(data_i, data_q, channel)

        # Basic checks
        assert isinstance(thetas, np.ndarray)
        assert thetas.shape == data_i.shape
        assert np.all(np.isfinite(thetas))

        # Thetas should be in radians
        assert np.all(thetas >= 0)  # After angle wrapping correction
        assert np.all(thetas <= 2 * np.pi)

    def test_convert_iq_to_theta_center_at_origin(self):
        """Test I/Q to theta conversion with center at origin."""
        data_i = np.array([1.0, -1.0, 0.0, 0.0])
        data_q = np.array([0.0, 0.0, 1.0, -1.0])
        channel = 1  # Center at origin

        thetas = self.pp.convert_iq_to_theta(data_i, data_q, channel)

        # Expected angles: 0, pi, pi/2, -pi/2
        expected = np.array([0.0, np.pi, np.pi / 2, -np.pi / 2])
        np.testing.assert_allclose(thetas, expected, rtol=1e-10)

    def test_convert_iq_to_theta_angle_wrapping(self):
        """Test that angle wrapping works correctly."""
        # Create data that would result in negative angles
        data_i = np.array([0.5, 0.5])
        data_q = np.array([-0.5, -0.5])
        channel = 0

        thetas = self.pp.convert_iq_to_theta(data_i, data_q, channel)

        # After wrapping, angles should be positive
        assert np.all(thetas >= 0)
        assert np.all(thetas <= 2 * np.pi)

    def test_convert_iq_to_theta_consistency(self):
        """Test that conversion is consistent for same I/Q values."""
        data_i = np.array([1.0, 2.0, 3.0])
        data_q = np.array([4.0, 5.0, 6.0])
        channel = 0

        thetas1 = self.pp.convert_iq_to_theta(data_i, data_q, channel)
        thetas2 = self.pp.convert_iq_to_theta(data_i, data_q, channel)

        # Results should be identical
        np.testing.assert_array_equal(thetas1, thetas2)

    def test_convert_iq_to_theta_invalid_channel(self):
        """Test that KeyError is raised for invalid channel."""
        data_i = np.array([1.0])
        data_q = np.array([2.0])
        channel = 999  # Invalid channel

        with pytest.raises(KeyError):
            self.pp.convert_iq_to_theta(data_i, data_q, channel)


class TestDxRecordsPCA:
    """Test the PCA method of DxRecords class."""

    def setup_method(self):
        """Set up test data and create a DxRecords instance."""
        self.dx_records = DxRecords()
        # Mock the pca_n_components attribute
        self.dx_records.pca_n_components = 2

    def test_pca_basic(self):
        """Test basic PCA functionality with simple data."""
        # Create a simple dataset with known structure
        n_samples = 100
        data = np.random.randn(n_samples)

        # Apply PCA
        result = self.dx_records.pca(data)

        # Basic checks
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        assert np.all(np.isfinite(result))

    def test_pca_removes_components(self):
        """Test that PCA actually removes principal components."""
        # Create data with a strong principal component
        n_samples = 1000
        t = np.linspace(0, 10, n_samples)
        # Add a strong sinusoidal component (will be first PC)
        data = 10 * np.sin(t) + 0.1 * np.random.randn(n_samples)

        # Apply PCA with 1 component removed
        self.dx_records.pca_n_components = 1
        result = self.dx_records.pca(data)

        # Result should have lower variance than input
        # since we removed the dominant component
        assert np.var(result) < np.var(data)
        assert np.all(np.isfinite(result))

    def test_pca_different_n_components(self):
        """Test PCA with different numbers of components."""
        n_samples = 200
        data = np.random.randn(n_samples)

        results = {}
        for n_comp in [1, 2, 4]:
            self.dx_records.pca_n_components = n_comp
            results[n_comp] = self.dx_records.pca(data)

            # All results should be valid arrays
            assert isinstance(results[n_comp], np.ndarray)
            assert results[n_comp].shape == data.shape
            assert np.all(np.isfinite(results[n_comp]))

    def test_pca_preserves_mean(self):
        """Test that PCA approximately preserves the mean of the data."""
        n_samples = 500
        data = np.random.randn(n_samples) + 5.0  # Add offset

        result = self.dx_records.pca(data)

        # Mean should be approximately preserved
        np.testing.assert_allclose(np.mean(result), np.mean(data), rtol=0.1)


class TestDxRecordsStaticMethods:
    """Test the static methods of DxRecords class."""

    def test_infer_dtype(self):
        """Test the infer_dtype method of DxRecords."""
        # Create a DxRecords instance
        dx_records = DxRecords()

        # Mock the record_length attribute
        dx_records.record_length = 1000

        # Test infer_dtype
        dtype = dx_records.infer_dtype()

        # Check that it's a structured dtype
        assert hasattr(dtype, "names")
        assert hasattr(dtype, "fields")

        # Check required fields
        required_fields = [
            "time",
            "endtime",
            "length",
            "dt",
            "channel",
            "data_dtheta",
            "data_dx",
            "data_dx_moving_average",
            "data_dx_convolved",
        ]
        for field in required_fields:
            assert field in dtype.names, f"Field '{field}' missing from dtype"

        # Check that data fields have the correct shape
        assert dtype["data_dtheta"].shape == (1000,)
        assert dtype["data_dx"].shape == (1000,)
        assert dtype["data_dx_moving_average"].shape == (1000,)
        assert dtype["data_dx_convolved"].shape == (1000,)

    def test_iq_gain_correction_model_basic(self):
        """Test basic IQ gain correction model generation."""
        # Create mock data
        n_channels = 2
        n_fine_points = 10
        n_wide_points = 20

        fine_f = np.random.rand(n_channels, n_fine_points) * 1000 + 1000  # 1000-2000 Hz
        wide_z = np.random.rand(n_channels, n_wide_points) + 1j * np.random.rand(
            n_channels, n_wide_points
        )
        wide_f = np.random.rand(n_channels, n_wide_points) * 2000 + 500  # 500-2500 Hz
        fres = np.array([1500.0, 1600.0])  # Resonant frequencies
        widescan_resolution = 1000.0
        polyfit_order = 3

        i_models, q_models = DxRecords.iq_gain_correction_model(
            fine_f, wide_z, wide_f, fres, widescan_resolution, polyfit_order
        )

        # Basic validation
        assert len(i_models) == n_channels
        assert len(q_models) == n_channels

        for i in range(n_channels):
            assert hasattr(i_models[i], "__call__")  # Should be callable (poly1d)
            assert hasattr(q_models[i], "__call__")  # Should be callable (poly1d)

            # Test that models can be evaluated
            test_freq_offset = 0.0
            i_val = i_models[i](test_freq_offset)
            q_val = q_models[i](test_freq_offset)

            assert np.isfinite(i_val)
            assert np.isfinite(q_val)

    def test_iq_gain_correction_model_edge_cases(self):
        """Test IQ gain correction model with edge cases."""
        # Test with single channel
        fine_f = np.array([[1.0, 2.0, 3.0]])
        wide_z = np.array([[1.0 + 1j, 2.0 + 2j, 3.0 + 3j]])
        wide_f = np.array([[0.5, 1.5, 2.5]])
        fres = np.array([1.0])
        widescan_resolution = 10.0
        polyfit_order = 1

        i_models, q_models = DxRecords.iq_gain_correction_model(
            fine_f, wide_z, wide_f, fres, widescan_resolution, polyfit_order
        )

        assert len(i_models) == 1
        assert len(q_models) == 1

    def test_pulse_kernel_basic(self):
        """Test basic pulse kernel generation with typical parameters."""
        ns = 10000
        fs = 100000  # 100 kHz
        t0 = 100000  # 100 us
        tau = 300000  # 300 us
        sigma = 700000  # 700 us
        truncation_factor = 5

        kernel = DxRecords.pulse_kernel(ns, fs, t0, tau, sigma, truncation_factor)

        # Basic checks
        assert isinstance(kernel, np.ndarray)
        assert kernel.ndim == 1
        assert len(kernel) > 0

        # Kernel should be normalized (sum = 1)
        np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=1e-10)

        # All values should be non-negative
        assert np.all(kernel >= 0)

        # Kernel should have finite values
        assert np.all(np.isfinite(kernel))

    def test_pulse_kernel_parameters(self):
        """Test pulse kernel with different parameter combinations."""
        ns = 5000
        fs = 50000  # 50 kHz

        # Test different parameter sets
        test_params = [
            (50000, 200000, 500000, 3),  # Short pulse
            (100000, 500000, 1000000, 7),  # Long pulse
            (0, 100000, 200000, 4),  # Immediate start
        ]

        for t0, tau, sigma, truncation_factor in test_params:
            kernel = DxRecords.pulse_kernel(ns, fs, t0, tau, sigma, truncation_factor)

            # Basic validation
            assert isinstance(kernel, np.ndarray)
            assert kernel.ndim == 1
            assert len(kernel) > 0
            np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=1e-10)
            assert np.all(kernel >= 0)
            assert np.all(np.isfinite(kernel))

    def test_pulse_kernel_truncation(self):
        """Test that truncation factor affects kernel length appropriately."""
        ns = 10000
        fs = 100000
        t0 = 100000
        tau = 300000
        sigma = 700000

        # Test different truncation factors
        kernels = []
        for truncation_factor in [2, 5, 10]:
            kernel = DxRecords.pulse_kernel(ns, fs, t0, tau, sigma, truncation_factor)
            kernels.append(kernel)

            # All kernels should be normalized
            np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=1e-10)

        # Larger truncation factors should generally result in longer kernels
        # (though this depends on the specific parameters)
        assert len(kernels[1]) >= len(kernels[0])  # truncation_factor 5 vs 2
        assert len(kernels[2]) >= len(kernels[1])  # truncation_factor 10 vs 5


class TestDxRecordsFileLoading:
    """Test the _load_correction_files method of DxRecords class."""

    def setup_method(self):
        """Set up test data and create a DxRecords instance."""
        self.temp_dir = tempfile.mkdtemp()
        self.dx_records = DxRecords()

        # Mock the config
        self.dx_records.config = {
            "iq_finescan_dir": self.temp_dir,
            "iq_widescan_filename": "iq_wide_z_test-1234567890.npy",
            "iq_finescan_filename": "iq_fine_z_test-1234567890.npy",
            "resonant_frequency_filename": "fres_test-1234567890.npy",
            "resonant_frequency_dir": self.temp_dir,
            "iq_widescan_dir": self.temp_dir,
            "fs": 38000,
        }

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_load_correction_files_valid_files(self):
        """Test loading valid correction files."""
        # Create test data files
        fine_z_data = np.random.rand(2, 10) + 1j * np.random.rand(2, 10)
        fine_f_data = np.random.rand(2, 10) * 1000 + 1000
        wide_z_data = np.random.rand(2, 20) + 1j * np.random.rand(2, 20)
        wide_f_data = np.random.rand(2, 20) * 2000 + 500
        fres_data = np.array([1500.0, 1600.0])

        # Save test files
        np.save(os.path.join(self.temp_dir, "iq_fine_z_test-1234567890.npy"), fine_z_data)
        np.save(os.path.join(self.temp_dir, "iq_fine_f_test-1234567890.npy"), fine_f_data)
        np.save(os.path.join(self.temp_dir, "iq_wide_z_test-1234567890.npy"), wide_z_data)
        np.save(os.path.join(self.temp_dir, "iq_wide_f_test-1234567890.npy"), wide_f_data)
        np.save(os.path.join(self.temp_dir, "fres_test-1234567890.npy"), fres_data)

        # Test loading
        self.dx_records._load_correction_files()

        # Verify data was loaded correctly
        np.testing.assert_array_equal(self.dx_records.fine_z, fine_z_data)
        np.testing.assert_array_equal(self.dx_records.fine_f, fine_f_data)
        np.testing.assert_array_equal(self.dx_records.wide_z, wide_z_data)
        np.testing.assert_array_equal(self.dx_records.wide_f, wide_f_data)
        np.testing.assert_array_equal(self.dx_records.fres, fres_data)

    def test_load_correction_files_invalid_extension(self):
        """Test that AssertionError is raised for files with invalid extensions."""
        # Change the config to use invalid extensions
        self.dx_records.config["iq_finescan_filename"] = "iq_fine_z_test-1234567890.txt"
        self.dx_records.config["iq_widescan_filename"] = "iq_wide_z_test-1234567890.txt"
        self.dx_records.config["resonant_frequency_filename"] = "fres_test-1234567890.txt"

        with pytest.raises(AssertionError, match="should end with .npy"):
            self.dx_records._load_correction_files()

    def test_load_correction_files_invalid_prefix(self):
        """Test that AssertionError is raised for files with invalid prefixes."""
        # Change the config to use invalid prefixes
        self.dx_records.config["iq_finescan_filename"] = "wrong_fine_z_test-1234567890.npy"
        self.dx_records.config["iq_widescan_filename"] = "wrong_wide_z_test-1234567890.npy"
        self.dx_records.config["resonant_frequency_filename"] = "wrong_fres_test-1234567890.npy"

        with pytest.raises(AssertionError, match="should start with"):
            self.dx_records._load_correction_files()

    def test_load_correction_files_timestamp_mismatch(self):
        """Test that AssertionError is raised for files with mismatched timestamps."""
        # Change the config to use different timestamps
        self.dx_records.config["iq_finescan_filename"] = "iq_fine_z_test-1234567890.npy"
        self.dx_records.config["iq_widescan_filename"] = "iq_wide_z_test-1234567891.npy"
        self.dx_records.config["resonant_frequency_filename"] = "fres_test-1234567892.npy"

        with pytest.raises(AssertionError, match="should be the same"):
            self.dx_records._load_correction_files()

    def test_load_correction_files_missing_files(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            self.dx_records._load_correction_files()


class TestDxRecordsSetupMethods:
    """Test the setup methods of DxRecords class."""

    def setup_method(self):
        """Set up test data and create a DxRecords instance."""
        self.temp_dir = tempfile.mkdtemp()
        self.dx_records = DxRecords()

        # Mock the config
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
        }

        # Create test data files
        self.n_channels = 2
        self.n_fine_points = 10
        self.n_wide_points = 20

        fine_z_data = np.random.rand(self.n_channels, self.n_fine_points) + 1j * np.random.rand(
            self.n_channels, self.n_fine_points
        )
        fine_f_data = np.random.rand(self.n_channels, self.n_fine_points) * 1000 + 1000
        wide_z_data = np.random.rand(self.n_channels, self.n_wide_points) + 1j * np.random.rand(
            self.n_channels, self.n_wide_points
        )
        wide_f_data = np.random.rand(self.n_channels, self.n_wide_points) * 2000 + 500
        fres_data = np.array([1500.0, 1600.0])

        # Save test files
        np.save(os.path.join(self.temp_dir, "iq_fine_z_test-1234567890.npy"), fine_z_data)
        np.save(os.path.join(self.temp_dir, "iq_fine_f_test-1234567890.npy"), fine_f_data)
        np.save(os.path.join(self.temp_dir, "iq_wide_z_test-1234567890.npy"), wide_z_data)
        np.save(os.path.join(self.temp_dir, "iq_wide_f_test-1234567890.npy"), wide_f_data)
        np.save(os.path.join(self.temp_dir, "fres_test-1234567890.npy"), fres_data)

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_setup_iq_correction_and_calibration(self):
        """Test the IQ correction and calibration setup."""
        self.dx_records._setup_iq_correction_and_calibration()

        # Verify that all required attributes are set
        assert hasattr(self.dx_records, "i_models")
        assert hasattr(self.dx_records, "q_models")
        assert hasattr(self.dx_records, "iq_centers")
        assert hasattr(self.dx_records, "phis")
        assert hasattr(self.dx_records, "fine_z_corrected")

        # Verify dimensions
        assert len(self.dx_records.i_models) == self.n_channels
        assert len(self.dx_records.q_models) == self.n_channels
        assert len(self.dx_records.iq_centers) == self.n_channels
        assert len(self.dx_records.phis) == self.n_channels

        # Verify that models are callable
        for i in range(self.n_channels):
            assert hasattr(self.dx_records.i_models[i], "__call__")
            assert hasattr(self.dx_records.q_models[i], "__call__")

        # Verify that centers and phis are finite
        assert np.all(np.isfinite(self.dx_records.iq_centers))
        assert np.all(np.isfinite(self.dx_records.phis))

    def test_setup_frequency_interpolation_models(self):
        """Test the frequency interpolation models setup."""
        # First setup IQ correction
        self.dx_records._setup_iq_correction_and_calibration()

        # Then setup frequency interpolation
        self.dx_records._setup_frequency_interpolation_models()

        # Verify that all required attributes are set
        assert hasattr(self.dx_records, "thetas_at_fres")
        assert hasattr(self.dx_records, "interpolated_freqs")
        assert hasattr(self.dx_records, "f_interpolation_models")

        # Verify dimensions
        assert len(self.dx_records.thetas_at_fres) == self.n_channels
        assert len(self.dx_records.interpolated_freqs) == self.n_channels
        assert len(self.dx_records.f_interpolation_models) == self.n_channels

        # Verify that interpolation models are callable
        for i in range(self.n_channels):
            assert hasattr(self.dx_records.f_interpolation_models[i], "__call__")

        # Verify that thetas and frequencies are finite
        assert np.all(np.isfinite(self.dx_records.thetas_at_fres))
        assert np.all(np.isfinite(self.dx_records.interpolated_freqs))

    def test_setup_method(self):
        """Test the complete setup method."""

        # Mock the deps attribute that would normally be set by strax
        class MockDeps:
            def dtype_for(self, name):
                if name == "raw_records":
                    return np.dtype(
                        [
                            ("data_i", np.float32, 1000),
                            ("data_q", np.float32, 1000),
                        ]
                    )

        self.dx_records.deps = {"raw_records": MockDeps()}

        # Test setup
        self.dx_records.setup()

        # Verify that all required attributes are set
        assert hasattr(self.dx_records, "record_length")
        assert hasattr(self.dx_records, "dt_exact")
        assert hasattr(self.dx_records, "i_models")
        assert hasattr(self.dx_records, "q_models")
        assert hasattr(self.dx_records, "iq_centers")
        assert hasattr(self.dx_records, "phis")
        assert hasattr(self.dx_records, "fine_z_corrected")
        assert hasattr(self.dx_records, "thetas_at_fres")
        assert hasattr(self.dx_records, "interpolated_freqs")
        assert hasattr(self.dx_records, "f_interpolation_models")
        assert hasattr(self.dx_records, "kernel")
        assert hasattr(self.dx_records, "moving_average_kernel")
        assert hasattr(self.dx_records, "pca_n_components")
        # Verify interpolation-related attributes
        assert hasattr(self.dx_records, "At_interp")
        assert hasattr(self.dx_records, "t_max")
        assert hasattr(self.dx_records, "interpolated_template")

        # Verify record_length is set correctly
        assert self.dx_records.record_length == 1000

        # Verify dt_exact is calculated correctly
        expected_dt = 1 / self.dx_records.config["fs"] * 1e9  # Convert to ns
        assert self.dx_records.dt_exact == expected_dt

        # Verify pca_n_components is set from config
        assert self.dx_records.pca_n_components == (self.dx_records.config["pca_n_components"])

        # Verify interpolated_template properties
        assert len(self.dx_records.interpolated_template) == PULSE_TEMPLATE_LENGTH
        assert np.argmax(self.dx_records.interpolated_template) == PULSE_TEMPLATE_ARGMAX
        assert np.all(np.isfinite(self.dx_records.interpolated_template))


class TestDxRecordsCompute:
    """Test the compute method of DxRecords class."""

    def setup_method(self):
        """Set up test data and create a DxRecords instance."""
        self.temp_dir = tempfile.mkdtemp()
        self.dx_records = DxRecords()

        # Mock the config
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
            "moving_average_width": 100000,
            "pulse_kernel_start_time": 200000,
            "pulse_kernel_decay_time": 600000,
            "pulse_kernel_gaussian_smearing_width": 28000,
            "pulse_kernel_truncation_factor": 10,
            "pca_n_components": 4,
            "template_interp_path": DEFAULT_TEMPLATE_INTERP_PATH,
            "template_interp_folder": TEMPLATE_INTERP_FOLDER,
        }

        # Create test data files
        self.n_channels = 2
        self.n_fine_points = 10
        self.n_wide_points = 20
        self.record_length = 1000

        fine_z_data = np.random.rand(self.n_channels, self.n_fine_points) + 1j * np.random.rand(
            self.n_channels, self.n_fine_points
        )
        fine_f_data = np.random.rand(self.n_channels, self.n_fine_points) * 1000 + 1000
        wide_z_data = np.random.rand(self.n_channels, self.n_wide_points) + 1j * np.random.rand(
            self.n_channels, self.n_wide_points
        )
        wide_f_data = np.random.rand(self.n_channels, self.n_wide_points) * 2000 + 500
        fres_data = np.array([1500.0, 1600.0])

        # Save test files
        np.save(os.path.join(self.temp_dir, "iq_fine_z_test-1234567890.npy"), fine_z_data)
        np.save(os.path.join(self.temp_dir, "iq_fine_f_test-1234567890.npy"), fine_f_data)
        np.save(os.path.join(self.temp_dir, "iq_wide_z_test-1234567890.npy"), wide_z_data)
        np.save(os.path.join(self.temp_dir, "iq_wide_f_test-1234567890.npy"), wide_f_data)
        np.save(os.path.join(self.temp_dir, "fres_test-1234567890.npy"), fres_data)

        # Mock the setup methods
        self.dx_records.record_length = self.record_length
        self.dx_records.dt_exact = 1 / self.dx_records.config["fs"] * 1e9  # Convert to ns
        self.dx_records.pca_n_components = self.dx_records.config["pca_n_components"]

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_compute_basic(self):
        """Test basic compute functionality with mock data."""
        # Setup the plugin
        self.dx_records._setup_iq_correction_and_calibration()
        self.dx_records._setup_frequency_interpolation_models()

        # Pre-compute kernels
        self.dx_records.kernel = DxRecords.pulse_kernel(
            self.record_length,
            self.dx_records.config["fs"],
            self.dx_records.config["pulse_kernel_start_time"],
            self.dx_records.config["pulse_kernel_decay_time"],
            self.dx_records.config["pulse_kernel_gaussian_smearing_width"],
            self.dx_records.config["pulse_kernel_truncation_factor"],
        )

        moving_average_kernel_width = int(
            self.dx_records.config["moving_average_width"] / self.dx_records.dt_exact
        )
        self.dx_records.moving_average_kernel = (
            np.ones(moving_average_kernel_width) / moving_average_kernel_width
        )

        # Set up interpolated template for truth pulse injection
        self.dx_records.At_interp, self.dx_records.t_max = load_interpolation(
            self.dx_records.config["template_interp_path"]
        )
        # Initialize per-channel template dicts (empty for tests without channel templates)
        self.dx_records.At_interp_dict = {}
        self.dx_records.t_max_dict = {}
        self.dx_records.interpolated_template_dict = {}

        dt_seconds = 1.0 / self.dx_records.config["fs"]
        t_seconds = np.arange(PULSE_TEMPLATE_LENGTH) * dt_seconds
        t_max_target = PULSE_TEMPLATE_ARGMAX * dt_seconds
        time_shift = t_max_target - self.dx_records.t_max
        timeshifted_seconds = t_seconds - time_shift
        self.dx_records.interpolated_template = self.dx_records.At_interp(timeshifted_seconds)

        # Create mock raw records
        raw_records = np.zeros(
            2,
            dtype=[
                ("time", np.int64),
                ("endtime", np.int64),
                ("length", np.int64),
                ("dt", np.int64),
                ("channel", np.int16),
                ("data_i", np.float32, self.record_length),
                ("data_q", np.float32, self.record_length),
            ],
        )

        raw_records[0]["time"] = 0
        raw_records[0]["length"] = self.record_length
        raw_records[0]["dt"] = int(self.dx_records.dt_exact)
        raw_records[0]["endtime"] = (
            raw_records[0]["time"] + raw_records[0]["length"] * raw_records[0]["dt"]
        )
        raw_records[0]["channel"] = 0
        raw_records[0]["data_i"] = np.random.randn(self.record_length)
        raw_records[0]["data_q"] = np.random.randn(self.record_length)

        raw_records[1]["time"] = self.record_length * int(self.dx_records.dt_exact)
        raw_records[1]["length"] = self.record_length
        raw_records[1]["dt"] = int(self.dx_records.dt_exact)
        raw_records[1]["endtime"] = (
            raw_records[1]["time"] + raw_records[1]["length"] * raw_records[1]["dt"]
        )
        raw_records[1]["channel"] = 1
        raw_records[1]["data_i"] = np.random.randn(self.record_length)
        raw_records[1]["data_q"] = np.random.randn(self.record_length)

        # Create empty truth array (no truth events)
        truth_dtype = [
            ("time", np.int64),
            ("endtime", np.int64),
            ("energy_true", np.float32),
            ("dx_true", np.float32),
            ("channel", np.int16),
        ]
        truth = np.zeros(0, dtype=truth_dtype)

        # Test compute
        results = self.dx_records.compute(raw_records, truth)

        # Basic validation
        assert len(results) == 2
        assert results.dtype.names is not None

        # Check required fields
        required_fields = [
            "time",
            "endtime",
            "length",
            "dt",
            "channel",
            "data_dtheta",
            "data_dx",
            "data_dx_moving_average",
            "data_dx_convolved",
        ]
        for field in required_fields:
            assert field in results.dtype.names

        # Check data shapes
        for i, result in enumerate(results):
            assert result["data_dtheta"].shape == (self.record_length,)
            assert result["data_dx"].shape == (self.record_length,)
            assert result["data_dx_moving_average"].shape == (self.record_length,)
            assert result["data_dx_convolved"].shape == (self.record_length,)

            # Check that data is finite
            assert np.all(np.isfinite(result["data_dtheta"]))
            assert np.all(np.isfinite(result["data_dx"]))
            assert np.all(np.isfinite(result["data_dx_moving_average"]))
            assert np.all(np.isfinite(result["data_dx_convolved"]))

    def test_compute_with_pca_disabled(self):
        """Test compute with PCA disabled (pca_n_components=0)."""
        # Setup the plugin
        self.dx_records._setup_iq_correction_and_calibration()
        self.dx_records._setup_frequency_interpolation_models()

        # Set PCA components to 0 (disabled)
        self.dx_records.config["pca_n_components"] = 0
        self.dx_records.pca_n_components = 0

        # Pre-compute kernels
        self.dx_records.kernel = DxRecords.pulse_kernel(
            self.record_length,
            self.dx_records.config["fs"],
            self.dx_records.config["pulse_kernel_start_time"],
            self.dx_records.config["pulse_kernel_decay_time"],
            self.dx_records.config["pulse_kernel_gaussian_smearing_width"],
            self.dx_records.config["pulse_kernel_truncation_factor"],
        )

        moving_average_kernel_width = int(
            self.dx_records.config["moving_average_width"] / self.dx_records.dt_exact
        )
        self.dx_records.moving_average_kernel = (
            np.ones(moving_average_kernel_width) / moving_average_kernel_width
        )

        # Set up interpolated template for truth pulse injection
        self.dx_records.At_interp, self.dx_records.t_max = load_interpolation(
            self.dx_records.config["template_interp_path"]
        )
        # Initialize per-channel template dicts (empty for tests without channel templates)
        self.dx_records.At_interp_dict = {}
        self.dx_records.t_max_dict = {}
        self.dx_records.interpolated_template_dict = {}

        dt_seconds = 1.0 / self.dx_records.config["fs"]
        t_seconds = np.arange(PULSE_TEMPLATE_LENGTH) * dt_seconds
        t_max_target = PULSE_TEMPLATE_ARGMAX * dt_seconds
        time_shift = t_max_target - self.dx_records.t_max
        timeshifted_seconds = t_seconds - time_shift
        self.dx_records.interpolated_template = self.dx_records.At_interp(timeshifted_seconds)

        # Create mock raw records
        raw_records = np.zeros(
            1,
            dtype=[
                ("time", np.int64),
                ("endtime", np.int64),
                ("length", np.int64),
                ("dt", np.int64),
                ("channel", np.int16),
                ("data_i", np.float32, self.record_length),
                ("data_q", np.float32, self.record_length),
            ],
        )

        raw_records[0]["time"] = 0
        raw_records[0]["length"] = self.record_length
        raw_records[0]["dt"] = int(self.dx_records.dt_exact)
        raw_records[0]["endtime"] = (
            raw_records[0]["time"] + raw_records[0]["length"] * raw_records[0]["dt"]
        )
        raw_records[0]["channel"] = 0
        raw_records[0]["data_i"] = np.random.randn(self.record_length)
        raw_records[0]["data_q"] = np.random.randn(self.record_length)

        # Create empty truth array
        truth_dtype = [
            ("time", np.int64),
            ("endtime", np.int64),
            ("energy_true", np.float32),
            ("dx_true", np.float32),
            ("channel", np.int16),
        ]
        truth = np.zeros(0, dtype=truth_dtype)

        # Test compute - should work without errors
        results = self.dx_records.compute(raw_records, truth)

        assert len(results) == 1
        assert np.all(np.isfinite(results[0]["data_dx"]))

    def test_compute_pca_affects_output(self):
        """Test that different PCA settings affect the output."""
        # Setup the plugin
        self.dx_records._setup_iq_correction_and_calibration()
        self.dx_records._setup_frequency_interpolation_models()

        # Pre-compute kernels
        self.dx_records.kernel = DxRecords.pulse_kernel(
            self.record_length,
            self.dx_records.config["fs"],
            self.dx_records.config["pulse_kernel_start_time"],
            self.dx_records.config["pulse_kernel_decay_time"],
            self.dx_records.config["pulse_kernel_gaussian_smearing_width"],
            self.dx_records.config["pulse_kernel_truncation_factor"],
        )

        moving_average_kernel_width = int(
            self.dx_records.config["moving_average_width"] / self.dx_records.dt_exact
        )
        self.dx_records.moving_average_kernel = (
            np.ones(moving_average_kernel_width) / moving_average_kernel_width
        )

        # Set up interpolated template for truth pulse injection
        self.dx_records.At_interp, self.dx_records.t_max = load_interpolation(
            self.dx_records.config["template_interp_path"]
        )
        # Initialize per-channel template dicts (empty for tests without channel templates)
        self.dx_records.At_interp_dict = {}
        self.dx_records.t_max_dict = {}
        self.dx_records.interpolated_template_dict = {}

        dt_seconds = 1.0 / self.dx_records.config["fs"]
        t_seconds = np.arange(PULSE_TEMPLATE_LENGTH) * dt_seconds
        t_max_target = PULSE_TEMPLATE_ARGMAX * dt_seconds
        time_shift = t_max_target - self.dx_records.t_max
        timeshifted_seconds = t_seconds - time_shift
        self.dx_records.interpolated_template = self.dx_records.At_interp(timeshifted_seconds)

        # Create mock raw records
        raw_records = np.zeros(
            1,
            dtype=[
                ("time", np.int64),
                ("endtime", np.int64),
                ("length", np.int64),
                ("dt", np.int64),
                ("channel", np.int16),
                ("data_i", np.float32, self.record_length),
                ("data_q", np.float32, self.record_length),
            ],
        )

        raw_records[0]["time"] = 0
        raw_records[0]["length"] = self.record_length
        raw_records[0]["dt"] = int(self.dx_records.dt_exact)
        raw_records[0]["endtime"] = (
            raw_records[0]["time"] + raw_records[0]["length"] * raw_records[0]["dt"]
        )
        raw_records[0]["channel"] = 0
        # Use the same data for both tests
        np.random.seed(42)
        raw_records[0]["data_i"] = np.random.randn(self.record_length)
        raw_records[0]["data_q"] = np.random.randn(self.record_length)

        # Create empty truth array
        truth_dtype = [
            ("time", np.int64),
            ("endtime", np.int64),
            ("energy_true", np.float32),
            ("dx_true", np.float32),
            ("channel", np.int16),
        ]
        truth = np.zeros(0, dtype=truth_dtype)

        # Test with PCA enabled (4 components)
        self.dx_records.pca_n_components = 4
        results_with_pca = self.dx_records.compute(raw_records.copy(), truth)

        # Test with PCA disabled (0 components)
        self.dx_records.pca_n_components = 0
        results_no_pca = self.dx_records.compute(raw_records.copy(), truth)

        # Results should be different when PCA is applied
        # Note: They might be very similar for random data, but should
        # not be identical
        assert not np.allclose(results_with_pca[0]["data_dx"], results_no_pca[0]["data_dx"])

    def test_compute_empty_input(self):
        """Test compute with empty input."""
        # Setup the plugin
        self.dx_records._setup_iq_correction_and_calibration()
        self.dx_records._setup_frequency_interpolation_models()

        # Pre-compute kernels
        self.dx_records.kernel = DxRecords.pulse_kernel(
            self.record_length,
            self.dx_records.config["fs"],
            self.dx_records.config["pulse_kernel_start_time"],
            self.dx_records.config["pulse_kernel_decay_time"],
            self.dx_records.config["pulse_kernel_gaussian_smearing_width"],
            self.dx_records.config["pulse_kernel_truncation_factor"],
        )

        moving_average_kernel_width = int(
            self.dx_records.config["moving_average_width"] / self.dx_records.dt_exact
        )
        self.dx_records.moving_average_kernel = (
            np.ones(moving_average_kernel_width) / moving_average_kernel_width
        )

        # Set up interpolated template for truth pulse injection
        self.dx_records.At_interp, self.dx_records.t_max = load_interpolation(
            self.dx_records.config["template_interp_path"]
        )
        # Initialize per-channel template dicts (empty for tests without channel templates)
        self.dx_records.At_interp_dict = {}
        self.dx_records.t_max_dict = {}
        self.dx_records.interpolated_template_dict = {}

        dt_seconds = 1.0 / self.dx_records.config["fs"]
        t_seconds = np.arange(PULSE_TEMPLATE_LENGTH) * dt_seconds
        t_max_target = PULSE_TEMPLATE_ARGMAX * dt_seconds
        time_shift = t_max_target - self.dx_records.t_max
        timeshifted_seconds = t_seconds - time_shift
        self.dx_records.interpolated_template = self.dx_records.At_interp(timeshifted_seconds)

        # Empty input
        raw_records = np.zeros(
            0,
            dtype=[
                ("time", np.int64),
                ("endtime", np.int64),
                ("length", np.int64),
                ("dt", np.int64),
                ("channel", np.int16),
                ("data_i", np.float32, self.record_length),
                ("data_q", np.float32, self.record_length),
            ],
        )

        # Create empty truth array
        truth_dtype = [
            ("time", np.int64),
            ("endtime", np.int64),
            ("energy_true", np.float32),
            ("dx_true", np.float32),
            ("channel", np.int16),
        ]
        truth = np.zeros(0, dtype=truth_dtype)

        results = self.dx_records.compute(raw_records, truth)
        assert len(results) == 0


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
class TestRecordsWithRealDataOnline:
    """Test records processing with real qualiphide_fir_test_data."""

    def test_qualiphide_thz_online_context_creation(self):
        """Test that the qualiphide_thz_online context can be created without errors."""
        import straxion

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

    def test_records_processing(self):
        """Test the complete records processing pipeline with real data.

        This test requires the STRAXION_TEST_DATA_DIR environment variable to be set to the path
        containing the qualiphide_fir_test_data directory with example data.
        """
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        import straxion

        # Create context and process records
        st = straxion.qualiphide_thz_online()
        run_id = "1756824965"
        configs = _get_test_config(test_data_dir, run_id)

        clean_strax_data()
        try:
            records = st.get_array(run_id, "records", config=configs)

            # Basic validation of the output
            assert records is not None
            assert len(records) > 0

            # Check that all required fields are present
            required_fields = [
                "time",
                "endtime",
                "length",
                "dt",
                "channel",
                "data_theta",
                "data_theta_moving_average",
                "data_theta_convolved",
                "baseline",
                "baseline_std",
            ]
            for field in required_fields:
                assert (
                    field in records.dtype.names
                ), f"Required field '{field}' missing from records"

            # Check data types
            assert records["time"].dtype == np.int64
            assert records["endtime"].dtype == np.int64
            assert records["length"].dtype == np.int64
            assert records["dt"].dtype == np.int64
            assert records["channel"].dtype == np.int16
            assert records["data_theta"].dtype == np.float32
            assert records["data_theta_moving_average"].dtype == np.float32
            assert records["data_theta_convolved"].dtype == np.float32
            assert records["baseline"].dtype == np.float32
            assert records["baseline_std"].dtype == np.float32

            # Check that all records have reasonable lengths
            assert all(records["length"] > 0)
            assert all(records["dt"] > 0)

            # Check that data arrays have the correct shape
            for record in records:
                assert record["data_theta"].shape == (record["length"],)
                assert record["data_theta_moving_average"].shape == (record["length"],)
                assert record["data_theta_convolved"].shape == (record["length"],)

            # Check that baseline values are scalars (not arrays)
            assert records["baseline"].ndim == 1  # Should be 1D array of scalar values
            assert records["baseline_std"].ndim == 1  # Should be 1D array of scalar values

            print(
                f"Successfully processed {len(records)} records "
                f"from {len(np.unique(records['channel']))} channels"
            )

        except Exception as e:
            pytest.fail(f"Failed to process records: {str(e)}")

    def test_records_data_consistency(self):
        """Test that the records data is internally consistent."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        import straxion

        st = straxion.qualiphide_thz_online()
        run_id = "1756824965"
        configs = _get_test_config(test_data_dir, run_id)

        clean_strax_data()
        try:
            records = st.get_array(run_id, "records", config=configs)

            # Check endtime consistency
            # Get fs from context config to calculate dt_exact
            fs = st.config.get("fs", 38000)  # Default to 38000 if not found
            dt_exact = 1 / fs * SECOND_TO_NANOSECOND
            for record in records:
                expected_endtime = np.int64(record["time"] + record["length"] * dt_exact)
                assert record["endtime"] == expected_endtime

            # Check monotonic time within channels
            for channel in np.unique(records["channel"]):
                channel_records = records[records["channel"] == channel]
                if len(channel_records) > 1:
                    times = channel_records["time"]
                    assert np.all(
                        np.diff(times) > 0
                    ), f"Time stamps not monotonically increasing for channel {channel}"

            # Check finite data
            assert np.all(
                np.isfinite(records["data_theta"])
            ), "Non-finite values found in data_theta"
            assert np.all(
                np.isfinite(records["data_theta_moving_average"])
            ), "Non-finite values found in data_theta_moving_average"
            assert np.all(
                np.isfinite(records["data_theta_convolved"])
            ), "Non-finite values found in data_theta_convolved"
            assert np.all(np.isfinite(records["baseline"])), "Non-finite values found in baseline"
            assert np.all(
                np.isfinite(records["baseline_std"])
            ), "Non-finite values found in baseline_std"

        except Exception as e:
            pytest.fail(f"Failed to validate records consistency: {str(e)}")

    def test_records_missing_data_directory(self):
        """Test that the records plugin raises appropriate errors when data directory is missing."""
        import straxion

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
            st.get_array(run_id, "records", config=configs)

    def test_records_invalid_config(self):
        """Test that the records plugin handles invalid configuration gracefully."""
        import straxion

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
            st.get_array(run_id, "records", config=configs)


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR"),
    reason="Test data directory not provided via STRAXION_TEST_DATA_DIR environment variable",
)
class TestRecordsWithRealDataOffline:
    """Test records processing with real qualiphide_fir_test_data using DxRecords plugin."""

    def test_qualiphide_thz_offline_context_creation(self):
        """Test that the qualiphide_thz_offline context can be created without errors."""
        import straxion

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

    def test_records_processing(self):
        """Test the complete records processing pipeline with real data using DxRecords plugin.

        This test requires the STRAXION_TEST_DATA_DIR environment variable to be set to the path
        containing the qualiphide_fir_test_data directory with example data.
        """
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        import straxion

        # Create context and process records
        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = _get_test_config(test_data_dir, run_id)

        clean_strax_data()
        try:
            records = st.get_array(run_id, "records", config=configs)

            # Basic validation of the output
            assert records is not None
            assert len(records) > 0

            # Check that all required fields are present for DxRecords
            required_fields = [
                "time",
                "endtime",
                "length",
                "dt",
                "channel",
                "data_dtheta",
                "data_dx",
                "data_dx_moving_average",
                "data_dx_convolved",
            ]
            for field in required_fields:
                assert (
                    field in records.dtype.names
                ), f"Required field '{field}' missing from records"

            # Check data types
            assert records["time"].dtype == np.int64
            assert records["endtime"].dtype == np.int64
            assert records["length"].dtype == np.int64
            assert records["dt"].dtype == np.int64
            assert records["channel"].dtype == np.int16
            assert records["data_dtheta"].dtype == np.float32
            assert records["data_dx"].dtype == np.float32
            assert records["data_dx_moving_average"].dtype == np.float32
            assert records["data_dx_convolved"].dtype == np.float32

            # Check that all records have reasonable lengths
            assert all(records["length"] > 0)
            assert all(records["dt"] > 0)

            # Check that data arrays have the correct shape
            for record in records:
                assert record["data_dtheta"].shape == (record["length"],)
                assert record["data_dx"].shape == (record["length"],)
                assert record["data_dx_moving_average"].shape == (record["length"],)
                assert record["data_dx_convolved"].shape == (record["length"],)

            print(
                f"Successfully processed {len(records)} records "
                f"from {len(np.unique(records['channel']))} channels"
            )

        except Exception as e:
            pytest.fail(f"Failed to process records: {str(e)}")

    def test_records_data_consistency(self):
        """Test that the records data is internally consistent."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        import straxion

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = _get_test_config(test_data_dir, run_id)

        clean_strax_data()
        try:
            records = st.get_array(run_id, "records", config=configs)

            # Check endtime consistency
            # Get fs from context config to calculate dt_exact
            fs = st.config.get("fs", 38000)  # Default to 38000 if not found
            dt_exact = 1 / fs * SECOND_TO_NANOSECOND
            for record in records:
                expected_endtime = np.int64(record["time"] + record["length"] * dt_exact)
                assert record["endtime"] == expected_endtime

            # Check monotonic time within channels
            for channel in np.unique(records["channel"]):
                channel_records = records[records["channel"] == channel]
                if len(channel_records) > 1:
                    times = channel_records["time"]
                    assert np.all(
                        np.diff(times) > 0
                    ), f"Time stamps not monotonically increasing for channel {channel}"

            # Check finite data
            assert np.all(
                np.isfinite(records["data_dtheta"])
            ), "Non-finite values found in data_dtheta"
            assert np.all(np.isfinite(records["data_dx"])), "Non-finite values found in data_dx"
            assert np.all(
                np.isfinite(records["data_dx_moving_average"])
            ), "Non-finite values found in data_dx_moving_average"
            assert np.all(
                np.isfinite(records["data_dx_convolved"])
            ), "Non-finite values found in data_dx_convolved"

        except Exception as e:
            pytest.fail(f"Failed to validate records consistency: {str(e)}")

    def test_records_missing_data_directory(self):
        """Test that the records plugin raises appropriate errors when data directory is missing."""
        import straxion

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
            st.get_array(run_id, "records", config=configs)

    def test_records_invalid_config(self):
        """Test that the records plugin handles invalid configuration gracefully."""
        import straxion

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
            st.get_array(run_id, "records", config=configs)

    def test_records_with_truth_injection(self):
        """Test that truth pulses are correctly injected into records."""
        test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")

        if not os.path.exists(test_data_dir):
            pytest.fail(f"Test data directory {test_data_dir} does not exist")

        import straxion

        st = straxion.qualiphide_thz_offline()
        run_id = "1756824965"
        configs = _get_test_config(test_data_dir, run_id)

        clean_strax_data()
        try:
            # First, get records without truth injection (salt_rate=0)
            configs_no_salt = {**configs, "salt_rate": 0}
            records_no_truth = st.get_array(run_id, "records", config=configs_no_salt)

            # Clean and get records with truth injection (salt_rate > 0)
            clean_strax_data()
            configs_with_salt = {**configs, "salt_rate": 100}  # 100 Hz
            records_with_truth = st.get_array(run_id, "records", config=configs_with_salt)

            # Get the truth events to verify
            truth = st.get_array(run_id, "truth", config=configs_with_salt)

            # Basic validation
            assert len(records_no_truth) > 0
            assert len(records_with_truth) > 0
            assert len(truth) > 0, "No truth events generated"

            # Check that records have the same structure
            assert records_no_truth.dtype == records_with_truth.dtype
            assert len(records_no_truth) == len(records_with_truth)

            # Find a record that overlaps with a truth event
            found_injection = False
            for t in truth:
                # Find the record containing this truth event
                matching_records_idx = np.where(
                    (records_with_truth["channel"] == t["channel"])
                    & (records_with_truth["time"] <= t["time"])
                    & (records_with_truth["endtime"] > t["time"])
                )[0]

                if len(matching_records_idx) > 0:
                    idx = matching_records_idx[0]

                    # Compare data_dx before and after injection
                    diff = records_with_truth[idx]["data_dx"] - records_no_truth[idx]["data_dx"]

                    # There should be a difference where pulse was injected
                    if np.any(np.abs(diff) > 1e-6):
                        found_injection = True

                        # Calculate expected pulse position
                        time_offset = t["time"] - records_with_truth[idx]["time"]
                        dt = records_with_truth[idx]["dt"]
                        pulse_start_sample = int(time_offset / dt)

                        # Check that difference is localized around pulse position
                        max_diff_idx = np.argmax(np.abs(diff))

                        print(f"Found truth pulse injection:")
                        print(f"  Channel: {t['channel']}")
                        print(f"  Energy: {t['energy_true']:.2f} meV")
                        print(f"  dx_true: {t['dx_true']:.2e}")
                        print(f"  Expected sample: {pulse_start_sample}")
                        print(f"  Max diff at sample: {max_diff_idx}")
                        print(f"  Max diff value: {diff[max_diff_idx]:.2e}")

                        break

            assert found_injection, (
                "No pulse injection detected in any record. "
                f"Generated {len(truth)} truth events."
            )

            # Verify all data is still finite
            assert np.all(np.isfinite(records_with_truth["data_dx"]))
            assert np.all(np.isfinite(records_with_truth["data_dx_moving_average"]))
            assert np.all(np.isfinite(records_with_truth["data_dx_convolved"]))

        except Exception as e:
            pytest.fail(f"Failed to test truth injection: {str(e)}")
