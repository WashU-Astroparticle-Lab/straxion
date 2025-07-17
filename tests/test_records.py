import numpy as np
import pytest
import os
import tempfile
import shutil
from straxion.plugins.records import PulseProcessing


class TestCircfit:
    """Test the circfit static method of PulseProcessing class."""

    def setup_method(self):
        """Set up random seed for deterministic tests."""
        np.random.seed(42)

    def test_circfit_perfect_circle(self):
        """Test circfit with a perfect circle dataset.

        Generate points on a perfect circle with known center and radius, then verify that circfit
        can accurately reproduce these parameters.

        """
        # Define known circle parameters
        center_x = 2.5
        center_y = -1.3
        radius = 3.7

        # Generate points on a perfect circle
        n_points = 100
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        # Generate x, y coordinates on the circle
        x = center_x + radius * np.cos(angles)
        y = center_y + radius * np.sin(angles)

        # Add small amount of noise to make it more realistic
        noise_level = 1e-6
        x += np.random.normal(0, noise_level, n_points)
        y += np.random.normal(0, noise_level, n_points)

        # Fit circle using circfit
        fitted_center_x, fitted_center_y, fitted_radius, rms_error = PulseProcessing.circfit(x, y)

        # Check that fitted parameters are close to true values
        np.testing.assert_allclose(fitted_center_x, center_x, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(fitted_center_y, center_y, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(fitted_radius, radius, rtol=1e-6, atol=1e-6)

        # Check that RMS error is very small (should be close to noise level)
        assert rms_error < noise_level * 10, f"RMS error {rms_error} is too large"

        # Verify that the fitted circle actually fits the data well
        # Calculate distances from points to fitted circle center
        distances = np.sqrt((x - fitted_center_x) ** 2 + (y - fitted_center_y) ** 2)
        # Check that distances are close to fitted radius
        np.testing.assert_allclose(distances, fitted_radius, rtol=1e-5, atol=1e-5)

    def test_circfit_origin_centered_circle(self):
        """Test circfit with a circle centered at origin."""
        # Circle centered at origin
        center_x = 0.0
        center_y = 0.0
        radius = 5.0

        # Generate points on the circle
        n_points = 50
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x = center_x + radius * np.cos(angles)
        y = center_y + radius * np.sin(angles)

        # Fit circle
        fitted_center_x, fitted_center_y, fitted_radius, rms_error = PulseProcessing.circfit(x, y)

        # Check results
        np.testing.assert_allclose(fitted_center_x, center_x, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(fitted_center_y, center_y, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(fitted_radius, radius, rtol=1e-10, atol=1e-10)
        assert rms_error < 1e-10

    def test_circfit_insufficient_points(self):
        """Test that circfit raises ValueError with insufficient points."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="At least three points are required"):
            PulseProcessing.circfit(x, y)

    def test_circfit_mismatched_lengths(self):
        """Test that circfit raises ValueError with mismatched array lengths."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="x and y must be the same length"):
            PulseProcessing.circfit(x, y)

    def test_circfit_collinear_points(self):
        """Test that circfit raises ValueError with collinear points."""
        # Create collinear points (all on a straight line)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="Points are collinear or nearly collinear"):
            PulseProcessing.circfit(x, y)

    def test_circfit_large_circle(self):
        """Test circfit with a large radius circle."""
        center_x = 1000.0
        center_y = -500.0
        radius = 10000.0

        # Generate points on the circle
        n_points = 200
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x = center_x + radius * np.cos(angles)
        y = center_y + radius * np.sin(angles)

        # Add small noise
        noise_level = 1e-3
        x += np.random.normal(0, noise_level, n_points)
        y += np.random.normal(0, noise_level, n_points)

        # Fit circle
        fitted_center_x, fitted_center_y, fitted_radius, rms_error = PulseProcessing.circfit(x, y)

        # Check results with appropriate tolerance for large numbers
        np.testing.assert_allclose(fitted_center_x, center_x, rtol=1e-6, atol=1e-3)
        np.testing.assert_allclose(fitted_center_y, center_y, rtol=1e-6, atol=1e-3)
        np.testing.assert_allclose(fitted_radius, radius, rtol=1e-6, atol=1e-3)
        assert rms_error < noise_level * 10

    def test_circfit_quarter_circle(self):
        """Test circfit with points from only a quarter of a circle."""
        center_x = 1.0
        center_y = 2.0
        radius = 3.0

        # Generate points from only a quarter circle (0 to pi/2)
        n_points = 25
        angles = np.linspace(0, np.pi / 2, n_points)
        x = center_x + radius * np.cos(angles)
        y = center_y + radius * np.sin(angles)

        # Fit circle
        fitted_center_x, fitted_center_y, fitted_radius, rms_error = PulseProcessing.circfit(x, y)

        # Check results
        np.testing.assert_allclose(fitted_center_x, center_x, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(fitted_center_y, center_y, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(fitted_radius, radius, rtol=1e-6, atol=1e-6)
        assert rms_error < 1e-6


class TestLoadFinescanFiles:
    """Test the load_finescan_files static method of PulseProcessing class."""

    def setup_method(self):
        """Set up test data directory and files."""
        self.test_dir = tempfile.mkdtemp()
        self.create_test_files()

    def teardown_method(self):
        """Clean up test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_test_files(self):
        """Create test finescan files with various formats and content."""
        # Create valid finescan files
        test_data = {
            "finescan-kid-2025042808-ch0.txt": np.array(
                [
                    [0, 1.0, 2.0],
                    [1, 1.1, 2.1],
                    [2, 1.2, 2.2],
                ]
            ),
            "finescan-kid-2025042808-ch1.csv": np.array(
                [
                    [0, 3.0, 4.0],
                    [1, 3.1, 4.1],
                    [2, 3.2, 4.2],
                ]
            ),
            "finescan-kid-2025042808-ch2.txt": np.array(
                [
                    [0, 5.0, 6.0],
                    [1, 5.1, 6.1],
                ]
            ),
        }

        for filename, data in test_data.items():
            filepath = os.path.join(self.test_dir, filename)
            if filename.endswith(".csv"):
                np.savetxt(filepath, data, delimiter=",", fmt="%.1f")
            else:
                np.savetxt(filepath, data, fmt="%.1f")

    def test_load_finescan_files_success(self):
        """Test successful loading of finescan files."""
        result = PulseProcessing.load_finescan_files(self.test_dir)

        # Check that all expected channels are present
        assert 0 in result
        assert 1 in result
        assert 2 in result

        # Check data structure
        for channel in [0, 1, 2]:
            assert isinstance(result[channel], np.ndarray)
            assert result[channel].ndim == 2
            assert result[channel].shape[1] >= 3

        # Check specific data values
        np.testing.assert_array_equal(result[0][0], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(result[1][0], [0.0, 3.0, 4.0])
        np.testing.assert_array_equal(result[2][0], [0.0, 5.0, 6.0])

    def test_load_finescan_files_nonexistent_directory(self):
        """Test that FileNotFoundError is raised for nonexistent directory."""
        with pytest.raises(FileNotFoundError, match="Fine scan directory not found"):
            PulseProcessing.load_finescan_files("/nonexistent/path")

    def test_load_finescan_files_empty_directory(self):
        """Test that FileNotFoundError is raised for directory with no matching files."""
        empty_dir = tempfile.mkdtemp()
        try:
            with pytest.raises(FileNotFoundError, match="No fine scan files found"):
                PulseProcessing.load_finescan_files(empty_dir)
        finally:
            shutil.rmtree(empty_dir)

    def test_load_finescan_files_invalid_filename(self):
        """Test that files with invalid names are ignored."""
        # Create a file with invalid name
        invalid_file = os.path.join(self.test_dir, "invalid-filename.txt")
        np.savetxt(invalid_file, np.array([[0, 1, 2]]))

        result = PulseProcessing.load_finescan_files(self.test_dir)

        # Should still load the valid files
        assert 0 in result
        assert 1 in result
        assert 2 in result
        # Invalid file should be ignored
        assert len(result) == 3

    def test_load_finescan_files_insufficient_columns(self):
        """Test that ValueError is raised for files with insufficient columns."""
        # Create a file with only 2 columns
        invalid_file = os.path.join(self.test_dir, "finescan-kid-2025042808-ch3.txt")
        np.savetxt(invalid_file, np.array([[0, 1]]))  # Only 2 columns

        with pytest.raises(ValueError, match="does not have at least 3 columns"):
            PulseProcessing.load_finescan_files(self.test_dir)

    def test_load_finescan_files_single_row(self):
        """Test loading files with single row data."""
        # Create a file with single row
        single_row_file = os.path.join(self.test_dir, "finescan-kid-2025042808-ch4.txt")
        np.savetxt(single_row_file, np.array([0, 1, 2]))  # Single row

        result = PulseProcessing.load_finescan_files(self.test_dir)

        # Should handle single row correctly
        assert 4 in result
        assert result[4].shape == (1, 3)
        np.testing.assert_array_equal(result[4][0], [0.0, 1.0, 2.0])

    def test_load_finescan_files_mixed_formats(self):
        """Test loading files with mixed .txt and .csv formats."""
        result = PulseProcessing.load_finescan_files(self.test_dir)

        # Should load both .txt and .csv files
        assert 0 in result  # .txt file
        assert 1 in result  # .csv file
        assert 2 in result  # .txt file

    def test_load_finescan_files_channel_number_extraction(self):
        """Test that channel numbers are correctly extracted from filenames."""
        result = PulseProcessing.load_finescan_files(self.test_dir)

        # Check that channel numbers are integers
        for channel in result.keys():
            assert isinstance(channel, int)
            assert channel >= 0

    def test_load_finescan_files_data_integrity(self):
        """Test that loaded data maintains integrity."""
        result = PulseProcessing.load_finescan_files(self.test_dir)

        for _, data in result.items():
            # Check data type
            assert data.dtype == np.float64

            # Check that all values are finite
            assert np.all(np.isfinite(data))

            # Check that data has expected shape
            assert data.shape[1] >= 3


class TestPulseKernelEMG:
    """Test the pulse_kernel_emg static method of PulseProcessing class."""

    def test_pulse_kernel_emg_basic(self):
        """Test basic pulse kernel generation with typical parameters."""
        ns = 10000
        fs = 100000  # 100 kHz
        t0 = 100000  # 100 us
        tau = 300000  # 300 us
        sigma = 700000  # 700 us
        truncation_factor = 5

        kernel = PulseProcessing.pulse_kernel_emg(ns, fs, t0, tau, sigma, truncation_factor)

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

    def test_pulse_kernel_emg_parameters(self):
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
            kernel = PulseProcessing.pulse_kernel_emg(ns, fs, t0, tau, sigma, truncation_factor)

            # Basic validation
            assert isinstance(kernel, np.ndarray)
            assert kernel.ndim == 1
            assert len(kernel) > 0
            np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=1e-10)
            assert np.all(kernel >= 0)
            assert np.all(np.isfinite(kernel))

    def test_pulse_kernel_emg_edge_cases(self):
        """Test pulse kernel with edge case parameters."""
        ns = 1000
        fs = 100000

        # Very small parameters
        kernel = PulseProcessing.pulse_kernel_emg(ns, fs, 1000, 5000, 10000, 2)
        assert isinstance(kernel, np.ndarray)
        assert len(kernel) > 0
        np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=1e-10)

        # Very large parameters
        kernel = PulseProcessing.pulse_kernel_emg(ns, fs, 500000, 1000000, 2000000, 10)
        assert isinstance(kernel, np.ndarray)
        assert len(kernel) > 0
        np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=1e-10)

    def test_pulse_kernel_emg_truncation(self):
        """Test that truncation factor affects kernel length appropriately."""
        ns = 10000
        fs = 100000
        t0 = 100000
        tau = 300000
        sigma = 700000

        # Test different truncation factors
        kernels = []
        for truncation_factor in [2, 5, 10]:
            kernel = PulseProcessing.pulse_kernel_emg(ns, fs, t0, tau, sigma, truncation_factor)
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

        # Expected angles: 0, pi, pi/2, -pi/2 (wrapped to 3pi/2)
        expected = np.array([0.0, np.pi, np.pi / 2, 3 * np.pi / 2])
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


@pytest.mark.skipif(
    not os.getenv("STRAXION_FINESCAN_DATA_DIR"),
    reason=(
        "Finescan test data directory not provided via "
        "STRAXION_FINESCAN_DATA_DIR environment variable"
    ),
)
class TestLoadFinescanFilesWithRealData:
    """Test load_finescan_files with real test data."""

    def test_finescan_data_dir_exists_and_not_empty(self):
        """Test that STRAXION_FINESCAN_DATA_DIR is set, exists, and is not empty."""
        test_data_dir = os.getenv("STRAXION_FINESCAN_DATA_DIR")
        if test_data_dir:
            assert os.path.exists(test_data_dir) and os.path.isdir(
                test_data_dir
            ), f"STRAXION_FINESCAN_DATA_DIR '{test_data_dir}' does not exist or is not a directory."
            contents = os.listdir(test_data_dir)
            print(f"Contents of STRAXION_FINESCAN_DATA_DIR ({test_data_dir}): {contents}")
            assert len(contents) > 0, f"STRAXION_FINESCAN_DATA_DIR '{test_data_dir}' is empty."
        else:
            pytest.fail("STRAXION_FINESCAN_DATA_DIR is not set.")

    def test_load_real_finescan_files(self):
        """Test loading real finescan files from test data."""
        test_data_dir = os.getenv("STRAXION_FINESCAN_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_FINESCAN_DATA_DIR environment variable is not set")

        result = PulseProcessing.load_finescan_files(test_data_dir)

        # Basic validation
        assert isinstance(result, dict)
        assert len(result) > 0

        # Check each channel's data
        for channel, data in result.items():
            assert isinstance(channel, int)
            assert isinstance(data, np.ndarray)
            assert data.ndim == 2
            assert data.shape[1] >= 3  # index, data_i, data_q

            # Check data integrity
            assert np.all(np.isfinite(data))
            assert data.dtype == np.float64

        print(f"Successfully loaded finescan data for {len(result)} channels")

    def test_finescan_data_consistency(self):
        """Test that finescan data is internally consistent."""
        test_data_dir = os.getenv("STRAXION_FINESCAN_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_FINESCAN_DATA_DIR environment variable is not set")

        result = PulseProcessing.load_finescan_files(test_data_dir)

        for channel, data in result.items():
            # Check that index column is sequential
            indices = data[:, 0]
            expected_indices = np.arange(len(indices))
            np.testing.assert_array_equal(indices, expected_indices)

            # Check that data_i and data_q are reasonable values
            data_i = data[:, 1]
            data_q = data[:, 2]

            # Should have some variation (not all identical)
            assert np.std(data_i) > 0, f"Channel {channel} data_i has no variation"
            assert np.std(data_q) > 0, f"Channel {channel} data_q has no variation"

            # Should be finite
            assert np.all(np.isfinite(data_i))
            assert np.all(np.isfinite(data_q))
