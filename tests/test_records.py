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

        # Test loading each .npy file
        npy_files = [f for f in os.listdir(test_data_dir) if f.endswith(".npy")]

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
            for record in records:
                expected_endtime = record["time"] + record["length"] * record["dt"]
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
            "iq_finescan_filename": "nonexistent.npy",
            "iq_widescan_dir": "/nonexistent",
            "iq_widescan_filename": "nonexistent.npy",
            "resonant_frequency_dir": "/nonexistent",
            "resonant_frequency_filename": "nonexistent.npy",
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
            "iq_finescan_filename": "nonexistent.npy",
            "iq_widescan_dir": "/nonexistent",
            "iq_widescan_filename": "nonexistent.npy",
            "resonant_frequency_dir": "/nonexistent",
            "resonant_frequency_filename": "nonexistent.npy",
        }

        clean_strax_data()
        with pytest.raises(Exception):
            st.get_array(run_id, "records", config=configs)
