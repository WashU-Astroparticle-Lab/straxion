import numpy as np
import pytest
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
