import pytest
import numpy as np
import straxion
import scipy.signal
from straxion.plugins.hits import Hits


def test_hits_plugin_registration():
    """Test that the Hits plugin is properly registered in the context."""
    st = straxion.qualiphide()
    assert "hits" in st._plugin_class_registry
    assert st._plugin_class_registry["hits"] == Hits


def test_hits_dtype_inference():
    """Test that Hits can infer the correct data type."""
    st = straxion.qualiphide()
    plugin = st.get_single_plugin("timeS429", "hits")
    dtype = plugin.infer_dtype()
    # List of expected fields (add/remove as needed)
    expected_fields = [
        "time",
        "endtime",
        "length",
        "dt",
        "channel",
        "width",
        "data_theta",
        "data_theta_moving_average",
        "data_theta_convolved",
        "hit_threshold",
        "aligned_at_records_i",
        "amplitude_max",
        "amplitude_min",
        "amplitude_max_ext",
        "amplitude_min_ext",
    ]
    field_names = [name[1] for name, *_ in dtype]
    for field in expected_fields:
        assert field in field_names


def test_hits_empty_input():
    """Test that Hits returns empty output for empty input."""
    st = straxion.qualiphide()
    plugin = st.get_single_plugin("timeS429", "hits")
    empty_records = np.array(
        [],
        dtype=[
            ("channel", np.int16),
            ("data_theta_convolved", np.float32, 10),
            ("data_theta_moving_average", np.float32, 10),
            ("data_theta", np.float32, 10),
        ],
    )
    result = plugin.compute(empty_records)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_hits_minimal_valid_input():
    """Test Hits with a minimal valid record that should produce no hits."""
    st = straxion.qualiphide()
    plugin = st.get_single_plugin("timeS429", "hits")
    # Minimal record with all zeros (should not trigger any hits)
    record = np.zeros(
        1,
        dtype=[
            ("channel", np.int16),
            ("data_theta_convolved", np.float32, 10),
            ("data_theta_moving_average", np.float32, 10),
            ("data_theta", np.float32, 10),
            ("time", np.int64),
        ],
    )
    result = plugin.compute(record)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_hits_malformed_input():
    """Test that Hits handles malformed input gracefully."""
    st = straxion.qualiphide()
    plugin = st.get_single_plugin("timeS429", "hits")
    # Missing required fields
    bad_record = np.zeros(1, dtype=[("channel", np.int16)])
    with pytest.raises(Exception):
        plugin.compute(bad_record)


def test_hits_detects_synthetic_hit():
    """Test that Hits detects a synthetic hit in the input using a noisy truncated exponential
    pulse."""
    st = straxion.qualiphide()
    plugin = st.get_single_plugin("timeS429", "hits")
    n_samples = 1000
    pulse_start = 400
    pulse_length = 100
    tau = 20  # decay constant

    # Generate truncated exponential pulse
    t = np.arange(pulse_length)
    pulse_shape = np.exp(-t / tau)
    pulse = np.zeros(n_samples)
    pulse[pulse_start : pulse_start + pulse_length] = pulse_shape

    # Add Gaussian noise
    noise_sigma = 0.05
    noisy_signal = pulse + np.random.normal(0, noise_sigma, n_samples)

    # Use the same noisy signal for all three fields
    record = np.zeros(
        1,
        dtype=[
            ("channel", np.int16),
            ("data_theta_convolved", np.float32, n_samples),
            ("data_theta_moving_average", np.float32, n_samples),
            ("data_theta", np.float32, n_samples),
            ("time", np.int64),
        ],
    )
    record["data_theta_convolved"][0] = noisy_signal

    result = plugin.compute(record)
    assert isinstance(result, np.ndarray)
    # Should find at least one hit
    assert result.size >= 1
    # Check required fields
    for field in ["time", "endtime", "length", "channel"]:
        assert field in result.dtype.names


def test_hits_invalid_config():
    """Test that Hits raises an error with invalid config."""
    st = straxion.qualiphide()
    # Set an invalid config (e.g., negative record_length)
    st.set_config({"record_length": -1})
    with pytest.raises(Exception):
        st.get_single_plugin("timeS429", "hits")
