import os
import pytest
import numpy as np
import straxion
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


def test_hits_invalid_config():
    """Test that Hits raises an error with invalid config."""
    st = straxion.qualiphide()
    # Set an invalid config (e.g., negative record_length)
    st.set_config({"record_length": -1})
    with pytest.raises(Exception):
        st.get_single_plugin("timeS429", "hits")


def test_find_hit_candidates_with_simulated_pulse():
    """Test _find_hit_candidates directly with a simulated noisy exponential pulse."""
    st = straxion.qualiphide()
    plugin = st.get_single_plugin("timeS429", "hits")
    n_samples = 2000
    pulse_start = 1000
    pulse_length = 200
    tau = 50

    # Set random seed for reproducible results
    np.random.seed(42)

    # Generate truncated exponential pulse
    t = np.arange(pulse_length)
    pulse_shape = np.exp(-t / tau)
    pulse = np.zeros(n_samples)
    pulse[pulse_start : pulse_start + pulse_length] = pulse_shape

    # Add Gaussian noise
    noise_sigma = 0.05
    noisy_signal = pulse + np.random.normal(0, noise_sigma, n_samples)

    # Set threshold and min_pulse_width
    hit_threshold = 3 * np.std(noisy_signal)
    min_pulse_width = 20

    hit_start_indices, hit_widths = plugin._find_hit_candidates(
        noisy_signal, hit_threshold, min_pulse_width
    )
    assert len(hit_start_indices) > 0, "No hit candidates found, but expected at least one."
    assert len(hit_start_indices) == len(
        hit_widths
    ), "Mismatch between hit start indices and widths."


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR") or not os.getenv("STRAXION_FINESCAN_DATA_DIR"),
    reason=(
        "Finescan test data directory not provided via "
        "STRAXION_FINESCAN_DATA_DIR environment variable"
    ),
)
def test_hits_processing():
    """Test the complete hits processing pipeline with real data.

    This test requires both STRAXION_TEST_DATA_DIR and STRAXION_FINESCAN_DATA_DIR environment
    variables to be set to the paths containing the test data directories.

    """
    test_data_dir = os.getenv("STRAXION_TEST_DATA_DIR")
    finescan_data_dir = os.getenv("STRAXION_FINESCAN_DATA_DIR")

    if not test_data_dir:
        pytest.fail("STRAXION_TEST_DATA_DIR environment variable is not set")
    if not finescan_data_dir:
        pytest.fail("STRAXION_FINESCAN_DATA_DIR environment variable is not set")

    if not os.path.exists(test_data_dir):
        pytest.fail(f"Test data directory {test_data_dir} does not exist")
    if not os.path.exists(finescan_data_dir):
        pytest.fail(f"Finescan data directory {finescan_data_dir} does not exist")

    # Create context and process hits
    st = straxion.qualiphide()
    st.set_config(
        dict(
            daq_input_dir=test_data_dir,
            iq_finescan_dir=finescan_data_dir,
            record_length=5_000_000,
            fs=500_000,
        )
    )

    try:
        hits = st.get_array("timeS429", "hits")

        # Basic validation of the output
        assert hits is not None
        assert len(hits) >= 0  # Can be empty if no hits found

        # Check that all required fields are present
        required_fields = [
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
        for field in required_fields:
            assert field in hits.dtype.names, f"Required field '{field}' missing from hits"

        # Check data types
        assert hits["time"].dtype == np.int64
        assert hits["endtime"].dtype == np.int64
        assert hits["length"].dtype == np.int64
        assert hits["dt"].dtype == np.int64
        assert hits["channel"].dtype == np.int16
        assert hits["width"].dtype == np.int32
        assert hits["data_theta"].dtype == np.float32
        assert hits["data_theta_moving_average"].dtype == np.float32
        assert hits["data_theta_convolved"].dtype == np.float32
        assert hits["hit_threshold"].dtype == np.float32
        assert hits["aligned_at_records_i"].dtype == np.int32
        assert hits["amplitude_max"].dtype == np.float32
        assert hits["amplitude_min"].dtype == np.float32
        assert hits["amplitude_max_ext"].dtype == np.float32
        assert hits["amplitude_min_ext"].dtype == np.float32

        # Check that all hits have the expected dt
        expected_dt = int(1 / 500_000 * 1_000_000_000)  # Convert to nanoseconds
        assert all(hits["dt"] == expected_dt)

        # Check that channels are within expected range (0-9 based on context config)
        if len(hits) > 0:
            assert all(0 <= hits["channel"]) and all(hits["channel"] <= 9)

        # Check that waveform data has the correct shape
        expected_waveform_length = 600  # HIT_WINDOW_LENGTH_LEFT + HIT_WINDOW_LENGTH_RIGHT
        for hit in hits:
            assert hit["data_theta"].shape == (expected_waveform_length,)
            assert hit["data_theta_moving_average"].shape == (expected_waveform_length,)
            assert hit["data_theta_convolved"].shape == (expected_waveform_length,)

        # Check that timing information is consistent
        for h_i, hit in enumerate(hits):
            expected_endtime = hit["time"] + hit["length"] * hit["dt"]
            assert hit["endtime"] == expected_endtime, f"Hit #{h_i} endtime mismatch."

        # Check that hit characteristics are reasonable
        for hit in hits:
            assert hit["amplitude_max"] >= hit["amplitude_min"]
            assert hit["amplitude_max_ext"] >= hit["amplitude_min_ext"]
            assert hit["width"] > 0
            assert hit["hit_threshold"] > 0

        print(
            f"Successfully processed {len(hits)} hits "
            f"from {len(np.unique(hits['channel'])) if len(hits) > 0 else 0} channels"
        )

    except Exception as e:
        pytest.fail(f"Failed to process hits: {str(e)}")
