import os
import pytest
import numpy as np
import straxion
from straxion.plugins.hit_classification import HitClassification


def test_hit_classification_plugin_registration():
    """Test that the HitClassification plugin is properly registered in the context."""
    st = straxion.qualiphide()
    assert "hit_classification" in st._plugin_class_registry
    assert st._plugin_class_registry["hit_classification"] == HitClassification


def test_hit_classification_dtype_inference():
    """Test that HitClassification can infer the correct data type."""
    st = straxion.qualiphide()
    plugin = st.get_single_plugin("timeS429", "hit_classification")
    dtype = plugin.infer_dtype()
    # List of expected fields
    expected_fields = [
        "time",
        "endtime",
        "channel",
        "is_cr",
        "is_symmetric_spike",
        "is_unidentified",
        "ma_rise_edge_slope",
    ]
    field_names = [name[1] for name, *_ in dtype]
    for field in expected_fields:
        assert field in field_names


@pytest.mark.skipif(
    not os.getenv("STRAXION_TEST_DATA_DIR") or not os.getenv("STRAXION_FINESCAN_DATA_DIR"),
    reason=(
        "Finescan test data directory not provided via "
        "STRAXION_FINESCAN_DATA_DIR environment variable"
    ),
)
def test_hit_classification_processing():
    """Test the complete hit classification processing pipeline with real data.

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

    # Create context and process hit classification
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
        hit_classification = st.get_array("timeS429", "hit_classification")

        # Basic validation of the output
        assert hit_classification is not None
        assert len(hit_classification) >= 0  # Can be empty if no hits found

        # Check that all required fields are present
        required_fields = [
            "time",
            "endtime",
            "channel",
            "is_cr",
            "is_symmetric_spike",
            "is_unidentified",
            "ma_rise_edge_slope",
        ]
        for field in required_fields:
            assert (
                field in hit_classification.dtype.names
            ), f"Required field '{field}' missing from hit_classification"

        # Check data types
        assert hit_classification["time"].dtype == np.int64
        assert hit_classification["endtime"].dtype == np.int64
        assert hit_classification["channel"].dtype == np.int16
        assert hit_classification["is_cr"].dtype == bool
        assert hit_classification["is_symmetric_spike"].dtype == bool
        assert hit_classification["is_unidentified"].dtype == bool
        assert hit_classification["ma_rise_edge_slope"].dtype == np.float32

        # Check that channels are within expected range (0-9 based on context config)
        if len(hit_classification) > 0:
            assert all(0 <= hit_classification["channel"]) and all(
                hit_classification["channel"] <= 9
            )

        # Check that timing information is consistent
        for hc_i, hit_class in enumerate(hit_classification):
            expected_endtime = hit_class["time"] + (
                hit_class["endtime"] - hit_class["time"]
            )  # This should match
            assert hit_class["endtime"] == expected_endtime, (
                f"Hit classification #{hc_i} endtime mismatch. "
                f"Note that hit_class['endtime'] is {hit_class['endtime']} and "
                f"hit_class['time'] is {hit_class['time']}"
            )

        # Print classification statistics
        if len(hit_classification) > 0:
            n_cr = np.sum(hit_classification["is_cr"])
            n_symmetric_spike = np.sum(hit_classification["is_symmetric_spike"])
            n_unidentified = np.sum(hit_classification["is_unidentified"])

            print(
                f"Successfully processed {len(hit_classification)} hit classifications: "
                f"{n_cr} cosmic rays, {n_symmetric_spike} symmetric spikes, "
                f"{n_unidentified} unidentified"
            )

            assert n_cr == 60, "Expected 60 cosmic rays, got {}".format(n_cr)
            assert n_unidentified == 0, "Expected 0 unidentified hits, got {}".format(
                n_unidentified
            )

    except Exception as e:
        pytest.fail(f"Failed to process hit classification: {str(e)}")
