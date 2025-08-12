import numpy as np
import pytest
import os
import straxion


class TestBaselineMonitor:
    """Test the BaselineMonitor plugin."""

    def test_baseline_monitor_processing(self):
        """Test the complete baseline monitor processing pipeline with real data.

        This test requires the STRAXION_FINESCAN_DATA_DIR environment variable to be set to the path
        containing the finescan test data directory.

        """
        test_data_dir = os.getenv("STRAXION_FINESCAN_DATA_DIR")
        if not test_data_dir:
            pytest.fail("STRAXION_FINESCAN_DATA_DIR environment variable is not set")
        if not os.path.exists(test_data_dir):
            pytest.fail(f"Finescan test data directory {test_data_dir} does not exist")

        # Create context and process baseline monitor data
        st = straxion.qualiphide()

        config = {
            "daq_input_dir": "timeS429",
            "iq_finescan_dir": test_data_dir,
            "record_length": 5_000_000,
            "fs": 50_000,
        }

        try:
            # Get baseline monitor data
            bm = st.get_array("timeS429", "baseline_monitor", config=config)

            # Basic validation of the output
            assert bm is not None
            assert len(bm) > 0

            # Check that all required fields are present
            required_fields = [
                "time",
                "endtime",
                "length",
                "dt",
                "channel",
                "baseline_monitor_interval",
                "baseline_monitor_std",
                "baseline_monitor_std_moving_average",
                "baseline_monitor_std_convolved",
            ]
            for field in required_fields:
                assert (
                    field in bm.dtype.names
                ), f"Required field '{field}' missing from baseline monitor"

            # Check data types
            assert bm["time"].dtype == np.int64
            assert bm["endtime"].dtype == np.int64
            assert bm["length"].dtype == np.int64
            assert bm["dt"].dtype == np.int64
            assert bm["channel"].dtype == np.int16
            assert bm["baseline_monitor_interval"].dtype == np.int64
            assert bm["baseline_monitor_std"].dtype == np.float32
            assert bm["baseline_monitor_std_moving_average"].dtype == np.float32
            assert bm["baseline_monitor_std_convolved"].dtype == np.float32

            # Check that all records have the expected length
            expected_length = config["record_length"]
            assert all(bm["length"] == expected_length)

            # Check that all records have the expected dt
            expected_dt = int(1 / config["fs"] * 1_000_000_000)  # Convert to nanoseconds
            assert all(bm["dt"] == expected_dt)

            # Check that channels are within expected range (0-9 based on context config)
            assert all(0 <= bm["channel"]) and all(bm["channel"] <= 9)

            # Check that baseline monitor arrays have the correct shape
            # Should have 100 intervals based on N_BASELINE_MONITOR_INTERVAL
            expected_intervals = 100
            for record in bm:
                assert record["baseline_monitor_std"].shape == (expected_intervals,)
                assert record["baseline_monitor_std_moving_average"].shape == (expected_intervals,)
                assert record["baseline_monitor_std_convolved"].shape == (expected_intervals,)

            # Check that baseline monitor interval is consistent across all records
            expected_interval = (
                expected_length // expected_intervals // config["fs"] * 1_000_000_000
            )
            assert all(bm["baseline_monitor_interval"] == expected_interval)

            # Check that baseline std values are reasonable (should be positive)
            assert np.all(bm["baseline_monitor_std"] >= 0)
            assert np.all(bm["baseline_monitor_std_moving_average"] >= 0)
            assert np.all(bm["baseline_monitor_std_convolved"] >= 0)

            # Check that baseline std values are not all identical (should have some variation)
            for channel in np.unique(bm["channel"]):
                channel_data = bm[bm["channel"] == channel]
                if len(channel_data) > 1:
                    # Check that std values vary across intervals
                    for record in channel_data:
                        std_values = record["baseline_monitor_std"]
                        std_ma_values = record["baseline_monitor_std_moving_average"]
                        std_conv_values = record["baseline_monitor_std_convolved"]

                        # Should not be all identical
                        assert not np.allclose(
                            std_values, std_values[0]
                        ), f"Channel {channel} std values are constant"
                        assert not np.allclose(
                            std_ma_values, std_ma_values[0]
                        ), f"Channel {channel} moving average std values are constant"
                        assert not np.allclose(
                            std_conv_values, std_conv_values[0]
                        ), f"Channel {channel} convolved std values are constant"

            print(
                f"Successfully processed {len(bm)} baseline monitor records "
                f"from {len(np.unique(bm['channel']))} channels"
            )

        except Exception as e:
            pytest.fail(f"Failed to process baseline monitor data: {str(e)}")
