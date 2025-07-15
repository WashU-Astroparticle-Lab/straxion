#!/usr/bin/env python3
"""Example script to test the DAQReader plugin with real data.

Usage:
    python run_test_example.py /path/to/timeS429/directory

This script demonstrates the same functionality as the test_raw_records_processing test
but can be run independently for debugging or validation.

"""

import sys
import os
import numpy as np
import straxion


def test_raw_records_processing(data_dir):
    """Test the raw_records processing with the provided data directory."""

    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist")
        return False

    print(f"Testing with data directory: {data_dir}")

    # Create context and process raw_records
    st = straxion.qualiphide()

    config = {"daq_input_dir": data_dir, "record_length": 5_000_000, "fs": 500_000}

    try:
        print("Processing raw_records...")
        rr = st.get_array("timeS429", "raw_records", config=config)

        # Basic validation of the output
        print(f"Successfully processed {len(rr)} records")
        print(f"Number of unique channels: {len(np.unique(rr['channel']))}")
        print(f"Channels found: {sorted(np.unique(rr['channel']))}")

        # Check that all required fields are present
        required_fields = ["time", "endtime", "length", "dt", "channel", "data_i", "data_q"]
        for field in required_fields:
            assert field in rr.dtype.names, f"Missing field: {field}"

        print("✓ All required fields present")

        # Check data types
        assert rr["time"].dtype == np.int64
        assert rr["channel"].dtype == np.int16
        assert rr["data_i"].dtype == np.dtype(">f8")
        assert rr["data_q"].dtype == np.dtype(">f8")
        print("✓ Data types are correct")

        # Check that all records have the expected length
        expected_length = config["record_length"]
        assert all(rr["length"] == expected_length)
        print(f"✓ All records have expected length: {expected_length}")

        # Check that all records have the expected dt
        expected_dt = int(1 / config["fs"] * 1_000_000_000)  # Convert to nanoseconds
        assert all(rr["dt"] == expected_dt)
        print(f"✓ All records have expected dt: {expected_dt} ns")

        # Check that channels are within expected range (0-9 based on context config)
        assert all(0 <= rr["channel"]) and all(rr["channel"] <= 9)
        print("✓ All channels are within expected range [0, 9]")

        # Check that data arrays have the correct shape
        for record in rr:
            assert record["data_i"].shape == (expected_length,)
            assert record["data_q"].shape == (expected_length,)
        print("✓ All data arrays have correct shape")

        # Show some statistics
        print("\nData Statistics:")
        print(f"  Time range: {rr['time'].min()} to {rr['time'].max()} ns")
        print(f"  I data range: {rr['data_i'].min():.2e} to {rr['data_i'].max():.2e}")
        print(f"  Q data range: {rr['data_q'].min():.2e} to {rr['data_q'].max():.2e}")

        print("\n✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_test_example.py /path/to/timeS429/directory")
        sys.exit(1)

    data_dir = sys.argv[1]
    success = test_raw_records_processing(data_dir)

    if success:
        print("\nTest completed successfully!")
        sys.exit(0)
    else:
        print("\nTest failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
