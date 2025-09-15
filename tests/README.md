# Testing Guide

Unless you are a developer for `straxion`, please feel free to skip reading. This directory contains comprehensive tests for the straxion package, including tests for all major plugins and processing pipelines.

## Test Structure

- `test_contexts.py`: Basic tests for the straxion context creation
- `test_raw_records.py`: Tests for the raw_records plugin (`QUALIPHIDETHzReader`) with qualiphide_thz_offline context
- `test_records.py`: Tests for the records plugin with both online and offline contexts
- `test_baseline_monitor.py`: Tests for the baseline_monitor plugin with qualiphide_thz_online context
- `test_hits.py`: Tests for the hits plugin with both online and offline contexts
- `test_hit_classification.py`: Tests for hit classification plugins (HitClassification and SpikeCoincidence)

## Running Tests Locally

### Basic Tests (No Test Data Required)

```bash
# Run all tests that don't require test data
pytest tests/ -v

# Run specific test file
pytest tests/test_contexts.py -v
pytest tests/test_raw_records.py -v
```

### Tests with Real Data

To run tests that require real data, you need to:

1. Set the `STRAXION_TEST_DATA_DIR` environment variable to point to the directory containing your test data
2. Ensure the test data follows the expected structure

```bash
# Set environment variable and run tests
export STRAXION_TEST_DATA_DIR=/path/to/your/test/data
pytest tests/ -v

# Or run in one command
STRAXION_TEST_DATA_DIR=/path/to/your/test/data pytest tests/ -v

# Run specific test files
pytest tests/test_raw_records.py -v
pytest tests/test_records.py -v
pytest tests/test_baseline_monitor.py -v
pytest tests/test_hits.py -v
pytest tests/test_hit_classification.py -v
```

### Expected Test Data Structure

The tests expect the following directory structure with `.npy` files:

```
/path/to/test/data/
└── qualiphide_fir_test_data/
    ├── fres_2dB-1756824887.npy
    ├── iq_fine_f_2dB_below_pcrit-1756824887.npy
    ├── iq_fine_z_2dB_below_pcrit-1756824887.npy
    ├── iq_wide_f_2dB_below_pcrit-1756824887.npy
    ├── iq_wide_z_2dB_below_pcrit-1756824887.npy
    └── ts_38kHz-1756824965.npy
```

**Note**: The test data is now provided as a single `qualiphide_fir_test_data.tar.gz` archive that contains all the required `.npy` files.

## GitHub Actions Integration

The tests are configured to work with GitHub Actions using GitHub Releases for test data.

### Setting Up Test Data

The workflow automatically downloads test data from GitHub Releases. To set this up:

1. **Prepare your test data**:
   - Create a directory named `qualiphide_fir_test_data` containing all the required `.npy` files
   - Compress it into a `tar.gz` archive: `tar -czf qualiphide_fir_test_data.tar.gz qualiphide_fir_test_data/`

2. **Create a draft release** in your GitHub repository with the tag `test-data`
3. **Upload the `qualiphide_fir_test_data.tar.gz` file** to the draft release
4. **The workflow will automatically download and use the data**

See `docs/SECURE_TEST_DATA_SETUP.md` for detailed step-by-step instructions.

### GitHub Actions Workflow

The tests are integrated into the existing `.github/workflows/pytest.yml` workflow, which will:

1. Run basic tests without test data on Python 3.11
2. Automatically download `qualiphide_fir_test_data.tar.gz` from GitHub Releases (if available)
3. Run comprehensive tests with real data for all plugins:
   - raw_records processing with `qualiphide_thz_offline` context (`qualiphide_thz_online` uses the same one)
   - records processing with both online and offline contexts
   - baseline_monitor processing with `qualiphide_thz_online` context
   - hits processing with both online and offline contexts
   - hit classification processing with both `HitClassification` and `SpikeCoincidence` plugins
4. Generate coverage reports for both basic and data-dependent tests



## Test Coverage

The tests provide comprehensive coverage for all major straxion components:

### Plugin Tests
- **raw_records**: NX3LikeReader plugin with qualiphide_thz_offline context
- **records**: Both online (phase angle processing) and offline (frequency shift processing) contexts
- **baseline_monitor**: Baseline monitoring with qualiphide_thz_online context
- **hits**: Hit finding with both online and offline contexts
- **hit_classification**: Both HitClassification and SpikeCoincidence plugins

### Test Categories
- **Unit Tests**: Plugin registration, dtype inference, configuration validation
- **Integration Tests**: Complete data processing pipelines with real data
- **Validation Tests**: Data consistency, timing validation, field validation
- **Error Handling**: Missing data directories, invalid configurations, malformed inputs

### Context Coverage
- **qualiphide_thz_online**: Online processing pipeline (phase angle data)
- **qualiphide_thz_offline**: Offline processing pipeline (frequency shift data)

## Debugging Failed Tests

If tests fail:

1. **Check test data structure**: Ensure your test data directory contains all required `.npy` files:
   - `fres_2dB-1756824887.npy`
   - `iq_fine_f_2dB_below_pcrit-1756824887.npy`
   - `iq_fine_z_2dB_below_pcrit-1756824887.npy`
   - `iq_wide_f_2dB_below_pcrit-1756824887.npy`
   - `iq_wide_z_2dB_below_pcrit-1756824887.npy`
   - `ts_38kHz-1756824965.npy`

2. **Verify file format**: Ensure all files are valid `.npy` files that can be loaded by numpy

3. **Check environment variable**: Ensure `STRAXION_TEST_DATA_DIR` points to the correct directory

4. **Context-specific issues**:
   - For online context tests: Check that phase angle data is properly processed
   - For offline context tests: Check that frequency shift data is properly processed

5. **Configuration parameters**: Verify that sampling frequency and other parameters match your data
