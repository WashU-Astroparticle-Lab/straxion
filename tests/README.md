# Testing Guide

Unless you are a developer for `straxion`, please feel free to skip reading. This directory contains tests for the straxion package, including comprehensive tests for the `NX3LikeReader` plugin.

## Test Structure

- `test_contexts.py`: Basic tests for the straxion context creation
- `test_raw_records.py`: Comprehensive tests for the NX3LikeReader plugin

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

To run tests that require real data (like `test_raw_records_processing`), you need to:

1. Set the `STRAXION_TEST_DATA_DIR` environment variable to point to the directory containing your test data
2. Ensure the test data follows the expected structure

```bash
# Set environment variable and run tests
export STRAXION_TEST_DATA_DIR=/path/to/your/test/data
pytest tests/ -v

# Or run in one command
STRAXION_TEST_DATA_DIR=/path/to/your/test/data pytest tests/ -v
```

### Expected Test Data Structure

The tests expect the following directory structure:

```
/path/to/test/data/
└── timeS429/
    ├── timeS429-ch0.bin
    ├── timeS429-ch1.bin
    ├── timeS429-ch2.bin
    └── ... (more channel files)
```

## GitHub Actions Integration

The tests are configured to work with GitHub Actions using GitHub Releases for test data.

### Setting Up Test Data

The workflow automatically downloads test data from GitHub Releases. To set this up:

1. **Prepare your test data**:
   ```bash
   python scripts/prepare_release_data.py /path/to/your/timeS429/directory
   ```

2. **Create a draft release** in your GitHub repository
3. **Upload the compressed file** to the draft release
4. **The workflow will automatically download and use the data**

See `docs/SECURE_TEST_DATA_SETUP.md` for detailed step-by-step instructions.

### GitHub Actions Workflow

The tests are integrated into the existing `.github/workflows/pytest.yml` workflow, which will:

1. Run basic tests without test data on Python 3.11
2. Automatically download test data from GitHub Releases (if available)
3. Run raw_records tests with real data
4. Generate coverage reports for both basic and data-dependent tests



## Test Coverage

The tests cover:

- Context creation and plugin registration
- Configuration validation
- Data type inference
- Complete data processing pipeline
- Data consistency checks
- Error handling for missing/invalid data

## Debugging Failed Tests

If tests fail:

1. Check that your test data directory exists and contains the expected files
2. Verify the file naming convention matches `<RUN>-ch<CHANNEL>.bin`
3. Ensure the binary files are in the correct format expected by the NX3LikeReader
4. Check that the configuration parameters (record_length, fs) match your data
