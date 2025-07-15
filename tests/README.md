# Testing Guide

This directory contains tests for the straxion package, including comprehensive tests for the DAQReader plugin.

## Test Structure

- `test_contexts.py`: Basic tests for the straxion context creation
- `test_raw_records.py`: Comprehensive tests for the DAQReader plugin

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

### Using the Example Script

You can also use the standalone example script to test the DAQReader:

```bash
python run_test_example.py /path/to/timeS429/directory
```

This script performs the same validation as the test but provides more detailed output for debugging.

## GitHub Actions Integration

The tests are configured to work with GitHub Actions using GitHub secrets for test data.

### Setting Up GitHub Secrets

1. Go to your repository settings
2. Navigate to "Secrets and variables" → "Actions"
3. Add a new repository secret:
   - **Name**: `STRAXION_TEST_DATA_DIR`
   - **Value**: The path to your test data directory on the GitHub Actions runner

### GitHub Actions Workflow

The tests are integrated into the existing `.github/workflows/pytest.yml` workflow, which will:

1. Run basic tests without test data on Python 3.11
2. Run raw_records tests with real data if the `STRAXION_TEST_DATA_DIR` secret is available
3. Generate coverage reports for both basic and data-dependent tests

### Providing Test Data to GitHub Actions

There are several ways to provide test data to GitHub Actions:

#### Option 1: Upload as Artifact (Recommended for small datasets)

```yaml
# In your workflow
- name: Upload test data
  uses: actions/upload-artifact@v3
  with:
    name: test-data
    path: /path/to/your/test/data

- name: Download test data
  uses: actions/download-artifact@v3
  with:
    name: test-data
    path: ./test-data

- name: Run tests with data
  env:
    STRAXION_TEST_DATA_DIR: ./test-data
  run: pytest tests/ -v
```

#### Option 2: Use GitHub Secrets for Paths

If your test data is stored in a known location on the runner:

```yaml
- name: Run tests with data
  env:
    STRAXION_TEST_DATA_DIR: ${{ secrets.STRAXION_TEST_DATA_DIR }}
  run: pytest tests/ -v
```

#### Option 3: Download from External Source

```yaml
- name: Download test data
  run: |
    # Download your test data from a secure location
    curl -L -o test-data.tar.gz ${{ secrets.TEST_DATA_URL }}
    tar -xzf test-data.tar.gz

- name: Run tests with data
  env:
    STRAXION_TEST_DATA_DIR: ./extracted-test-data
  run: pytest tests/ -v
```

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
3. Ensure the binary files are in the correct format expected by the DAQReader
4. Check that the configuration parameters (record_length, fs) match your data

For more detailed debugging, use the standalone example script:

```bash
python run_test_example.py /path/to/your/test/data
```

This will provide step-by-step validation and detailed error messages.
