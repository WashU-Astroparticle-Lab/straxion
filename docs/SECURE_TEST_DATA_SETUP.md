# Secure Test Data Setup Guide

This guide explains how to securely upload and use test data in GitHub Actions without exposing it publicly.

## GitHub Releases Method (Recommended)

### Step 1: Create a Draft Release

1. Go to your GitHub repository
2. Click **Releases** in the right sidebar
3. Click **Create a new release**
4. **Tag version**: `test-data` (or any version)
5. **Release title**: `Test Data for CI`
6. **Description**: `Test data for automated testing`
7. **Important**: Check **"Set as a draft release"** (this keeps it private)
8. Click **Publish release** (it will be published as a draft)

### Step 2: Upload Test Data

1. In the draft release, click **Edit**
2. Scroll down to **Attach binaries by dropping them here or selecting them**
3. Upload your compressed test data file (e.g., `timeS429.tar.gz`)
4. Click **Update release**

### Step 3: The Workflow

The workflow in `.github/workflows/pytest.yml` is already configured to automatically download test data from GitHub Releases. No additional configuration is needed.

## Why GitHub Releases?

This method is recommended because:

1. **No external dependencies** - Uses GitHub's built-in features
2. **Secure** - Draft releases are private
3. **Simple setup** - Just upload and go
4. **Large file support** - Up to 2GB per file
5. **Free** - No additional costs

## File Size Support

- **GitHub Secrets**: 64KB limit ❌ (Too small for test data)
- **GitHub Releases**: 2GB limit ✅ (Perfect for your 20MB file)

## Quick Setup Commands

### Prepare Your Test Data

```bash
# Use the helper script
python scripts/prepare_release_data.py /path/to/your/timeS429/directory

# Or manually
tar -czf timeS429.tar.gz timeS429/
ls -lh timeS429.tar.gz
```

### Test Locally

```bash
# Extract and test locally
tar -xzf timeS429.tar.gz
export STRAXION_TEST_DATA_DIR=$(pwd)/timeS429
pytest tests/test_raw_records.py::test_raw_records_processing -v
```

## Troubleshooting

### Common Issues

1. **Release not found**: Make sure the release is published as a draft
2. **Asset not found**: Check the asset name contains "timeS429"
3. **Permission denied**: Ensure the workflow has proper permissions

### Debug Commands

Add this to your workflow for debugging:

```yaml
- name: Debug test data setup
  run: |
    echo "Current directory: $(pwd)"
    echo "Directory contents:"
    ls -la
    if [ -d "timeS429" ]; then
      echo "timeS429 contents:"
      ls -la timeS429/
      echo "Number of files: $(find timeS429 -type f | wc -l)"
    fi
    echo "STRAXION_TEST_DATA_DIR: $STRAXION_TEST_DATA_DIR"
```

## Summary

This setup provides a secure, simple, and free way to test your NX3LikeReader plugin with real data in GitHub Actions. The workflow will automatically download test data from a private draft release and run comprehensive tests without exposing the data publicly.
