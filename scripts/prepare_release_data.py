#!/usr/bin/env python3
"""Helper script to prepare test data for GitHub Releases.

This script compresses your test data directory for upload to a GitHub draft release.
The workflow will automatically download this data during CI runs.

Usage:
    python scripts/prepare_release_data.py /path/to/timeS429/directory

"""

import os
import sys
import tarfile
import argparse


def compress_directory(directory_path, output_file):
    """Compress a directory into a tar.gz file."""
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(directory_path, arcname=os.path.basename(directory_path))

    print(f"✓ Compressed {directory_path} to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare test data for GitHub Releases")
    parser.add_argument("directory", help="Path to the timeS429 directory")
    parser.add_argument(
        "--output",
        "-o",
        default="timeS429.tar.gz",
        help="Output file for compressed data (default: timeS429.tar.gz)",
    )

    args = parser.parse_args()

    # Check if directory exists
    if not os.path.exists(args.directory):
        print(f"❌ Error: Directory {args.directory} does not exist")
        sys.exit(1)

    if not os.path.isdir(args.directory):
        print(f"❌ Error: {args.directory} is not a directory")
        sys.exit(1)

    # Get directory name
    dir_name = os.path.basename(args.directory)
    if dir_name != "timeS429":
        print(f"⚠️  Warning: Directory name is '{dir_name}', expected 'timeS429'")

    try:
        # Compress the directory
        compress_directory(args.directory, args.output)

        # Get file size
        file_size = os.path.getsize(args.output)
        print(f"✓ File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")

        # Check GitHub releases size limit
        if file_size > 2 * 1024 * 1024 * 1024:  # 2GB limit
            print("❌ Error: File is too large for GitHub releases (2GB limit)")
            print("Consider splitting the data or using external storage")
            sys.exit(1)

        print(f"\n🎉 Test data prepared successfully!")
        print(f"\nNext steps:")
        print("1. Go to your GitHub repository")
        print("2. Click 'Releases' in the right sidebar")
        print("3. Click 'Create a new release'")
        print("4. Set tag version (e.g., 'test-data')")
        print("5. Set release title (e.g., 'Test Data for CI')")
        print("6. **IMPORTANT**: Check 'Set as a draft release'")
        print("7. Click 'Publish release' (it will be published as draft)")
        print("8. In the draft release, click 'Edit'")
        print("9. Upload the file: " + args.output)
        print("10. Click 'Update release'")
        print("\nThe workflow will automatically download this data during CI runs.")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
