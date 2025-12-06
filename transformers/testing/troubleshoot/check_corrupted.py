#!/usr/bin/env python3

import os
import subprocess
import argparse
import logging
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def is_ffmpeg_installed() -> bool:
    """Checks if ffmpeg/ffprobe is installed and available in PATH."""
    try:
        # Try running ffprobe with -version to check its availability
        subprocess.run(["ffprobe", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_media_integrity(filepath: str) -> Tuple[bool, str]:
    """
    Checks the integrity of a media file (.mp4 or .wav) using ffprobe.
    Returns (True, "") if the file appears well-formed, or (False, error_message) otherwise.
    """
    try:
        # Run ffprobe with '-v error' to only show errors, and capture stderr.
        # A successful run (returncode 0) indicates the file is likely well-formed.
        # A non-zero returncode indicates an issue.
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-i", filepath],
            check=False, # We handle the return code ourselves, don't raise CalledProcessError
            stdout=subprocess.PIPE, # Discard stdout as we only care about stderr for errors
            stderr=subprocess.PIPE, # Capture stderr for error messages
            text=True,              # Decode stdout/stderr as text
            timeout=60              # Timeout for each file in seconds (prevent hangs on severely corrupted files)
        )

        if result.returncode == 0:
            # ffprobe returned 0, indicating no severe errors.
            return True, ""
        else:
            # ffprobe returned a non-zero exit code.
            # The error message should be in stderr.
            error_message = result.stderr.strip()
            if not error_message:
                error_message = f"ffprobe exited with code {result.returncode} but provided no specific error message."
            return False, error_message

    except FileNotFoundError:
        return False, "ffprobe not found. Please ensure ffmpeg is installed and in your system's PATH."
    except subprocess.TimeoutExpired:
        # ffprobe command timed out.
        return False, "Timed out during ffprobe check. File might be corrupted or excessively large."
    except Exception as e:
        # Catch any other unexpected errors.
        return False, f"An unexpected error occurred: {e}"

def scan_recordings_directory(recordings_dir: str, extensions: List[str]) -> List[Tuple[str, str]]:
    """
    Scans a directory for specified media files and checks their integrity.
    Returns a list of (filepath, error_message) for all malformed files.
    """
    if not is_ffmpeg_installed():
        logging.error("FFmpeg (ffprobe) is not installed or not in your system's PATH. Please install it to use this script.")
        return []

    if not os.path.isdir(recordings_dir):
        logging.error(f"Error: Directory '{recordings_dir}' not found.")
        return []

    broken_files: List[Tuple[str, str]] = []
    total_files_checked = 0

    logging.info(f"Scanning directory: {recordings_dir} for {', '.join(extensions)} files (recursively)...")

    # Walk through the directory recursively
    for root, _, files in os.walk(recordings_dir):
        for filename in files:
            file_extension = os.path.splitext(filename)[1].lower() # Get file extension and convert to lowercase

            if file_extension in extensions:
                filepath = os.path.join(root, filename)
                total_files_checked += 1
                logging.info(f"Checking: {filepath}")

                is_ok, error_msg = check_media_integrity(filepath)
                if not is_ok:
                    broken_files.append((filepath, error_msg))
                    logging.warning(f"  --> BROKEN: {error_msg}")
    
    logging.info(f"\n--- Scan Summary ---")
    logging.info(f"Total {', '.join(extensions)} files checked: {total_files_checked}")
    if broken_files:
        logging.error(f"Found {len(broken_files)} broken files:")
        for filepath, error_msg in broken_files:
            logging.error(f"- {filepath}\n  Error: {error_msg}")
    else:
        logging.info("No broken media files found.")
    
    return broken_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan a recordings directory for .mp4 and .wav files and check their integrity using ffprobe."
    )
    parser.add_argument(
        "recordings_dir",
        type=str,
        help="The path to the directory containing recordings (e.g., /mnt/trainingdata/dataset0/recordings)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging for more detailed output (mostly for script's internal logic)."
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Define the file extensions to check
    extensions_to_check = [".mp4", ".wav"]

    broken_files_list = scan_recordings_directory(args.recordings_dir, extensions_to_check)
    
    # Exit with a non-zero code if any broken files were found
    if broken_files_list:
        exit(1)
