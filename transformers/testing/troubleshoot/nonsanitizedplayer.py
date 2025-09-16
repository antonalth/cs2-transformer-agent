#!/usr/bin/env python3
"""
Scans a directory of CS2 pipeline SQLite databases (.db files) to identify
and summarize all unique player names that do not conform to the safe
filename sanitization logic.

This script is used to assess the impact of a bugfix where player names
were not being sanitized before being used in filenames. It categorizes
problematic names into 'CRITICAL' (error-causing) and 'INCONSISTENT'.

Usage:
    python find_offending_playernames.py --datadir /path/to/your/db_files
"""
import argparse
import re
import sqlite3
from pathlib import Path

def sanitize_player_name(player_name: str) -> str:
    """
    Sanitizes a player name to make it safe for use in filenames.
    This function MUST be identical to the one used in the fixed scripts.

    - Replaces spaces with underscores.
    - Removes any character that is NOT an alphanumeric character, underscore, or hyphen.
    - Strips leading/trailing whitespace.
    """
    if not player_name:
        return "unknown_player"
    
    # Replace spaces with underscores
    name = player_name.replace(' ', '_')
    
    # Remove any character that is NOT an alphanumeric character, an underscore, or a hyphen.
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    
    return sanitized_name.strip()

def find_nonconforming_names(directory_path: Path):
    """
    Searches through all .db files in a directory to find all unique player names
    that would be changed by the sanitization logic and categorizes them.
    """
    print(f"[*] Scanning for .db files in: {directory_path.resolve()}")
    db_files = list(directory_path.glob('*.db'))

    if not db_files:
        print("[!] No .db files found in the specified directory.")
        return

    # A set to store unique tuples of (original, sanitized)
    unique_offending_names = set()
    # Regex to find characters illegal in Windows filenames (excluding space, which we handle)
    error_causing_pattern = re.compile(r'[:*?"<>|/\\]')

    for db_path in sorted(db_files):
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT playername FROM RECORDING")
            all_names = cursor.fetchall()
            
            for name_tuple in all_names:
                original_name = name_tuple[0]
                if not original_name:
                    continue

                sanitized = sanitize_player_name(original_name)

                if original_name != sanitized:
                    unique_offending_names.add((original_name, sanitized))
            
        except sqlite3.OperationalError as e:
            if "no such table: RECORDING" in str(e):
                print(f"[!] Warning: '{db_path.name}' does not have a 'RECORDING' table. Skipping.")
            else:
                print(f"[!] Error processing '{db_path.name}': {e}")
        except Exception as e:
            print(f"[!] An unexpected error occurred with '{db_path.name}': {e}")
        finally:
            if 'conn' in locals():
                conn.close()

    print("-" * 70)
    
    if not unique_offending_names:
        print(f"\n[SUCCESS] Scan complete. All player names across {len(db_files)} files already conform to the sanitization rules.")
        return

    print(f"\n[INFO] Scan complete. Found {len(unique_offending_names)} unique player name formats that require fixing.")

    # --- Categorize and Print the Summary ---
    critical_names = []
    inconsistent_names = []

    for original, sanitized in sorted(list(unique_offending_names)):
        if error_causing_pattern.search(original):
            critical_names.append({'orig': original, 'san': sanitized})
        else:
            inconsistent_names.append({'orig': original, 'san': sanitized})

    if critical_names:
        print("\n--- CRITICAL (Error-Causing) Names ---")
        print("These names contain characters that crash the recording script on Windows.")
        for item in critical_names:
            print(f"  - Original: '{item['orig']}'  ->  Sanitized: '{item['san']}'")
    
    if inconsistent_names:
        print("\n--- INCONSISTENT Names (Require Fixing) ---")
        print("These names did not cause errors but will be changed by the new logic.")
        for item in inconsistent_names:
            print(f"  - Original: '{item['orig']}'  ->  Sanitized: '{item['san']}'")
    
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Find and summarize all non-conforming player names in CS2 pipeline databases.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--datadir", 
        required=True, 
        type=Path, 
        help="The path to the directory containing your .db files."
    )
    args = parser.parse_args()

    if not args.datadir.is_dir():
        print(f"[ERROR] The provided path is not a valid directory: {args.datadir}")
        return

    find_nonconforming_names(args.datadir)


if __name__ == "__main__":
    main()