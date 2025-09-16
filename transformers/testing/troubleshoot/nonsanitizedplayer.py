#!/usr/bin/env python3
"""
Scans a directory of CS2 pipeline SQLite databases (.db files) to identify
player names in the 'RECORDING' table that do not conform to the safe
filename sanitization logic.

This script is used to assess the impact of a bugfix where player names
were not being sanitized before being used in filenames, causing errors on
certain operating systems (e.g., Windows).

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
    Searches through all .db files in a directory to find player names
    that would be changed by the sanitization logic.
    """
    print(f"[*] Scanning for .db files in: {directory_path.resolve()}")
    db_files = list(directory_path.glob('*.db'))

    if not db_files:
        print("[!] No .db files found in the specified directory.")
        return

    total_issues_found = 0
    print("-" * 70)

    for db_path in sorted(db_files):
        issues_in_file = 0
        
        try:
            # Connect in read-only mode for safety
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            
            # Get all unique player names from the RECORDING table
            cursor.execute("SELECT DISTINCT playername FROM RECORDING")
            # The result is a list of tuples, e.g., [('Player1',), ('mahar:>',)]
            all_names = cursor.fetchall()
            
            for name_tuple in all_names:
                original_name = name_tuple[0]
                if not original_name:
                    continue # Skip null or empty names

                sanitized = sanitize_player_name(original_name)

                # The core logic: if the sanitized name is different, report it.
                if original_name != sanitized:
                    if issues_in_file == 0:
                        print(f"\n[*] Issues found in: {db_path.name}")
                    print(f"  - Original: '{original_name}'  ->  Sanitized: '{sanitized}'")
                    issues_in_file += 1
            
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
            total_issues_found += issues_in_file

    print("-" * 70)
    if total_issues_found > 0:
        print(f"\n[SUCCESS] Scan complete. Found a total of {total_issues_found} player names that require sanitization.")
    else:
        print(f"\n[SUCCESS] Scan complete. All player names across {len(db_files)} files already conform to the sanitization rules.")


def main():
    parser = argparse.ArgumentParser(
        description="Find non-conforming player names in CS2 pipeline databases.",
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