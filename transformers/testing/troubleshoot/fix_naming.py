# --- START OF FILE repair_filenames.py ---
import argparse
import re
import sqlite3
from pathlib import Path

def sanitize_player_name(player_name: str) -> str:
    """
    Sanitizes a player name to make it safe for use in filenames.
    This function MUST be identical to the one we will use in the fixed scripts.
    """
    if not player_name:
        return "unknown_player"
    name = player_name.replace(' ', '_')
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    return sanitized_name.strip()

def repair_database_and_files(db_path: Path, recdir: Path, dry_run: bool):
    """
    Scans a single database for recorded entries and synchronizes their
    filenames and DB paths with the new sanitization logic.
    """
    print(f"\n[*] Processing database: {db_path.name}")
    updated_count = 0
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Select all entries that are marked as recorded
        cursor.execute("SELECT rowid, playername, recording_filepath, roundnumber, team, starttick, stoptick FROM RECORDING WHERE is_recorded = TRUE")
        records_to_check = cursor.fetchall()
        
        if not records_to_check:
            print("  -> No recorded entries found. Nothing to do.")
            return

        for rowid, pname, old_filepath_str, r_num, team, s_tick, e_tick in records_to_check:
            if not old_filepath_str or not pname:
                continue

            old_filepath = Path(old_filepath_str)
            
            # Reconstruct what the filename SHOULD BE with the new logic
            sanitized_pname = sanitize_player_name(pname)
            expected_filename_base = f"{r_num:02d}_{team}_{sanitized_pname}_{s_tick}_{e_tick}"
            
            # Check if the current filename matches the expected one
            if old_filepath.stem != expected_filename_base:
                print(f"  [MISMATCH FOUND] For player: '{pname}'")
                
                # Construct the new, correct path
                new_filepath = old_filepath.with_name(f"{expected_filename_base}{old_filepath.suffix}")
                
                print(f"    -  Old Path: {old_filepath}")
                print(f"    -  New Path: {new_filepath}")

                if not dry_run:
                    # --- RENAME FILES ON DISK ---
                    old_wav_path = old_filepath.with_suffix('.wav')
                    new_wav_path = new_filepath.with_suffix('.wav')

                    try:
                        if old_filepath.exists():
                            old_filepath.rename(new_filepath)
                            print("      -> Renamed MP4 file.")
                        else:
                            print(f"      -> WARNING: MP4 file not found at {old_filepath}. Cannot rename.")

                        if old_wav_path.exists():
                            old_wav_path.rename(new_wav_path)
                            print("      -> Renamed WAV file.")
                        else:
                            print(f"      -> WARNING: WAV file not found at {old_wav_path}. Cannot rename.")
                            
                        # --- UPDATE DATABASE ---
                        cursor.execute("UPDATE RECORDING SET recording_filepath = ? WHERE rowid = ?", (str(new_filepath.resolve()), rowid))
                        print("      -> Updated database record.")
                        conn.commit()

                    except Exception as e:
                        print(f"      -> ERROR: An error occurred during rename/update: {e}")
                        conn.rollback() # Rollback DB change on any error
                
                updated_count += 1

    except Exception as e:
        print(f"  [ERROR] Failed to process database {db_path.name}: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
    
    if updated_count == 0 and records_to_check:
        print("  -> All recorded entries already conform to the new naming scheme.")

def main():
    parser = argparse.ArgumentParser(
        description="Repair inconsistent filenames and database entries for CS2 recordings.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--datadir", required=True, type=Path, help="Directory containing the .db files.")
    parser.add_-argument("--recdir", required=True, type=Path, help="Base directory where final video folders are stored.")
    parser.add_argument("--dry-run", action="store_true", help="Show what changes would be made without actually renaming files or updating the DB.")
    args = parser.parse_args()

    if args.dry_run:
        print("="*50)
        print("=== DRY RUN MODE: No files will be changed. ===")
        print("="*50)

    for db_file in sorted(args.datadir.glob("*.db")):
        repair_database_and_files(db_file, args.recdir, args.dry_run)

    print("\n[SUCCESS] Repair script finished.")
    if args.dry_run:
        print("Rerun without --dry-run to apply the changes.")

if __name__ == "__main__":
    main()