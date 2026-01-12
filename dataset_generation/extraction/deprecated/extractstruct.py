"""
Copyright 2025 Anton Althoff

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------
"""
import os
import sqlite3
import argparse

def extract_schema_and_first_row(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get user-defined tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]

    results = {}
    for table in tables:
        # Get column info
        cursor.execute(f"PRAGMA table_info('{table}')")
        cols = [(col[1], col[2]) for col in cursor.fetchall()]

        # Get first row, if any
        cursor.execute(f"SELECT * FROM '{table}' LIMIT 1")
        first_row = cursor.fetchone()

        results[table] = {
            'columns': cols,
            'first_row': first_row
        }

    conn.close()
    return results


def main(directory):
    for fname in os.listdir(directory):
        if fname.endswith(('.db', '.sqlite', '.sqlite3')):
            path = os.path.join(directory, fname)
            print(f"Database: {fname}")
            schema_data = extract_schema_and_first_row(path)

            if not schema_data:
                print("  (No tables found)")
            for table, info in schema_data.items():
                cols = info['columns']
                first = info['first_row']
                cols_str = ', '.join(f"{name} ({dtype})" for name, dtype in cols)
                print(f"  Table '{table}': {cols_str}")

                # Print first entry
                if first:
                    values = ', '.join(str(v) for v in first)
                    print(f"    First row: {values}")
                else:
                    print("    First row: (table is empty)")
            print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract table schemas and first entries from SQLite DBs in a directory.')
    parser.add_argument('directory', help='Path to directory containing .db/.sqlite files')
    args = parser.parse_args()
    main(args.directory)
