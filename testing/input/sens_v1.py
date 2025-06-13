import argparse
import sqlite3
import pandas as pd
from demoparser2 import DemoParser
from tqdm import tqdm

def setup_database(db_path: str, table_name: str) -> sqlite3.Connection:
    """
    Connects to the SQLite database and creates the specified table.
    If the table already exists, it is dropped and recreated.

    Args:
        db_path (str): The path to the SQLite database file.
        table_name (str): The name of the table to create.

    Returns:
        sqlite3.Connection: The connection object to the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop the table if it already exists to ensure a fresh start
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    # Create the new table
    create_table_query = f"""
    CREATE TABLE {table_name} (
        tick INTEGER,
        player_name TEXT,
        x REAL,
        y REAL,
        PRIMARY KEY (tick, player_name)
    );
    """
    cursor.execute(create_table_query)
    print(f"Database '{db_path}' set up with table '{table_name}'.")
    return conn

def process_demo(parser: DemoParser, conn: sqlite3.Connection, table_name: str):
    """
    Parses the demo, calculates mouse deltas, and writes them to the database.

    Args:
        parser (DemoParser): An initialized DemoParser instance.
        conn (sqlite3.Connection): An active SQLite database connection.
        table_name (str): The name of the table to insert data into.
    """
    
    # Reference for `parse_ticks` signature from demoparser-main/src/python/demoparser2.pyi
    # def parse_ticks(self, wanted_props: Sequence[str], ...) -> pd.DataFrame:
    print("Parsing tick data... This may take a while.")
    
    # We request the view angles (pitch, yaw) and the aim punch angle.
    # The `player_name` is also explicitly requested.
    props_to_parse = ["player_name", "pitch", "yaw", "aim_punch_angle"]
    
    try:
        ticks_df = parser.parse_ticks(props_to_parse)
        print("Tick data parsed successfully.")
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return

    # --- Crucial Fix: Expand 'aim_punch_angle' into 'aim_punch_angle_x' and 'aim_punch_angle_y' ---
    # The `aim_punch_angle` column typically contains a list/tuple of [pitch, yaw] values.
    # We need to split this into two separate columns for calculations.
    # Assuming the first element is pitch (vertical) and the second is yaw (horizontal).
    if 'aim_punch_angle' in ticks_df.columns:
        print("Expanding 'aim_punch_angle' column...")
        # Use .apply(pd.Series) to split the list/tuple into new columns
        aim_punch_components = ticks_df['aim_punch_angle'].apply(pd.Series)
        
        # Rename the new columns and drop the original 'aim_punch_angle' column
        # Based on typical CS/Source engine conventions, index 0 is pitch (vertical), index 1 is yaw (horizontal).
        ticks_df['aim_punch_angle_pitch'] = aim_punch_components[0]
        ticks_df['aim_punch_angle_yaw'] = aim_punch_components[1]
        ticks_df.drop(columns=['aim_punch_angle'], inplace=True)
        print("Successfully expanded 'aim_punch_angle'.")
    else:
        print("Error: 'aim_punch_angle' column not found in parsed data. Cannot proceed.")
        return

    # Check for expected columns after expansion
    expected_cols = ["tick", "player_name", "pitch", "yaw", "aim_punch_angle_pitch", "aim_punch_angle_yaw"]
    if not all(col in ticks_df.columns for col in expected_cols):
        print("Error: The parsed DataFrame does not contain all expected columns after expansion.")
        print(f"Expected: {expected_cols}")
        print(f"Got: {ticks_df.columns.tolist()}")
        print("Cannot proceed with delta calculation.")
        return

    print("Calculating mouse deltas from view angles and aim punch...")
    
    # Sort by player and tick to ensure correct diff calculation
    ticks_df = ticks_df.sort_values(by=['player_name', 'tick']).reset_index(drop=True)
    
    # Group by player to calculate diffs independently for each
    grouped = ticks_df.groupby('player_name')
    
    # Use a progress bar for the processing step
    with tqdm(total=4, desc="Calculating Deltas") as pbar:
        ticks_df['delta_pitch'] = grouped['pitch'].transform('diff')
        pbar.update(1)
        ticks_df['delta_yaw'] = grouped['yaw'].transform('diff')
        pbar.update(1)
        ticks_df['delta_aim_punch_pitch'] = grouped['aim_punch_angle_pitch'].transform('diff')
        pbar.update(1)
        ticks_df['delta_aim_punch_yaw'] = grouped['aim_punch_angle_yaw'].transform('diff')
        pbar.update(1)

    # The actual mouse movement is the change in view angle minus the change in aim punch
    # Vertical movement (mouse y) corresponds to pitch (view angle on X-axis in angle terms)
    # Horizontal movement (mouse x) corresponds to yaw (view angle on Y-axis in angle terms)
    # Note: For pitch, a positive change in view angle is usually looking up,
    # while a positive mouse Y input is usually looking down. We negate delta_pitch
    # to align with conventional mouse Y movement (down is positive).
    ticks_df['y'] = - (ticks_df['delta_pitch'] - ticks_df['delta_aim_punch_pitch']) 
    ticks_df['x'] = ticks_df['delta_yaw'] - ticks_df['delta_aim_punch_yaw']
    
    # Create the final DataFrame with the required schema
    result_df = ticks_df[['tick', 'player_name', 'x', 'y']].copy()
    
    # Drop rows with NaN values, which are the first tick for each player (no previous tick to diff against)
    result_df.dropna(inplace=True)

    print(f"Writing {len(result_df)} records to the database...")
    try:
        # Use pandas.to_sql for efficient bulk insertion
        result_df.to_sql(table_name, conn, if_exists='append', index=False)
        print("Data successfully written to the database.")
    except Exception as e:
        print(f"An error occurred while writing to the database: {e}")


def main():
    """Main function to parse arguments and run the processing."""
    arg_parser = argparse.ArgumentParser(
        description="Parse a CS2 demo file to calculate sensitivity-independent mouse deltas and store them in a SQLite database."
    )
    arg_parser.add_argument("demofile", help="Path to the .dem file")
    arg_parser.add_argument("database", help="Path to the SQLite database file")
    arg_parser.add_argument("table", help="Name of the table to store the mouse data")
    
    args = arg_parser.parse_args()

    # --- Database and Parser Setup ---
    # Reference for DemoParser constructor from demoparser-main/src/python/demoparser2.pyi
    # def __init__(self, path: str) -> None:
    try:
        parser = DemoParser(args.demofile)
        db_connection = setup_database(args.database, args.table)
    except Exception as e:
        print(f"Failed to initialize parser or database. Error: {e}")
        return

    # --- Processing ---
    process_demo(parser, db_connection, args.table)

    # --- Cleanup ---
    db_connection.close()
    print("Process finished.")

if __name__ == "__main__":
    main()