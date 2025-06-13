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

def normalize_angle_delta(delta: pd.Series) -> pd.Series:
    """
    Corrects angular deltas that wrap around 360 degrees.

    Args:
        delta (pd.Series): A Series of angle differences.

    Returns:
        pd.Series: The corrected Series.
    """
    delta_copy = delta.copy()
    delta_copy[delta_copy > 180] -= 360
    delta_copy[delta_copy < -180] += 360
    return delta_copy

def process_demo(parser: DemoParser, conn: sqlite3.Connection, table_name: str):
    """
    Parses the demo, calculates mouse deltas, and writes them to the database.

    Args:
        parser (DemoParser): An initialized DemoParser instance.
        conn (sqlite3.Connection): An active SQLite database connection.
        table_name (str): The name of the table to insert data into.
    """
    
    print("Parsing tick data... This may take a while.")
    props_to_parse = ["player_name", "pitch", "yaw", "aim_punch_angle"]
    
    try:
        ticks_df = parser.parse_ticks(props_to_parse)
        print("Tick data parsed successfully.")
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return

    if 'aim_punch_angle' in ticks_df.columns:
        print("Expanding 'aim_punch_angle' column...")
        aim_punch_components = ticks_df['aim_punch_angle'].apply(pd.Series)
        ticks_df['aim_punch_angle_pitch'] = aim_punch_components[0]
        ticks_df['aim_punch_angle_yaw'] = aim_punch_components[1]
        ticks_df.drop(columns=['aim_punch_angle'], inplace=True)
        print("Successfully expanded 'aim_punch_angle'.")
    else:
        print("Error: 'aim_punch_angle' column not found. Cannot proceed.")
        return

    print("Calculating mouse deltas with wrap-around correction...")
    
    ticks_df = ticks_df.sort_values(by=['player_name', 'tick']).reset_index(drop=True)
    grouped = ticks_df.groupby('player_name')
    
    with tqdm(total=4, desc="Calculating and Normalizing Deltas") as pbar:
        delta_pitch = grouped['pitch'].transform('diff')
        ticks_df['delta_pitch'] = normalize_angle_delta(delta_pitch)
        pbar.update(1)
        
        delta_yaw = grouped['yaw'].transform('diff')
        ticks_df['delta_yaw'] = normalize_angle_delta(delta_yaw)
        pbar.update(1)
        
        delta_aim_punch_pitch = grouped['aim_punch_angle_pitch'].transform('diff')
        ticks_df['delta_aim_punch_pitch'] = normalize_angle_delta(delta_aim_punch_pitch)
        pbar.update(1)
        
        delta_aim_punch_yaw = grouped['aim_punch_angle_yaw'].transform('diff')
        ticks_df['delta_aim_punch_yaw'] = normalize_angle_delta(delta_aim_punch_yaw)
        pbar.update(1)

    # The actual mouse movement is the change in view angle minus the change in aim punch
    # We negate both results to align with standard mouse input conventions.
    ticks_df['y'] = - (ticks_df['delta_pitch'] - ticks_df['delta_aim_punch_pitch']) 
    ticks_df['x'] = - (ticks_df['delta_yaw'] - ticks_df['delta_aim_punch_yaw'])
    
    result_df = ticks_df[['tick', 'player_name', 'x', 'y']].copy()
    result_df.dropna(inplace=True)

    print(f"Writing {len(result_df)} records to the database...")
    try:
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

    try:
        parser = DemoParser(args.demofile)
        db_connection = setup_database(args.database, args.table)
    except Exception as e:
        print(f"Failed to initialize parser or database. Error: {e}")
        return

    process_demo(parser, db_connection, args.table)

    db_connection.close()
    print("Process finished.")

if __name__ == "__main__":
    main()