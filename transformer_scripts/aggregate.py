import argparse
import subprocess
import os
import glob
import sys

def run_command(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run full demo processing pipeline.")
    parser.add_argument('--demo', required=True, help='Path to demo file (.dem)')
    parser.add_argument('--out', default=None, help='Output directory (default: data_<demobasename>)')
    args = parser.parse_args()

    demo_path = os.path.abspath(args.demo)
    if not os.path.isfile(demo_path):
        print(f"Error: Demo file not found: {demo_path}", file=sys.stderr)
        sys.exit(1)

    demo_name = os.path.splitext(os.path.basename(demo_path))[0]
    out_dir = os.path.abspath(args.out or f"data_{demo_name}")

    # Create or clean output directory
    os.makedirs(out_dir, exist_ok=True)
    print(f"Using output directory: {out_dir}")
    for db_file in glob.glob(os.path.join(out_dir, '*.db')):
        print(f"Removing existing DB: {db_file}")
        os.remove(db_file)

    # Locate transformer_scripts directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    commands = [
        (['python', os.path.join(script_dir, 'mouse.py'), demo_path, os.path.join(out_dir, 'mouse.db'), 'MOUSE'], None),
        (['python', os.path.join(script_dir, 'rounds.py'), '--sqlout', os.path.join(out_dir, 'rounds.db'), demo_path], None),
        (['python', os.path.join(script_dir, 'keyboard_location.py'), '--sqlout', os.path.join(out_dir, 'keyboard_location.db'), demo_path, '--optimize'], None),
        (['python', os.path.join(script_dir, 'buy_sell_drop.py'), '--sqlin', os.path.join(out_dir, 'keyboard_location.db'), '--sqlout', os.path.join(out_dir, 'buy_sell_drop.db'), demo_path], None),
    ]

    for cmd, cwd in commands:
        try:
            run_command(cmd, cwd=cwd)
        except subprocess.CalledProcessError as e:
            print(f"Error: Command {' '.join(cmd)} failed with exit code {e.returncode}", file=sys.stderr)
            sys.exit(e.returncode)

    print("All processing steps completed successfully.")

if __name__ == '__main__':
    main()
