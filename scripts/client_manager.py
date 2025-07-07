import os
import sys
import subprocess
import signal
import time
import argparse

# --- Configuration ---
# This script manages the lifecycle of multiple sandboxed CS2 clients.

# Path to the Sandboxie-Plus executable.
# IMPORTANT: Update this path if your installation is different.
SANDBOXIE_EXE_PATH = r'C:\Program Files\Sandboxie-Plus\Start.exe'

# Relative path to the broker script that starts the Node.js server.
BROKER_SCRIPT_PATH = os.path.join('broker', 'broker.py')

# Relative path to the batch file that launches CS2.
CS2_BAT_PATH = 'start_cs2.bat'

# --- Global variable for the broker process ---
broker_process = None

def set_working_directory():
    """Sets the CWD to the script's location for reliable pathing."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"--- Working directory set to: {os.getcwd()} ---")

def parse_arguments():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Launch and manage multiple sandboxed Counter-Strike 2 clients.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--num',
        type=int,
        default=3,
        choices=range(1, 11),
        metavar='[1-10]',
        help="Number of CS2 clients to launch. Default is 3."
    )
    parser.add_argument(
        '--delay',
        type=int,
        default=60,
        metavar='SECONDS',
        help="Delay in seconds after starting Steam clients to allow for login. Default is 60."
    )
    return parser.parse_args()

def run_command(command_list, wait=True):
    """A helper function to run external commands."""
    print(f"  Executing: {' '.join(command_list)}")
    try:
        if wait:
            subprocess.run(command_list, check=True)
        else:
            return subprocess.Popen(command_list)
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Executable not found: '{command_list[0]}'")
        print("Please ensure the path is correct in the script's configuration.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Command failed with exit code {e.returncode}")
        # Continue execution as some failures might be non-critical (e.g., terminate on empty sandbox)

def terminate_all_sandboxes():
    """Terminates all processes running in any sandbox."""
    print("\n--- Terminating all processes in all sandboxes ---")
    command = [SANDBOXIE_EXE_PATH, '/terminate_all']
    run_command(command)
    print("--- Termination command sent. ---")

def signal_handler(sig, frame):
    """Handles Ctrl+C to gracefully shut down everything."""
    print("\n\n--- Ctrl+C detected. Initiating shutdown... ---")
    
    global broker_process
    if broker_process and broker_process.poll() is None:
        print("--- Terminating broker server... ---")
        broker_process.terminate()
        broker_process.wait()
    
    terminate_all_sandboxes()
    sys.exit(0)

def main():
    """Main execution logic."""
    set_working_directory()
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_arguments()

    print(f"--- Starting Client Manager ---")
    print(f"  Clients to launch: {args.num}")
    print(f"  Steam init delay: {args.delay} seconds")

    # 1. Initial Cleanup
    terminate_all_sandboxes()
    time.sleep(10)

    # 2. Start Steam in each sandbox
    print(f"\n--- Starting {args.num} Steam client(s) in sandboxes [game1...game{args.num}] ---")
    for i in range(1, args.num + 1):
        box_name = f"game{i}"
        command = [SANDBOXIE_EXE_PATH, '/hide_window', f'/box:{box_name}', 'steam://open']
        run_command(command)
    
    # 3. Wait for Steam to initialize
    print(f"\n--- Waiting {args.delay} seconds for Steam clients to log in and initialize ---")
    time.sleep(args.delay)

    # 4. Start the broker server
    print("\n--- Starting the broker server (broker/broker.py) ---")
    global broker_process
    # Use sys.executable to ensure we use the same python interpreter
    broker_command = [sys.executable, BROKER_SCRIPT_PATH]
    broker_process = run_command(broker_command, wait=False)
    print("--- Broker server process started. Waiting 5 seconds for it to initialize... ---")
    time.sleep(5) # Give the server time to start its HTTP/WebSocket listeners

    # 5. Start CS2 in each sandbox, one by one
    print(f"\n--- Launching {args.num} CS2 client(s) sequentially ---")
    for i in range(1, args.num + 1):
        box_name = f"game{i}"
        print(f"\n--- Launching CS2 in sandbox '{box_name}' ---")
        command = [SANDBOXIE_EXE_PATH, f'/box:{box_name}', CS2_BAT_PATH]
        run_command(command)

        if i < args.num:
            print("--- Waiting 15 seconds before launching the next client to ensure connection order ---")
            time.sleep(15)

    print("\n\n🚀 All clients launched. The broker server is running.")
    print("Press Ctrl+C to shut down all processes.")

    # Wait for the broker process to exit. If it crashes, the script will proceed to cleanup.
    if broker_process:
        broker_process.wait()

    # Final cleanup if the script exits for any reason other than Ctrl+C
    print("\n--- Broker process has finished. Performing final cleanup. ---")
    terminate_all_sandboxes()

if __name__ == '__main__':
    main()