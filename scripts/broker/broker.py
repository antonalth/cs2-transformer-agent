import os
import sys
import shutil
import subprocess
import signal
import time

# --- Configuration ---
SERVER_SCRIPT_NAME = 'server.js'
REQUIRED_NODE_PACKAGE = 'simple-websockets'
NODE_MODULES_DIR = 'node_modules'

# --- Global variable to hold the server process ---
node_process = None

def set_working_directory():
    """
    Sets the current working directory to the script's actual location.
    This makes the script independent of where it's called from.
    """
    # Get the absolute path of the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Change the current working directory to the script's directory
    os.chdir(script_dir)
    print(f"--- Working directory set to: {os.getcwd()} ---")

def check_prerequisites():
    """Checks if Node.js and npm are installed and in the PATH."""
    print("--- Checking for prerequisites ---")
    if not shutil.which("node"):
        print("ERROR: Node.js is not installed or not in your PATH.")
        print("Please install Node.js from https://nodejs.org/")
        sys.exit(1)
    
    if not shutil.which("npm"):
        print("ERROR: npm is not installed or not in your PATH.")
        print("npm is typically installed with Node.js.")
        sys.exit(1)

    print("✅ Node.js and npm are found.")
    return True

def setup_project():
    """
    Initializes the npm project and installs dependencies if needed.
    Assumes the CWD is already the project directory.
    """
    print("\n--- Setting up Node.js project ---")

    # 1. Check for package.json
    if not os.path.exists('package.json'):
        print("'package.json' not found. Creating it now...")
        try:
            subprocess.run(['npm', 'init', '-y'], check=True, capture_output=True, text=True)
            print("✅ 'package.json' created successfully.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to create 'package.json'.\n{e.stderr}")
            sys.exit(1)

    # 2. Check for required dependency
    package_path = os.path.join(NODE_MODULES_DIR, REQUIRED_NODE_PACKAGE)
    if not os.path.exists(package_path):
        print(f"Node package '{REQUIRED_NODE_PACKAGE}' not found. Installing now...")
        try:
            subprocess.run(
                ['npm', 'install', REQUIRED_NODE_PACKAGE], 
                check=True, 
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"✅ Package '{REQUIRED_NODE_PACKAGE}' installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install '{REQUIRED_NODE_PACKAGE}'.\n{e.stderr}")
            sys.exit(1)
    else:
        print(f"✅ Package '{REQUIRED_NODE_PACKAGE}' is already installed.")

def run_server():
    """
    Runs the Node.js server script as a subprocess.
    Assumes the CWD is already the project directory.
    """
    global node_process
    
    if not os.path.exists(SERVER_SCRIPT_NAME):
        print(f"\nERROR: Server script '{SERVER_SCRIPT_NAME}' not found in this directory.")
        sys.exit(1)
    
    print(f"\n--- Starting Node.js server ({SERVER_SCRIPT_NAME}) ---")
    
    command = ['node', SERVER_SCRIPT_NAME]
    
    try:
        node_process = subprocess.Popen(command)
        print("\n🚀 Server is running. Press Ctrl+C to stop.")
        node_process.wait()

    except FileNotFoundError:
        print(f"ERROR: Could not execute command: '{' '.join(command)}'.")
        print("Please ensure Node.js is correctly installed.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def signal_handler(sig, frame):
    """Handles Ctrl+C to gracefully shut down the child process."""
    global node_process
    print("\n--- Shutting down server ---")
    if node_process:
        node_process.terminate()
        try:
            node_process.wait(timeout=5)
            print("✅ Server shut down gracefully.")
        except subprocess.TimeoutExpired:
            print("Server did not terminate in time, forcing kill.")
            node_process.kill()
            print("✅ Server killed.")
    sys.exit(0)

def main():
    """Main execution function."""
    # This is the most important step for making the script location-independent.
    set_working_directory()
    
    # Register the signal handler for Ctrl+C (SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    check_prerequisites()
    setup_project()
    run_server()

if __name__ == '__main__':
    main()