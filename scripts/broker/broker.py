import os
import sys
import shutil
import subprocess
import signal

# --- Configuration ---
# This wrapper script automates the setup and execution of the TypeScript server.
# On Windows, it will automatically run the server within the Windows Subsystem for Linux (WSL).
# On Linux/macOS, it will run the server directly.
#
# REQUIREMENTS FOR WINDOWS USERS:
# 1. WSL must be installed (https://learn.microsoft.com/en-us/windows/wsl/install).
# 2. Node.js and npm must be installed *inside your WSL distribution*. You can test this by
#    opening a WSL terminal and running `node --version`.

SERVER_SCRIPT_PATH = os.path.join('src', 'server.ts')
REQUIRED_NODE_PACKAGES = ['simple-websockets', 'tsx']
NODE_MODULES_DIR = 'node_modules'
SRC_DIR = 'src'

# --- Global variable to hold the server process ---
node_process = None

def set_working_directory():
    """
    Sets the current working directory to the script's actual location.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"--- Working directory set to: {os.getcwd()} ---")

def check_prerequisites():
    """
    Checks for host prerequisites (Node/npm) and WSL on Windows.
    """
    print("--- Checking for prerequisites ---")
    # Check for Node/npm on the host system for dependency installation
    if not shutil.which("node") or not shutil.which("npm"):
        print("ERROR: Node.js and/or npm are not installed on the host system.")
        print("Please install them from https://nodejs.org/")
        sys.exit(1)
    print("✅ Node.js and npm are found on the host system.")

    # If on Windows, specifically check for the WSL executable
    if sys.platform == 'win32':
        if not shutil.which("wsl"):
            print("\nERROR: Running on Windows, but 'wsl.exe' was not found.")
            print("This script requires WSL to run the server on Windows.")
            print("Please install WSL: https://learn.microsoft.com/en-us/windows/wsl/install")
            sys.exit(1)
        print("✅ WSL executable is found.")

def setup_project():
    """
    Initializes the npm project and installs dependencies on the host system.
    """
    print("\n--- Setting up Node.js project ---")

    if not os.path.exists(SRC_DIR):
        print(f"Creating source directory: '{SRC_DIR}'...")
        os.makedirs(SRC_DIR)

    if not os.path.exists('package.json'):
        print("'package.json' not found. Creating it now...")
        subprocess.run(['npm', 'init', '-y'], check=True, capture_output=True, text=True)
        print("✅ 'package.json' created successfully.")

    missing_packages = [
        pkg for pkg in REQUIRED_NODE_PACKAGES 
        if not os.path.exists(os.path.join(NODE_MODULES_DIR, pkg))
    ]
            
    if missing_packages:
        packages_str = ' '.join(missing_packages)
        print(f"Missing required packages: {packages_str}. Installing now...")
        try:
            command = ['npm', 'install'] + missing_packages
            subprocess.run(
                command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
            )
            print("✅ All required packages installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install packages.\n{e.stderr}")
            sys.exit(1)
    else:
        print("✅ All required packages are already installed.")

def run_server():
    """
    Runs the TypeScript server, using WSL if on Windows.
    """
    global node_process
    
    if not os.path.exists(SERVER_SCRIPT_PATH):
        print(f"\nERROR: Server script '{SERVER_SCRIPT_PATH}' not found.")
        sys.exit(1)
    
    command = []
    
    # Construct the command based on the operating system
    if sys.platform == 'win32':
        print(f"\n--- Starting TypeScript server via WSL ---")
        print("    NOTE: This requires Node.js, npm, and npx to be installed")
        print("          *inside your WSL distribution*.")
        # 'wsl -e' executes the command in the current directory, handling path conversion
        command = ['wsl', '-e', 'npx', 'tsx', SERVER_SCRIPT_PATH]
    else:
        print(f"\n--- Starting TypeScript server directly ---")
        command = ['npx', 'tsx', SERVER_SCRIPT_PATH]

    print(f"Executing command: {' '.join(command)}")

    try:
        node_process = subprocess.Popen(command)
        print("\n🚀 Server is running. Press Ctrl+C to stop.")
        node_process.wait()
    except FileNotFoundError:
        print(f"ERROR: Could not execute command: '{' '.join(command)}'.")
        print("Please check your PATH and ensure all tools are installed correctly.")
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
    set_working_directory()
    signal.signal(signal.SIGINT, signal_handler)
    check_prerequisites()
    setup_project()
    run_server()

if __name__ == '__main__':
    main()