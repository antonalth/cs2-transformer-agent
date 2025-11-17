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
import sys
import subprocess
import signal
import platform

# --- Configuration ---
# This script runs the TypeScript server, assuming all prerequisites are met.
# It uses the `wsl --cd` command on Windows to ensure the correct working directory.

# The path to the TypeScript server file, relative to the project root.
# Using forward slashes is safer for cross-platform compatibility.
SERVER_SCRIPT_FILENAME = 'src/server.ts'

# --- Global variable to hold the server process ---
server_process = None

def run_server():
    """
    Runs the TypeScript server, automatically using WSL on Windows.
    This version explicitly sets the working directory inside WSL.
    """
    global server_process

    # Determine the script's directory and set it as the CWD for this Python script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"--- Working directory set to: {script_dir} ---")

    if not os.path.exists(SERVER_SCRIPT_FILENAME):
        print(f"\nERROR: Server script not found at '{SERVER_SCRIPT_FILENAME}'")
        sys.exit(1)

    command = []
    
    # --- Command Construction ---
    if platform.system() == "Windows":
        print("\n--- Detected Windows. Preparing to launch server via WSL. ---")
        
        # KEY FIX: Use `wsl --cd <dir>` to explicitly set the working directory inside WSL.
        # This is the most reliable way to solve pathing issues.
        # `os.getcwd()` gets the current directory (e.g., C:\Users\...\project),
        # and `--cd` correctly translates it to /mnt/c/Users/.../project within WSL.
        # The command to run is now relative to that new CWD.
        command = [
            'wsl',
            '--cd', os.getcwd(),
            'npx', 'tsx', SERVER_SCRIPT_FILENAME
        ]
    else:
        print("\n--- Detected Linux/macOS. Launching server directly. ---")
        # For non-Windows systems, the CWD is already set by os.chdir(), so we can
        # execute the command directly.
        command = ['npx', 'tsx', SERVER_SCRIPT_FILENAME]

    print(f"Executing command: {' '.join(command)}\n")

    try:
        # Launch the subprocess and stream its output in real-time.
        server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        for line in iter(server_process.stdout.readline, ''):
            sys.stdout.write(line)
        
        server_process.wait()
        return_code = server_process.returncode
        
        if return_code != 0:
            print(f"\n--- Server exited with a non-zero status code: {return_code} ---")

    except FileNotFoundError:
        tool = "wsl" if platform.system() == "Windows" else "npx"
        print(f"ERROR: The command '{tool}' was not found.")
        print("Please ensure it is installed and available in your system's PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while running the server: {e}")
        sys.exit(1)

def signal_handler(sig, frame):
    """Handles Ctrl+C to gracefully shut down the child process."""
    global server_process
    print("\n--- Ctrl+C detected. Shutting down server... ---")
    if server_process and server_process.poll() is None:
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
            print("✅ Server shut down.")
        except subprocess.TimeoutExpired:
            print("Server did not respond, forcing kill.")
            server_process.kill()
            print("✅ Server killed.")
    sys.exit(0)

def main():
    """Main execution function."""
    signal.signal(signal.SIGINT, signal_handler)
    run_server()

if __name__ == '__main__':
    main()