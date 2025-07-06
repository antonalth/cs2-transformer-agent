import os
import sys
import subprocess
import signal
import platform

# --- Configuration ---
# This script focuses solely on running the TypeScript server.
# It assumes Node.js, npm, and required packages are already installed.

# The path to the TypeScript server file, relative to this script's location.
SERVER_SCRIPT_PATH = os.path.join('src', 'server.ts')

# --- Global variable to hold the server process ---
server_process = None

def run_server():
    """
    Runs the TypeScript server, automatically using WSL on Windows.
    This function now streams the server's output in real-time.
    """
    global server_process

    # Determine the script's directory and set it as the CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"--- Working directory set to: {script_dir} ---")

    if not os.path.exists(SERVER_SCRIPT_PATH):
        print(f"\nERROR: Server script not found at '{SERVER_SCRIPT_PATH}'")
        sys.exit(1)

    command = []
    
    # --- Command Construction ---
    # Construct the appropriate command based on the operating system.
    if platform.system() == "Windows":
        print("\n--- Detected Windows. Preparing to launch server via WSL. ---")
        print("    NOTE: Node.js, npm, and tsx must be installed in your WSL distribution.")
        
        # This is the key change. By not using '-e', we invoke the user's default
        # WSL shell (e.g., bash), which correctly loads the PATH environment
        # variable from .bashrc or .zshrc, allowing it to find 'npx'.
        # The command arguments are passed directly to WSL.
        command = ['wsl', 'npx', 'tsx', SERVER_SCRIPT_PATH]
    else:
        print("\n--- Detected Linux/macOS. Launching server directly. ---")
        command = ['npx', 'tsx', SERVER_SCRIPT_PATH]

    print(f"Executing command: {' '.join(command)}\n")

    try:
        # Launch the subprocess with stdout/stderr piped so we can stream it.
        # bufsize=1 enables line-buffered output.
        # universal_newlines=True decodes output as text.
        server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirect stderr to stdout for combined output
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream the server's output to the console in real-time
        for line in iter(server_process.stdout.readline, ''):
            sys.stdout.write(line)
        
        # Wait for the process to finish and get the exit code
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
        # The process is still running, terminate it
        server_process.terminate()
        try:
            # Give it a moment to terminate gracefully
            server_process.wait(timeout=5)
            print("✅ Server shut down.")
        except subprocess.TimeoutExpired:
            # If it doesn't respond, force kill it
            print("Server did not respond to terminate, forcing kill.")
            server_process.kill()
            print("✅ Server killed.")
    sys.exit(0)

def main():
    """Main execution function."""
    # Register the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    run_server()

if __name__ == '__main__':
    main()