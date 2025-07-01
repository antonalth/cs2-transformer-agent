import os
import subprocess
import sys
import argparse

# --- CONFIGURATION SECTION ---

# 1. ENVIRONMENT NAME
ENV_NAME = "cs2_ai_env"

# 2. PYTHON VERSION - Using 3.11 for performance and broad support.
PYTHON_VERSION = "3.11"

# 3. PYTORCH INSTALLATION COMMAND for CUDA 12.x
PYTORCH_COMMAND = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"

# 4. TENSORRT WHEEL FILE CONFIGURATION
TENSORRT_WHEEL_FILENAME = "tensorrt-10.10.0.31-cp311-none-win_amd64.whl"
TENSORRT_PYTHON_WHEEL_PATH = f"C:\\tools\\TensorRT-10.10.0.31\\python\\{TENSORRT_WHEEL_FILENAME}"

# 5. LIST OF OTHER REQUIRED PACKAGES
OTHER_PIP_PACKAGES = [
    "transformers==4.41.2",
    "onnx",
    "opencv-python",
    "onnxruntime-gpu",
    "tqdm"  # For progress bars
]

# --- END OF CONFIGURATION ---


def run_command(command, description):
    """
    Runs a command, captures and prints its output, and exits on failure.
    This version is more robust for seeing command output.
    """
    print(f"\n--- {description} ---")
    print(f"Executing: {command}")
    try:
        # Use capture_output=True to grab stdout/stderr from the process
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8' # Be explicit about encoding for Windows
        )
        # Manually print the captured output
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            # Print stderr as well, as some tools print status here
            print(result.stderr.strip())

    except subprocess.CalledProcessError as e:
        # If the command fails, print all its output for easy debugging
        print(f"\n[FATAL ERROR] Command failed with exit code {e.returncode}")
        if e.stdout:
            print("--- STDOUT ---")
            print(e.stdout.strip())
        if e.stderr:
            print("--- STDERR ---")
            print(e.stderr.strip())
        print("----------------")
        print("Aborting script.")
        sys.exit(1)

def run_in_env(env_name, command, description):
    """Runs a command inside the specified conda environment using 'conda run'."""
    full_command = f"conda run -n {env_name} {command}"
    run_command(full_command, description)


def setup_environment():
    """Creates the conda environment and installs all packages."""
    print(">>> Starting Environment Setup for Python 3.11 <<<")

    if not os.path.exists(TENSORRT_PYTHON_WHEEL_PATH):
        print(f"\n[FATAL ERROR] TensorRT wheel file not found at path: {TENSORRT_PYTHON_WHEEL_PATH}")
        sys.exit(1)

    env_list_proc = subprocess.run("conda env list", shell=True, capture_output=True, text=True)
    if f"\n{ENV_NAME} " not in env_list_proc.stdout:
        run_command(f"conda create -n {ENV_NAME} python={PYTHON_VERSION} -y", f"Creating Conda environment '{ENV_NAME}'")
    else:
        print(f"\n--- Conda environment '{ENV_NAME}' already exists. Skipping creation. ---")

    # The rest of the installation process...
    run_in_env(ENV_NAME, PYTORCH_COMMAND, "Installing PyTorch for CUDA 12.x")
    pip_install_command = "pip install " + " ".join(OTHER_PIP_PACKAGES)
    run_in_env(ENV_NAME, pip_install_command, "Installing other required packages")
    
    tensorrt_dir = os.path.dirname(TENSORRT_PYTHON_WHEEL_PATH)
    tensorrt_filename = os.path.basename(TENSORRT_PYTHON_WHEEL_PATH)
    pip_command_for_trt = f"pip install {tensorrt_filename}"
    full_conda_command = f'conda run --cwd "{tensorrt_dir}" -n {ENV_NAME} {pip_command_for_trt}'
    run_command(full_conda_command, "Installing TensorRT Python bindings")

    # Basic and Detailed Verification Steps
    verify_installation()

    print("\n\n--- SETUP COMPLETE ---")
    print(f"To activate your new environment, open a new terminal and run:")
    print(f"conda activate {ENV_NAME}")

def verify_installation():
    """Runs all verification checks inside the conda environment."""
    # This is the Python script to be executed
    # Use python -c to run the script. The quotes must be handled carefully.
    # The outer quotes for the shell, inner quotes for python strings.
    command_to_run = f"python check_env.py"
    run_in_env(
        ENV_NAME,
        command_to_run,
        "Verifying installation and available GPUs"
    )
    
def cleanup_environment():
    """Removes the conda environment."""
    print(f">>> Starting Environment Cleanup for '{ENV_NAME}' <<<")
    confirm = input(f"Are you sure you want to permanently remove the conda environment '{ENV_NAME}'? (y/n): ")
    if confirm.lower() != 'y':
        print("Cleanup cancelled.")
        return
    run_command(f"conda env remove -n {ENV_NAME} -y", f"Removing Conda environment '{ENV_NAME}'")
    print("\n--- CLEANUP COMPLETE ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup or Cleanup the Conda environment for the CS2 AI project.")
    parser.add_argument('--cleanup', action='store_true', help='If set, removes the Conda environment instead of setting it up.')
    args = parser.parse_args()
    if args.cleanup:
        cleanup_environment()
    else:
        setup_environment()