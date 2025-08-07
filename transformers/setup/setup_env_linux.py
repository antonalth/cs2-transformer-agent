#!/usr/bin/env python3
import os
import subprocess
import sys
import argparse

#TODO BEFORE:
#install cuda correctly (use .run cuda installer, fails once to disable original driver)
#download tensorrt tarfile, add lib folder to LD_LIB local variable
#test with nvidia-smi + nvcc --version

# --- CONFIGURATION SECTION ---

# 1. ENVIRONMENT NAME
ENV_NAME = "cs2_ai_env"

# 2. PYTHON VERSION - Using 3.11 for performance and broad support.
PYTHON_VERSION = "3.12"

# 3. PYTORCH INSTALLATION COMMAND for CUDA 12.x
# This command is cross-platform and works for Linux with CUDA 12.1
PYTORCH_COMMAND = "pip3 install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"

# --- ACTION REQUIRED for TENSORRT ---
# 4. TENSORRT WHEEL FILE CONFIGURATION FOR LINUX
#    a. Download the TensorRT TAR file for your Linux distribution and CUDA version
#       from NVIDIA's developer website: https://developer.nvidia.com/tensorrt
#    b. Extract the archive, e.g., `tar -xzvf TensorRT-10.x.y.z.Linux.x86_64-gnu.cuda-12.x.tar.gz`
#    c. Locate the Python wheel file inside the extracted directory, typically in the `python/` subfolder.
#       Its name will be similar to 'tensorrt-10.x.y.z-cp311-none-linux_x86_64.whl'.
#    d. UPDATE THE PATH BELOW to the full, absolute path of that .whl file.

# https://nvidia.github.io/TensorRT-LLM/installation/linux.html
#sudo apt-get -y install libopenmpi-dev && pip3 install --upgrade pip setuptools && pip3 install tensorrt_llm #for tenssort_LLM (benchmark script for now)

TENSORRT_WHEEL_PATH = "/home/unknown/TensorRT-10.10.0.31/python/tensorrt-10.10.0.31-cp312-none-linux_x86_64.whl"

# 5. LIST OF OTHER REQUIRED PACKAGES
OTHER_PIP_PACKAGES = [
    "lmdb",
    "msgpack",
    "flash-attn",
    "transformers",
    "onnx",
    "matplotlib",
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
            encoding='utf-8'
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
    print(">>> Starting Environment Setup for Python 3.12 on Linux <<<")

    # Verify the user has updated the TensorRT path
    if "/path/to/your/" in TENSORRT_WHEEL_PATH:
        print("\n[FATAL ERROR] Please edit the `TENSORRT_WHEEL_PATH` variable in this script.")
        print("You must provide the full path to your downloaded TensorRT .whl file.")
        sys.exit(1)

    if not os.path.exists(TENSORRT_WHEEL_PATH):
        print(f"\n[FATAL ERROR] TensorRT wheel file not found at the specified path: {TENSORRT_WHEEL_PATH}")
        print("Please ensure the path is correct and the file exists.")
        sys.exit(1)

    # Check if the environment already exists
    env_list_proc = subprocess.run("conda env list", shell=True, capture_output=True, text=True)
    # Using a space after the env name ensures we don't match a substring, e.g., "myenv" matching "myenv_old"
    if f"\n{ENV_NAME} " not in env_list_proc.stdout and not env_list_proc.stdout.startswith(f"{ENV_NAME} "):
        run_command(f"conda create -n {ENV_NAME} python={PYTHON_VERSION} -y", f"Creating Conda environment '{ENV_NAME}'")
    else:
        print(f"\n--- Conda environment '{ENV_NAME}' already exists. Skipping creation. ---")

    # Install packages
    run_in_env(ENV_NAME, PYTORCH_COMMAND, "Installing PyTorch for CUDA 12.x")
    
    pip_install_command = "pip install " + " ".join(OTHER_PIP_PACKAGES)
    run_in_env(ENV_NAME, pip_install_command, "Installing other required packages")
    
    # Install TensorRT from the specified wheel file path
    # pip can handle a full path to a wheel file directly.
    pip_command_for_trt = f'pip install "{TENSORRT_WHEEL_PATH}"'
    run_in_env(ENV_NAME, pip_command_for_trt, "Installing TensorRT Python bindings")

    # Basic and Detailed Verification Steps
    verify_installation()

    print("\n\n--- SETUP COMPLETE ---")
    print(f"To activate your new environment, open a new terminal and run:")
    print(f"conda activate {ENV_NAME}")

def verify_installation():
    """Runs all verification checks inside the conda environment."""
    # This assumes a 'check_env.py' script exists in the same directory.
    print("\n--- Verifying installation (requires check_env.py) ---")
    command_to_run = "python check_env.py"
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