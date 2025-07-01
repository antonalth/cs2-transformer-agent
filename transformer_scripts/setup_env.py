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
# Using 'pip' is standard inside a specific conda environment.
PYTORCH_COMMAND = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"

# 4. TENSORRT WHEEL FILE CONFIGURATION
TENSORRT_WHEEL_FILENAME = "tensorrt-10.10.0.31-cp311-none-win_amd64.whl"
TENSORRT_PYTHON_WHEEL_PATH = f"C:\\tools\\TensorRT-10.10.0.31\\python\\{TENSORRT_WHEEL_FILENAME}"

# 5. LIST OF OTHER REQUIRED PACKAGES
OTHER_PIP_PACKAGES = [
    "transformers",
    "onnx",
    "onnxruntime-gpu",
    "tqdm"  # For progress bars
]

# --- END OF CONFIGURATION ---


def run_command(command, description):
    """Runs a command directly, used for managing conda itself."""
    print(f"\n--- {description} ---")
    print(f"Executing: {command}")
    try:
        # Using shell=True is necessary for conda commands on Windows
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code {e.returncode}")
        print("Aborting script.")
        sys.exit(1)

def run_in_env(env_name, command, description):
    """Runs a command inside the specified conda environment using 'conda run'."""
    # This is a more robust method for scripting than 'conda activate'
    full_command = f"conda run -n {env_name} {command}"
    run_command(full_command, description)


def setup_environment():
    """Creates the conda environment and installs all packages."""
    print(">>> Starting Environment Setup for Python 3.11 <<<")

    # Critical pre-flight check for TensorRT wheel file
    if not os.path.exists(TENSORRT_PYTHON_WHEEL_PATH):
        print(f"\n[FATAL ERROR] TensorRT wheel file not found at the specified path:")
        print(f"PATH: {TENSORRT_PYTHON_WHEEL_PATH}")
        sys.exit(1)

    # 1. Create Conda environment
    run_command(
        f"conda create -n {ENV_NAME} python={PYTHON_VERSION} -y",
        f"Creating Conda environment '{ENV_NAME}' with Python {PYTHON_VERSION}"
    )

    # 2. Install PyTorch with CUDA
    run_in_env(
        ENV_NAME,
        PYTORCH_COMMAND,
        "Installing PyTorch for CUDA 12.x"
    )

    # 3. Install other packages
    pip_install_command = "pip install " + " ".join(OTHER_PIP_PACKAGES)
    run_in_env(
        ENV_NAME,
        pip_install_command,
        "Installing other required packages (transformers, onnx, etc.)"
    )

    # 4. Install local TensorRT wheel
    # We need to wrap the path in extra quotes for conda run
    tensorrt_install_command = f"pip install \\\"{TENSORRT_PYTHON_WHEEL_PATH}\\\""
    run_in_env(
        ENV_NAME,
        tensorrt_install_command,
        "Installing TensorRT Python bindings for Python 3.11"
    )

    # 5. Verify installation
    verify_command = "python -c \"import torch; import tensorrt; print('\\n--- Verification ---'); print('PyTorch version:', torch.__version__); print('TensorRT version:', tensorrt.__version__); print('CUDA available for PyTorch:', torch.cuda.is_available()); print('--------------------')\""
    run_in_env(
        ENV_NAME,
        verify_command,
        "Verifying installation"
    )

    print("\n\n--- SETUP COMPLETE ---")
    print(f"To activate your new environment, open a new terminal and run:")
    print(f"conda activate {ENV_NAME}")

def cleanup_environment():
    """Removes the conda environment."""
    print(f">>> Starting Environment Cleanup for '{ENV_NAME}' <<<")
    confirm = input(f"Are you sure you want to permanently remove the conda environment '{ENV_NAME}'? (y/n): ")
    if confirm.lower() != 'y':
        print("Cleanup cancelled.")
        return
    run_command(
        f"conda env remove -n {ENV_NAME} -y",
        f"Removing Conda environment '{ENV_NAME}'"
    )
    print("\n--- CLEANUP COMPLETE ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup or Cleanup the Conda environment for the CS2 AI project.")
    parser.add_argument('--cleanup', action='store_true', help='If set, removes the Conda environment instead of setting it up.')
    args = parser.parse_args()
    if args.cleanup:
        cleanup_environment()
    else:
        setup_environment()