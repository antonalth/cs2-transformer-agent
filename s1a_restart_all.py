import subprocess
import time
import os

def main():
    try:
        # Run stop_cs2.bat
        subprocess.run(r".\stop_cs2.bat", check=False)

        # Wait for 1 second
        time.sleep(1)

        # Run start_cs2.bat
        subprocess.run(r".\start_cs2.bat", check=True)

        # Run WSL command
        wsl_command = (
            'wsl bash -c "cd /home/unknonw/advancedfx-main/misc/mirv-script && '
            'node dist/4-advanced-websockets/server.js"'
        )
        subprocess.run(wsl_command, shell=True, check=True)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
