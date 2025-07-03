import psutil
import sys

def check_process_psutil(process_name):
    """
    Checks if a process with the given name is running using psutil.

    Args:
        process_name (str): The name of the process (e.g., "notepad.exe", "chrome.exe").

    Returns:
        bool: True if the process is found, False otherwise.
    """
    if not sys.platform.startswith('win'):
        print("This script is intended for Windows. psutil is cross-platform but process names might differ.")
        # You can still proceed, but be aware of platform specifics
        # return False # Or raise an error, depending on desired strictness

    # Ensure the input name is case-insensitive for Windows
    process_name_lower = process_name.lower()

    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'].lower() == process_name_lower:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Handle cases where a process might disappear or be inaccessible during iteration
            continue
    return False

if __name__ == "__main__":
    print("--- Using psutil ---")

    # Example 1: Check for Notepad (likely running if you open it)
    process_to_check_1 = "notepad.exe"
    if check_process_psutil(process_to_check_1):
        print(f"'{process_to_check_1}' is running.")
    else:
        print(f"'{process_to_check_1}' is NOT running.")

    # Example 2: Check for Chrome (if you have it open)
    process_to_check_2 = "chrome.exe"
    if check_process_psutil(process_to_check_2):
        print(f"'{process_to_check_2}' is running.")
    else:
        print(f"'{process_to_check_2}' is NOT running.")

    # Example 3: Check for a non-existent process
    process_to_check_3 = "nonexistentprocess.exe"
    if check_process_psutil(process_to_check_3):
        print(f"'{process_to_check_3}' is running.")
    else:
        print(f"'{process_to_check_3}' is NOT running.")

    print("\n--- Interactive Test ---")
    user_input_process = input("Enter a process name to check (e.g., explorer.exe, cmd.exe): ")
    if check_process_psutil(user_input_process):
        print(f"'{user_input_process}' is running.")
    else:
        print(f"'{user_input_process}' is NOT running.")