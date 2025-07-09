#!/usr/bin/env python3
"""
Orchestrator for the CS2 Data Collection Pipeline.

This script manages a two-phase process:
1. Data Generation: Runs `demo_extract/extract.py` on demo files to produce SQLite databases.
2. Video Recording: Runs `recording/record2.py` using the generated databases to record video clips.

It supports parallel execution for both phases to improve performance.
This script should be located in the project's root directory.

GRACEFUL SHUTDOWN (Ctrl+C) LOGIC:
1. The main process catches SIGINT and calls `signal_handler`.
2. `signal_handler` ONLY sets a global `SHUTDOWN_EVENT`. It does not terminate processes.
3. Worker processes are configured to IGNORE SIGINT. This prevents them from dying immediately.
4. The `run_subprocess` function in each worker periodically checks `SHUTDOWN_EVENT`.
5. If the event is set, `run_subprocess` sends SIGINT to its specific child process
   (e.g., `extract.py`), allowing it to perform its own cleanup.
6. The worker waits for its child to exit, then the worker itself exits.
7. The `main()` function, which is waiting on `pool.join()` or `proc.join()`, unblocks
   and the script exits cleanly.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import threading
from pathlib import Path
from multiprocessing import Pool, Manager, Event, Process
import queue # Added for queue.Empty exception

# --- Third-party libraries that need to be installed: ---
# pip install requests
try:
    import requests
except ImportError:
    print("Error: 'requests' library not found. Please install it using 'pip install requests'", file=sys.stderr)
    sys.exit(1)

# --- Globals ---
HTTP_SERVER_URL = "http://localhost:8080"
SHUTDOWN_EVENT = Event()
ACTIVE_PROCESSES = [] # Kept for tracking/debugging purposes

# --- Script Path Resolution ---
SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACT_SCRIPT_PATH = SCRIPT_DIR / "demo_extract" / "extract.py"
RECORD_SCRIPT_PATH = SCRIPT_DIR / "recording" / "record2.py"


def initialize_worker_pool():
    """Initializer for processes within a Pool to make them ignore SIGINT."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def signal_handler(signum, frame):
    """
    Gracefully handle Ctrl+C by setting a global shutdown event.
    This function's ONLY job is to set the event. The main loop will handle cleanup.
    """
    print("\n\n! CTRL+C DETECTED! INITIATING GRACEFUL SHUTDOWN...\n"
          "! Workers will finish their current task and then exit. Please wait.\n", flush=True)
    SHUTDOWN_EVENT.set()


def run_subprocess(command_list, worker_prefix):
    """
    Runs a command in a new process group, streaming its output in a separate thread,
    while remaining responsive to a shutdown event.
    """
    def stream_output(pipe, prefix):
        try:
            for line in iter(pipe.readline, ''):
                print(f"{prefix} {line.strip()}", flush=True)
        finally:
            pipe.close()

    print(f"{worker_prefix} Starting command: {' '.join(map(str, command_list))}", flush=True)

    child_env = os.environ.copy()
    child_env['PYTHONUTF8'] = '1'

    # This is critical for correct signal handling.
    # On Windows, CREATE_NEW_PROCESS_GROUP detaches the child from the parent's
    # console, preventing it from receiving the initial Ctrl+C event.
    # On POSIX, preexec_fn=os.setsid does the same by putting the child in a new session.
    # This gives our orchestrator full control over when to send the shutdown signal.
    popen_kwargs = {
        'stdout': subprocess.PIPE,
        'stderr': subprocess.STDOUT,
        'text': True,
        'encoding': 'utf-8',
        'errors': 'replace',
        'env': child_env,
    }
    if sys.platform == "win32":
        popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs['preexec_fn'] = os.setsid

    process = subprocess.Popen(command_list, **popen_kwargs)

    output_thread = threading.Thread(target=stream_output, args=(process.stdout, worker_prefix))
    output_thread.daemon = True
    output_thread.start()

    while process.poll() is None:
        if SHUTDOWN_EVENT.is_set():
            print(f"{worker_prefix} Shutdown detected. Sending signal to child process {process.pid}...", flush=True)
            if sys.platform == "win32":
                # Send Ctrl+C event on Windows
                process.send_signal(signal.CTRL_C_EVENT)
            else:
                # Send SIGINT to the entire process group on POSIX
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
            break
        time.sleep(0.1)

    return_code = process.wait()
    output_thread.join()

    if SHUTDOWN_EVENT.is_set():
        print(f"{worker_prefix} Child process terminated due to shutdown request.", flush=True)
    elif return_code == 0:
        print(f"{worker_prefix} Command finished successfully.", flush=True)
    else:
        print(f"{worker_prefix} ERROR: Command failed with exit code {return_code}.", flush=True)

    return return_code


def extract_worker(task_queue, datadir, extract_script_path):
    """Worker function for the extraction phase. (Runs inside a Pool)."""
    worker_id = os.getpid()
    prefix = f"[EXTRACT-WORKER-{worker_id}]"

    while not SHUTDOWN_EVENT.is_set():
        try:
            demo_path = task_queue.get(timeout=0.5) # Use a timeout to remain responsive
            if demo_path is None: break

            db_filename = demo_path.stem + '.db'
            db_path = datadir / db_filename

            command = [
                sys.executable, str(extract_script_path),
                "--demo", str(demo_path), "--out", str(db_path)
            ]
            run_subprocess(command, prefix)

        except queue.Empty:
            # Queue is empty, but we might be shutting down. Loop again to check event.
            continue
        except Exception as e:
            print(f"{prefix} Worker error: {e}", flush=True)
            break


def record_worker(task_queue, datadir, recdir, override_level, client_id, record_script_path):
    """Worker function for the recording phase (for single Process instances)."""
    # CRITICAL: A worker process created via `multiprocessing.Process` does not
    # use the Pool's initializer. We must manually tell this worker process
    # to ignore SIGINT so that it doesn't die instantly upon Ctrl+C.
    # This allows it to check the SHUTDOWN_EVENT and perform a graceful exit.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    prefix = f"[RECORD-WORKER-ID-{client_id}]"

    while not SHUTDOWN_EVENT.is_set():
        try:
            # Use a timeout to remain responsive to the shutdown event
            demo_path = task_queue.get(timeout=0.5)

            db_filename = demo_path.stem + '.db'
            db_path = datadir / db_filename

            print(f"{prefix} Processing demo: {demo_path.name}", flush=True)

            command = [
                sys.executable, str(record_script_path),
                "--id", str(client_id), "--demofile", str(demo_path),
                "--sql", str(db_path), "--out", str(recdir),
                "--override", str(override_level)
            ]
            run_subprocess(command, prefix)
        except queue.Empty:
            # Queue is empty, our job is done.
            break
        except Exception as e:
            print(f"{prefix} Unhandled worker error: {e}", flush=True)
            break


def get_available_clients():
    """Queries the HTTP server to get a list of available recording client IDs."""
    list_url = f"{HTTP_SERVER_URL}/list"
    try:
        print(f"\n> Querying recording server at {list_url} for available clients...")
        response = requests.get(list_url, timeout=5)
        response.raise_for_status()
        clients = response.json()
        if not isinstance(clients, list) or not clients:
            print("! ERROR: Server responded, but no available clients found. Response:", clients, file=sys.stderr)
            return []
        print(f"> Found {len(clients)} available recording clients: {clients}")
        return clients
    except requests.exceptions.RequestException as e:
        print(f"! ERROR: Could not connect to the recording server at {HTTP_SERVER_URL}.", file=sys.stderr)
        print(f"! Details: {e}", file=sys.stderr)
        return []


def main():
    if not EXTRACT_SCRIPT_PATH.is_file():
        print(f"FATAL ERROR: `extract.py` not found at expected path: {EXTRACT_SCRIPT_PATH}", file=sys.stderr)
        sys.exit(1)
    if not RECORD_SCRIPT_PATH.is_file():
        print(f"FATAL ERROR: `record2.py` not found at expected path: {RECORD_SCRIPT_PATH}", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Orchestrator for the CS2 data collection pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--demodir", required=True, type=Path, help="Directory containing .dem files.")
    parser.add_argument("--datadir", required=True, type=Path, help="Directory to store generated .db files.")
    parser.add_argument("--recdir", required=True, type=Path, help="Directory to store final video folders.")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers for data extraction.")
    parser.add_argument("--override", type=int, choices=[0, 1, 2], default=0, help="Override level for re-recording (passed to record2.py).")
    parser.add_argument("--no_data_gen", action="store_true", help="Skip data generation and only run recording for demos with existing .db files.")
    args = parser.parse_args()

    # Set up signal handling
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    args.demodir.mkdir(exist_ok=True)
    args.datadir.mkdir(exist_ok=True)
    args.recdir.mkdir(exist_ok=True)

    print(f"Project Directory: {SCRIPT_DIR}")
    print(f"Extractor Script:  {EXTRACT_SCRIPT_PATH}")
    print(f"Recorder Script:   {RECORD_SCRIPT_PATH}")

    all_demo_files = sorted(list(args.demodir.glob("*.dem")))
    if not all_demo_files:
        print(f"No .dem files found in {args.demodir}. Exiting.")
        return

    try:
        # PHASE 1: DATA GENERATION
        print("\n" + "="*50 + "\n### PHASE 1: DATA GENERATION ###\n" + "="*50)
        if not args.no_data_gen:
            demos_to_extract = [p for p in all_demo_files if not (args.datadir / (p.stem + '.db')).exists()]
            if not demos_to_extract:
                print("> All demo files already have a corresponding .db file.")
            else:
                print(f"> Queuing {len(demos_to_extract)} demos for data extraction.")
                with Manager() as manager:
                    task_queue = manager.Queue()
                    for demo in demos_to_extract: task_queue.put(demo)
                    # Add sentinels for workers to know when to stop
                    for _ in range(args.workers): task_queue.put(None)

                    with Pool(processes=args.workers, initializer=initialize_worker_pool) as pool:
                        ACTIVE_PROCESSES.append(pool)
                        tasks = [(task_queue, args.datadir, EXTRACT_SCRIPT_PATH) for _ in range(args.workers)]
                        pool.starmap(extract_worker, tasks)
                        # The 'with' statement handles pool.close() and pool.join()
                if pool in ACTIVE_PROCESSES: ACTIVE_PROCESSES.remove(pool)
        else:
            print("> --no_data_gen is set. Skipping data generation.")

        if SHUTDOWN_EVENT.is_set(): return
        print("\n### PHASE 1 COMPLETE ###")

        # PHASE 2: VIDEO RECORDING
        print("\n" + "="*50 + "\n### PHASE 2: VIDEO RECORDING ###\n" + "="*50)
        available_clients = get_available_clients()
        if not available_clients: return

        demos_to_record = [p for p in all_demo_files if (args.datadir / (p.stem + '.db')).exists()]
        if not demos_to_record:
            print("> No demos with corresponding .db files found. Nothing to record.")
        else:
            print(f"> Queuing {len(demos_to_record)} demos for video recording across {len(available_clients)} workers.")
            with Manager() as manager:
                task_queue = manager.Queue()
                for demo in demos_to_record: task_queue.put(demo)

                processes = []
                for client_id in available_clients:
                    # Note: `Process` does NOT take an `initializer` argument.
                    # The signal handling is done inside the target function `record_worker`.
                    proc = Process(
                        target=record_worker,
                        args=(task_queue, args.datadir, args.recdir, args.override, client_id, RECORD_SCRIPT_PATH)
                    )
                    processes.append(proc)
                    ACTIVE_PROCESSES.append(proc)
                    proc.start()

                for proc in processes:
                    proc.join() # Wait for each process to complete
                for proc in processes:
                    if proc in ACTIVE_PROCESSES: ACTIVE_PROCESSES.remove(proc)

        if SHUTDOWN_EVENT.is_set(): return
        print("\n### PHASE 2 COMPLETE ###")

    except KeyboardInterrupt:
        # This block will run after the signal_handler sets the event.
        # The main loop's .join() calls will be interrupted, or the workers
        # will terminate gracefully, causing the joins to complete.
        print("\n> Main process interrupted. Waiting for running tasks to gracefully shutdown...", file=sys.stderr)
    finally:
        # Restore the original signal handler before exiting.
        signal.signal(signal.SIGINT, original_sigint_handler)
        if SHUTDOWN_EVENT.is_set():
            print("\n! Pipeline shutdown was triggered by user.", file=sys.stderr)
            sys.exit(1)

    print("\n" + "="*50 + "\n>>> Orchestration finished successfully. <<<\n" + "="*50)

if __name__ == '__main__':
    # 'spawn' is a safer start method, especially for cross-platform apps.
    # It's the default on Windows and macOS.
    from multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass # Already set
    main()