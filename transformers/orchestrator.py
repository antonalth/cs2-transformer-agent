#!/usr/bin/env python3
"""
Orchestrator for the CS2 Data Collection Pipeline.

This script manages a two-phase process:
1. Data Generation: Runs `demo_extract/extract.py` on demo files to produce SQLite databases.
2. Video Recording: Runs `recording/record2.py` using the generated databases to record video clips.

It supports parallel execution for both phases and features a robust, graceful shutdown
mechanism that works reliably on both Windows and POSIX-based systems.
This script should be located in the project's root directory.

GRACEFUL SHUTDOWN (Ctrl+C) LOGIC:
1. The main process catches SIGINT via a signal handler. The main thread is kept
   responsive by using non-blocking waits (e.g., `async_result.wait(timeout=...)`).
2. The handler's ONLY job is to set a global `SHUTDOWN_EVENT`.
3. Worker processes (spawned by Pool or Process) are configured to IGNORE SIGINT.
   This prevents them from dying immediately and allows them to obey the SHUTDOWN_EVENT.
4. The child scripts (`extract.py`, `record2.py`) are spawned in a new process
   group/session, shielding them from the initial Ctrl+C broadcast.
5. The `run_subprocess` function in each worker periodically checks SHUTDOWN_EVENT.
   If set, it sends a SIGINT/Ctrl+C to its specific child process.
6. The worker waits for its child to exit, then the worker itself exits cleanly.
7. The main process, which is waiting in its non-blocking loop, sees that all
   workers have finished and exits cleanly.
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
import queue

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

    # This is critical for correct signal handling. It puts the child process
    # in a new group, shielding it from the initial Ctrl+C from the terminal.
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
                process.send_signal(signal.CTRL_C_EVENT)
            else:
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
    """Worker function for the extraction phase (runs inside a Pool)."""
    worker_id = os.getpid()
    prefix = f"[EXTRACT-WORKER-{worker_id}]"

    while not SHUTDOWN_EVENT.is_set():
        try:
            demo_path = task_queue.get(timeout=1)
            if demo_path is None:  # Sentinel value means no more tasks.
                break

            db_filename = demo_path.stem + '.db'
            db_path = datadir / db_filename
            command = [
                sys.executable, str(extract_script_path),
                "--demo", str(demo_path), "--out", str(db_path)
            ]
            run_subprocess(command, prefix)

        except queue.Empty:
            continue # Timeout, loop again to check SHUTDOWN_EVENT
        except Exception as e:
            print(f"{prefix} Worker error: {e}", flush=True)
            break


def record_worker(task_queue, datadir, recdir, override_level, client_id, record_script_path):
    """Worker function for the recording phase (for single Process instances)."""
    # CRITICAL: Manually tell this worker process to ignore SIGINT so that
    # it doesn't die instantly upon Ctrl+C. This is done by the Pool initializer
    # for Pool workers, but must be done here for Process workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    prefix = f"[RECORD-WORKER-ID-{client_id}]"

    while not SHUTDOWN_EVENT.is_set():
        try:
            demo_path = task_queue.get(timeout=1)
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
            break # No more tasks.
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
                    for _ in range(args.workers): task_queue.put(None)

                    with Pool(processes=args.workers, initializer=initialize_worker_pool) as pool:
                        async_result = pool.starmap_async(extract_worker, [(task_queue, args.datadir, EXTRACT_SCRIPT_PATH) for _ in range(args.workers)])
                        while not async_result.ready():
                            async_result.wait(timeout=1)
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
                    proc = Process(
                        target=record_worker,
                        args=(task_queue, args.datadir, args.recdir, args.override, client_id, RECORD_SCRIPT_PATH)
                    )
                    processes.append(proc)
                    proc.start()

                active_procs = list(processes)
                while any(p.is_alive() for p in active_procs):
                    for p in active_procs:
                        p.join(timeout=0.2)

        if SHUTDOWN_EVENT.is_set(): return
        print("\n### PHASE 2 COMPLETE ###")

    except KeyboardInterrupt:
        print("\n> Main process caught KeyboardInterrupt. Pipeline is shutting down.", file=sys.stderr)
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        if SHUTDOWN_EVENT.is_set():
            print("\n! Pipeline shutdown was triggered by user.", file=sys.stderr)
            sys.exit(1)

    print("\n" + "="*50 + "\n>>> Orchestration finished successfully. <<<\n" + "="*50)

if __name__ == '__main__':
    from multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()