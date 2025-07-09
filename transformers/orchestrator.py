#!/usr/bin/env python3
"""
Orchestrator for the CS2 Data Collection Pipeline.

This script manages a two-phase process:
1. Data Generation: Runs `demo_extract/extract.py` on demo files to produce SQLite databases.
2. Video Recording: Runs `recording/record2.py` using the generated databases to record video clips.

It supports parallel execution and features a robust, graceful shutdown mechanism.
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

try:
    import requests
except ImportError:
    print("Error: 'requests' library not found. Please install it using 'pip install requests'", file=sys.stderr)
    sys.exit(1)

# --- Globals ---
# This global is for the main process's signal handler. Workers will get this
# object passed to them as an argument.
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
    """Gracefully handle Ctrl+C by setting the global shutdown event."""
    print("\n\n! CTRL+C DETECTED! INITIATING GRACEFUL SHUTDOWN...\n"
          "! Workers will finish their current task and then exit. Please wait.\n", flush=True)
    SHUTDOWN_EVENT.set()


def run_subprocess(command_list, worker_prefix, shutdown_event): # MODIFIED: Accepts event
    """
    Runs a command, streaming its output, while remaining responsive to a shutdown event.
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
    
    popen_kwargs = {
        'stdout': subprocess.PIPE, 'stderr': subprocess.STDOUT, 'text': True,
        'encoding': 'utf-8', 'errors': 'replace', 'env': child_env,
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
        # MODIFIED: Checks the passed-in event object, not a global one.
        if shutdown_event.is_set():
            print(f"{worker_prefix} Shutdown detected. Sending signal to child process {process.pid}...", flush=True)
            if sys.platform == "win32":
                process.send_signal(signal.CTRL_C_EVENT)
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
            break
        time.sleep(0.1)

    return_code = process.wait()
    output_thread.join()

    # MODIFIED: Checks the passed-in event object.
    if shutdown_event.is_set():
        print(f"{worker_prefix} Child process terminated due to shutdown request.", flush=True)
    elif return_code == 0:
        print(f"{worker_prefix} Command finished successfully.", flush=True)
    else:
        print(f"{worker_prefix} ERROR: Command failed with exit code {return_code}.", flush=True)

    return return_code


# MODIFIED: Accepts shutdown_event as an argument
def extract_worker(task_queue, datadir, extract_script_path, shutdown_event):
    """Worker function for the extraction phase."""
    worker_id = os.getpid()
    prefix = f"[EXTRACT-WORKER-{worker_id}]"

    # MODIFIED: Checks the passed-in event object.
    while not shutdown_event.is_set():
        try:
            demo_path = task_queue.get(timeout=1)
            if demo_path is None: break

            db_filename = demo_path.stem + '.db'
            db_path = datadir / db_filename
            command = [
                sys.executable, str(extract_script_path),
                "--demo", str(demo_path), "--out", str(db_path)
            ]
            # MODIFIED: Passes the event down to the subprocess runner.
            run_subprocess(command, prefix, shutdown_event)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"{prefix} Worker error: {e}", flush=True)
            break


# MODIFIED: Accepts shutdown_event as an argument
def record_worker(task_queue, datadir, recdir, override_level, client_id, record_script_path, shutdown_event):
    """Worker function for the recording phase."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    prefix = f"[RECORD-WORKER-ID-{client_id}]"

    # MODIFIED: Checks the passed-in event object.
    while not shutdown_event.is_set():
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
            # MODIFIED: Passes the event down to the subprocess runner.
            run_subprocess(command, prefix, shutdown_event)
        except queue.Empty:
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
    # ... (Path checking and argument parsing is unchanged) ...
    if not EXTRACT_SCRIPT_PATH.is_file(): sys.exit(f"FATAL ERROR: `extract.py` not found at expected path: {EXTRACT_SCRIPT_PATH}")
    if not RECORD_SCRIPT_PATH.is_file(): sys.exit(f"FATAL ERROR: `record2.py` not found at expected path: {RECORD_SCRIPT_PATH}")
    parser = argparse.ArgumentParser(description="Orchestrator for the CS2 data collection pipeline.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
                        # MODIFIED: Pass the SHUTDOWN_EVENT to each worker.
                        tasks = [(task_queue, args.datadir, EXTRACT_SCRIPT_PATH, SHUTDOWN_EVENT) for _ in range(args.workers)]
                        async_result = pool.starmap_async(extract_worker, tasks)
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
                    # MODIFIED: Pass the SHUTDOWN_EVENT to each worker process.
                    proc_args = (task_queue, args.datadir, args.recdir, args.override, client_id, RECORD_SCRIPT_PATH, SHUTDOWN_EVENT)
                    proc = Process(target=record_worker, args=proc_args)
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