#!/usr/bin/env python3
"""
Orchestrator for the CS2 Data Collection Pipeline.

This script manages a two-phase process:
1. Data Generation: Runs `demo_extract/extract.py` on demo files to produce SQLite databases.
2. Video Recording: Runs `recording/record2.py` using the generated databases to record video clips.

It supports parallel execution for both phases to improve performance.
This script should be located in the project's root directory.
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
ACTIVE_PROCESSES = []

# --- Script Path Resolution ---
SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACT_SCRIPT_PATH = SCRIPT_DIR / "demo_extract" / "extract.py"
RECORD_SCRIPT_PATH = SCRIPT_DIR / "recording" / "record2.py"


def signal_handler(signum, frame):
    """Gracefully handle Ctrl+C by setting an event and terminating processes."""
    # Import type inside handler to avoid scope issues in some contexts
    from multiprocessing.pool import Pool

    print("\n\n! CTRL+C DETECTED! INITIATING GRACEFUL SHUTDOWN...\n", flush=True)
    SHUTDOWN_EVENT.set()
    
    time.sleep(1) # Give workers a moment to see the event
    
    # Use a copy for safe iteration while modifying the list
    for p in list(ACTIVE_PROCESSES):
        if isinstance(p, Pool):
            print(f"! Terminating Pool {p}...", flush=True)
            p.terminate()
        elif isinstance(p, Process):
            print(f"! Terminating Process {p.name}...", flush=True)
            p.terminate()
    
    for p in list(ACTIVE_PROCESSES):
        p.join(timeout=5)
        
    print("\n! All workers terminated. Exiting now.", flush=True)
    # Use os._exit for a more forceful exit from a signal handler
    os._exit(1)


def run_subprocess(command_list, worker_prefix):
    """
    Runs a command, streaming its output in a separate thread, while remaining
    responsive to a shutdown event in the main thread. This ensures graceful
    shutdown propagation.
    """
    def stream_output(pipe, prefix):
        """Reads from a pipe and prints lines with a prefix."""
        try:
            for line in iter(pipe.readline, ''):
                print(f"{prefix} {line.strip()}", flush=True)
        finally:
            pipe.close()

    print(f"{worker_prefix} Starting command: {' '.join(map(str, command_list))}", flush=True)
    
    child_env = os.environ.copy()
    child_env['PYTHONUTF8'] = '1'

    process = subprocess.Popen(
        command_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=child_env
    )

    output_thread = threading.Thread(
        target=stream_output, 
        args=(process.stdout, worker_prefix)
    )
    output_thread.daemon = True
    output_thread.start()

    try:
        # Loop to check for shutdown signal or process completion
        while process.poll() is None:
            if SHUTDOWN_EVENT.is_set():
                print(f"{worker_prefix} Shutdown detected. Sending SIGINT to child process {process.pid}...", flush=True)
                # Send Ctrl+C to the child, which triggers its own cleanup function
                process.send_signal(signal.SIGINT)
                break
            time.sleep(0.1)
    except (KeyboardInterrupt, SystemExit):
        # This worker process itself was interrupted
        print(f"{worker_prefix} Worker interrupted. Terminating child {process.pid}...", flush=True)
        process.terminate()

    return_code = process.wait()
    output_thread.join()

    if return_code == 0:
        print(f"{worker_prefix} Command finished successfully.", flush=True)
    else:
        # On Windows, a SIGINT termination often results in code 1.
        # Check if shutdown was requested to avoid printing a scary error.
        if not SHUTDOWN_EVENT.is_set():
            print(f"{worker_prefix} ERROR: Command failed with exit code {return_code}.", flush=True)

    return return_code


def extract_worker(task_queue, datadir, extract_script_path):
    """Worker function for the extraction phase."""
    worker_id = os.getpid()
    prefix = f"[EXTRACT-WORKER-{worker_id}]"
    
    while not SHUTDOWN_EVENT.is_set():
        try:
            demo_path = task_queue.get(timeout=1)
            if demo_path is None:
                break
            
            db_filename = demo_path.stem + '.db'
            db_path = datadir / db_filename
            
            command = [
                sys.executable, str(extract_script_path),
                "--demo", str(demo_path), "--out", str(db_path)
            ]
            run_subprocess(command, prefix)
            
        except queue.Empty:
            break
        except Exception as e:
            print(f"{prefix} Worker error: {e}", flush=True)
            break


def record_worker(task_queue, datadir, recdir, override_level, client_id, record_script_path):
    """Worker function for the recording phase."""
    prefix = f"[RECORD-WORKER-ID-{client_id}]"

    while not SHUTDOWN_EVENT.is_set():
        try:
            demo_path = task_queue.get_nowait()
            if demo_path is None:
                task_queue.put(None)
                break
            
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

        except Exception:
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
        print(f"! Make sure the 'server.ts' script is running and accessible.", file=sys.stderr)
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
        # ========================================================================
        # PHASE 1: DATA GENERATION (extract.py)
        # ========================================================================
        print("\n" + "="*50)
        print("### PHASE 1: DATA GENERATION ###")
        print("="*50)

        demos_to_extract = []
        if args.no_data_gen:
            print("> --no_data_gen is set. Skipping data generation.")
        else:
            print(f"> Checking {len(all_demo_files)} demos for existing database files...")
            for demo_path in all_demo_files:
                expected_db_path = args.datadir / (demo_path.stem + '.db')
                if not expected_db_path.exists():
                    demos_to_extract.append(demo_path)
            
            if not demos_to_extract:
                print("> All demo files already have a corresponding .db file.")
            else:
                print(f"> Queuing {len(demos_to_extract)} demos for data extraction.")
                
                manager = Manager()
                task_queue = manager.Queue()
                for demo in demos_to_extract:
                    task_queue.put(demo)
                
                for _ in range(args.workers):
                    task_queue.put(None)
                
                pool = Pool(processes=args.workers)
                ACTIVE_PROCESSES.append(pool)

                tasks = [(task_queue, args.datadir, EXTRACT_SCRIPT_PATH) for _ in range(args.workers)]
                pool.starmap(extract_worker, tasks)
                
                pool.close()
                pool.join()
                ACTIVE_PROCESSES.remove(pool)

        if SHUTDOWN_EVENT.is_set(): return
        print("\n### PHASE 1 COMPLETE ###")

        # ========================================================================
        # PHASE 2: VIDEO RECORDING (record2.py)
        # ========================================================================
        print("\n" + "="*50)
        print("### PHASE 2: VIDEO RECORDING ###")
        print("="*50)

        available_clients = get_available_clients()
        if not available_clients:
            print("! Cannot proceed with recording phase. Exiting.", file=sys.stderr)
            return

        demos_to_record = []
        print("> Determining which demos to process for recording...")
        for demo_path in all_demo_files:
            expected_db_path = args.datadir / (demo_path.stem + '.db')
            if expected_db_path.exists():
                demos_to_record.append(demo_path)
            elif not args.no_data_gen:
                print(f"! WARNING: DB file for {demo_path.name} was not generated. It will be skipped.", file=sys.stderr)

        if not demos_to_record:
            print("> No demos with corresponding .db files found. Nothing to record.")
        else:
            print(f"> Queuing {len(demos_to_record)} demos for video recording across {len(available_clients)} workers.")
            
            manager = Manager()
            task_queue = manager.Queue()
            for demo in demos_to_record:
                task_queue.put(demo)

            processes = []
            for client_id in available_clients:
                proc = Process(
                    target=record_worker,
                    args=(task_queue, args.datadir, args.recdir, args.override, client_id, RECORD_SCRIPT_PATH)
                )
                processes.append(proc)
                ACTIVE_PROCESSES.append(proc)
                proc.start()

            for proc in processes:
                proc.join()
            
            for proc in processes:
                if proc in ACTIVE_PROCESSES:
                    ACTIVE_PROCESSES.remove(proc)
        
        if SHUTDOWN_EVENT.is_set(): return
        print("\n### PHASE 2 COMPLETE ###")

    except (KeyboardInterrupt, SystemExit):
        print("\n> Main process interrupted. Pipeline shutting down.", file=sys.stderr)
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        if SHUTDOWN_EVENT.is_set():
            sys.exit(1)

    print("\n" + "="*50)
    print(">>> Orchestration finished successfully. <<<")
    print("="*50)

if __name__ == '__main__':
    import queue
    from multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    main()