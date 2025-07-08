#!/usr/bin/env python3
"""
Orchestrator for the CS2 Data Collection Pipeline.
This version implements a robust, non-blocking graceful shutdown mechanism.
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
import requests

# --- Globals ---
HTTP_SERVER_URL = "http://localhost:8080"
SHUTDOWN_EVENT = Event()
# No longer need ACTIVE_PROCESSES, shutdown is managed by the main loop

# --- Script Path Resolution ---
SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACT_SCRIPT_PATH = SCRIPT_DIR / "demo_extract" / "extract.py"
RECORD_SCRIPT_PATH = SCRIPT_DIR / "recording" / "record2.py"


def initialize_worker():
    """Initializer for all worker processes to make them ignore SIGINT (Ctrl+C)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def signal_handler(signum, frame):
    """
    The simplest possible signal handler. Its only job is to set the shutdown event.
    The main program loop will handle the rest.
    """
    print("\n\n! CTRL+C DETECTED! INITIATING GRACEFUL SHUTDOWN (please wait for workers to finish)...", flush=True)
    SHUTDOWN_EVENT.set()


def run_subprocess(command_list, worker_prefix):
    """
    Runs a command, streaming its output in a separate thread, while remaining
    responsive to a shutdown event in the main thread.
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

    process = subprocess.Popen(
        command_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=child_env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )

    output_thread = threading.Thread(target=stream_output, args=(process.stdout, worker_prefix))
    output_thread.daemon = True
    output_thread.start()

    while process.poll() is None:
        if SHUTDOWN_EVENT.is_set():
            print(f"{worker_prefix} Shutdown detected. Sending signal to child process {process.pid}...", flush=True)
            if sys.platform == "win32":
                process.send_signal(signal.CTRL_C_EVENT)
            else:
                process.send_signal(signal.SIGINT)
            break
        time.sleep(0.1)

    return_code = process.wait()
    output_thread.join()

    if return_code == 0:
        print(f"{worker_prefix} Command finished successfully.", flush=True)
    elif not SHUTDOWN_EVENT.is_set():
        print(f"{worker_prefix} ERROR: Command failed with exit code {return_code}.", flush=True)

    return return_code


def extract_worker(task_queue, datadir, extract_script_path):
    """Worker function for the extraction phase."""
    worker_id = os.getpid()
    prefix = f"[EXTRACT-WORKER-{worker_id}]"
    
    while not SHUTDOWN_EVENT.is_set():
        try:
            demo_path = task_queue.get(timeout=1)
            if demo_path is None: break
            
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
    """Worker function for the recording phase (for single Process instances)."""
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

    # Set the master signal handler
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
                manager = Manager()
                task_queue = manager.Queue()
                for demo in demos_to_extract: task_queue.put(demo)
                for _ in range(args.workers): task_queue.put(None)
                
                pool = Pool(processes=args.workers, initializer=initialize_worker)
                tasks = [(task_queue, args.datadir, EXTRACT_SCRIPT_PATH) for _ in range(args.workers)]
                pool.starmap(extract_worker, tasks)
                pool.close()
                pool.join()
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
            manager = Manager()
            task_queue = manager.Queue()
            for demo in demos_to_record: task_queue.put(demo)

            processes = []
            # Apply initializer to each single Process worker
            worker_initializer = initialize_worker
            for client_id in available_clients:
                proc = Process(
                    target=record_worker,
                    args=(task_queue, args.datadir, args.recdir, args.override, client_id, RECORD_SCRIPT_PATH),
                    initializer=worker_initializer
                )
                processes.append(proc)
                proc.start()

            for proc in processes:
                proc.join()
        
        if SHUTDOWN_EVENT.is_set(): return
        print("\n### PHASE 2 COMPLETE ###")

    except KeyboardInterrupt:
        print("\n> Main process received KeyboardInterrupt. Exiting.", file=sys.stderr)
    finally:
        if SHUTDOWN_EVENT.is_set():
            print("> Shutdown due to signal.", file=sys.stderr)
            sys.exit(1)

    print("\n" + "="*50 + "\n>>> Orchestration finished successfully. <<<\n" + "="*50)

if __name__ == '__main__':
    import queue
    # Needed for compatibility on different OS
    from multiprocessing import set_start_method
    try: set_start_method('spawn')
    except RuntimeError: pass
    main()