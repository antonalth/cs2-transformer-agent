#!/usr/bin/env python3
"""
Orchestrator for the CS2 Data Collection Pipeline.

This script manages a two-phase process:
1. Data Generation: Runs `demo_extract/extract.py` on demo files to produce SQLite databases.
2. Video Recording: Runs `recording/record2.py` using the generated databases to record video clips.

It supports parallel execution for both phases. When Ctrl+C is pressed, the script
will terminate all running workers and their child processes before exiting. This
relies on the default OS behavior for signal propagation and requires child scripts
to handle KeyboardInterrupt gracefully. This script should be located in the
project's root directory.
"""

import argparse
import os
import subprocess
import sys
import threading
import queue
from pathlib import Path
from multiprocessing import Pool, Manager, Process

# --- Third-party libraries that need to be installed: ---
# pip install requests
try:
    import requests
except ImportError:
    print("Error: 'requests' library not found. Please install it using 'pip install requests'", file=sys.stderr)
    sys.exit(1)

# --- Globals ---
HTTP_SERVER_URL = "http://localhost:8080"

# --- Script Path Resolution ---
SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACT_SCRIPT_PATH = SCRIPT_DIR / "demo_extract" / "extract.py"
RECORD_SCRIPT_PATH = SCRIPT_DIR / "recording" / "record2.py"


def run_subprocess(command_list, worker_prefix):
    """
    Runs a command, streaming its output, and allows Ctrl+C to pass through.
    This function blocks until the command completes or is interrupted.
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
        'stdout': subprocess.PIPE,
        'stderr': subprocess.STDOUT,
        'text': True,
        'encoding': 'utf-8',
        'errors': 'replace',
        'env': child_env,
    }

    process = subprocess.Popen(command_list, **popen_kwargs)

    output_thread = threading.Thread(target=stream_output, args=(process.stdout, worker_prefix))
    output_thread.daemon = True
    output_thread.start()

    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        print(f"{worker_prefix} Interrupted by user. Terminating child process...", flush=True)
        process.terminate() # Ensure the child is terminated
        process.wait()      # Wait for it to die
        raise               # Re-raise the exception to signal the worker to stop

    output_thread.join()

    # The child script should handle its own KeyboardInterrupt traceback.
    # The orchestrator just reports the final status.
    if return_code == 0:
        print(f"{worker_prefix} Command finished successfully.", flush=True)
    else:
        print(f"{worker_prefix} Command finished with a non-zero exit code: {return_code}.", flush=True)

    return return_code


def extract_worker(demo_path, datadir, extract_script_path):
    """Worker function for the extraction phase. Processes one demo file."""
    worker_id = os.getpid()
    prefix = f"[EXTRACT-WORKER-{worker_id}]"

    try:
        db_filename = demo_path.stem + '.db'
        db_path = datadir / db_filename
        command = [
            sys.executable, str(extract_script_path),
            "--demo", str(demo_path), "--out", str(db_path)
        ]
        run_subprocess(command, prefix)
    except KeyboardInterrupt:
        # This worker was interrupted. Simply return to allow the pool to shut down cleanly.
        return
    except Exception as e:
        print(f"{prefix} Worker error on demo {demo_path.name}: {e}", flush=True)


def extract_worker_wrapper(args):
    """Helper function to unpack arguments for use with pool.imap_unordered."""
    return extract_worker(*args)


def record_worker(task_queue, datadir, recdir, override_level, client_id, record_script_path):
    """Worker function for the recording phase. Drains tasks from a queue."""
    prefix = f"[RECORD-WORKER-ID-{client_id}]"
    while True:
        try:
            demo_path = task_queue.get_nowait()
        except queue.Empty:
            print(f"{prefix} No more tasks. Worker exiting.", flush=True)
            break
        except (KeyboardInterrupt, SystemExit):
            break

        try:
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
        except KeyboardInterrupt:
            print(f"{prefix} Interrupted during task. Worker exiting.", flush=True)
            break
        except Exception as e:
            print(f"{prefix} Unhandled error processing {demo_path.name}: {e}", flush=True)


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

    # --- PHASE 1: DATA GENERATION ---
    print("\n" + "="*50 + "\n### PHASE 1: DATA GENERATION ###\n" + "="*50)
    if not args.no_data_gen:
        demos_to_extract = [p for p in all_demo_files if not (args.datadir / (p.stem + '.db')).exists()]
        if not demos_to_extract:
            print("> All demo files already have a corresponding .db file.")
        else:
            print(f"> Queuing {len(demos_to_extract)} demos for data extraction.")
            tasks = [(demo, args.datadir, EXTRACT_SCRIPT_PATH) for demo in demos_to_extract]
            try:
                # Use `imap_unordered` which is cleanly interruptible. We must iterate
                # through its results for the tasks to actually be processed.
                with Pool(processes=args.workers) as pool:
                    # On Ctrl+C, this loop is broken, the `with` block exits,
                    # and the pool is correctly terminated without starting new tasks.
                    for _ in pool.imap_unordered(extract_worker_wrapper, tasks):
                        pass
            except KeyboardInterrupt:
                print("\n! CTRL+C DETECTED! Terminating data extraction workers...", file=sys.stderr)
                # The 'with Pool' context manager handles pool.terminate() automatically on any exit from the block.
                sys.exit(1)
    else:
        print("> --no_data_gen is set. Skipping data generation.")
    print("\n### PHASE 1 COMPLETE ###")

    # --- PHASE 2: VIDEO RECORDING ---
    print("\n" + "="*50 + "\n### PHASE 2: VIDEO RECORDING ###\n" + "="*50)
    available_clients = get_available_clients()
    if not available_clients:
        sys.exit(1)

    demos_to_record = [p for p in all_demo_files if (args.datadir / (p.stem + '.db')).exists()]
    if not demos_to_record:
        print("> No demos with corresponding .db files found. Nothing to record.")
    else:
        print(f"> Queuing {len(demos_to_record)} demos for video recording across {len(available_clients)} workers.")
        processes = []
        try:
            with Manager() as manager:
                task_queue = manager.Queue()
                for demo in demos_to_record:
                    task_queue.put(demo)

                for client_id in available_clients:
                    proc = Process(
                        target=record_worker,
                        args=(task_queue, args.datadir, args.recdir, args.override, client_id, RECORD_SCRIPT_PATH)
                    )
                    processes.append(proc)
                    proc.start()

                # Wait for all processes to complete
                for p in processes:
                    p.join()

        except KeyboardInterrupt:
            print("\n! CTRL+C DETECTED! Terminating recording workers...", file=sys.stderr)
            for p in processes:
                if p.is_alive():
                    p.terminate() # Send SIGTERM to the worker process
            for p in processes:
                p.join() # Wait for the process to die
            sys.exit(1)

    print("\n### PHASE 2 COMPLETE ###")
    print("\n" + "="*50 + "\n>>> Orchestration finished successfully. <<<\n" + "="*50)

if __name__ == '__main__':
    from multiprocessing import set_start_method
    try:
        # 'spawn' is a safer start method, especially on macOS and Windows.
        set_start_method('spawn')
    except RuntimeError:
        # This will be raised if the start method has already been set.
        pass
    main()