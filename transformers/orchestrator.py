#!/usr/bin/env python3
"""
Orchestrator for the CS2 Data Collection and Processing Pipeline.

This script manages a three-phase process:
1. Data Generation: Runs `extract.py` on demo files to produce SQLite databases.
2. Video Recording: Runs `record2.py` using the generated databases to record video clips.
3. LMDB Generation: Runs `injection_mold.py` to compile recordings and databases into LMDBs.

It supports parallel execution for all phases and handles Ctrl+C for graceful shutdown.
This script should be located in the project's root directory.
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
INJECTION_MOLD_SCRIPT_PATH = SCRIPT_DIR / "transformers" / "injection_mold.py"


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
        process.terminate()
        process.wait()
        raise

    output_thread.join()

    if return_code == 0:
        print(f"{worker_prefix} Command finished successfully.", flush=True)
    else:
        print(f"{worker_prefix} Command finished with a non-zero exit code: {return_code}.", flush=True)

    return return_code


def extract_worker(demo_path, datadir):
    """Worker function for the extraction phase."""
    worker_id = os.getpid()
    prefix = f"[EXTRACT-WORKER-{worker_id}]"
    try:
        db_path = datadir / (demo_path.stem + '.db')
        command = [sys.executable, str(EXTRACT_SCRIPT_PATH), "--demo", str(demo_path), "--out", str(db_path)]
        run_subprocess(command, prefix)
    except KeyboardInterrupt:
        return
    except Exception as e:
        print(f"{prefix} Worker error on demo {demo_path.name}: {e}", flush=True)

def extract_worker_wrapper(args):
    return extract_worker(*args)


def record_worker(task_queue, datadir, recdir, override_level, client_id):
    """Worker function for the recording phase."""
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
            db_path = datadir / (demo_path.stem + '.db')
            print(f"{prefix} Processing demo: {demo_path.name}", flush=True)
            command = [
                sys.executable, str(RECORD_SCRIPT_PATH), "--id", str(client_id),
                "--demofile", str(demo_path), "--sql", str(db_path),
                "--out", str(recdir), "--override", str(override_level)
            ]
            run_subprocess(command, prefix)
        except KeyboardInterrupt:
            print(f"{prefix} Interrupted during task. Worker exiting.", flush=True)
            break
        except Exception as e:
            print(f"{prefix} Unhandled error processing {demo_path.name}: {e}", flush=True)


def injection_mold_worker(demo_path, args):
    """Worker function for the LMDB generation phase."""
    worker_id = os.getpid()
    prefix = f"[INJECT-WORKER-{worker_id}]"
    demo_name = demo_path.stem

    try:
        recdir_path = args.recdir / demo_name
        db_path = args.datadir / (demo_name + '.db')
        lmdb_out_path = args.lmdbpath / (demo_name + '.lmdb')

        command = [
            sys.executable, str(INJECTION_MOLD_SCRIPT_PATH),
            "--recdir", str(recdir_path),
            "--dbfile", str(db_path),
            "--outlmdb", str(lmdb_out_path),
            "--workers", str(args.lmdbworkers),
            "--quality", str(args.lmdbquality)
        ]
        if args.lmdboverwrite:
            command.append("--overwrite")
        if args.lmdboverridesql:
            command.append("--overridesql")
        if args.lmdbblockfile:
            command.extend(["--blockfile", str(args.lmdbblockfile)])

        run_subprocess(command, prefix)
    except KeyboardInterrupt:
        return
    except Exception as e:
        print(f"{prefix} Worker error on demo {demo_name}: {e}", flush=True)

def injection_mold_worker_wrapper(args):
    return injection_mold_worker(*args)


def get_available_clients():
    """Queries the HTTP server to get a list of available recording client IDs."""
    try:
        print(f"\n> Querying recording server at {HTTP_SERVER_URL}/list for available clients...")
        response = requests.get(f"{HTTP_SERVER_URL}/list", timeout=5)
        response.raise_for_status()
        clients = response.json()
        if not isinstance(clients, list) or not clients:
            print("! ERROR: Server responded, but no available clients found.", file=sys.stderr)
            return []
        print(f"> Found {len(clients)} available recording clients: {clients}")
        return clients
    except requests.exceptions.RequestException as e:
        print(f"! ERROR: Could not connect to the recording server at {HTTP_SERVER_URL}. Details: {e}", file=sys.stderr)
        return []


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Orchestrator for the CS2 data collection and processing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Core Paths
    parser.add_argument("--demodir", required=True, type=Path, help="Directory containing .dem files.")
    parser.add_argument("--datadir", required=True, type=Path, help="Directory to store generated .db files.")
    parser.add_argument("--recdir", required=True, type=Path, help="Directory to store final video folders.")
    # Control Flags
    parser.add_argument("--runsteps", type=str, default="123", help="Which pipeline steps to run (e.g., '1', '12', '23', '3').")
    # Phase 1: Extraction
    parser.add_argument("--extractworkers", type=int, default=2, help="Number of parallel workers for data extraction.")
    # Phase 2: Recording
    parser.add_argument("--override", type=int, choices=[0, 1, 2], default=0, help="Override level for re-recording (passed to record2.py).")
    # Phase 3: LMDB Generation
    lmdb_group = parser.add_argument_group('Phase 3: LMDB Generation Parameters')
    lmdb_group.add_argument("--lmdbpath", type=Path, help="Directory to store final LMDB folders. Required for step 3.")
    lmdb_group.add_argument("--step3workers", type=int, default=1, help="Number of parallel injection_mold.py processes to run.")
    lmdb_group.add_argument("--lmdbworkers", type=int, default=5, help="Number of workers for each injection_mold.py instance.")
    lmdb_group.add_argument("--lmdbquality", type=int, default=85, help="JPEG quality for injection_mold.py.")
    lmdb_group.add_argument("--lmdbblockfile", type=Path, help="Path to blockfile for injection_mold.py.")
    lmdb_group.add_argument("--lmdboverwrite", action="store_true", help="Pass --overwrite to injection_mold.py.")
    lmdb_group.add_argument("--lmdboverridesql", action="store_true", help="Pass --overridesql to injection_mold.py.")

    args = parser.parse_args()

    # --- Script and Path Validation ---
    if not EXTRACT_SCRIPT_PATH.is_file():
        print(f"FATAL: `extract.py` not found at: {EXTRACT_SCRIPT_PATH}", file=sys.stderr); sys.exit(1)
    if not RECORD_SCRIPT_PATH.is_file():
        print(f"FATAL: `record2.py` not found at: {RECORD_SCRIPT_PATH}", file=sys.stderr); sys.exit(1)
    if '3' in args.runsteps and not INJECTION_MOLD_SCRIPT_PATH.is_file():
        print(f"FATAL: `injection_mold.py` not found at: {INJECTION_MOLD_SCRIPT_PATH}", file=sys.stderr); sys.exit(1)
    if '3' in args.runsteps and not args.lmdbpath:
        print("FATAL: --lmdbpath is required when running step 3.", file=sys.stderr); sys.exit(1)

    args.demodir.mkdir(exist_ok=True)
    args.datadir.mkdir(exist_ok=True)
    args.recdir.mkdir(exist_ok=True)
    if args.lmdbpath: args.lmdbpath.mkdir(exist_ok=True)

    print(f"Project Directory: {SCRIPT_DIR}\nExtractor Script:  {EXTRACT_SCRIPT_PATH}\nRecorder Script:   {RECORD_SCRIPT_PATH}\nInjector Script:   {INJECTION_MOLD_SCRIPT_PATH}")

    all_demo_files = sorted(list(args.demodir.glob("*.dem")))
    if not all_demo_files:
        print(f"No .dem files found in {args.demodir}. Exiting."); return

    # --- PHASE 1: DATA GENERATION ---
    if '1' in args.runsteps:
        print("\n" + "="*50 + "\n### PHASE 1: DATA GENERATION ###\n" + "="*50)
        demos_to_extract = [p for p in all_demo_files if not (args.datadir / (p.stem + '.db')).exists()]
        if not demos_to_extract:
            print("> All demos already have a .db file.")
        else:
            print(f"> Queuing {len(demos_to_extract)} demos for data extraction.")
            tasks = [(demo, args.datadir) for demo in demos_to_extract]
            try:
                with Pool(processes=args.extractworkers) as pool:
                    for _ in pool.imap_unordered(extract_worker_wrapper, tasks): pass
            except KeyboardInterrupt:
                print("\n! CTRL+C: Terminating data extraction...", file=sys.stderr); sys.exit(1)
        print("\n### PHASE 1 COMPLETE ###")

    # --- PHASE 2: VIDEO RECORDING ---
    if '2' in args.runsteps:
        print("\n" + "="*50 + "\n### PHASE 2: VIDEO RECORDING ###\n" + "="*50)
        available_clients = get_available_clients()
        if not available_clients: sys.exit(1)

        demos_to_record = [p for p in all_demo_files if (args.datadir / (p.stem + '.db')).exists()]
        if not demos_to_record:
            print("> No demos with .db files found to record.")
        else:
            print(f"> Queuing {len(demos_to_record)} demos for recording across {len(available_clients)} clients.")
            processes = []
            try:
                with Manager() as manager:
                    task_queue = manager.Queue()
                    for demo in demos_to_record: task_queue.put(demo)
                    for client_id in available_clients:
                        proc = Process(target=record_worker, args=(task_queue, args.datadir, args.recdir, args.override, client_id))
                        processes.append(proc); proc.start()
                    for p in processes: p.join()
            except KeyboardInterrupt:
                print("\n! CTRL+C: Terminating recording workers...", file=sys.stderr)
                for p in processes:
                    if p.is_alive(): p.terminate()
                for p in processes: p.join()
                sys.exit(1)
        print("\n### PHASE 2 COMPLETE ###")

    # --- PHASE 3: LMDB GENERATION ---
    if '3' in args.runsteps:
        print("\n" + "="*50 + "\n### PHASE 3: LMDB GENERATION ###\n" + "="*50)
        # A demo is ready for injection if its .db and recording folder exist
        demos_to_inject = [p for p in all_demo_files if (args.datadir / (p.stem + '.db')).exists() and (args.recdir / p.stem).is_dir()]
        if not demos_to_inject:
            print("> No demos with required .db and recording folders found.")
        else:
            print(f"> Queuing {len(demos_to_inject)} demos for LMDB generation.")
            tasks = [(demo, args) for demo in demos_to_inject]
            try:
                with Pool(processes=args.step3workers) as pool:
                    for _ in pool.imap_unordered(injection_mold_worker_wrapper, tasks): pass
            except KeyboardInterrupt:
                print("\n! CTRL+C: Terminating LMDB generation...", file=sys.stderr); sys.exit(1)
        print("\n### PHASE 3 COMPLETE ###")


    print("\n" + "="*50 + "\n>>> Orchestration finished successfully. <<<\n" + "="*50)

if __name__ == '__main__':
    from multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()