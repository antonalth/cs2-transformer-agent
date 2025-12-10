"""
debug.py
------------------------------------------------------------------------
Memory profiling tools for tracking down OOM errors.
"""
import torch
import os
import psutil
import gc
import logging

logger = logging.getLogger("Debug")
_DEBUG_FLAG = False

def enable():
    """Enables debug mode and starts recording CUDA memory history."""
    global _DEBUG_FLAG
    _DEBUG_FLAG = True
    try:
        # Start recording memory history for visualization (pytorch.org/memory_viz)
        torch.cuda.memory._record_memory_history(max_entries=100000)
        logger.info("[DEBUG] Memory recording enabled. Snapshot will be saved on OOM.")
    except AttributeError:
        logger.warning("[DEBUG] torch.cuda.memory._record_memory_history not available.")

def log(tag: str):
    """Prints current GPU/RAM usage."""
    if not _DEBUG_FLAG:
        return
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_alloc = torch.cuda.max_memory_allocated() / 1024**3
    else:
        alloc = reserved = max_alloc = 0.0

    mem = psutil.Process(os.getpid()).memory_info()
    ram = mem.rss / 1024**3
    
    print(f"[MEM] {tag:<25} | Alloc: {alloc:6.2f}GB | Rsrv: {reserved:6.2f}GB | Max: {max_alloc:6.2f}GB | RAM: {ram:6.2f}GB")

def save_snapshot(filename="oom_snapshot.pickle"):
    """Dumps the recorded memory history to a pickle file."""
    if not _DEBUG_FLAG: 
        return
    try:
        print(f"[DEBUG] Saving memory snapshot to {filename}...")
        torch.cuda.memory._dump_snapshot(filename)
        print(f"[DEBUG] Snapshot saved. Upload to https://pytorch.org/memory_viz")
    except Exception as e:
        print(f"[DEBUG] Failed to save snapshot: {e}")