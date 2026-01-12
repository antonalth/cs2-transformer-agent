#!/usr/bin/env python3
import argparse
import sys
import os
import re
import time
from datetime import datetime

# Avoid importing local 'wandb' folder if it exists
if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())

try:
    import wandb
except ImportError:
    print("Error: wandb module not found. Please install it or run inside the docker container.")
    sys.exit(1)

def get_latest_runs(api, project=None, entity=None, limit=5):
    """Fetches the most recent runs."""
    target_project = project if project else "cs2-behavior-cloning"
    try:
        runs = api.runs(path=f"{entity}/{target_project}" if entity else target_project)
        # Convert to list and sort
        run_list = list(runs)
        run_list.sort(key=lambda x: x.created_at, reverse=True)
        return run_list[:limit]
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return []

def get_latest_running_run(api, project=None, entity=None):
    """Finds the most recently started run that is currently running."""
    runs = get_latest_runs(api, project, entity, limit=20)
    running_runs = [r for r in runs if r.state == "running"]
    return running_runs[0] if running_runs else None

def match_metrics(keys, patterns):
    """Returns keys that match any of the regex patterns."""
    if not patterns:
        return sorted(list(keys))
    
    matches = []
    for k in keys:
        for p in patterns:
            if re.search(p, k, re.IGNORECASE):
                matches.append(k)
                break
    return sorted(list(set(matches)))

def print_run_info(run):
    print(f"Connected to Run: {run.name} ({run.id})")
    print(f"State: {run.state} | URL: {run.url}")
    print("-" * 60)

def display_summary(run, filter_patterns, brief=False):
    summary = run.summary
    
    if brief:
        keys = ["train/loss", "val/loss", "train/lr", "train/step", "_step"]
        keys = [k for k in keys if k in summary]
    else:
        keys = match_metrics(summary.keys(), filter_patterns)
    
    if not keys:
        print("No matching summary metrics found.")
        return

    print("Summary Metrics:")
    max_len = max([len(k) for k in keys])
    keys.sort()
    
    for k in keys:
        val = summary[k]
        if isinstance(val, float):
            val_str = f"{val:.6f}"
        else:
            val_str = str(val)
        print(f"{k.ljust(max_len)} : {val_str}")

def display_history(run, filter_patterns, last_n, show_all=False):
    history = run.history() 
    
    if isinstance(history, list):
        if not history:
            print("No history data available yet.")
            return
        
        all_keys = set()
        for row in history:
            all_keys.update(row.keys())
        
        cols = match_metrics(list(all_keys), filter_patterns)
        if '_step' in all_keys:
            history.sort(key=lambda x: x.get('_step', 0))

        data = history if show_all else history[-last_n:]
        
        display_cols = [c for c in cols]
        if '_step' not in display_cols and '_step' in all_keys:
            display_cols.insert(0, '_step')
        
        widths = {c: max(len(c), 8) for c in display_cols}
        formatted_data = []
        for row in data:
            f_row = {}
            for c in display_cols:
                val = row.get(c, "")
                val_s = f"{val:.6f}" if isinstance(val, float) else str(val)
                widths[c] = max(widths[c], len(val_s))
                f_row[c] = val_s
            formatted_data.append(f_row)

        header = "  ".join([c.ljust(widths[c]) for c in display_cols])
        print(header)
        print("-" * len(header))
        for row in formatted_data:
            print("  ".join([row[c].ljust(widths[c]) for c in display_cols]))

    else:
        # Pandas logic (fallback)
        if history.empty:
            print("No history data available yet.")
            return
        cols = match_metrics(history.columns, filter_patterns)
        if '_step' in cols or '_step' in history.columns:
             history = history.sort_values('_step')
        data = history if show_all else history.tail(last_n)
        display_cols = [c for c in cols if c in data.columns]
        if '_step' not in display_cols and '_step' in data.columns:
            display_cols.insert(0, '_step')
        print(data[display_cols].to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Monitor WandB runs from CLI")
    parser.add_argument("--project", type=str, default="cs2-behavior-cloning", help="WandB project name")
    parser.add_argument("--entity", type=str, default=None, help="WandB entity/username")
    parser.add_argument("--run-id", type=str, default=None, help="Specific run ID to query")
    parser.add_argument("--last", type=int, default=10, help="Number of history lines to show")
    parser.add_argument("--history", action="store_true", help="Show history data")
    parser.add_argument("--filter", type=str, default=None, help="Comma-separated regex patterns")
    parser.add_argument("--all", action="store_true", help="Show all history data")
    parser.add_argument("--watch", type=int, const=30, nargs='?', help="Watch mode (default 30s)")
    parser.add_argument("--val", action="store_true", help="Shortcut for validation metrics")
    parser.add_argument("--brief", action="store_true", help="Show only key metrics")
    parser.add_argument("--list", action="store_true", help="List recent runs and exit")
    
    args = parser.parse_args()
    api = wandb.Api()

    if args.list:
        print(f"Recent runs in {args.project}:")
        for r in get_latest_runs(api, args.project, args.entity, limit=10):
            print(f" - {r.id}: {r.name} [{r.state}] (Started: {r.created_at})")
        return

    filter_patterns = args.filter.split(",") if args.filter else []
    if args.val:
        filter_patterns.append("val/")

    def run_once():
        if args.run_id:
            try:
                run = api.run(f"{args.entity}/{args.project}/{args.run_id}" if args.entity else f"{args.project}/{args.run_id}")
            except:
                print(f"Could not find run {args.run_id}")
                return False
        else:
            run = get_latest_running_run(api, args.project, args.entity)
        
        if not run:
            print("No active runs found.")
            return False

        if not args.watch:
            print_run_info(run)
        else:
            # Clear screen for watch mode
            sys.stdout.write("\033[H\033[J")
            print(f"Watching Run: {run.name} ({run.id}) | {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 60)

        if args.history:
            display_history(run, filter_patterns, args.last, args.all)
        else:
            display_summary(run, filter_patterns, args.brief)
        return True

    if args.watch:
        try:
            while True:
                if not run_once():
                    break
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nWatch stopped.")
    else:
        run_once()

if __name__ == "__main__":
    main()