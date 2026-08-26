"""
cleanup_benchmark.py
────────────────────
Deletes all benchmark-generated heatmaps, plan files, and waiting-room files
(matching 'bench_' or 'batch_other_') without touching general evaluation plots.

Usage:
    python cleanup_benchmark.py                   # dry run — shows what would be deleted
    python cleanup_benchmark.py --confirm         # actually deletes
    python cleanup_benchmark.py --all --confirm   # force deletes all heatmaps/plans
"""

import os
import sys
import glob
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from airflow_local_settings import PLAN_DIR, WAITING_ROOM_DIR, AIRFLOW_BASE_DIR
#states are in a directory benchmarks_results/states inside the config files
CLUSTER_FILE     = os.path.join(AIRFLOW_BASE_DIR, "config", "benchmark_results", "states", "*.json")



def find_files(delete_all: bool) -> list[str]:
    # 1. Gather files from their respective directories
    heatmaps = glob.glob(os.path.join(PLAN_DIR, "heatmap_*.png"))
    plans    = glob.glob(os.path.join(PLAN_DIR, "plan_*.json"))
    waiting  = glob.glob(os.path.join(WAITING_ROOM_DIR, "waiting_*.json"))
    states   = glob.glob(CLUSTER_FILE)

    candidates = heatmaps + plans + waiting + states

    if delete_all:
        return candidates

    # FIXED: Include BOTH "bench_" and "batch_other_" filename patterns, 
    # ensuring general summary comparison plots (plot_*.png) are never included.
    return [
        f for f in candidates 
        if "bench_" in os.path.basename(f) or "batch_" in os.path.basename(f) or "oracle_" in os.path.basename(f)
    ]


def main():
    parser = argparse.ArgumentParser(description="Clean up benchmark-generated files")
    parser.add_argument("--confirm", action="store_true",
                        help="Actually delete files (default is dry run)")
    parser.add_argument("--all", action="store_true", dest="delete_all",
                        help="Force delete all heatmaps and plans in the directory")
    args = parser.parse_args()

    files = find_files(args.delete_all)
    print(files)
    if not files:
        print("[CLEANUP] Nothing to delete.")
        return

    total_size = sum(os.path.getsize(f) for f in files if os.path.exists(f)) / 1024 / 1024

    print(f"[CLEANUP] Found {len(files)} target file(s) — {total_size:.2f} MB")
    print()

    by_type = {"heatmap": [], "plan": [], "waiting": [], "state": []}
    for f in files:
        name = os.path.basename(f)
        if name.startswith("heatmap_"):       by_type["heatmap"].append(f)
        elif name.startswith("plan_"):        by_type["plan"].append(f)
        elif name.startswith("waiting_"):     by_type["waiting"].append(f)


    for kind, flist in by_type.items():
        if flist:
            print(f"  {kind:8s} ({len(flist):4d} files)")
            for f in flist[:5]:
                print(f"             {os.path.basename(f)}")
            if len(flist) > 5:
                print(f"             ... and {len(flist) - 5} more")

    print()

    if not args.confirm:
        print("[CLEANUP] Dry run — nothing deleted. Add --confirm to actually delete.")
        return

    deleted = 0
    errors  = 0
    for f in files:
        try:
            if os.path.exists(f):
                os.remove(f)
                deleted += 1
        except Exception as e:
            print(f"[ERROR] Could not delete {f}: {e}")
            errors += 1


    #delete the states directory if it exists
    states_dir = os.path.dirname(CLUSTER_FILE)
    if os.path.exists(states_dir):
        try: # removes everything inside the states directory
            for f in os.listdir(states_dir):
                file_path = os.path.join(states_dir, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"[CLEANUP] Deleted all files in states directory: {states_dir}")
        except OSError as e:
            print(f"[CLEANUP] Could not delete states directory (not empty?): {e}")

    print(f"[CLEANUP] Deleted {deleted} file(s). Errors: {errors}.")


if __name__ == "__main__":
    main()