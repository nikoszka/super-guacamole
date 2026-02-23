#!/usr/bin/env python3
"""
W&B Run Manager — list, categorize, and clean up wandb runs (cloud + local).

Organizes runs into: Size (Small/Large/XLarge) -> Dataset -> Model.
Works with the W&B cloud API and/or local wandb run directories.

Usage — cloud (default):
    python manage_wandb_runs.py                            # List cloud runs
    python manage_wandb_runs.py --delete-unfinished        # Delete unfinished from cloud
    python manage_wandb_runs.py --delete-unfinished --yes  # Skip confirmation

Usage — local:
    python manage_wandb_runs.py --local                    # List local run dirs
    python manage_wandb_runs.py --local --clean-local      # Delete unfinished local dirs
    python manage_wandb_runs.py --local --clean-local -y   # Skip confirmation
    python manage_wandb_runs.py --local --organize         # Reorganize into Size/Dataset/Model/
    python manage_wandb_runs.py --local --organize -y      # Skip confirmation

Filters (work with both modes):
    --size Small|Large|XLarge    --family Llama|Qwen|Mistral
    --dataset trivia_qa          --status finished|crashed|failed|killed
"""

import argparse
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Model categorisation (shared with compute_multi_model_auroc.py logic)
# ---------------------------------------------------------------------------

def parse_model_info(model_name):
    name_lower = model_name.lower()

    if "llama" in name_lower:
        family = "Llama"
    elif "qwen" in name_lower:
        family = "Qwen"
    elif "mistral" in name_lower or "mixtral" in name_lower:
        family = "Mistral"
    else:
        family = "Other"

    if "1b" in name_lower or "1.5b" in name_lower:
        size = "Small"
    elif "7b" in name_lower or "8b" in name_lower:
        size = "Large"
    elif "70b" in name_lower or "72b" in name_lower:
        size = "XLarge"
    else:
        size = "Unknown"

    quantization = None
    if "8bit" in name_lower:
        quantization = "8-bit"
    elif "4bit" in name_lower:
        quantization = "4-bit"

    return family, size, quantization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_duration(seconds):
    if seconds is None or seconds < 0:
        return "-"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def format_size(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def dir_size(path):
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


STATUS_SYMBOLS = {
    "finished": "OK",
    "running": "..",
    "crashed": "XX",
    "failed": "XX",
    "killed": "KL",
}

SIZE_ORDER = {"Small": 0, "Large": 1, "XLarge": 2, "Unknown": 3}


# ---------------------------------------------------------------------------
# Cloud mode — fetch from W&B API
# ---------------------------------------------------------------------------

def fetch_cloud_runs(entity, project):
    import wandb
    api = wandb.Api()
    path = f"{entity}/{project}"
    print(f"Fetching runs from {path} ...")
    runs = list(api.runs(path))
    print(f"Found {len(runs)} runs.\n")
    return runs


def enrich_cloud_run(run):
    config = run.config or {}
    model_name = config.get("model_name", "unknown")
    dataset = config.get("dataset", "unknown")
    experiment_lot = config.get("experiment_lot", "")
    family, size, quantization = parse_model_info(model_name)

    duration = None
    if run.summary and "_wandb" in run.summary:
        runtime = run.summary["_wandb"].get("runtime")
        if runtime is not None:
            duration = runtime

    created = run.created_at if hasattr(run, "created_at") else None

    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "model_name": model_name,
        "dataset": dataset,
        "family": family,
        "size": size,
        "quantization": quantization,
        "experiment_lot": experiment_lot,
        "duration": duration,
        "created": created,
        "local_dir": None,
        "disk_size": None,
        "run_obj": run,
    }


# ---------------------------------------------------------------------------
# Local mode — scan wandb run directories on disk
# ---------------------------------------------------------------------------

def detect_wandb_dir():
    """Try common local wandb directory locations."""
    candidates = [
        Path("src/boldis/uncertainty/wandb"),
        Path("src/nikos/uncertainty/wandb"),
        Path("wandb"),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def infer_local_state(run_dir):
    """Determine if a local run finished by checking for output artifacts."""
    files_dir = run_dir / "files"
    if not files_dir.is_dir():
        return "crashed"
    has_generations = (files_dir / "validation_generations.pkl").exists()
    has_summary = (files_dir / "wandb-summary.json").exists()
    if has_generations and has_summary:
        return "finished"
    if has_summary:
        return "failed"
    return "crashed"


def scan_local_runs(wandb_dir):
    wandb_path = Path(wandb_dir)
    if not wandb_path.is_dir():
        print(f"Error: directory not found: {wandb_dir}")
        sys.exit(1)

    runs = []
    for entry in sorted(wandb_path.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("run-"):
            continue

        parts = entry.name.split("-")
        run_id = parts[-1] if len(parts) >= 3 else entry.name

        config = {}
        config_file = entry / "files" / "config.yaml"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    raw = yaml.safe_load(f) or {}
                config = {k: v.get("value", v) if isinstance(v, dict) else v
                          for k, v in raw.items()}
            except Exception:
                pass

        model_name = config.get("model_name", "unknown")
        dataset = config.get("dataset", "unknown")
        experiment_lot = config.get("experiment_lot", "")
        family, size, quantization = parse_model_info(model_name)
        state = infer_local_state(entry)
        disk = dir_size(entry)

        timestamp = "-".join(parts[1:-1]) if len(parts) >= 3 else ""

        runs.append({
            "id": run_id,
            "name": entry.name,
            "state": state,
            "model_name": model_name,
            "dataset": dataset,
            "family": family,
            "size": size,
            "quantization": quantization,
            "experiment_lot": experiment_lot,
            "duration": None,
            "created": timestamp.replace("_", " "),
            "local_dir": str(entry),
            "disk_size": disk,
            "run_obj": None,
        })

    print(f"Scanned {wandb_path} — found {len(runs)} local run(s).\n")
    return runs


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_filters(runs, *, size=None, family=None, dataset=None, status=None):
    filtered = runs
    if size:
        filtered = [r for r in filtered if r["size"].lower() == size.lower()]
    if family:
        filtered = [r for r in filtered if r["family"].lower() == family.lower()]
    if dataset:
        filtered = [r for r in filtered if r["dataset"].lower() == dataset.lower()]
    if status:
        filtered = [r for r in filtered if r["state"].lower() == status.lower()]
    return filtered


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_runs(runs, *, local_mode=False):
    tree = defaultdict(lambda: defaultdict(list))
    for r in runs:
        tree[r["size"]][r["dataset"]].append(r)

    sorted_sizes = sorted(tree.keys(), key=lambda s: SIZE_ORDER.get(s, 99))

    counts = defaultdict(int)
    total_disk = 0
    for r in runs:
        counts[r["state"]] += 1
        if r["disk_size"]:
            total_disk += r["disk_size"]

    sep = "-" * 100

    for size in sorted_sizes:
        datasets = tree[size]
        print(f"\n{'=' * 100}")
        print(f"  {size} Models")
        print(f"{'=' * 100}")

        for ds in sorted(datasets.keys()):
            ds_runs = datasets[ds]
            print(f"\n  Dataset: {ds}")
            print(f"  {sep}")
            if local_mode:
                header = f"  {'Status':<8} {'Model':<34} {'Run ID':<12} {'Disk':<10} {'Timestamp':<20}"
            else:
                header = f"  {'Status':<8} {'Model':<34} {'Run ID':<12} {'Duration':<10} {'Created':<20}"
            print(header)
            print(f"  {sep}")

            for r in sorted(ds_runs, key=lambda x: (x["family"], x["model_name"])):
                sym = STATUS_SYMBOLS.get(r["state"], "??")
                quant = f" ({r['quantization']})" if r["quantization"] else ""
                model_display = f"{r['model_name']}{quant}"
                if local_mode:
                    disk_str = format_size(r["disk_size"]) if r["disk_size"] else "-"
                    ts = r["created"] or "-"
                    print(
                        f"  [{sym}]   {model_display:<34} {r['id']:<12} "
                        f"{disk_str:<10} {ts}"
                    )
                else:
                    created_str = r["created"][:19] if r["created"] else "-"
                    print(
                        f"  [{sym}]   {model_display:<34} {r['id']:<12} "
                        f"{format_duration(r['duration']):<10} {created_str}"
                    )

    print(f"\n{'=' * 100}")
    print("  Summary")
    print(f"{'=' * 100}")
    parts = [f"{state}: {n}" for state, n in sorted(counts.items())]
    summary = f"  Total: {len(runs)}  |  " + "  |  ".join(parts)
    if local_mode and total_disk:
        summary += f"  |  Disk: {format_size(total_disk)}"
    print(summary)
    print()


# ---------------------------------------------------------------------------
# Deletion — cloud
# ---------------------------------------------------------------------------

def delete_unfinished_cloud(runs, skip_confirm=False):
    unfinished = [r for r in runs if r["state"] != "finished"]
    if not unfinished:
        print("All cloud runs are finished — nothing to delete.")
        return

    print(f"\n{len(unfinished)} cloud run(s) did NOT finish:\n")
    for r in unfinished:
        quant = f" ({r['quantization']})" if r["quantization"] else ""
        print(
            f"  [{r['state']:<8}]  {r['model_name']}{quant:<30}  "
            f"{r['dataset']:<12}  {r['id']}"
        )

    print()
    if not skip_confirm:
        answer = input(f"Delete these {len(unfinished)} cloud run(s)? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    print()
    for r in unfinished:
        try:
            r["run_obj"].delete()
            print(f"  Deleted {r['id']}  ({r['model_name']} / {r['dataset']})")
        except Exception as e:
            print(f"  FAILED  {r['id']}  — {e}")

    print(f"\nDone. Deleted {len(unfinished)} cloud run(s).")


# ---------------------------------------------------------------------------
# Deletion — local
# ---------------------------------------------------------------------------

def clean_local(runs, skip_confirm=False):
    unfinished = [r for r in runs if r["state"] != "finished" and r["local_dir"]]
    if not unfinished:
        print("All local runs are finished — nothing to clean.")
        return

    total_bytes = sum(r["disk_size"] or 0 for r in unfinished)
    print(f"\n{len(unfinished)} local run dir(s) did NOT finish "
          f"({format_size(total_bytes)} on disk):\n")
    for r in unfinished:
        quant = f" ({r['quantization']})" if r["quantization"] else ""
        disk_str = format_size(r["disk_size"]) if r["disk_size"] else "-"
        print(
            f"  [{r['state']:<8}]  {r['model_name']}{quant:<30}  "
            f"{r['dataset']:<12}  {r['id']}  {disk_str}"
        )

    print()
    if not skip_confirm:
        answer = input(
            f"Delete these {len(unfinished)} local dir(s) ({format_size(total_bytes)})? [y/N] "
        ).strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    print()
    deleted = 0
    for r in unfinished:
        try:
            shutil.rmtree(r["local_dir"])
            print(f"  Removed {r['local_dir']}")
            deleted += 1
        except Exception as e:
            print(f"  FAILED  {r['local_dir']}  — {e}")

    print(f"\nDone. Removed {deleted} local dir(s), freed ~{format_size(total_bytes)}.")


# ---------------------------------------------------------------------------
# Organize — move local dirs into Size/Dataset/Model/ hierarchy
# ---------------------------------------------------------------------------

def organize_local(runs, wandb_dir, skip_confirm=False):
    """Move run dirs into a Size/Dataset/Model/ folder structure."""
    movable = [r for r in runs if r["local_dir"]]
    if not movable:
        print("No local runs to organize.")
        return

    wandb_path = Path(wandb_dir)
    plan = []
    for r in movable:
        src = Path(r["local_dir"])
        dest_dir = wandb_path / r["size"] / r["dataset"] / r["model_name"]
        dest = dest_dir / src.name
        if src == dest:
            continue
        plan.append((r, src, dest_dir, dest))

    if not plan:
        print("All runs are already organized — nothing to move.")
        return

    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r, _src, _dest_dir, _dest in plan:
        tree[r["size"]][r["dataset"]][r["model_name"]].append(r["id"])

    print(f"\nWill reorganize {len(plan)} run dir(s) into:\n")
    for size in sorted(tree, key=lambda s: SIZE_ORDER.get(s, 99)):
        print(f"  {size}/")
        for ds in sorted(tree[size]):
            print(f"    {ds}/")
            for model in sorted(tree[size][ds]):
                ids = tree[size][ds][model]
                print(f"      {model}/  ({len(ids)} run(s))")
    print()

    if not skip_confirm:
        answer = input(f"Move {len(plan)} run dir(s)? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    print()
    moved = 0
    for r, src, dest_dir, dest in plan:
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            print(f"  {src.name}  ->  {dest.relative_to(wandb_path)}")
            moved += 1
        except Exception as e:
            print(f"  FAILED  {src.name}  — {e}")

    print(f"\nDone. Moved {moved} run dir(s).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="List, categorize, and clean up W&B runs (cloud + local).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_argument_group("mode")
    mode.add_argument(
        "--local", action="store_true",
        help="Scan local wandb run directories instead of the cloud API",
    )
    mode.add_argument(
        "--wandb-dir", type=str, default=None,
        help="Path to local wandb directory (auto-detected if omitted)",
    )

    cloud = parser.add_argument_group("cloud options")
    cloud.add_argument(
        "--entity", default="nikosteam", help="W&B entity (default: nikosteam)",
    )
    cloud.add_argument(
        "--project", default="super_guacamole", help="W&B project (default: super_guacamole)",
    )

    filters = parser.add_argument_group("filters")
    filters.add_argument("--size", help="Filter: Small, Large, XLarge")
    filters.add_argument("--family", help="Filter: Llama, Qwen, Mistral")
    filters.add_argument("--dataset", help="Filter: trivia_qa, squad, coqa, ...")
    filters.add_argument("--status", help="Filter: finished, crashed, failed, running, killed")

    actions = parser.add_argument_group("actions")
    actions.add_argument(
        "--delete-unfinished", action="store_true",
        help="Delete unfinished runs from cloud",
    )
    actions.add_argument(
        "--clean-local", action="store_true",
        help="Delete local run directories that did not finish",
    )
    actions.add_argument(
        "--organize", action="store_true",
        help="Reorganize local run dirs into Size/Dataset/Model/ hierarchy",
    )
    actions.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompts",
    )

    args = parser.parse_args()

    # --- Gather runs ---
    is_local = args.local or args.clean_local or args.organize
    if is_local:
        wandb_dir = args.wandb_dir
        if not wandb_dir:
            detected = detect_wandb_dir()
            if not detected:
                print("Error: could not auto-detect wandb directory. Use --wandb-dir.")
                sys.exit(1)
            wandb_dir = str(detected)
        runs = scan_local_runs(wandb_dir)
    else:
        wandb_dir = None
        raw = fetch_cloud_runs(args.entity, args.project)
        runs = [enrich_cloud_run(r) for r in raw]

    runs = apply_filters(
        runs, size=args.size, family=args.family, dataset=args.dataset, status=args.status,
    )

    if not runs:
        print("No runs match the given filters.")
        sys.exit(0)

    print_runs(runs, local_mode=is_local)

    # --- Actions ---
    if args.delete_unfinished:
        delete_unfinished_cloud(runs, skip_confirm=args.yes)

    if args.clean_local:
        clean_local(runs, skip_confirm=args.yes)

    if args.organize:
        organize_local(runs, wandb_dir, skip_confirm=args.yes)


if __name__ == "__main__":
    main()
