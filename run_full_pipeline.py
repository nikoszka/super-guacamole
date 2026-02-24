#!/usr/bin/env python3
"""
Full Long-Answer Experiment Pipeline.

Orchestrates: LLM Judge -> Phase 6 Token Weighting -> POS Analysis -> Dashboard.
Works on the 19 latest runs (1 per model/dataset combo) from the organized
local wandb directory (Size/Dataset/Model/run-*/).

Usage:
    python run_full_pipeline.py                        # Run everything
    python run_full_pipeline.py --step judge            # Only LLM judge
    python run_full_pipeline.py --step phase6           # Only token weighting
    python run_full_pipeline.py --step pos              # Only POS analysis
    python run_full_pipeline.py --step dashboard        # Only visualization
    python run_full_pipeline.py --size Small            # Filter by size
    python run_full_pipeline.py --dataset trivia_qa     # Filter by dataset
    python run_full_pipeline.py --dry-run               # Show what would be done
"""

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WANDB_BASE = Path("src/boldis/uncertainty/wandb")
RESULTS_BASE = Path("results/pipeline")
STATE_FILE = RESULTS_BASE / "pipeline_state.json"
VALID_SIZES = {"Small", "Large", "XLarge"}
WANDB_PROJECT = "super_guacamole_pipeline"
WANDB_ENTITY = "nikosteam"

# ============================================================================
# Run discovery
# ============================================================================

def discover_latest_runs(wandb_base, size_filter=None, dataset_filter=None):
    """Walk Size/Dataset/Model/ tree and pick the latest run per combo."""
    runs = []
    for size_dir in sorted(wandb_base.iterdir()):
        if not size_dir.is_dir() or size_dir.name not in VALID_SIZES:
            continue
        if size_filter and size_dir.name.lower() != size_filter.lower():
            continue
        for ds_dir in sorted(size_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            if dataset_filter and ds_dir.name.lower() != dataset_filter.lower():
                continue
            for model_dir in sorted(ds_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                run_dirs = sorted(
                    [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
                )
                if not run_dirs:
                    continue
                latest = run_dirs[-1]
                pickle_path = latest / "files" / "validation_generations.pkl"
                if not pickle_path.exists():
                    logger.warning("No pickle in %s — skipping", latest)
                    continue
                runs.append({
                    "size": size_dir.name,
                    "dataset": ds_dir.name,
                    "model": model_dir.name,
                    "run_dir": str(latest),
                    "pickle_path": str(pickle_path),
                    "key": f"{size_dir.name}/{ds_dir.name}/{model_dir.name}/{latest.name}",
                })
    return runs


def print_run_table(runs):
    sep = "-" * 90
    print(f"\n{'=' * 90}")
    print(f"  Pipeline will process {len(runs)} run(s)")
    print(f"{'=' * 90}")
    print(f"  {'Size':<8} {'Dataset':<12} {'Model':<32} {'Run ID'}")
    print(f"  {sep}")
    for r in runs:
        run_id = Path(r["run_dir"]).name.split("-")[-1]
        print(f"  {r['size']:<8} {r['dataset']:<12} {r['model']:<32} {run_id}")
    print()


# ============================================================================
# State management
# ============================================================================

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def is_done(state, key, step):
    return state.get(key, {}).get(step) == "done"


def mark_done(state, key, step):
    state.setdefault(key, {})[step] = "done"
    save_state(state)


# ============================================================================
# Step 1: LLM Judge Evaluation
# ============================================================================

def run_judge_step(runs, state, force_rejudge=False):
    """Evaluate all runs with llm_llama-3-8b judge (8-bit), loading the model once."""
    if force_rejudge:
        pending = runs
        for r in runs:
            state.setdefault(r["key"], {}).pop("judge", None)
        save_state(state)
    else:
        pending = [r for r in runs if not is_done(state, r["key"], "judge")]
    if not pending:
        logger.info("Judge step: all %d runs already evaluated.", len(runs))
        return

    logger.info("Judge step: %d/%d runs need evaluation.", len(pending), len(runs))
    logger.info("Loading LLM judge model (Meta-Llama-3-8B, 8-bit) — this may take a minute...")

    src_dir = os.path.join(os.getcwd(), "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from models.huggingface_models import HuggingfaceModel
    from utils.utils import model_based_metric

    judge_model = HuggingfaceModel(
        "Meta-Llama-3-8B-8bit", stop_sequences="default", max_new_tokens=10
    )

    def metric_fn(predicted_answer, example, _model):
        return model_based_metric(predicted_answer, example, judge_model)

    logger.info("Judge model loaded (8-bit, ~8GB VRAM).")

    global_example_idx = 0
    total_examples = len(pending) * 400  # approximate

    for i, run in enumerate(pending, 1):
        run_label = f"{run['size']}/{run['dataset']}/{run['model']}"
        logger.info("[%d/%d] Judging %s ...", i, len(pending), run_label)
        t0 = time.time()
        with open(run["pickle_path"], "rb") as f:
            data = pickle.load(f)

        evaluated = 0
        skipped = 0
        correct = 0
        errors = 0
        for _eid, entry in data.items():
            mla = entry.get("most_likely_answer")
            if mla is None:
                continue
            existing = mla.get("accuracy")
            if not force_rejudge and existing is not None and existing != 0.0:
                skipped += 1
                if existing > 0.5:
                    correct += 1
                continue
            response = mla.get("response", "").strip()
            if not response:
                continue
            try:
                acc = metric_fn(response, entry, None)
                mla["accuracy"] = float(acc)
                evaluated += 1
                if acc > 0.5:
                    correct += 1
            except Exception as e:
                logger.warning("Error evaluating example: %s", e)
                mla["accuracy"] = 0.0
                evaluated += 1
                errors += 1

            global_example_idx += 1
            if wandb.run and global_example_idx % 10 == 0:
                total_done = evaluated + skipped
                wandb.log({
                    "judge/global_example": global_example_idx,
                    "judge/run_progress": i / len(pending),
                    "judge/current_accuracy": correct / max(total_done, 1),
                    "judge/examples_evaluated": evaluated,
                    "judge/examples_skipped": skipped,
                    "judge/errors": errors,
                    "judge/current_run": run_label,
                })

        with open(run["pickle_path"], "wb") as f:
            pickle.dump(data, f)

        elapsed = time.time() - t0
        total_done = evaluated + skipped
        run_accuracy = correct / max(total_done, 1)
        logger.info(
            "  Done: %d evaluated, %d skipped, accuracy=%.1f%% (%.1fs)",
            evaluated, skipped, run_accuracy * 100, elapsed,
        )
        if wandb.run:
            wandb.log({
                "judge/run_completed": i,
                "judge/run_accuracy": run_accuracy,
                "judge/run_evaluated": evaluated,
                "judge/run_skipped": skipped,
                "judge/run_errors": errors,
                "judge/run_time_s": elapsed,
                "judge/run_label": run_label,
            })
        mark_done(state, run["key"], "judge")


# ============================================================================
# Step 2: Phase 6 — Token Weighting Schemes
# ============================================================================

def run_phase6_step(runs, state):
    """Run phase 6 weighting schemes analysis for each run."""
    pending = [r for r in runs if not is_done(state, r["key"], "phase6")]
    if not pending:
        logger.info("Phase 6 step: all %d runs already analysed.", len(runs))
        return

    logger.info("Phase 6 step: %d/%d runs need analysis.", len(pending), len(runs))

    for i, run in enumerate(pending, 1):
        output_dir = RESULTS_BASE / run["size"] / run["dataset"] / run["model"] / "phase6"
        run_label = f"{run['size']}/{run['dataset']}/{run['model']}"
        logger.info("[%d/%d] Phase 6: %s ...", i, len(pending), run_label)
        t0 = time.time()
        cmd = [
            sys.executable,
            "src/analysis/phase6_weighting_schemes_comparison.py",
            "--pickle-path", run["pickle_path"],
            "--model-name", run["model"],
            "--output-dir", str(output_dir),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=3600)
            mark_done(state, run["key"], "phase6")
            elapsed = time.time() - t0
            logger.info("  Phase 6 complete -> %s (%.1fs)", output_dir, elapsed)

            # Log AUROC results to W&B
            csv_path = output_dir / "weighting_schemes_auroc.csv"
            if wandb.run and csv_path.exists():
                import pandas as pd
                df = pd.read_csv(csv_path)
                top5 = df.nlargest(5, "AUROC")
                log_data = {
                    "phase6/run_completed": i,
                    "phase6/run_time_s": elapsed,
                    "phase6/run_label": run_label,
                    "phase6/best_scheme": top5.iloc[0]["Scheme"],
                    "phase6/best_auroc": top5.iloc[0]["AUROC"],
                }
                for _, row in top5.iterrows():
                    log_data[f"phase6/auroc_{row['Scheme']}"] = row["AUROC"]
                wandb.log(log_data)

                # Log per-run AUROC plots as images
                for img_name in ["auroc_comparison.png", "roc_curves_smooth.png"]:
                    img_path = output_dir / img_name
                    if img_path.exists():
                        wandb.log({
                            f"phase6/{run_label}/{img_name}": wandb.Image(str(img_path))
                        })

        except subprocess.CalledProcessError as e:
            logger.error("  Phase 6 FAILED for %s: %s", run["key"], e)
        except subprocess.TimeoutExpired:
            logger.error("  Phase 6 TIMEOUT for %s (>1h)", run["key"])


# ============================================================================
# Step 3: POS Analysis
# ============================================================================

def run_pos_step(runs, state):
    """Run POS tagging analysis for each run."""
    pending = [r for r in runs if not is_done(state, r["key"], "pos")]
    if not pending:
        logger.info("POS step: all %d runs already analysed.", len(runs))
        return

    logger.info("POS step: %d/%d runs need analysis.", len(pending), len(runs))

    for i, run in enumerate(pending, 1):
        output_dir = RESULTS_BASE / run["size"] / run["dataset"] / run["model"] / "pos"
        run_label = f"{run['size']}/{run['dataset']}/{run['model']}"
        logger.info("[%d/%d] POS: %s ...", i, len(pending), run_label)
        t0 = time.time()
        cmd = [
            sys.executable,
            "src/analysis/phase1_7_pos_analysis.py",
            "--pickle-path", run["pickle_path"],
            "--model-name", run["model"],
            "--output-dir", str(output_dir),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=1800)
            mark_done(state, run["key"], "pos")
            elapsed = time.time() - t0
            logger.info("  POS complete -> %s (%.1fs)", output_dir, elapsed)

            if wandb.run:
                log_data = {
                    "pos/run_completed": i,
                    "pos/run_time_s": elapsed,
                    "pos/run_label": run_label,
                }
                for img_name in ["pos_nll_boxplot_token.png", "pos_nll_boxplot_word.png"]:
                    img_path = output_dir / img_name
                    if img_path.exists():
                        log_data[f"pos/{run_label}/{img_name}"] = wandb.Image(str(img_path))
                wandb.log(log_data)

        except subprocess.CalledProcessError as e:
            logger.error("  POS FAILED for %s: %s", run["key"], e)
        except subprocess.TimeoutExpired:
            logger.error("  POS TIMEOUT for %s (>30m)", run["key"])


# ============================================================================
# Step 4: Cross-Model Dashboard
# ============================================================================

def run_dashboard_step(runs):
    """Aggregate all per-run Phase 6 results and create thesis-ready visualizations."""
    import numpy as np

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    except ImportError as e:
        logger.error("Dashboard requires matplotlib, seaborn, pandas: %s", e)
        return

    dashboard_dir = RESULTS_BASE / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    # -- Collect all Phase 6 AUROC CSVs --
    all_rows = []
    for run in runs:
        csv_path = RESULTS_BASE / run["size"] / run["dataset"] / run["model"] / "phase6" / "weighting_schemes_auroc.csv"
        if not csv_path.exists():
            logger.warning("No phase6 CSV for %s — skipping in dashboard", run["key"])
            continue
        df = pd.read_csv(csv_path)
        df["size"] = run["size"]
        df["dataset"] = run["dataset"]
        df["model"] = run["model"]
        all_rows.append(df)

    if not all_rows:
        logger.error("No phase 6 results found — run --step phase6 first.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(dashboard_dir / "all_aurocs.csv", index=False)
    logger.info("Combined %d AUROC rows from %d runs.", len(combined), len(all_rows))

    # Identify top-N schemes globally
    avg_auroc = combined.groupby("Scheme")["AUROC"].mean().sort_values(ascending=False)
    top_schemes = list(avg_auroc.head(10).index)

    # ----- Viz 1: AUROC Heatmap (top 10 schemes x runs) -----
    top_df = combined[combined["Scheme"].isin(top_schemes)].copy()
    top_df["run_label"] = top_df["model"] + "\n" + top_df["dataset"]
    pivot = top_df.pivot_table(index="Scheme", columns="run_label", values="AUROC")
    scheme_order = [s for s in top_schemes if s in pivot.index]
    pivot = pivot.reindex(scheme_order)

    fig, ax = plt.subplots(figsize=(max(14, len(all_rows) * 1.2), 8))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.4, vmax=1.0,
        linewidths=0.5, ax=ax, cbar_kws={"label": "AUROC"},
    )
    ax.set_title("Token Weighting Schemes — AUROC Across Models & Datasets (Top 10)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Weighting Scheme")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    fig.savefig(dashboard_dir / "auroc_heatmap.png", dpi=200)
    plt.close(fig)
    logger.info("Saved auroc_heatmap.png")

    # ----- Viz 2: Best scheme consistency -----
    best_per_run = combined.loc[combined.groupby(["size", "dataset", "model"])["AUROC"].idxmax()]
    best_counts = best_per_run["Scheme"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    best_counts.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Number of runs where this scheme ranks #1")
    ax.set_title("Best Scheme Consistency — How Often Does Each Scheme Win?", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(dashboard_dir / "best_scheme_consistency.png", dpi=200)
    plt.close(fig)
    logger.info("Saved best_scheme_consistency.png")

    # ----- Viz 3: Model family comparison (top scheme AUROC) -----
    def get_family(model_name):
        ml = model_name.lower()
        if "llama" in ml:
            return "Llama"
        if "qwen" in ml:
            return "Qwen"
        if "mistral" in ml:
            return "Mistral"
        return "Other"

    top1_scheme = avg_auroc.index[0]
    top1_df = combined[combined["Scheme"] == top1_scheme].copy()
    top1_df["family"] = top1_df["model"].apply(get_family)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top1_df, x="dataset", y="AUROC", hue="family", ax=ax)
    ax.set_title(f"Model Family Comparison — {top1_scheme} AUROC", fontsize=14, fontweight="bold")
    ax.set_ylabel("AUROC")
    ax.set_xlabel("Dataset")
    ax.legend(title="Model Family")
    plt.tight_layout()
    fig.savefig(dashboard_dir / "model_family_comparison.png", dpi=200)
    plt.close(fig)
    logger.info("Saved model_family_comparison.png")

    # ----- Viz 4: Size effect -----
    top1_by_size = combined[combined["Scheme"] == top1_scheme].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top1_by_size, x="dataset", y="AUROC", hue="size", ax=ax)
    ax.set_title(f"Size Effect — {top1_scheme} AUROC (Small vs Large)", fontsize=14, fontweight="bold")
    ax.set_ylabel("AUROC")
    ax.set_xlabel("Dataset")
    ax.legend(title="Model Size")
    plt.tight_layout()
    fig.savefig(dashboard_dir / "size_effect.png", dpi=200)
    plt.close(fig)
    logger.info("Saved size_effect.png")

    # ----- Viz 5: Dataset effect (top 5 schemes, faceted) -----
    top5 = top_schemes[:5]
    top5_df = combined[combined["Scheme"].isin(top5)].copy()
    g = sns.catplot(
        data=top5_df, x="Scheme", y="AUROC", hue="dataset",
        kind="bar", height=5, aspect=1.8,
    )
    g.set_xticklabels(rotation=30, ha="right")
    g.fig.suptitle("Dataset Effect — Top 5 Schemes", fontsize=14, fontweight="bold", y=1.02)
    g.fig.savefig(dashboard_dir / "dataset_effect.png", dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    logger.info("Saved dataset_effect.png")

    # ----- Viz 6: Per-model bar chart of top scheme AUROC -----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, sz in zip(axes, ["Small", "Large"]):
        sz_df = top1_df[top1_df["size"] == sz]
        if sz_df.empty:
            ax.set_title(f"{sz} Models — No Data")
            continue
        sns.barplot(data=sz_df, x="model", y="AUROC", hue="dataset", ax=ax)
        ax.set_title(f"{sz} Models — {top1_scheme}", fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    plt.suptitle("AUROC by Model and Dataset", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(dashboard_dir / "per_model_auroc.png", dpi=200)
    plt.close(fig)
    logger.info("Saved per_model_auroc.png")

    # ----- Summary table -----
    summary = avg_auroc.reset_index()
    summary.columns = ["Scheme", "Mean_AUROC"]
    summary["Std_AUROC"] = combined.groupby("Scheme")["AUROC"].std().values
    summary["Min_AUROC"] = combined.groupby("Scheme")["AUROC"].min().values
    summary["Max_AUROC"] = combined.groupby("Scheme")["AUROC"].max().values
    summary["Num_Runs"] = combined.groupby("Scheme")["AUROC"].count().values
    summary = summary.sort_values("Mean_AUROC", ascending=False)
    summary.to_csv(dashboard_dir / "scheme_summary.csv", index=False)
    logger.info("Saved scheme_summary.csv")

    print(f"\n{'=' * 80}")
    print("  Dashboard Summary — Top 10 Schemes (mean AUROC across all runs)")
    print(f"{'=' * 80}")
    for _, row in summary.head(10).iterrows():
        print(f"  {row['Scheme']:<35} {row['Mean_AUROC']:.4f}  (std={row['Std_AUROC']:.4f}, range=[{row['Min_AUROC']:.3f}, {row['Max_AUROC']:.3f}])")
    print(f"\n  Dashboard saved to: {dashboard_dir}/\n")

    # Log dashboard artifacts and summary to W&B
    if wandb.run:
        for _, row in summary.head(10).iterrows():
            wandb.run.summary[f"dashboard/mean_auroc/{row['Scheme']}"] = row["Mean_AUROC"]
        wandb.run.summary["dashboard/best_scheme"] = summary.iloc[0]["Scheme"]
        wandb.run.summary["dashboard/best_mean_auroc"] = summary.iloc[0]["Mean_AUROC"]
        wandb.run.summary["dashboard/num_runs"] = len(all_rows)

        wandb.log({"dashboard/scheme_summary": wandb.Table(dataframe=summary.head(20))})

        for img_name in [
            "auroc_heatmap.png", "best_scheme_consistency.png",
            "model_family_comparison.png", "size_effect.png",
            "dataset_effect.png", "per_model_auroc.png",
        ]:
            img_path = dashboard_dir / img_name
            if img_path.exists():
                wandb.log({f"dashboard/{img_name}": wandb.Image(str(img_path))})


# ============================================================================
# Main
# ============================================================================

STEPS = ["judge", "phase6", "pos", "dashboard"]


def main():
    parser = argparse.ArgumentParser(
        description="Full long-answer experiment pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--step", choices=STEPS, help="Run only this step")
    parser.add_argument("--size", help="Filter: Small, Large, XLarge")
    parser.add_argument("--dataset", help="Filter: trivia_qa, squad, coqa")
    parser.add_argument("--dry-run", action="store_true", help="Show runs, don't execute")
    parser.add_argument(
        "--force-rejudge", action="store_true",
        help="Re-evaluate all examples with the judge, even those already scored",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B tracking",
    )
    parser.add_argument(
        "--wandb-dir", type=str, default=str(WANDB_BASE),
        help=f"Local wandb directory (default: {WANDB_BASE})",
    )
    args = parser.parse_args()

    wandb_base = Path(args.wandb_dir)
    runs = discover_latest_runs(wandb_base, size_filter=args.size, dataset_filter=args.dataset)

    if not runs:
        logger.error("No runs found in %s", wandb_base)
        sys.exit(1)

    print_run_table(runs)

    if args.dry_run:
        state = load_state()
        for r in runs:
            statuses = state.get(r["key"], {})
            done = [s for s in STEPS if statuses.get(s) == "done"]
            todo = [s for s in STEPS if statuses.get(s) != "done"]
            print(f"  {r['key']}")
            print(f"    done: {', '.join(done) or 'none'}  |  todo: {', '.join(todo)}")
        return

    # Initialize W&B
    if not args.no_wandb:
        step_name = args.step or "full"
        run_name = f"pipeline_{step_name}"
        if args.size:
            run_name += f"_{args.size}"
        if args.dataset:
            run_name += f"_{args.dataset}"

        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=run_name,
            config={
                "step": args.step or "all",
                "size_filter": args.size,
                "dataset_filter": args.dataset,
                "num_runs": len(runs),
                "force_rejudge": args.force_rejudge,
                "models": list({r["model"] for r in runs}),
                "datasets": list({r["dataset"] for r in runs}),
                "sizes": list({r["size"] for r in runs}),
            },
            tags=["pipeline", args.step or "full"],
        )

    state = load_state()
    steps_to_run = [args.step] if args.step else STEPS

    if "judge" in steps_to_run:
        run_judge_step(runs, state, force_rejudge=args.force_rejudge)

    if "phase6" in steps_to_run:
        run_phase6_step(runs, state)

    if "pos" in steps_to_run:
        run_pos_step(runs, state)

    if "dashboard" in steps_to_run:
        run_dashboard_step(runs)

    if wandb.run:
        wandb.finish()
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
