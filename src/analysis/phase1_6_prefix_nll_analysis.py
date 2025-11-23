"""Phase 1.6: Prefix-level NLL analysis.

This module:
1. Computes prefix NLL / mean NLL curves for each answer
2. Compares correct vs incorrect answers (short or long) at prefix level
3. Computes AUROC using only early tokens (first token / first-k tokens)
"""

import argparse
import json
import logging
import os
import pickle
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pickle_data(pickle_path: str) -> Dict[str, Any]:
    """Load validation generations from pickle file."""
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")

    logger.info("Loading data from: %s", pickle_path)
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    logger.info("Loaded %d examples", len(data))
    return data


def get_correctness_labels(
    data: Dict[str, Any],
    use_rouge: bool = False,
    rouge_threshold: float = 0.3,
) -> Dict[str, int]:
    """Get correctness labels for all examples.

    If use_rouge=True, uses ROUGE-L against reference answers.
    Otherwise uses LLM judge accuracy (accuracy > 0.5).
    """
    from rouge_score import rouge_scorer

    labels: Dict[str, int] = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True) if use_rouge else None

    for example_id, entry in data.items():
        if "most_likely_answer" not in entry:
            continue

        mla = entry["most_likely_answer"]

        if use_rouge and scorer:
            if "reference" in entry and "answers" in entry["reference"]:
                true_answers = entry["reference"]["answers"]["text"]
                pred_answer = mla.get("response", "").strip()
                if not true_answers or not pred_answer:
                    continue
                best_rougeL = 0.0
                for ref in true_answers:
                    score = scorer.score(ref.strip(), pred_answer)
                    best_rougeL = max(best_rougeL, score["rougeL"].fmeasure)
                labels[example_id] = int(best_rougeL >= rouge_threshold)
        else:
            acc = mla.get("accuracy", None)
            if acc is not None:
                labels[example_id] = int(acc > 0.5)

    return labels


def compute_prefix_stats_for_example(
    token_log_likelihoods: List[float],
    max_prefix_len: Optional[int] = None,
) -> Dict[str, List[float]]:
    """Compute prefix NLL and mean NLL for a single example."""
    if not token_log_likelihoods:
        return {"prefix_nll": [], "prefix_mean_nll": []}

    log_probs = np.array(token_log_likelihoods, dtype=float)
    nlls = -log_probs
    prefix_nll = np.cumsum(nlls)

    if max_prefix_len is not None:
        prefix_nll = prefix_nll[:max_prefix_len]

    ks = np.arange(1, len(prefix_nll) + 1, dtype=float)
    prefix_mean_nll = prefix_nll / ks

    return {
        "prefix_nll": prefix_nll.tolist(),
        "prefix_mean_nll": prefix_mean_nll.tolist(),
    }


def aggregate_prefix_curves(
    per_example_stats: Dict[str, Dict[str, List[float]]],
    labels: Dict[str, int],
    max_prefix_len: int,
) -> Dict[str, Any]:
    """Aggregate prefix curves across correct vs incorrect answers."""
    def collect(group_label: int) -> Dict[str, np.ndarray]:
        curves = [
            stats["prefix_mean_nll"]
            for eid, stats in per_example_stats.items()
            if labels.get(eid) == group_label and stats["prefix_mean_nll"]
        ]
        if not curves:
            return {
                "mean": np.zeros(max_prefix_len),
                "std": np.zeros(max_prefix_len),
                "count_per_k": np.zeros(max_prefix_len),
            }

        # Pad curves with NaN to same length and compute mean over available ones at each k
        padded = np.full((len(curves), max_prefix_len), np.nan, dtype=float)
        for i, c in enumerate(curves):
            L = min(len(c), max_prefix_len)
            padded[i, :L] = c[:L]

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        count_per_k = np.sum(~np.isnan(padded), axis=0)
        return {"mean": mean, "std": std, "count_per_k": count_per_k}

    correct = collect(1)
    incorrect = collect(0)

    return {
        "correct": {
            "mean_prefix_mean_nll": correct["mean"].tolist(),
            "std_prefix_mean_nll": correct["std"].tolist(),
            "count_per_k": correct["count_per_k"].tolist(),
        },
        "incorrect": {
            "mean_prefix_mean_nll": incorrect["mean"].tolist(),
            "std_prefix_mean_nll": incorrect["std"].tolist(),
            "count_per_k": incorrect["count_per_k"].tolist(),
        },
    }


def compute_early_token_aurocs(
    data: Dict[str, Any],
    labels: Dict[str, int],
    ks: List[int],
) -> Dict[str, float]:
    """Compute AUROC using only early tokens (first, first-k) as predictors."""
    results: Dict[str, float] = {}

    # Prepare containers
    y_true_all: List[int] = []
    first_token_scores: List[float] = []
    prefix_scores: Dict[int, List[float]] = {k: [] for k in ks}

    for example_id, entry in data.items():
        if example_id not in labels:
            continue
        if "most_likely_answer" not in entry:
            continue

        mla = entry["most_likely_answer"]
        token_log_likelihoods = mla.get("token_log_likelihoods", [])
        if not token_log_likelihoods:
            continue

        log_probs = np.array(token_log_likelihoods, dtype=float)
        nlls = -log_probs

        y_true_all.append(labels[example_id])
        # First token NLL (higher NLL = more uncertain)
        first_token_scores.append(nlls[0])

        for k in ks:
            L = min(len(nlls), k)
            prefix_mean = float(np.mean(nlls[:L]))
            prefix_scores[k].append(prefix_mean)

    if not y_true_all:
        return results

    y_true_arr = np.array(y_true_all, dtype=int)

    # For AUROC, higher score should predict incorrect; roc_auc_score expects higher scores
    # to predict positive class (correct=1). So we negate.
    try:
        results["first_token_nll"] = roc_auc_score(y_true_arr, -np.array(first_token_scores))
    except ValueError:
        results["first_token_nll"] = float("nan")

    for k in ks:
        scores_k = np.array(prefix_scores[k])
        if len(scores_k) != len(y_true_arr):
            continue
        try:
            results[f"first_{k}_tokens_mean_nll"] = roc_auc_score(y_true_arr, -scores_k)
        except ValueError:
            results[f"first_{k}_tokens_mean_nll"] = float("nan")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.6: Prefix-level NLL analysis (short or long answers)"
    )
    parser.add_argument(
        "--pickle-path",
        type=str,
        required=True,
        help="Path to validation_generations.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (if not provided, will be auto-generated from wandb-run-id and context-type)",
    )
    parser.add_argument(
        "--use-rouge",
        action="store_true",
        help="Use ROUGE-based correctness (for short answers). If False, use LLM judge accuracy.",
    )
    parser.add_argument(
        "--rouge-threshold",
        type=float,
        default=0.3,
        help="ROUGE-L threshold for correctness (default: 0.3)",
    )
    parser.add_argument(
        "--max-prefix-len",
        type=int,
        default=50,
        help="Maximum prefix length (in tokens) to include in aggregated curves (default: 50)",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Prefix lengths (k) for early-token AUROC analysis (default: 1 3 5)",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="WandB run ID to include in output folder name",
    )
    parser.add_argument(
        "--context-type",
        type=str,
        choices=["short", "long"],
        default=None,
        help="Context type: short or long (used in output folder naming)",
    )

    args = parser.parse_args()
    
    # Auto-generate output directory if not provided
    if args.output_dir is None:
        dir_parts = ["results", "phase1_6"]
        if args.context_type:
            dir_parts.append(args.context_type)
        if args.wandb_run_id:
            dir_parts.append(args.wandb_run_id)
        args.output_dir = "_".join(dir_parts)
        logger.info(f"Auto-generated output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data and labels
    data = load_pickle_data(args.pickle_path)
    labels = get_correctness_labels(
        data,
        use_rouge=args.use_rouge,
        rouge_threshold=args.rouge_threshold,
    )
    logger.info("Got correctness labels for %d examples", len(labels))

    # Compute per-example prefix stats
    per_example_stats: Dict[str, Dict[str, List[float]]] = {}
    for example_id, entry in tqdm(data.items(), desc="Computing prefix stats"):
        if example_id not in labels:
            continue
        if "most_likely_answer" not in entry:
            continue

        mla = entry["most_likely_answer"]
        token_log_likelihoods = mla.get("token_log_likelihoods", [])
        if not token_log_likelihoods:
            continue

        stats = compute_prefix_stats_for_example(
            token_log_likelihoods,
            max_prefix_len=args.max_prefix_len,
        )
        if stats["prefix_mean_nll"]:
            per_example_stats[example_id] = stats

    logger.info("Computed prefix stats for %d examples", len(per_example_stats))

    # Aggregate curves
    aggregated = aggregate_prefix_curves(
        per_example_stats,
        labels,
        max_prefix_len=args.max_prefix_len,
    )

    # Compute early-token AUROCs
    aurocs = compute_early_token_aurocs(
        data,
        labels,
        ks=args.ks,
    )

    # Save summary JSON
    summary = {
        "aggregated_prefix_curves": aggregated,
        "early_token_aurocs": aurocs,
        "num_examples_with_stats": len(per_example_stats),
        "num_labeled_examples": len(labels),
    }
    summary_path = os.path.join(args.output_dir, "prefix_nll_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("✅ Prefix NLL summary saved to: %s", summary_path)

    # Plot prefix mean NLL curves
    correct = aggregated["correct"]
    incorrect = aggregated["incorrect"]
    ks = np.arange(1, args.max_prefix_len + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(
        ks,
        correct["mean_prefix_mean_nll"],
        label="Correct",
        linewidth=2,
        color="tab:blue",
    )
    plt.plot(
        ks,
        incorrect["mean_prefix_mean_nll"],
        label="Incorrect",
        linewidth=2,
        color="tab:red",
    )
    plt.xlabel("Prefix Length k (tokens)", fontsize=12)
    plt.ylabel("Mean Prefix NLL", fontsize=12)
    plt.title("Prefix Mean NLL Curves: Correct vs Incorrect", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = os.path.join(args.output_dir, "prefix_mean_nll_curves.png")
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("✅ Prefix mean NLL curves saved to: %s", curve_path)

    # Plot first-token NLL distributions
    first_token_nlls_correct: List[float] = []
    first_token_nlls_incorrect: List[float] = []
    for example_id, entry in data.items():
        if example_id not in labels:
            continue
        if "most_likely_answer" not in entry:
            continue
        mla = entry["most_likely_answer"]
        token_log_likelihoods = mla.get("token_log_likelihoods", [])
        if not token_log_likelihoods:
            continue
        nll_first = -float(token_log_likelihoods[0])
        if labels[example_id] == 1:
            first_token_nlls_correct.append(nll_first)
        else:
            first_token_nlls_incorrect.append(nll_first)

    if first_token_nlls_correct and first_token_nlls_incorrect:
        plt.figure(figsize=(10, 6))
        plt.boxplot(
            [first_token_nlls_correct, first_token_nlls_incorrect],
            labels=["Correct", "Incorrect"],
        )
        plt.ylabel("First Token NLL", fontsize=12)
        plt.title("First Token NLL by Correctness", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        box_path = os.path.join(args.output_dir, "first_token_nll_boxplot.png")
        plt.savefig(box_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("✅ First-token NLL boxplot saved to: %s", box_path)

    # Log AUROC summary
    logger.info("\n" + "=" * 80)
    logger.info("PREFIX-LEVEL NLL ANALYSIS SUMMARY (PHASE 1.6)")
    logger.info("=" * 80)
    for name, auc in aurocs.items():
        logger.info("AUROC [%s]: %s", name, f"{auc:.4f}" if np.isfinite(auc) else "N/A")


if __name__ == "__main__":
    main()


