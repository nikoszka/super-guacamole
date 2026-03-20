#!/usr/bin/env python3
"""
Diagnostic Analysis: Llama-3.1-8B vs Qwen3-8B on CoQA

Investigates WHY nll_proportional weighting works on Llama but fails on Qwen.
Compares per-token NLL distributions, weighting behavior, and per-example patterns
on the same CoQA questions.
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import roc_auc_score


def load_data(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def get_nll_and_labels(data):
    """Extract per-token NLLs and correctness labels."""
    results = {}
    for eid, entry in data.items():
        mla = entry.get("most_likely_answer", {})
        token_lls = mla.get("token_log_likelihoods", [])
        accuracy = mla.get("accuracy")
        response = mla.get("response", "")
        if not token_lls or accuracy is None:
            continue
        nlls = [-ll for ll in token_lls]
        correct = int(accuracy > 0.5)
        results[eid] = {
            "nlls": nlls,
            "correct": correct,
            "response": response,
            "g_nll": sum(nlls),
            "avg_nll": np.mean(nlls),
            "seq_len": len(nlls),
        }
    return results


def compute_weighted_nll(nlls, scheme="uniform"):
    """Compute weighted NLL for a single example."""
    n = len(nlls)
    if n == 0:
        return 0.0
    arr = np.array(nlls)

    if scheme == "uniform":
        weights = np.ones(n)
    elif scheme == "nll_proportional":
        weights = arr.copy()
        weights = weights - weights.min() + 1e-8
    elif scheme == "linear_decreasing":
        weights = np.linspace(1.0, 0.1, n)
    elif scheme == "first_last_k":
        k = max(1, n // 3)
        weights = np.zeros(n)
        weights[:k] = 1.0
        weights[-k:] = 1.0
    else:
        weights = np.ones(n)

    return float(np.sum(weights * arr) / np.sum(weights))


def main():
    parser = argparse.ArgumentParser(description="Llama vs Qwen NLL diagnostic")
    parser.add_argument("--llama-pickle", required=True)
    parser.add_argument("--qwen-pickle", required=True)
    parser.add_argument("--output-dir", default="results/pipeline/diagnostic_llama_vs_qwen")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Llama data...")
    llama_data = load_data(args.llama_pickle)
    print("Loading Qwen data...")
    qwen_data = load_data(args.qwen_pickle)

    llama_results = get_nll_and_labels(llama_data)
    qwen_results = get_nll_and_labels(qwen_data)

    common_ids = sorted(set(llama_results.keys()) & set(qwen_results.keys()))
    print(f"Common examples: {len(common_ids)} (Llama: {len(llama_results)}, Qwen: {len(qwen_results)})")

    # ====================================================================
    # 1. NLL Distribution Comparison
    # ====================================================================
    all_llama_nlls = np.concatenate([llama_results[eid]["nlls"] for eid in common_ids])
    all_qwen_nlls = np.concatenate([qwen_results[eid]["nlls"] for eid in common_ids])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("NLL Distribution: Llama-3.1-8B vs Qwen3-8B (CoQA)", fontsize=16, fontweight="bold")

    # 1a: Overall NLL histogram
    ax = axes[0, 0]
    ax.hist(all_llama_nlls, bins=100, alpha=0.6, label=f"Llama (mean={np.mean(all_llama_nlls):.3f})", density=True, color="steelblue")
    ax.hist(all_qwen_nlls, bins=100, alpha=0.6, label=f"Qwen (mean={np.mean(all_qwen_nlls):.3f})", density=True, color="coral")
    ax.set_xlabel("Token NLL")
    ax.set_ylabel("Density")
    ax.set_title("All Token NLLs")
    ax.legend()
    ax.set_xlim(0, 5)

    # 1b: NLL by correctness
    ax = axes[0, 1]
    for model, results, color_base in [("Llama", llama_results, "steelblue"), ("Qwen", qwen_results, "coral")]:
        correct_nlls = np.concatenate([results[eid]["nlls"] for eid in common_ids if results[eid]["correct"]])
        incorrect_nlls = np.concatenate([results[eid]["nlls"] for eid in common_ids if not results[eid]["correct"]])
        ax.hist(correct_nlls, bins=80, alpha=0.4, density=True, label=f"{model} correct (mean={np.mean(correct_nlls):.3f})")
        ax.hist(incorrect_nlls, bins=80, alpha=0.4, density=True, label=f"{model} incorrect (mean={np.mean(incorrect_nlls):.3f})", linestyle="--")
    ax.set_xlabel("Token NLL")
    ax.set_ylabel("Density")
    ax.set_title("NLL by Correctness")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 5)

    # 1c: Sequence length comparison
    ax = axes[1, 0]
    llama_lens = [llama_results[eid]["seq_len"] for eid in common_ids]
    qwen_lens = [qwen_results[eid]["seq_len"] for eid in common_ids]
    ax.hist(llama_lens, bins=50, alpha=0.6, label=f"Llama (mean={np.mean(llama_lens):.1f})", color="steelblue")
    ax.hist(qwen_lens, bins=50, alpha=0.6, label=f"Qwen (mean={np.mean(qwen_lens):.1f})", color="coral")
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Response Length Distribution")
    ax.legend()

    # 1d: Per-example G-NLL scatter
    ax = axes[1, 1]
    llama_gnlls = [llama_results[eid]["g_nll"] for eid in common_ids]
    qwen_gnlls = [qwen_results[eid]["g_nll"] for eid in common_ids]
    llama_correct = [llama_results[eid]["correct"] for eid in common_ids]
    colors = ["green" if c else "red" for c in llama_correct]
    ax.scatter(llama_gnlls, qwen_gnlls, c=colors, alpha=0.5, s=20)
    ax.set_xlabel("Llama G-NLL")
    ax.set_ylabel("Qwen G-NLL")
    ax.set_title("Per-Example G-NLL (green=Llama correct)")
    ax.plot([0, max(max(llama_gnlls), max(qwen_gnlls))],
            [0, max(max(llama_gnlls), max(qwen_gnlls))], "k--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "nll_distributions.png"), dpi=200)
    plt.close(fig)
    print("Saved nll_distributions.png")

    # ====================================================================
    # 2. NLL Variance Analysis — the key to NLL-prop weighting
    # ====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("NLL Variance Analysis: Why Does NLL-Prop Weighting Differ?", fontsize=16, fontweight="bold")

    # 2a: Intra-example NLL variance (std of NLLs within each response)
    ax = axes[0, 0]
    llama_stds = [np.std(llama_results[eid]["nlls"]) for eid in common_ids]
    qwen_stds = [np.std(qwen_results[eid]["nlls"]) for eid in common_ids]
    ax.hist(llama_stds, bins=50, alpha=0.6, label=f"Llama (mean={np.mean(llama_stds):.3f})", color="steelblue")
    ax.hist(qwen_stds, bins=50, alpha=0.6, label=f"Qwen (mean={np.mean(qwen_stds):.3f})", color="coral")
    ax.set_xlabel("Within-Example NLL Std Dev")
    ax.set_ylabel("Count")
    ax.set_title("Intra-Example NLL Variance")
    ax.legend()

    # 2b: NLL coefficient of variation
    ax = axes[0, 1]
    llama_cvs = [np.std(llama_results[eid]["nlls"]) / (np.mean(llama_results[eid]["nlls"]) + 1e-8) for eid in common_ids]
    qwen_cvs = [np.std(qwen_results[eid]["nlls"]) / (np.mean(qwen_results[eid]["nlls"]) + 1e-8) for eid in common_ids]
    ax.hist(llama_cvs, bins=50, alpha=0.6, label=f"Llama (mean={np.mean(llama_cvs):.3f})", color="steelblue")
    ax.hist(qwen_cvs, bins=50, alpha=0.6, label=f"Qwen (mean={np.mean(qwen_cvs):.3f})", color="coral")
    ax.set_xlabel("NLL Coefficient of Variation")
    ax.set_ylabel("Count")
    ax.set_title("NLL CV (how spread out are token uncertainties?)")
    ax.legend()

    # 2c: NLL max/min ratio
    ax = axes[1, 0]
    llama_ranges = [np.max(llama_results[eid]["nlls"]) - np.min(llama_results[eid]["nlls"]) for eid in common_ids]
    qwen_ranges = [np.max(qwen_results[eid]["nlls"]) - np.min(qwen_results[eid]["nlls"]) for eid in common_ids]
    ax.hist(llama_ranges, bins=50, alpha=0.6, label=f"Llama (mean={np.mean(llama_ranges):.3f})", color="steelblue")
    ax.hist(qwen_ranges, bins=50, alpha=0.6, label=f"Qwen (mean={np.mean(qwen_ranges):.3f})", color="coral")
    ax.set_xlabel("NLL Range (max - min)")
    ax.set_ylabel("Count")
    ax.set_title("NLL Dynamic Range per Example")
    ax.legend()

    # 2d: Fraction of high-NLL tokens (>1.0) per example
    ax = axes[1, 1]
    llama_high_frac = [np.mean(np.array(llama_results[eid]["nlls"]) > 1.0) for eid in common_ids]
    qwen_high_frac = [np.mean(np.array(qwen_results[eid]["nlls"]) > 1.0) for eid in common_ids]
    ax.hist(llama_high_frac, bins=30, alpha=0.6, label=f"Llama (mean={np.mean(llama_high_frac):.3f})", color="steelblue")
    ax.hist(qwen_high_frac, bins=30, alpha=0.6, label=f"Qwen (mean={np.mean(qwen_high_frac):.3f})", color="coral")
    ax.set_xlabel("Fraction of Tokens with NLL > 1.0")
    ax.set_ylabel("Count")
    ax.set_title("High-Uncertainty Token Fraction")
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "nll_variance_analysis.png"), dpi=200)
    plt.close(fig)
    print("Saved nll_variance_analysis.png")

    # ====================================================================
    # 3. Weighting Scheme AUROC Breakdown
    # ====================================================================
    schemes = ["uniform", "nll_proportional", "linear_decreasing", "first_last_k"]
    scheme_results = {model: {} for model in ["Llama", "Qwen"]}

    for scheme in schemes:
        for model_name, results in [("Llama", llama_results), ("Qwen", qwen_results)]:
            scores = []
            labels = []
            for eid in common_ids:
                r = results[eid]
                scores.append(compute_weighted_nll(r["nlls"], scheme))
                labels.append(r["correct"])
            try:
                auroc = roc_auc_score(labels, [-s for s in scores])
                scheme_results[model_name][scheme] = auroc
            except ValueError:
                scheme_results[model_name][scheme] = 0.5

    print("\n" + "=" * 70)
    print("  Weighting Scheme AUROC (Common Examples Only)")
    print("=" * 70)
    print(f"  {'Scheme':<25} {'Llama':>10} {'Qwen':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    for scheme in schemes:
        l = scheme_results["Llama"][scheme]
        q = scheme_results["Qwen"][scheme]
        print(f"  {scheme:<25} {l:>10.4f} {q:>10.4f} {l-q:>+10.4f}")

    # ====================================================================
    # 4. Per-Example NLL-Prop Score Difference Analysis
    # ====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Why NLL-Prop Works on Llama but Not Qwen", fontsize=16, fontweight="bold")

    # 4a: NLL-prop weighted score by correctness
    ax = axes[0, 0]
    for model_name, results, color in [("Llama", llama_results, "steelblue"), ("Qwen", qwen_results, "coral")]:
        correct_scores = [compute_weighted_nll(results[eid]["nlls"], "nll_proportional") for eid in common_ids if results[eid]["correct"]]
        incorrect_scores = [compute_weighted_nll(results[eid]["nlls"], "nll_proportional") for eid in common_ids if not results[eid]["correct"]]
        positions = [0, 1] if model_name == "Llama" else [2, 3]
        bp = ax.boxplot([correct_scores, incorrect_scores], positions=positions, widths=0.6,
                        patch_artist=True, boxprops=dict(facecolor=color, alpha=0.6))
    ax.set_xticks([0.5, 2.5])
    ax.set_xticklabels(["Llama", "Qwen"])
    ax.set_ylabel("NLL-Prop Weighted Score")
    ax.set_title("NLL-Prop Score: Correct (left) vs Incorrect (right)")
    ax.legend(["Correct", "Incorrect"], loc="upper right")

    # 4b: Separation analysis — how much do correct/incorrect overlap?
    ax = axes[0, 1]
    for model_name, results, color in [("Llama", llama_results, "steelblue"), ("Qwen", qwen_results, "coral")]:
        correct_scores = np.array([compute_weighted_nll(results[eid]["nlls"], "nll_proportional") for eid in common_ids if results[eid]["correct"]])
        incorrect_scores = np.array([compute_weighted_nll(results[eid]["nlls"], "nll_proportional") for eid in common_ids if not results[eid]["correct"]])
        mean_sep = np.mean(incorrect_scores) - np.mean(correct_scores)
        pooled_std = np.sqrt((np.var(correct_scores) + np.var(incorrect_scores)) / 2)
        cohens_d = mean_sep / pooled_std if pooled_std > 0 else 0
        ax.bar(model_name, cohens_d, color=color, alpha=0.7)
        ax.text(model_name, cohens_d + 0.02, f"d={cohens_d:.3f}", ha="center", fontsize=11)
    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect Size: Correct vs Incorrect Separation")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # 4c: NLL-prop vs Uniform improvement per example
    ax = axes[1, 0]
    for model_name, results, color in [("Llama", llama_results, "steelblue"), ("Qwen", qwen_results, "coral")]:
        uniform_scores = np.array([compute_weighted_nll(results[eid]["nlls"], "uniform") for eid in common_ids])
        nllprop_scores = np.array([compute_weighted_nll(results[eid]["nlls"], "nll_proportional") for eid in common_ids])
        ratio = nllprop_scores / (uniform_scores + 1e-8)
        labels_arr = np.array([results[eid]["correct"] for eid in common_ids])
        ax.scatter(uniform_scores, ratio, c=[("green" if c else "red") for c in labels_arr],
                   alpha=0.3, s=15, label=model_name)
    ax.set_xlabel("Uniform (G-NLL) Score")
    ax.set_ylabel("NLL-Prop / Uniform Ratio")
    ax.set_title("How NLL-Prop Rescales G-NLL (green=correct)")
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    # 4d: NLL position profile (mean NLL at each relative position)
    ax = axes[1, 1]
    n_bins = 20
    for model_name, results, color in [("Llama", llama_results, "steelblue"), ("Qwen", qwen_results, "coral")]:
        position_nlls = defaultdict(list)
        for eid in common_ids:
            nlls = results[eid]["nlls"]
            n = len(nlls)
            for i, nll in enumerate(nlls):
                bin_idx = int(i / n * n_bins) if n > 0 else 0
                bin_idx = min(bin_idx, n_bins - 1)
                position_nlls[bin_idx].append(nll)
        positions = sorted(position_nlls.keys())
        means = [np.mean(position_nlls[p]) for p in positions]
        ax.plot([p / n_bins for p in positions], means, "-o", color=color, label=model_name, markersize=4)
    ax.set_xlabel("Relative Position in Response")
    ax.set_ylabel("Mean NLL")
    ax.set_title("NLL Position Profile")
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "nll_prop_diagnostic.png"), dpi=200)
    plt.close(fig)
    print("Saved nll_prop_diagnostic.png")

    # ====================================================================
    # 5. Token-Level Deep Dive: Same Question, Different Models
    # ====================================================================
    fig = plt.figure(figsize=(18, 24))
    fig.suptitle("Same Questions, Different Models: Token-Level NLL Comparison", fontsize=16, fontweight="bold")

    # Pick 6 examples where models disagree on correctness or NLL-prop ranking differs
    disagree_examples = []
    nllprop_diverge = []
    for eid in common_ids:
        l_correct = llama_results[eid]["correct"]
        q_correct = qwen_results[eid]["correct"]
        if l_correct != q_correct:
            disagree_examples.append(eid)
        l_nllprop = compute_weighted_nll(llama_results[eid]["nlls"], "nll_proportional")
        q_nllprop = compute_weighted_nll(qwen_results[eid]["nlls"], "nll_proportional")
        l_uniform = compute_weighted_nll(llama_results[eid]["nlls"], "uniform")
        q_uniform = compute_weighted_nll(qwen_results[eid]["nlls"], "uniform")
        l_boost = l_nllprop - l_uniform
        q_boost = q_nllprop - q_uniform
        nllprop_diverge.append((eid, l_boost, q_boost, abs(l_boost - q_boost)))

    nllprop_diverge.sort(key=lambda x: -x[3])
    sample_ids = []
    if disagree_examples:
        sample_ids.extend(disagree_examples[:3])
    sample_ids.extend([x[0] for x in nllprop_diverge[:6 - len(sample_ids)]])
    sample_ids = sample_ids[:6]

    for idx, eid in enumerate(sample_ids):
        ax1 = fig.add_subplot(6, 1, idx + 1)
        llama_nlls = llama_results[eid]["nlls"]
        qwen_nlls = qwen_results[eid]["nlls"]

        max_len = max(len(llama_nlls), len(qwen_nlls))
        x_llama = np.arange(len(llama_nlls))
        x_qwen = np.arange(len(qwen_nlls))

        ax1.bar(x_llama - 0.2, llama_nlls, width=0.4, alpha=0.7, color="steelblue", label="Llama")
        ax1.bar(x_qwen + 0.2, qwen_nlls, width=0.4, alpha=0.7, color="coral", label="Qwen")

        l_label = "correct" if llama_results[eid]["correct"] else "incorrect"
        q_label = "correct" if qwen_results[eid]["correct"] else "incorrect"
        l_nllp = compute_weighted_nll(llama_nlls, "nll_proportional")
        q_nllp = compute_weighted_nll(qwen_nlls, "nll_proportional")

        title = (f"Example {eid[:20]}... | Llama: {l_label} (NLL-prop={l_nllp:.3f}) | "
                 f"Qwen: {q_label} (NLL-prop={q_nllp:.3f})")
        ax1.set_title(title, fontsize=10)
        ax1.set_ylabel("Token NLL")
        if idx == 0:
            ax1.legend()
        if idx == 5:
            ax1.set_xlabel("Token Position")

    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "token_level_comparison.png"), dpi=150)
    plt.close(fig)
    print("Saved token_level_comparison.png")

    # ====================================================================
    # 6. Key Statistics Summary
    # ====================================================================
    summary = {
        "common_examples": len(common_ids),
        "llama": {
            "total_examples": len(llama_results),
            "accuracy": np.mean([llama_results[eid]["correct"] for eid in common_ids]),
            "mean_token_nll": float(np.mean(all_llama_nlls)),
            "std_token_nll": float(np.std(all_llama_nlls)),
            "median_token_nll": float(np.median(all_llama_nlls)),
            "mean_seq_len": float(np.mean(llama_lens)),
            "mean_intra_nll_std": float(np.mean(llama_stds)),
            "mean_nll_cv": float(np.mean(llama_cvs)),
            "mean_nll_range": float(np.mean(llama_ranges)),
            "mean_high_nll_frac": float(np.mean(llama_high_frac)),
        },
        "qwen": {
            "total_examples": len(qwen_results),
            "accuracy": np.mean([qwen_results[eid]["correct"] for eid in common_ids]),
            "mean_token_nll": float(np.mean(all_qwen_nlls)),
            "std_token_nll": float(np.std(all_qwen_nlls)),
            "median_token_nll": float(np.median(all_qwen_nlls)),
            "mean_seq_len": float(np.mean(qwen_lens)),
            "mean_intra_nll_std": float(np.mean(qwen_stds)),
            "mean_nll_cv": float(np.mean(qwen_cvs)),
            "mean_nll_range": float(np.mean(qwen_ranges)),
            "mean_high_nll_frac": float(np.mean(qwen_high_frac)),
        },
        "auroc_comparison": scheme_results,
    }

    with open(os.path.join(args.output_dir, "diagnostic_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved diagnostic_summary.json")

    # Print key findings
    print("\n" + "=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)
    print(f"  Llama accuracy: {summary['llama']['accuracy']:.1%}")
    print(f"  Qwen accuracy:  {summary['qwen']['accuracy']:.1%}")
    print(f"")
    print(f"  Llama mean token NLL: {summary['llama']['mean_token_nll']:.4f} (std={summary['llama']['std_token_nll']:.4f})")
    print(f"  Qwen  mean token NLL: {summary['qwen']['mean_token_nll']:.4f} (std={summary['qwen']['std_token_nll']:.4f})")
    print(f"")
    print(f"  Llama intra-example NLL std: {summary['llama']['mean_intra_nll_std']:.4f}")
    print(f"  Qwen  intra-example NLL std: {summary['qwen']['mean_intra_nll_std']:.4f}")
    print(f"")
    print(f"  Llama NLL CV: {summary['llama']['mean_nll_cv']:.4f}")
    print(f"  Qwen  NLL CV: {summary['qwen']['mean_nll_cv']:.4f}")
    print(f"")
    print(f"  Llama NLL range: {summary['llama']['mean_nll_range']:.4f}")
    print(f"  Qwen  NLL range: {summary['qwen']['mean_nll_range']:.4f}")
    print(f"")
    print(f"  Llama high-NLL fraction (>1.0): {summary['llama']['mean_high_nll_frac']:.4f}")
    print(f"  Qwen  high-NLL fraction (>1.0): {summary['qwen']['mean_high_nll_frac']:.4f}")
    print(f"")
    print(f"  Llama mean seq len: {summary['llama']['mean_seq_len']:.1f}")
    print(f"  Qwen  mean seq len: {summary['qwen']['mean_seq_len']:.1f}")


if __name__ == "__main__":
    main()
