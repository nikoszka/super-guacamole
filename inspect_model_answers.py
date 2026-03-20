#!/usr/bin/env python3
"""
Quick inspection script: load the latest pickle for each Small model/dataset
combo, print LLM-Judge accuracy, and show sample correct/incorrect answers
so you can visually verify the model outputs are sensible sentences.
"""

import pickle
import os
import sys
import json
import types
from pathlib import Path
from collections import defaultdict

# Minimal torch stub so pickle.load works without a real torch installation
if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.Tensor = type("Tensor", (), {
        "__reduce_ex__": lambda self, proto: (list, (list(self),)),
    })

    class _FakeStorage:
        def __init__(self, *a, **kw):
            pass

    _torch_storage = types.ModuleType("torch.storage")
    _torch_storage._load_from_bytes = lambda b: b

    _torch_types = types.ModuleType("torch._utils")
    _torch_types._rebuild_tensor_v2 = lambda *args, **kwargs: []

    sys.modules["torch"] = _torch
    sys.modules["torch.storage"] = _torch_storage
    sys.modules["torch._utils"] = _torch_types

import numpy as np

WANDB_BASE = Path("src/boldis/uncertainty/wandb")
VALID_SIZES = {"Small", "Large", "XLarge"}

NUM_SAMPLES = 5  # correct + incorrect samples to show each


def discover_latest_runs(wandb_base, size_filter=None, dataset_filter=None):
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
                    [d for d in model_dir.iterdir()
                     if d.is_dir() and d.name.startswith("run-")]
                )
                if not run_dirs:
                    continue
                latest = run_dirs[-1]
                pickle_path = latest / "files" / "validation_generations.pkl"
                if not pickle_path.exists():
                    continue
                runs.append({
                    "size": size_dir.name,
                    "dataset": ds_dir.name,
                    "model": model_dir.name,
                    "run_dir": str(latest),
                    "pickle_path": str(pickle_path),
                })
    return runs


def inspect_run(run_info):
    """Load pickle and print accuracy + sample answers."""
    label = f"{run_info['size']}/{run_info['dataset']}/{run_info['model']}"
    pkl_path = run_info["pickle_path"]

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    correct_examples = []
    incorrect_examples = []
    accuracies = []
    missing_accuracy = 0
    total = 0

    for eid, entry in data.items():
        mla = entry.get("most_likely_answer")
        if mla is None:
            continue
        total += 1

        acc = mla.get("accuracy")
        if acc is None:
            missing_accuracy += 1
            continue
        accuracies.append(float(acc))

        response = mla.get("response", "").strip()
        tokens = mla.get("tokens", [])
        token_lls = mla.get("token_log_likelihoods", [])
        question = entry.get("question", "N/A")
        ref_answers = entry.get("reference", {}).get("answers", {}).get("text", [])
        ref_str = " | ".join(ref_answers[:3]) if ref_answers else "N/A"

        gnll = -sum(token_lls) if token_lls else float("nan")

        sample = {
            "eid": eid,
            "question": question,
            "reference": ref_str,
            "response": response,
            "accuracy": float(acc),
            "num_tokens": len(tokens),
            "gnll": gnll,
        }

        if acc > 0.5:
            correct_examples.append(sample)
        else:
            incorrect_examples.append(sample)

    # Print summary
    print(f"\n{'='*100}")
    print(f"  {label}")
    print(f"  Pickle: {pkl_path}")
    print(f"{'='*100}")
    print(f"  Total examples: {total}")
    print(f"  Missing accuracy (no judge): {missing_accuracy}")

    if accuracies:
        n_correct = sum(1 for a in accuracies if a > 0.5)
        n_incorrect = len(accuracies) - n_correct
        mean_acc = np.mean(accuracies)
        print(f"  Judge Accuracy: {mean_acc:.4f}  ({n_correct} correct / {n_incorrect} incorrect out of {len(accuracies)})")
    else:
        print(f"  No accuracy values found!")
        return

    # Check response lengths
    all_resp_lens = [len(s["response"]) for s in correct_examples + incorrect_examples]
    if all_resp_lens:
        print(f"  Response length (chars): min={min(all_resp_lens)}, mean={np.mean(all_resp_lens):.0f}, max={max(all_resp_lens)}")

    all_tok_counts = [s["num_tokens"] for s in correct_examples + incorrect_examples]
    if all_tok_counts:
        print(f"  Token count: min={min(all_tok_counts)}, mean={np.mean(all_tok_counts):.0f}, max={max(all_tok_counts)}")

    # G-NLL stats by correctness
    correct_gnlls = [s["gnll"] for s in correct_examples if not np.isnan(s["gnll"])]
    incorrect_gnlls = [s["gnll"] for s in incorrect_examples if not np.isnan(s["gnll"])]
    if correct_gnlls and incorrect_gnlls:
        print(f"  Mean G-NLL (correct):   {np.mean(correct_gnlls):.2f}  (std={np.std(correct_gnlls):.2f})")
        print(f"  Mean G-NLL (incorrect): {np.mean(incorrect_gnlls):.2f}  (std={np.std(incorrect_gnlls):.2f})")
        print(f"  G-NLL separation: {np.mean(incorrect_gnlls) - np.mean(correct_gnlls):+.2f} (incorrect - correct)")

    # Sample correct answers
    print(f"\n  --- CORRECT ANSWERS (sample {min(NUM_SAMPLES, len(correct_examples))}) ---")
    for s in correct_examples[:NUM_SAMPLES]:
        print(f"    Q: {s['question'][:120]}")
        print(f"    Reference: {s['reference'][:120]}")
        print(f"    Model:     {s['response'][:200]}")
        print(f"    (tokens={s['num_tokens']}, gnll={s['gnll']:.2f}, acc={s['accuracy']:.2f})")
        print()

    # Sample incorrect answers
    print(f"  --- INCORRECT ANSWERS (sample {min(NUM_SAMPLES, len(incorrect_examples))}) ---")
    for s in incorrect_examples[:NUM_SAMPLES]:
        print(f"    Q: {s['question'][:120]}")
        print(f"    Reference: {s['reference'][:120]}")
        print(f"    Model:     {s['response'][:200]}")
        print(f"    (tokens={s['num_tokens']}, gnll={s['gnll']:.2f}, acc={s['accuracy']:.2f})")
        print()

    return {
        "label": label,
        "total": total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "accuracy": mean_acc,
        "missing_accuracy": missing_accuracy,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="Small", help="Filter by size (default: Small)")
    parser.add_argument("--dataset", default=None, help="Filter by dataset")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples per category")
    parser.add_argument("--json-out", type=str, default=None, help="Save summary to JSON")
    args = parser.parse_args()

    global NUM_SAMPLES
    NUM_SAMPLES = args.samples

    runs = discover_latest_runs(WANDB_BASE, size_filter=args.size, dataset_filter=args.dataset)
    if not runs:
        print(f"No runs found in {WANDB_BASE}")
        sys.exit(1)

    print(f"\nFound {len(runs)} runs for size={args.size or 'all'}, dataset={args.dataset or 'all'}")

    summaries = []
    for r in runs:
        s = inspect_run(r)
        if s:
            summaries.append(s)

    # Final summary table
    print(f"\n{'='*100}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"  {'Model+Dataset':<55} {'Accuracy':>10} {'Correct':>8} {'Incorrect':>10} {'Total':>6}")
    print(f"  {'-'*95}")
    for s in summaries:
        print(f"  {s['label']:<55} {s['accuracy']:>10.4f} {s['n_correct']:>8} {s['n_incorrect']:>10} {s['total']:>6}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nSaved summary to {args.json_out}")


if __name__ == "__main__":
    main()
