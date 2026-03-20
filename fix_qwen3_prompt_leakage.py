#!/usr/bin/env python3
"""
Fix Qwen3-8B prompt leakage in existing pickle files.

The model sometimes appends prompt text like:
  "Answer the following question in one complete sentence."
  "Answer in one complete sentence."
after its actual answer. This script:
  1. Detects the leaked suffix in each response
  2. Strips it from the response text
  3. Truncates tokens + token_log_likelihoods to match
  4. Recalculates sequence_nll and sequence_prob
  5. Saves a backup, then overwrites the pickle

Usage:
    python fix_qwen3_prompt_leakage.py                  # dry-run (default)
    python fix_qwen3_prompt_leakage.py --apply           # actually fix files
    python fix_qwen3_prompt_leakage.py --apply --all     # fix all Qwen models, not just Qwen3-8B
"""

import argparse
import copy
import os
import pickle
import re
import shutil
import sys
import types
from pathlib import Path

# Torch stub for unpickling
if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.Tensor = type("Tensor", (), {})
    _s = types.ModuleType("torch.storage")
    _s._load_from_bytes = lambda b: b
    _u = types.ModuleType("torch._utils")
    _u._rebuild_tensor_v2 = lambda *a, **kw: []
    sys.modules["torch"] = _torch
    sys.modules["torch.storage"] = _s
    sys.modules["torch._utils"] = _u

sys.path.insert(0, str(Path(__file__).parent / ".pylibs"))
import numpy as np

WANDB_BASE = Path("src/boldis/uncertainty/wandb")

LEAK_PATTERNS = [
    "Answer the following question in one complete sentence.",
    "Answer in one complete sentence.",
    "Answer the following question in a complete sentence.",
    "Answer the following question in a single complete sentence.",
    "Answer the following question in a complete, informative sentence.",
    "Provide a detailed, well-structured answer to the following question.",
]


def find_leak_position(response: str) -> int:
    """Return character index where leaked prompt text starts, or -1 if none found."""
    for pattern in LEAK_PATTERNS:
        idx = response.find(pattern)
        if idx > 0:
            return idx
    return -1


def clean_entry(entry: dict, key: str = "most_likely_answer") -> dict:
    """Clean a single entry (mla or high-temp response). Returns info dict."""
    mla = entry if key is None else entry.get(key)
    if mla is None:
        return {"changed": False}

    resp = mla.get("response", "")
    leak_pos = find_leak_position(resp)
    if leak_pos < 0:
        return {"changed": False}

    clean_resp = resp[:leak_pos].rstrip()
    tokens = mla.get("tokens", [])
    token_lls = mla.get("token_log_likelihoods", [])

    # Find cut point in token list by reconstructing text character-by-character
    char_count = 0
    cut_idx = len(tokens)
    for i, tok in enumerate(tokens):
        char_count += len(tok)
        if char_count >= leak_pos:
            # This token is at or past the leak boundary.
            # Check if this token itself is part of the clean text
            partial = "".join(tokens[: i + 1])
            if find_leak_position(partial) >= 0:
                cut_idx = i
            else:
                cut_idx = i + 1
            break

    old_n = len(tokens)
    new_tokens = tokens[:cut_idx]
    new_lls = token_lls[:cut_idx]

    # Also strip any trailing newline token that's just whitespace
    while new_tokens and new_tokens[-1].strip() == "" and new_tokens[-1] in ("\n", "\n\n"):
        new_tokens = new_tokens[:-1]
        new_lls = new_lls[:-1]

    mla["response"] = clean_resp
    mla["tokens"] = new_tokens
    mla["token_log_likelihoods"] = new_lls

    if new_lls:
        mla["sequence_nll"] = -sum(new_lls)
        mla["sequence_prob"] = float(np.exp(sum(new_lls)))
    
    # Also truncate token_ids if present
    if "token_ids" in mla and isinstance(mla["token_ids"], list):
        mla["token_ids"] = mla["token_ids"][:cut_idx]
        while len(mla["token_ids"]) > len(new_tokens):
            mla["token_ids"] = mla["token_ids"][:-1]

    return {
        "changed": True,
        "old_len": old_n,
        "new_len": len(new_tokens),
        "removed_tokens": old_n - len(new_tokens),
    }


def process_pickle(pkl_path: str, apply: bool = False):
    """Process a single pickle file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    total = 0
    fixed_mla = 0
    fixed_resp = 0

    for eid, entry in data.items():
        total += 1

        # Fix most_likely_answer
        info = clean_entry(entry, key="most_likely_answer")
        if info["changed"]:
            fixed_mla += 1

        # Fix high-temp responses too
        responses = entry.get("responses", [])
        for resp_entry in responses:
            if isinstance(resp_entry, dict):
                ri = clean_entry(resp_entry, key=None)
                if ri["changed"]:
                    fixed_resp += 1

    pct = fixed_mla / total * 100 if total else 0
    print(f"  {pkl_path}")
    print(f"    Total: {total}, MLA fixed: {fixed_mla} ({pct:.1f}%), High-T responses fixed: {fixed_resp}")

    if apply and (fixed_mla > 0 or fixed_resp > 0):
        backup = pkl_path + ".bak_before_leak_fix"
        if not os.path.exists(backup):
            shutil.copy2(pkl_path, backup)
            print(f"    Backup: {backup}")
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)
        print(f"    SAVED (fixed)")
    elif not apply and (fixed_mla > 0 or fixed_resp > 0):
        print(f"    [DRY RUN] Would fix {fixed_mla} MLA + {fixed_resp} high-T responses")

    return fixed_mla


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually modify files (default: dry run)")
    parser.add_argument("--all", action="store_true", help="Fix all Qwen models, not just Qwen3-8B")
    args = parser.parse_args()

    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"Scope: {'All Qwen models' if args.all else 'Qwen3-8B only'}\n")

    total_fixed = 0
    for size_dir in sorted(WANDB_BASE.iterdir()):
        if not size_dir.is_dir():
            continue
        for ds_dir in sorted(size_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            for model_dir in sorted(ds_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                name = model_dir.name.lower()
                if args.all:
                    if "qwen" not in name:
                        continue
                else:
                    if "qwen3-8b" not in name:
                        continue

                runs = sorted(
                    [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
                )
                if not runs:
                    continue
                latest = runs[-1]
                pkl = latest / "files" / "validation_generations.pkl"
                if not pkl.exists():
                    continue

                fixed = process_pickle(str(pkl), apply=args.apply)
                total_fixed += fixed

    print(f"\nTotal MLA entries fixed: {total_fixed}")
    if not args.apply and total_fixed > 0:
        print("Run with --apply to actually fix the files.")


if __name__ == "__main__":
    main()
