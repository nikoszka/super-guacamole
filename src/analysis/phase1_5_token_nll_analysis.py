"""Phase 1.5: Token-level NLL Analysis (without SAR relevance weighting).

This module:
1. Uses per-token log-likelihoods from the G-NLL baseline as \"importance\" scores
2. Analyzes patterns: position vs NLL, POS vs NLL, NLL distributions
3. Produces visualizations and JSON outputs for further qualitative analysis

This is analogous to Phase 2, but without SAR / semantic relevance weighting.
"""

import argparse
import json
import logging
import os
import pickle
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Try to import POS tagging libraries
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning(
        "spaCy not available. Install with: pip install spacy && "
        "python -m spacy download en_core_web_sm"
    )

try:
    import nltk
    from nltk import pos_tag
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE = True
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        logging.warning("NLTK data not found. Downloading required data...")
        nltk.download("punkt", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Install with: pip install nltk")

# Imports from project (for tokenizer)
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer  # type: ignore
from models.huggingface_models import get_hf_cache_dir  # type: ignore

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


def get_pos_tags(text: str, use_spacy: bool = True) -> List[Tuple[str, str]]:
    """Get POS tags for tokens in text.

    Args:
        text: Input text
        use_spacy: If True, use spaCy; otherwise use NLTK

    Returns:
        List of (token, pos_tag) tuples
    """
    if use_spacy and SPACY_AVAILABLE:
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            return [(token.text, token.pos_) for token in doc]
        except OSError:
            logger.warning("spaCy model not found, falling back to NLTK")
            use_spacy = False

    if not use_spacy and NLTK_AVAILABLE:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        return pos_tags

    # Fallback: return empty list
    logger.warning("No POS tagging available")
    return []


def compute_token_nll_analysis(
    entry: Dict[str, Any],
    tokenizer,
    use_pos_tagging: bool = True,
) -> Optional[Dict[str, Any]]:
    """Compute token-level NLL analysis for a single entry.

    Uses per-token negative log-likelihoods (from G-NLL) as importance scores.
    """
    if "most_likely_answer" not in entry:
        return None

    mla = entry["most_likely_answer"]
    response = mla.get("response", "").strip()
    token_log_likelihoods = mla.get("token_log_likelihoods", [])

    if not response or not token_log_likelihoods:
        return None

    # Tokenize response to approximate alignment with token_log_likelihoods
    token_ids = tokenizer.encode(response, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    if len(tokens) != len(token_log_likelihoods):
        logger.warning(
            "Token count mismatch for response (len(tokens)=%d, len(log_liks)=%d). "
            "Skipping.",
            len(tokens),
            len(token_log_likelihoods),
        )
        return None

    # Negative log-likelihoods (NLL) per token
    log_probs = np.array(token_log_likelihoods, dtype=float)
    nlls = -log_probs
    probs = np.exp(log_probs)

    # Positions (normalized 0-1)
    positions = np.linspace(0, 1, len(tokens))

    # POS tags
    if use_pos_tagging:
        pos_tagged = get_pos_tags(response, use_spacy=True)
        pos_dict = {word: pos for word, pos in pos_tagged}
        pos_tags: List[str] = []
        for tok in tokens:
            pos = pos_dict.get(tok, "UNKNOWN")
            if pos == "UNKNOWN":
                clean_tok = tok.strip(".,!?;:\"'()[]{}")
                pos = pos_dict.get(clean_tok, "UNKNOWN")
            pos_tags.append(pos)
    else:
        pos_tags = ["UNKNOWN"] * len(tokens)

    # Normalized importance (z-score of NLL) for visualization
    if len(nlls) > 1:
        nll_mean = float(nlls.mean())
        nll_std = float(nlls.std()) if nlls.std() > 0 else 1.0
        nll_z = ((nlls - nll_mean) / nll_std).tolist()
    else:
        nll_z = [0.0] * len(nlls)

    return {
        "tokens": tokens,
        "token_log_likelihoods": token_log_likelihoods,
        "nlls": nlls.tolist(),
        "probs": probs.tolist(),
        "positions": positions.tolist(),
        "pos_tags": pos_tags,
        "nll_z": nll_z,
        "response": response,
        "sequence_length": len(tokens),
    }


def analyze_token_nll_patterns(
    results: List[Dict[str, Any]],
    output_dir: str,
    num_sentence_plots: int = 10,
) -> Dict[str, Any]:
    """Analyze global patterns in token-level NLL data and create plots."""
    if not results:
        logger.warning("No results to analyze")
        return {}

    # Flatten data
    all_nlls: List[float] = []
    all_probs: List[float] = []
    all_positions: List[float] = []
    all_pos_tags: List[str] = []

    for res in results:
        all_nlls.extend(res["nlls"])
        all_probs.extend(res["probs"])
        all_positions.extend(res["positions"])
        all_pos_tags.extend(res["pos_tags"])

    df = pd.DataFrame(
        {
            "nll": all_nlls,
            "prob": all_probs,
            "position": all_positions,
            "pos_tag": all_pos_tags,
        }
    )

    os.makedirs(output_dir, exist_ok=True)

    # 1) NLL vs position
    position_corr = df[["nll", "position"]].corr().iloc[0, 1]
    plt.figure(figsize=(10, 6))
    plt.scatter(df["position"], df["nll"], alpha=0.3, s=10)
    plt.xlabel("Token Position (normalized 0-1)", fontsize=12)
    plt.ylabel("Negative Log-Likelihood (NLL)", fontsize=12)
    plt.title(
        f"Token NLL vs Position\n(Correlation: {position_corr:.3f})",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "nll_vs_position.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2) NLL distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["nll"], bins=50, kde=True)
    plt.xlabel("Negative Log-Likelihood (NLL)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Distribution of Token NLLs", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "nll_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 3) NLL by POS tag (top 10 tags)
    pos_stats = df.groupby("pos_tag")["nll"].agg(["mean", "std", "count"])
    top_pos = pos_stats.sort_values("count", ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_pos)), top_pos["mean"].values)
    plt.xticks(range(len(top_pos)), top_pos.index, rotation=45, ha="right")
    plt.xlabel("Part of Speech Tag", fontsize=12)
    plt.ylabel("Mean NLL", fontsize=12)
    plt.title("Average Token NLL by Part of Speech", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "nll_by_pos.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 4) Heatmap: position bins vs mean NLL
    df["position_bin"] = pd.cut(
        df["position"],
        bins=10,
        labels=[f"{i*10}-{(i+1)*10}%" for i in range(10)],
    )
    heatmap_data = (
        df.groupby("position_bin")["nll"].mean().reindex(df["position_bin"].cat.categories)
    )

    plt.figure(figsize=(10, 4))
    sns.heatmap(
        heatmap_data.to_frame().T,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Mean NLL"},
    )
    plt.yticks([], [])
    plt.xlabel("Token Position Bin", fontsize=12)
    plt.title("Mean NLL by Token Position Bin", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "position_nll_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 5) Sentence-level visualizations: per-token NLL for a few examples
    #    We save simple bar plots and a JSON with token + NLL for qualitative inspection.
    sentence_examples: List[Dict[str, Any]] = []
    num_plots = min(num_sentence_plots, len(results))
    for i in range(num_plots):
        res = results[i]
        tokens = res["tokens"]
        nlls = res["nlls"]
        response = res["response"]

        # Bar plot of NLL per token
        plt.figure(figsize=(min(14, 0.5 * len(tokens) + 4), 4))
        plt.bar(range(len(tokens)), nlls)
        plt.xticks(
            range(len(tokens)),
            [t.replace("\n", "\\n") for t in tokens],
            rotation=90,
            fontsize=8,
        )
        plt.ylabel("NLL", fontsize=12)
        plt.title(f"Token NLLs for Example {i+1}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"example_{i+1:02d}_token_nlls.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        sentence_examples.append(
            {
                "index": i + 1,
                "response": response,
                "tokens": tokens,
                "nlls": nlls,
            }
        )

    # Save sentence-level examples for downstream HTML/notebook visualization
    examples_path = os.path.join(output_dir, "sentence_level_nll_examples.json")
    with open(examples_path, "w") as f:
        json.dump(sentence_examples, f, indent=2)
    logger.info("✅ Sentence-level NLL examples saved to: %s", examples_path)

    summary = {
        "total_tokens_analyzed": int(len(df)),
        "position_correlation_nll": float(position_corr),
        "mean_nll": float(df["nll"].mean()),
        "std_nll": float(df["nll"].std()),
        "mean_prob": float(df["prob"].mean()),
        "std_prob": float(df["prob"].std()),
        "pos_tag_statistics": {
            k: {
                "mean_nll": float(v["mean"]),
                "std_nll": float(v["std"]),
                "count": int(v["count"]),
            }
            for k, v in pos_stats.to_dict("index").items()
        },
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1.5: Token-level NLL analysis without SAR relevance weighting"
        )
    )
    parser.add_argument(
        "--pickle-path",
        type=str,
        required=True,
        help="Path to validation_generations.pkl (preferably long answers)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name used for generation (e.g., Llama-3.2-1B)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of examples to analyze (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase1_5",
        help="Output directory for results (default: results/phase1_5)",
    )
    parser.add_argument(
        "--no-pos-tagging",
        action="store_true",
        help="Skip POS tagging analysis",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data = load_pickle_data(args.pickle_path)
    example_ids = list(data.keys())[: args.sample_size]
    logger.info("Analyzing %d examples", len(example_ids))

    # Initialize tokenizer
    logger.info("Loading tokenizer for model: %s", args.model_name)
    cache_dir = get_hf_cache_dir()
    if "llama" in args.model_name.lower():
        if (
            "Llama-3" in args.model_name
            or "Llama-3.1" in args.model_name
            or "Meta-Llama-3" in args.model_name
            or "Llama-2" in args.model_name
        ):
            base = "meta-llama"
        else:
            base = "huggyllama"
        tokenizer = AutoTokenizer.from_pretrained(
            f"{base}/{args.model_name}",
            token_type_ids=None,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_name}")

    # Compute token-level NLL analysis
    results: List[Dict[str, Any]] = []
    for example_id in tqdm(example_ids, desc="Processing examples"):
        entry = data[example_id]
        res = compute_token_nll_analysis(
            entry,
            tokenizer,
            use_pos_tagging=not args.no_pos_tagging,
        )
        if res is not None:
            res["example_id"] = example_id
            results.append(res)

    logger.info("Successfully analyzed %d examples", len(results))

    # Save raw (lightweight) results
    results_path = os.path.join(args.output_dir, "token_nll_results.json")
    json_results = []
    for r in results:
        json_r = {
            "example_id": r["example_id"],
            "sequence_length": r["sequence_length"],
            "mean_nll": float(np.mean(r["nlls"])),
            "std_nll": float(np.std(r["nlls"])),
        }
        json_results.append(json_r)

    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info("✅ Token NLL summary results saved to: %s", results_path)

    # Analyze patterns and create plots
    logger.info("Analyzing global patterns...")
    summary = analyze_token_nll_patterns(results, args.output_dir)

    # Save summary
    summary_path = os.path.join(args.output_dir, "analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("✅ Analysis summary saved to: %s", summary_path)

    # Log brief summary
    logger.info("\n" + "=" * 80)
    logger.info("TOKEN-LEVEL NLL ANALYSIS SUMMARY (PHASE 1.5)")
    logger.info("=" * 80)
    logger.info("Total tokens analyzed: %d", summary.get("total_tokens_analyzed", 0))
    logger.info(
        "Position vs NLL correlation: %.4f",
        summary.get("position_correlation_nll", 0.0),
    )
    logger.info("Mean NLL: %.4f", summary.get("mean_nll", 0.0))
    logger.info("Std NLL: %.4f", summary.get("std_nll", 0.0))


if __name__ == "__main__":
    main()


