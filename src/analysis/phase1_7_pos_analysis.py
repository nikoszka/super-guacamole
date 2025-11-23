"""Phase 1.7: POS & Word-Level Analysis (Syntax vs Uncertainty).

This module:
1. Performs context-aware POS tagging on full responses.
2. Groups tokens into words to compare Token NLL vs Word NLL.
3. Analyzes if uncertainty is driven by content words (Nouns/Verbs) or function words.
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

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from models.huggingface_models import get_hf_cache_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pickle_data(pickle_path: str) -> Dict[str, Any]:
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def align_tokens_to_spacy(response: str, model_tokens: List[str], nlp) -> List[str]:
    """Aligns model tokens to Spacy POS tags using character spans."""
    doc = nlp(response)
    
    # Map character indices to POS tags
    char_to_pos = {}
    for token in doc:
        for i in range(token.idx, token.idx + len(token)):
            char_to_pos[i] = token.pos_
            
    # Align model tokens
    aligned_pos = []
    
    # We need to handle spaces that might be implicit in model tokens
    # This is a heuristic alignment
    full_text_ptr = 0
    
    for tok in model_tokens:
        # Clean token for matching (remove potential leading space marker like ' ')
        clean_tok = tok.strip('Ġ ').replace('##', '')
        if not clean_tok:
            aligned_pos.append("SPACE")
            continue
            
        # Find this token in the response starting from full_text_ptr
        start = response.find(clean_tok, full_text_ptr)
        
        if start == -1:
            # Fallback
            aligned_pos.append("UNKNOWN")
            continue
            
        # Pick the POS tag of the middle character of the token
        mid_point = start + (len(clean_tok) // 2)
        pos = char_to_pos.get(mid_point, "UNKNOWN")
        aligned_pos.append(pos)
        
        full_text_ptr = start + len(clean_tok)
        
    return aligned_pos

def group_tokens_into_words(tokens: List[str], nlls: List[float], pos_tags: List[str]) -> List[Dict[str, Any]]:
    """Groups sub-word tokens into words and aggregates NLL (sum) and POS (majority)."""
    words = []
    current_word = ""
    current_nlls = []
    current_pos = []
    
    for i, (tok, nll, pos) in enumerate(zip(tokens, nlls, pos_tags)):
        # Heuristic: new word if token starts with space (often ' ' or 'Ġ') or is punctuation
        # We check if the token starts with a space marker
        is_start = tok.startswith(' ') or tok.startswith('Ġ') or (i==0)
        
        # Clean token representation
        clean_tok = tok.replace('Ġ', '').replace(' ', '')
        
        if is_start and current_word:
            # Save previous word
            words.append({
                "word": current_word,
                "nll": sum(current_nlls), # Sum NLL for word probability
                "avg_nll": np.mean(current_nlls),
                "pos": max(set(current_pos), key=current_pos.count) if current_pos else "UNKNOWN",
                "token_count": len(current_nlls)
            })
            current_word = ""
            current_nlls = []
            current_pos = []
            
        current_word += clean_tok
        current_nlls.append(nll)
        current_pos.append(pos)
        
    # Append last word
    if current_word:
        words.append({
            "word": current_word,
            "nll": sum(current_nlls),
            "avg_nll": np.mean(current_nlls),
            "pos": max(set(current_pos), key=current_pos.count) if current_pos else "UNKNOWN",
            "token_count": len(current_nlls)
        })
        
    return words

def analyze_pos_patterns(results: List[Dict[str, Any]], output_dir: str):
    # Flatten data
    all_tokens = []
    all_words = []
    
    for res in results:
        all_tokens.extend(res['token_data'])
        all_words.extend(res['word_data'])
        
    df_tokens = pd.DataFrame(all_tokens)
    df_words = pd.DataFrame(all_words)
    
    # 1. NLL by POS (Token Level)
    if not df_tokens.empty:
        plt.figure(figsize=(12, 6))
        pos_order = df_tokens.groupby('pos')['nll'].mean().sort_values(ascending=False).index
        sns.boxplot(data=df_tokens, x='pos', y='nll', order=pos_order, showfliers=False)
        plt.xticks(rotation=45)
        plt.title("Token NLL Distribution by POS Tag")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pos_nll_boxplot_token.png'))
        plt.close()
    
    # 2. NLL by POS (Word Level)
    if not df_words.empty:
        plt.figure(figsize=(12, 6))
        word_pos_order = df_words.groupby('pos')['nll'].mean().sort_values(ascending=False).index
        sns.boxplot(data=df_words, x='pos', y='nll', order=word_pos_order, showfliers=False)
        plt.xticks(rotation=45)
        plt.title("Word NLL (Sum) Distribution by POS Tag")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pos_nll_boxplot_word.png'))
        plt.close()
    
    # Summary Stats
    summary = {
        "token_stats": df_tokens.groupby('pos')['nll'].agg(['mean', 'std', 'count']).to_dict('index') if not df_tokens.empty else {},
        "word_stats": df_words.groupby('pos')['nll'].agg(['mean', 'std', 'count']).to_dict('index') if not df_words.empty else {}
    }
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--context-type", default="long")
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()
    
    # Setup Output Dir
    if args.output_dir is None:
        dir_parts = ["results", "phase1_7"]
        if args.context_type: dir_parts.append(args.context_type)
        if args.wandb_run_id: dir_parts.append(args.wandb_run_id)
        args.output_dir = "_".join(dir_parts)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load resources
    if not SPACY_AVAILABLE:
        logger.error("SpaCy is required for Phase 1.7 alignment. Please install spacy and download en_core_web_sm.")
        return
    
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading en_core_web_sm model...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # Tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    cache_dir = get_hf_cache_dir()
    if 'llama' in args.model_name.lower():
        if 'Llama-3' in args.model_name or 'Llama-3.1' in args.model_name or 'Meta-Llama-3' in args.model_name:
            base = 'meta-llama'
        else:
            base = 'huggyllama'
    else:
        base = 'huggyllama' # Default assumption
        
    tokenizer = AutoTokenizer.from_pretrained(f"{base}/{args.model_name}", cache_dir=cache_dir)
    
    data = load_pickle_data(args.pickle_path)
    example_ids = list(data.keys())[:args.sample_size]
    
    results = []
    vis_examples = []
    
    for eid in tqdm(example_ids):
        entry = data[eid]
        if "most_likely_answer" not in entry: continue
        
        mla = entry["most_likely_answer"]
        response = mla.get("response", "")
        token_log_liks = mla.get("token_log_likelihoods", [])
        
        # Get tokens
        if "token_ids" in mla:
            tokens = [tokenizer.decode([tid]) for tid in mla["token_ids"]]
        else:
            tokens = [tokenizer.decode([tid]) for tid in tokenizer.encode(response, add_special_tokens=False)]
            
        if len(tokens) != len(token_log_liks): continue
        
        nlls = [-x for x in token_log_liks]
        
        # Align POS
        pos_tags = align_tokens_to_spacy(response, tokens, nlp)
        
        # Group into Words
        word_data = group_tokens_into_words(tokens, nlls, pos_tags)
        
        # Store Data
        token_data = [{"token": t, "nll": n, "pos": p} for t, n, p in zip(tokens, nlls, pos_tags)]
        
        results.append({
            "example_id": eid,
            "token_data": token_data,
            "word_data": word_data
        })
        
        # Save for visualization (structure matches app expectations + new fields)
        vis_examples.append({
            "example_id": eid,
            "question": entry.get("question", ""),
            "correct_answer": str(entry.get("reference", {}).get("answers", {}).get("text", "N/A")),
            "response": response,
            "tokens": tokens,
            "nlls": nlls,
            "probs": np.exp(token_log_liks).tolist(),
            "accuracy": mla.get("accuracy", 0.0),
            "pos_tags": pos_tags,
            "words": word_data
        })
        
    # Analysis
    summary = analyze_pos_patterns(results, args.output_dir)
    
    with open(os.path.join(args.output_dir, "pos_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    # Save Visualization Data
    with open(os.path.join(args.output_dir, "pos_visualization_examples.json"), "w") as f:
        json.dump(vis_examples, f, indent=2)
        
    logger.info(f"Done! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()

