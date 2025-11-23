"""Phase 2: Qualitative Token Importance Analysis.

This module:
1. Computes token importance scores using relevance formula from SAR paper
2. Analyzes patterns: position correlation, POS correlation, NLL correlation
3. Generates visualizations and reports on "generative inequality" problem
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
    logging.warning("spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    import nltk
    from nltk import pos_tag
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        logging.warning("NLTK data not found. Downloading required data...")
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Install with: pip install nltk")

# Import RW-G-NLL functions
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uncertainty_measures.rw_gnll import (
    initialize_similarity_model,
    compute_token_relevance_weights
)
from transformers import AutoTokenizer
from models.huggingface_models import get_hf_cache_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pickle_data(pickle_path: str) -> Dict[str, Any]:
    """Load validation generations from pickle file."""
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")
    
    logger.info(f"Loading data from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded {len(data)} examples")
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


def compute_token_importance_analysis(
    entry: Dict[str, Any],
    similarity_model,
    tokenizer,
    cache: Optional[Dict] = None,
    use_pos_tagging: bool = True
) -> Optional[Dict[str, Any]]:
    """Compute token importance and related analysis for a single entry.
    
    Args:
        entry: Dictionary containing question, context, and most_likely_answer
        similarity_model: Initialized similarity model
        tokenizer: Tokenizer for the model
        cache: Optional cache for similarity computations
        use_pos_tagging: Whether to compute POS tags
        
    Returns:
        Dictionary with analysis results or None if computation fails
    """
    if 'most_likely_answer' not in entry:
        return None
    
    mla = entry['most_likely_answer']
    response = mla.get('response', '').strip()
    token_log_likelihoods = mla.get('token_log_likelihoods', [])
    
    if not response or not token_log_likelihoods:
        return None
    
    # Construct prompt
    question = entry.get('question', '')
    context = entry.get('context', '')
    if context:
        prompt_x = f"{context} {question}".strip()
    else:
        prompt_x = question
    
    try:
        # Compute relevance weights
        relevance_weights = compute_token_relevance_weights(
            prompt_x, response, tokenizer, similarity_model,
            cache=cache, show_progress=False
        )
        
        if len(relevance_weights) != len(token_log_likelihoods):
            logger.warning(f"Token count mismatch: {len(relevance_weights)} vs {len(token_log_likelihoods)}")
            return None
        
        # Compute NLLs (negative log-likelihoods)
        nlls = [-log_prob for log_prob in token_log_likelihoods]
        
        # Try to use stored tokens/token_ids if available (exact alignment guaranteed)
        if "tokens" in mla and mla["tokens"] and len(mla["tokens"]) == len(token_log_likelihoods):
            tokens = mla["tokens"]
            logger.debug("Using stored tokens from pickle (exact alignment)")
        elif "token_ids" in mla and mla["token_ids"] and len(mla["token_ids"]) == len(token_log_likelihoods):
            # Reconstruct tokens from token_ids
            token_ids = mla["token_ids"]
            tokens = [tokenizer.decode([tid]) for tid in token_ids]
            logger.debug("Using stored token_ids from pickle (exact alignment)")
        else:
            # Fallback: re-tokenize response (may have alignment issues with old pickles)
            logger.debug("No stored tokens/token_ids found, re-tokenizing response")
            token_ids = tokenizer.encode(response, add_special_tokens=False)
            tokens = [tokenizer.decode([tid]) for tid in token_ids]
            
            if len(tokens) != len(token_log_likelihoods):
                logger.warning(
                    f"Token count mismatch after re-tokenization: {len(tokens)} vs {len(token_log_likelihoods)}. "
                    "Re-generate pickle with token_ids for exact alignment."
                )
                return None
        
        # Get positions (normalized 0-1)
        positions = np.linspace(0, 1, len(tokens))
        
        # Get POS tags if available
        pos_tags = []
        if use_pos_tagging:
            pos_tagged = get_pos_tags(response, use_spacy=True)
            # Map tokens to POS tags (approximate matching)
            token_text = ' '.join(tokens)
            pos_dict = {word: pos for word, pos in pos_tagged}
            for token in tokens:
                # Try to find matching POS tag
                pos = pos_dict.get(token, 'UNKNOWN')
                if pos == 'UNKNOWN':
                    # Try without punctuation
                    clean_token = token.strip('.,!?;:')
                    pos = pos_dict.get(clean_token, 'UNKNOWN')
                pos_tags.append(pos)
        else:
            pos_tags = ['UNKNOWN'] * len(tokens)
        
        return {
            'tokens': tokens,
            'relevance_weights': relevance_weights,
            'nlls': nlls,
            'token_log_likelihoods': token_log_likelihoods,
            'positions': positions.tolist(),
            'pos_tags': pos_tags,
            'response': response,
            'sequence_length': len(tokens)
        }
    
    except Exception as e:
        logger.warning(f"Error computing token importance: {e}")
        return None


def analyze_token_importance_patterns(
    results: List[Dict[str, Any]],
    output_dir: str
) -> Dict[str, Any]:
    """Analyze patterns in token importance data.
    
    Args:
        results: List of token importance analysis results
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with analysis statistics
    """
    if not results:
        logger.warning("No results to analyze")
        return {}
    
    # Flatten data for analysis
    all_relevance = []
    all_nlls = []
    all_positions = []
    all_pos_tags = []
    
    for result in results:
        all_relevance.extend(result['relevance_weights'])
        all_nlls.extend(result['nlls'])
        all_positions.extend(result['positions'])
        all_pos_tags.extend(result['pos_tags'])
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'relevance_weight': all_relevance,
        'nll': all_nlls,
        'position': all_positions,
        'pos_tag': all_pos_tags
    })
    
    # 1. Position correlation analysis
    position_corr = df[['relevance_weight', 'position']].corr().iloc[0, 1]
    
    # 2. NLL correlation analysis
    nll_corr = df[['relevance_weight', 'nll']].corr().iloc[0, 1]
    
    # 3. POS analysis
    pos_stats = df.groupby('pos_tag')['relevance_weight'].agg(['mean', 'std', 'count']).to_dict('index')
    
    # Create visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Relevance vs Position
    plt.figure(figsize=(10, 6))
    plt.scatter(df['position'], df['relevance_weight'], alpha=0.3, s=10)
    plt.xlabel('Token Position (normalized 0-1)', fontsize=12)
    plt.ylabel('Relevance Weight R_T(y_t)', fontsize=12)
    plt.title(f'Token Importance vs Position\n(Correlation: {position_corr:.3f})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relevance_vs_position.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Relevance vs NLL
    plt.figure(figsize=(10, 6))
    plt.scatter(df['nll'], df['relevance_weight'], alpha=0.3, s=10)
    plt.xlabel('Negative Log-Likelihood (NLL)', fontsize=12)
    plt.ylabel('Relevance Weight R_T(y_t)', fontsize=12)
    plt.title(f'Token Importance vs Uncertainty (NLL)\n(Correlation: {nll_corr:.3f})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relevance_vs_nll.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: POS tag analysis (top 10 most common)
    pos_counts = df['pos_tag'].value_counts().head(10)
    pos_means = df.groupby('pos_tag')['relevance_weight'].mean().loc[pos_counts.index]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(pos_counts)), pos_means.values)
    plt.xticks(range(len(pos_counts)), pos_counts.index, rotation=45, ha='right')
    plt.xlabel('Part of Speech Tag', fontsize=12)
    plt.ylabel('Mean Relevance Weight', fontsize=12)
    plt.title('Average Token Importance by Part of Speech', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relevance_by_pos.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Heatmap of position bins vs relevance
    df['position_bin'] = pd.cut(df['position'], bins=10, labels=[f'{i*10}-{(i+1)*10}%' for i in range(10)])
    heatmap_data = df.groupby(['position_bin', 'pos_tag'])['relevance_weight'].mean().unstack(fill_value=0)
    
    if heatmap_data.shape[1] > 0:
        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Mean Relevance Weight'})
        plt.xlabel('Part of Speech Tag', fontsize=12)
        plt.ylabel('Token Position (normalized)', fontsize=12)
        plt.title('Token Importance Heatmap: Position vs POS', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'position_pos_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Summary statistics
    summary = {
        'total_tokens_analyzed': len(df),
        'position_correlation': float(position_corr),
        'nll_correlation': float(nll_corr),
        'mean_relevance_weight': float(df['relevance_weight'].mean()),
        'std_relevance_weight': float(df['relevance_weight'].std()),
        'mean_nll': float(df['nll'].mean()),
        'std_nll': float(df['nll'].std()),
        'pos_tag_statistics': {k: {
            'mean_relevance': float(v['mean']),
            'std_relevance': float(v['std']),
            'count': int(v['count'])
        } for k, v in pos_stats.items()}
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: Token importance qualitative analysis'
    )
    parser.add_argument(
        '--pickle-path',
        type=str,
        required=True,
        help='Path to validation_generations.pkl (preferably long answers)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='Number of examples to analyze (default: 100)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Model name used for generation (e.g., Llama-3.2-1B)'
    )
    parser.add_argument(
        '--similarity-model',
        type=str,
        default='cross-encoder/stsb-roberta-large',
        help='Similarity model for relevance computation (default: cross-encoder/stsb-roberta-large)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (if not provided, will be auto-generated from wandb-run-id and context-type)'
    )
    parser.add_argument(
        '--no-pos-tagging',
        action='store_true',
        help='Skip POS tagging analysis'
    )
    parser.add_argument(
        '--wandb-run-id',
        type=str,
        default=None,
        help='WandB run ID to include in output folder name'
    )
    parser.add_argument(
        '--context-type',
        type=str,
        choices=['short', 'long'],
        default=None,
        help='Context type: short or long (used in output folder naming)'
    )
    
    args = parser.parse_args()
    
    # Auto-generate output directory if not provided
    if args.output_dir is None:
        dir_parts = ['results', 'phase2']
        if args.context_type:
            dir_parts.append(args.context_type)
        if args.wandb_run_id:
            dir_parts.append(args.wandb_run_id)
        args.output_dir = '_'.join(dir_parts)
        logger.info(f"Auto-generated output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data = load_pickle_data(args.pickle_path)
    
    # Sample examples (prefer long answers)
    example_ids = list(data.keys())[:args.sample_size]
    logger.info(f"Analyzing {len(example_ids)} examples")
    
    # Initialize models
    logger.info("Initializing similarity model...")
    similarity_model = initialize_similarity_model(args.similarity_model)
    
    logger.info(f"Loading tokenizer for model: {args.model_name}")
    cache_dir = get_hf_cache_dir()
    from transformers import AutoTokenizer
    
    # Determine base path for model
    if 'llama' in args.model_name.lower():
        if 'Llama-3' in args.model_name or 'Llama-3.1' in args.model_name or 'Meta-Llama-3' in args.model_name or 'Llama-2' in args.model_name:
            base = 'meta-llama'
        else:
            base = 'huggyllama'
        tokenizer = AutoTokenizer.from_pretrained(
            f"{base}/{args.model_name}",
            token_type_ids=None,
            cache_dir=cache_dir
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_name}")
    
    # Compute token importance for each example
    results = []
    similarity_cache = {}
    
    logger.info("Computing token importance scores...")
    for example_id in tqdm(example_ids, desc="Processing examples"):
        entry = data[example_id]
        result = compute_token_importance_analysis(
            entry, similarity_model, tokenizer,
            cache=similarity_cache,
            use_pos_tagging=not args.no_pos_tagging
        )
        if result:
            result['example_id'] = example_id
            results.append(result)
    
    logger.info(f"Successfully analyzed {len(results)} examples")
    
    # Save raw results (without tokens) for global analysis
    results_path = os.path.join(args.output_dir, 'token_importance_results.json')
    json_results = []
    for r in results:
        json_r = {k: v for k, v in r.items() if k != 'tokens'}
        json_r['num_tokens'] = len(r['tokens'])
        json_results.append(json_r)
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"✅ Results saved to: {results_path}")

    # Additionally save a small set of examples with full token information
    # for qualitative visualization (e.g., in a Streamlit app).
    max_examples_for_vis = min(10, len(results))
    vis_examples = []
    for r in results[:max_examples_for_vis]:
        vis_examples.append({
            'example_id': r.get('example_id'),
            'response': r.get('response', ''),
            'tokens': r.get('tokens', []),
            'relevance_weights': r.get('relevance_weights', []),
            'nlls': r.get('nlls', []),
            'positions': r.get('positions', []),
            'pos_tags': r.get('pos_tags', []),
        })
    
    vis_path = os.path.join(args.output_dir, 'token_importance_examples.json')
    with open(vis_path, 'w') as f:
        json.dump(vis_examples, f, indent=2)
    logger.info(f"✅ Visualization examples saved to: {vis_path}")
    
    # Analyze patterns
    logger.info("Analyzing patterns...")
    summary = analyze_token_importance_patterns(results, args.output_dir)
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✅ Analysis summary saved to: {summary_path}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TOKEN IMPORTANCE ANALYSIS SUMMARY")
    logger.info("="*80)
    logger.info(f"Total tokens analyzed: {summary.get('total_tokens_analyzed', 0)}")
    logger.info(f"Position correlation: {summary.get('position_correlation', 0):.4f}")
    logger.info(f"NLL correlation: {summary.get('nll_correlation', 0):.4f}")
    logger.info(f"Mean relevance weight: {summary.get('mean_relevance_weight', 0):.4f}")
    logger.info(f"Mean NLL: {summary.get('mean_nll', 0):.4f}")
    logger.info("\nTop POS tags by mean relevance:")
    pos_stats = summary.get('pos_tag_statistics', {})
    sorted_pos = sorted(pos_stats.items(), key=lambda x: x[1]['mean_relevance'], reverse=True)[:10]
    for pos, stats in sorted_pos:
        logger.info(f"  {pos}: mean={stats['mean_relevance']:.4f}, count={stats['count']}")


if __name__ == '__main__':
    main()

