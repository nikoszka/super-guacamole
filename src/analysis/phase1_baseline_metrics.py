"""Phase 1: Expand Baseline Metrics & Data Analysis.

This module:
1. Analyzes token counts (min/mean/max) for short and long answer datasets
2. Computes baseline uncertainty metrics: G-NLL, Average NLL, Perplexity, Average Token Probability
3. Saves results to structured format for downstream analysis
"""

import argparse
import json
import logging
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pickle_data(pickle_path: str) -> Dict[str, Any]:
    """Load validation generations from pickle file.
    
    Args:
        pickle_path: Path to validation_generations.pkl file
        
    Returns:
        Dictionary mapping example IDs to entries
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")
    
    logger.info(f"Loading data from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded {len(data)} examples")
    return data


def compute_token_statistics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute token count statistics for dataset.
    
    Args:
        data: Dictionary of examples
        
    Returns:
        Dictionary with min/mean/max token counts
    """
    token_counts = []
    
    for example_id, entry in data.items():
        if 'most_likely_answer' not in entry:
            continue
        
        mla = entry['most_likely_answer']
        if 'token_log_likelihoods' in mla:
            token_counts.append(len(mla['token_log_likelihoods']))
    
    if not token_counts:
        return {
            'min': 0,
            'mean': 0.0,
            'max': 0,
            'std': 0.0,
            'median': 0.0,
            'total_examples': len(data),
            'examples_with_tokens': 0
        }
    
    return {
        'min': int(np.min(token_counts)),
        'mean': float(np.mean(token_counts)),
        'max': int(np.max(token_counts)),
        'std': float(np.std(token_counts)),
        'median': float(np.median(token_counts)),
        'total_examples': len(data),
        'examples_with_tokens': len(token_counts),
        'percentiles': {
            'p25': float(np.percentile(token_counts, 25)),
            'p50': float(np.percentile(token_counts, 50)),
            'p75': float(np.percentile(token_counts, 75)),
            'p90': float(np.percentile(token_counts, 90)),
            'p95': float(np.percentile(token_counts, 95)),
            'p99': float(np.percentile(token_counts, 99))
        }
    }


def compute_baseline_metrics(token_log_likelihoods: List[float]) -> Dict[str, float]:
    """Compute baseline uncertainty metrics for a single answer.
    
    Args:
        token_log_likelihoods: List of token log-likelihoods
        
    Returns:
        Dictionary with computed metrics
    """
    if not token_log_likelihoods:
        return {
            'g_nll': 0.0,
            'average_nll': 0.0,
            'perplexity': 1.0,
            'avg_token_probability': 0.0,
            'sequence_length': 0
        }
    
    # Convert to numpy array for efficient computation
    log_probs = np.array(token_log_likelihoods)
    sequence_length = len(log_probs)
    
    # G-NLL (Total NLL): sum of negative log-likelihoods
    g_nll = -np.sum(log_probs)
    
    # Average NLL: G-NLL normalized by sequence length
    average_nll = g_nll / sequence_length if sequence_length > 0 else 0.0
    
    # Perplexity: exponential of Average NLL
    perplexity = np.exp(average_nll) if average_nll < 700 else np.inf  # Avoid overflow
    
    # Average Token Probability: mean of token probabilities (not log-probs)
    token_probs = np.exp(log_probs)
    avg_token_probability = float(np.mean(token_probs))
    
    return {
        'g_nll': float(g_nll),
        'average_nll': float(average_nll),
        'perplexity': float(perplexity) if not np.isinf(perplexity) else float('inf'),
        'avg_token_probability': avg_token_probability,
        'sequence_length': sequence_length
    }


def process_dataset(data: Dict[str, Any], dataset_type: str = 'unknown') -> pd.DataFrame:
    """Process entire dataset and compute all baseline metrics.
    
    Args:
        data: Dictionary of examples
        dataset_type: Type of dataset ('short' or 'long')
        
    Returns:
        DataFrame with metrics for each example
    """
    results = []
    
    logger.info(f"Processing {len(data)} examples for {dataset_type} answers...")
    
    for example_id, entry in tqdm(data.items(), desc=f"Computing metrics ({dataset_type})"):
        if 'most_likely_answer' not in entry:
            continue
        
        mla = entry['most_likely_answer']
        
        if 'token_log_likelihoods' not in mla or not mla['token_log_likelihoods']:
            continue
        
        # Compute baseline metrics
        metrics = compute_baseline_metrics(mla['token_log_likelihoods'])
        
        # Add metadata
        result = {
            'example_id': example_id,
            'dataset_type': dataset_type,
            'response': mla.get('response', ''),
            'accuracy': mla.get('accuracy', None),
            **metrics
        }
        
        results.append(result)
    
    df = pd.DataFrame(results)
    logger.info(f"Computed metrics for {len(df)} examples")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Compute baseline metrics and token statistics'
    )
    parser.add_argument(
        '--short-pickle',
        type=str,
        help='Path to short answers validation_generations.pkl'
    )
    parser.add_argument(
        '--long-pickle',
        type=str,
        help='Path to long answers validation_generations.pkl'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/phase1',
        help='Output directory for results (default: results/phase1)'
    )
    parser.add_argument(
        '--output-format',
        type=str,
        choices=['json', 'csv', 'both'],
        default='both',
        help='Output format: json, csv, or both (default: both)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_stats = {}
    all_metrics = {}
    
    # Process short answers if provided
    if args.short_pickle:
        logger.info("="*80)
        logger.info("PROCESSING SHORT ANSWERS")
        logger.info("="*80)
        
        short_data = load_pickle_data(args.short_pickle)
        short_stats = compute_token_statistics(short_data)
        short_metrics_df = process_dataset(short_data, dataset_type='short')
        
        all_stats['short'] = short_stats
        all_metrics['short'] = short_metrics_df
        
        logger.info("\nShort Answer Token Statistics:")
        logger.info(f"  Min tokens: {short_stats['min']}")
        logger.info(f"  Mean tokens: {short_stats['mean']:.2f}")
        logger.info(f"  Max tokens: {short_stats['max']}")
        logger.info(f"  Std tokens: {short_stats['std']:.2f}")
        logger.info(f"  Median tokens: {short_stats['median']:.2f}")
        logger.info(f"  Examples with tokens: {short_stats['examples_with_tokens']}/{short_stats['total_examples']}")
    
    # Process long answers if provided
    if args.long_pickle:
        logger.info("\n" + "="*80)
        logger.info("PROCESSING LONG ANSWERS")
        logger.info("="*80)
        
        long_data = load_pickle_data(args.long_pickle)
        long_stats = compute_token_statistics(long_data)
        long_metrics_df = process_dataset(long_data, dataset_type='long')
        
        all_stats['long'] = long_stats
        all_metrics['long'] = long_metrics_df
        
        logger.info("\nLong Answer Token Statistics:")
        logger.info(f"  Min tokens: {long_stats['min']}")
        logger.info(f"  Mean tokens: {long_stats['mean']:.2f}")
        logger.info(f"  Max tokens: {long_stats['max']}")
        logger.info(f"  Std tokens: {long_stats['std']:.2f}")
        logger.info(f"  Median tokens: {long_stats['median']:.2f}")
        logger.info(f"  Examples with tokens: {long_stats['examples_with_tokens']}/{long_stats['total_examples']}")
    
    # Save statistics
    stats_path = os.path.join(args.output_dir, 'token_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"\n✅ Token statistics saved to: {stats_path}")
    
    # Save metrics
    if args.output_format in ['csv', 'both']:
        for dataset_type, df in all_metrics.items():
            csv_path = os.path.join(args.output_dir, f'baseline_metrics_{dataset_type}.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"✅ Metrics CSV saved to: {csv_path}")
    
    if args.output_format in ['json', 'both']:
        # Convert DataFrames to dict for JSON serialization
        metrics_dict = {}
        for dataset_type, df in all_metrics.items():
            metrics_dict[dataset_type] = df.to_dict('records')
        
        json_path = os.path.join(args.output_dir, 'baseline_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f"✅ Metrics JSON saved to: {json_path}")
    
    # Print summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    
    for dataset_type, df in all_metrics.items():
        logger.info(f"\n{dataset_type.upper()} Answers - Metric Statistics:")
        logger.info(f"  G-NLL: mean={df['g_nll'].mean():.4f}, std={df['g_nll'].std():.4f}")
        logger.info(f"  Average NLL: mean={df['average_nll'].mean():.4f}, std={df['average_nll'].std():.4f}")
        logger.info(f"  Perplexity: mean={df['perplexity'].replace([np.inf, -np.inf], np.nan).mean():.4f}, "
                   f"std={df['perplexity'].replace([np.inf, -np.inf], np.nan).std():.4f}")
        logger.info(f"  Avg Token Prob: mean={df['avg_token_probability'].mean():.6f}, "
                   f"std={df['avg_token_probability'].std():.6f}")


if __name__ == '__main__':
    main()

