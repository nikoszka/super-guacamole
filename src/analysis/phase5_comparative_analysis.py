"""Phase 5: Final Comparative Analysis.

This module:
1. Computes AUROC for all metrics (baselines, SAR, SE, RW-G-NLL)
2. Generates cost vs performance plot
3. Creates comprehensive comparison tables and visualizations
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
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

# Import all uncertainty measures
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uncertainty_measures.rw_gnll import (
    initialize_similarity_model,
    compute_rw_gnll
)
from uncertainty_measures.sar import compute_sar
from uncertainty_measures.semantic_entropy import (
    EntailmentDeberta,
    compute_semantic_entropy_from_entry
)
from transformers import AutoTokenizer
from models.huggingface_models import get_hf_cache_dir


def compute_baseline_metrics(token_log_likelihoods):
    """Compute baseline uncertainty metrics for a single answer."""
    if not token_log_likelihoods:
        return {
            'g_nll': 0.0,
            'average_nll': 0.0,
            'perplexity': 1.0,
            'avg_token_probability': 0.0,
        }
    
    log_probs = np.array(token_log_likelihoods)
    sequence_length = len(log_probs)
    
    g_nll = -np.sum(log_probs)
    average_nll = g_nll / sequence_length if sequence_length > 0 else 0.0
    perplexity = np.exp(average_nll) if average_nll < 700 else np.inf
    token_probs = np.exp(log_probs)
    avg_token_probability = float(np.mean(token_probs))
    
    return {
        'g_nll': float(g_nll),
        'average_nll': float(average_nll),
        'perplexity': float(perplexity) if not np.isinf(perplexity) else float('inf'),
        'avg_token_probability': avg_token_probability,
    }

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


def get_correctness_labels(data: Dict[str, Any], use_rouge: bool = False, 
                          rouge_threshold: float = 0.3) -> Dict[str, int]:
    """Get correctness labels for all examples.
    
    Args:
        data: Dictionary of examples
        use_rouge: If True, use ROUGE scores for correctness
        rouge_threshold: ROUGE-L threshold for correctness
        
    Returns:
        Dictionary mapping example_id to correctness (1=correct, 0=incorrect)
    """
    from rouge_score import rouge_scorer
    
    labels = {}
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) if use_rouge else None
    
    for example_id, entry in data.items():
        if 'most_likely_answer' not in entry:
            continue
        
        mla = entry['most_likely_answer']
        
        if use_rouge and scorer:
            # Use ROUGE score
            if 'reference' in entry and 'answers' in entry['reference']:
                true_answers = entry['reference']['answers']['text']
                pred_answer = mla.get('response', '').strip()
                if not true_answers or not pred_answer:
                    continue
                best_rougeL = 0.0
                for ref in true_answers:
                    score = scorer.score(ref.strip(), pred_answer)
                    best_rougeL = max(best_rougeL, score['rougeL'].fmeasure)
                labels[example_id] = int(best_rougeL >= rouge_threshold)
        else:
            # Use LLM judge accuracy
            accuracy = mla.get('accuracy', None)
            if accuracy is not None:
                labels[example_id] = int(accuracy > 0.5)
    
    return labels


def compute_all_metrics(
    data: Dict[str, Any],
    similarity_model,
    tokenizer,
    entailment_model,
    labels: Dict[str, int],
    use_sar: bool = True,
    use_se: bool = True,
    use_rw_gnll: bool = True,
    num_samples_sar: Optional[int] = None,
    num_samples_se: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """Compute all uncertainty metrics for all examples.
    
    Returns:
        Dictionary mapping metric_name to {example_id: score}
    """
    results = {
        'g_nll': {},
        'average_nll': {},
        'perplexity': {},
        'avg_token_prob': {},
        'rw_gnll': {},
        'sar': {},
        'semantic_entropy': {}
    }
    
    similarity_cache = {}
    
    logger.info("Computing metrics for all examples...")
    for example_id, entry in tqdm(data.items(), desc="Processing examples"):
        if example_id not in labels:
            continue
        
        if 'most_likely_answer' not in entry:
            continue
        
        mla = entry['most_likely_answer']
        token_log_likelihoods = mla.get('token_log_likelihoods', [])
        
        if not token_log_likelihoods:
            continue
        
        # Baseline metrics
        baseline_metrics = compute_baseline_metrics(token_log_likelihoods)
        results['g_nll'][example_id] = baseline_metrics['g_nll']
        results['average_nll'][example_id] = baseline_metrics['average_nll']
        results['perplexity'][example_id] = baseline_metrics['perplexity']
        results['avg_token_prob'][example_id] = baseline_metrics['avg_token_probability']
        
        # RW-G-NLL (single sample)
        if use_rw_gnll:
            try:
                rw_gnll_score, _ = compute_rw_gnll(
                    entry, similarity_model, tokenizer, cache=similarity_cache
                )
                results['rw_gnll'][example_id] = rw_gnll_score
            except Exception as e:
                logger.warning(f"Error computing RW-G-NLL for {example_id}: {e}")
        
        # SAR (multi-sample)
        if use_sar and 'responses' in entry and entry['responses']:
            try:
                sar_score, _ = compute_sar(
                    entry, similarity_model, tokenizer,
                    cache=similarity_cache,
                    num_samples=num_samples_sar
                )
                results['sar'][example_id] = sar_score
            except Exception as e:
                logger.warning(f"Error computing SAR for {example_id}: {e}")
        
        # Semantic Entropy (multi-sample)
        if use_se and 'responses' in entry and entry['responses']:
            try:
                se_score, _ = compute_semantic_entropy_from_entry(
                    entry, entailment_model, num_samples=num_samples_se
                )
                results['semantic_entropy'][example_id] = se_score
            except Exception as e:
                logger.warning(f"Error computing Semantic Entropy for {example_id}: {e}")
    
    return results


def compute_aurocs(
    results: Dict[str, Dict[str, float]],
    labels: Dict[str, int]
) -> Dict[str, float]:
    """Compute AUROC for all metrics.
    
    Args:
        results: Dictionary mapping metric_name to {example_id: score}
        labels: Dictionary mapping example_id to correctness (1=correct, 0=incorrect)
        
    Returns:
        Dictionary mapping metric_name to AUROC score
    """
    aurocs = {}
    
    for metric_name, scores in results.items():
        if not scores:
            continue
        
        # Get common example IDs
        common_ids = set(scores.keys()) & set(labels.keys())
        if len(common_ids) < 2:
            continue
        
        y_true = [labels[eid] for eid in common_ids]
        y_scores = [scores[eid] for eid in common_ids]
        
        # For AUROC, higher uncertainty should predict incorrect (label=0)
        # So we need to negate scores for metrics where lower is better (NLL, perplexity)
        # For metrics where higher is better (avg_token_prob), we use as-is
        if metric_name in ['g_nll', 'average_nll', 'perplexity', 'rw_gnll', 'sar', 'semantic_entropy']:
            # Higher uncertainty (higher score) should predict incorrect
            # roc_auc_score expects higher scores to predict positive class (correct=1)
            # So we negate: higher confidence (lower NLL) should predict correct
            y_scores_auc = [-s for s in y_scores]
        else:
            # For avg_token_prob, higher is better (higher confidence)
            y_scores_auc = y_scores
        
        try:
            auroc = roc_auc_score(y_true, y_scores_auc)
            aurocs[metric_name] = auroc
        except ValueError as e:
            logger.warning(f"Error computing AUROC for {metric_name}: {e}")
            aurocs[metric_name] = None
    
    return aurocs


def create_comparison_table(aurocs: Dict[str, float]) -> pd.DataFrame:
    """Create comparison table of AUROC scores."""
    df = pd.DataFrame([
        {'Metric': 'G-NLL', 'AUROC': aurocs.get('g_nll', None)},
        {'Metric': 'Average NLL', 'AUROC': aurocs.get('average_nll', None)},
        {'Metric': 'Perplexity', 'AUROC': aurocs.get('perplexity', None)},
        {'Metric': 'Avg Token Prob', 'AUROC': aurocs.get('avg_token_prob', None)},
        {'Metric': 'RW-G-NLL', 'AUROC': aurocs.get('rw_gnll', None)},
        {'Metric': 'SAR', 'AUROC': aurocs.get('sar', None)},
        {'Metric': 'Semantic Entropy', 'AUROC': aurocs.get('semantic_entropy', None)},
    ])
    
    # Sort by AUROC (descending)
    df = df.sort_values('AUROC', ascending=False, na_last=True)
    
    return df


def create_cost_performance_plot(
    aurocs: Dict[str, float],
    output_path: str,
    num_samples: Optional[Dict[str, int]] = None
):
    """Create cost vs performance plot.
    
    Args:
        aurocs: Dictionary mapping metric_name to AUROC
        output_path: Path to save plot
        num_samples: Optional dictionary mapping metric_name to number of samples M
    """
    if num_samples is None:
        num_samples = {
            'g_nll': 1,
            'average_nll': 1,
            'perplexity': 1,
            'avg_token_prob': 1,
            'rw_gnll': 1,
            'sar': 10,  # Default: assume 10 samples
            'semantic_entropy': 10  # Default: assume 10 samples
        }
    
    metric_labels = {
        'g_nll': 'G-NLL',
        'average_nll': 'Avg NLL',
        'perplexity': 'Perplexity',
        'avg_token_prob': 'Avg Token Prob',
        'rw_gnll': 'RW-G-NLL',
        'sar': 'SAR',
        'semantic_entropy': 'Semantic Entropy'
    }
    
    # Prepare data
    plot_data = []
    for metric_name, auroc in aurocs.items():
        if auroc is None:
            continue
        plot_data.append({
            'Metric': metric_labels.get(metric_name, metric_name),
            'AUROC': auroc,
            'Num_Samples': num_samples.get(metric_name, 1),
            'Cost': num_samples.get(metric_name, 1)  # Use num_samples as proxy for cost
        })
    
    if not plot_data:
        logger.warning("No data for cost-performance plot")
        return
    
    df = pd.DataFrame(plot_data)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    scatter = plt.scatter(df['Cost'], df['AUROC'], s=200, alpha=0.7, c=df['AUROC'], 
                         cmap='viridis', edgecolors='black', linewidths=1.5)
    
    # Add labels
    for _, row in df.iterrows():
        plt.annotate(row['Metric'], (row['Cost'], row['AUROC']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.xlabel('Computational Cost (Number of Samples M)', fontsize=14, fontweight='bold')
    plt.ylabel('AUROC (Performance)', fontsize=14, fontweight='bold')
    plt.title('Cost vs Performance: Uncertainty Metrics Comparison', 
             fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='AUROC')
    
    # Set x-axis to log scale if range is large
    if df['Cost'].max() / df['Cost'].min() > 10:
        plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Cost-performance plot saved to: {output_path}")


def create_roc_curves(
    results: Dict[str, Dict[str, float]],
    labels: Dict[str, int],
    output_dir: str
):
    """Create ROC curves for all metrics."""
    plt.figure(figsize=(10, 8))
    
    for metric_name, scores in results.items():
        if not scores:
            continue
        
        common_ids = set(scores.keys()) & set(labels.keys())
        if len(common_ids) < 2:
            continue
        
        y_true = [labels[eid] for eid in common_ids]
        y_scores = [scores[eid] for eid in common_ids]
        
        # Negate scores for metrics where lower is better
        if metric_name in ['g_nll', 'average_nll', 'perplexity', 'rw_gnll', 'sar', 'semantic_entropy']:
            y_scores_auc = [-s for s in y_scores]
        else:
            y_scores_auc = y_scores
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores_auc)
            auroc = roc_auc_score(y_true, y_scores_auc)
            
            metric_labels = {
                'g_nll': 'G-NLL',
                'average_nll': 'Avg NLL',
                'perplexity': 'Perplexity',
                'avg_token_prob': 'Avg Token Prob',
                'rw_gnll': 'RW-G-NLL',
                'sar': 'SAR',
                'semantic_entropy': 'Semantic Entropy'
            }
            label = f"{metric_labels.get(metric_name, metric_name)} (AUC={auroc:.3f})"
            plt.plot(fpr, tpr, label=label, linewidth=2)
        except Exception as e:
            logger.warning(f"Error creating ROC curve for {metric_name}: {e}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: All Metrics', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ ROC curves saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Phase 5: Final comparative analysis of all uncertainty metrics'
    )
    parser.add_argument(
        '--pickle-path',
        type=str,
        required=True,
        help='Path to validation_generations.pkl'
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
        help='Similarity model for RW-G-NLL and SAR (default: cross-encoder/stsb-roberta-large)'
    )
    parser.add_argument(
        '--use-rouge',
        action='store_true',
        help='Use ROUGE scores for correctness (for short answers)'
    )
    parser.add_argument(
        '--rouge-threshold',
        type=float,
        default=0.3,
        help='ROUGE-L threshold for correctness (default: 0.3)'
    )
    parser.add_argument(
        '--no-sar',
        action='store_true',
        help='Skip SAR computation'
    )
    parser.add_argument(
        '--no-se',
        action='store_true',
        help='Skip Semantic Entropy computation'
    )
    parser.add_argument(
        '--no-rw-gnll',
        action='store_true',
        help='Skip RW-G-NLL computation'
    )
    parser.add_argument(
        '--num-samples-sar',
        type=int,
        default=None,
        help='Limit number of samples for SAR (default: use all)'
    )
    parser.add_argument(
        '--num-samples-se',
        type=int,
        default=None,
        help='Limit number of samples for Semantic Entropy (default: use all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (if not provided, will be auto-generated from wandb-run-id and context-type)'
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
        dir_parts = ['results', 'phase5']
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
    
    # Get correctness labels
    labels = get_correctness_labels(data, use_rouge=args.use_rouge, 
                                   rouge_threshold=args.rouge_threshold)
    logger.info(f"Got correctness labels for {len(labels)} examples")
    
    # Initialize models
    logger.info("Initializing models...")
    
    # Similarity model for RW-G-NLL and SAR
    similarity_model = initialize_similarity_model(args.similarity_model)
    
    from analysis.utils import load_tokenizer
    tokenizer = load_tokenizer(args.model_name)
    
    # Entailment model for Semantic Entropy
    entailment_model = EntailmentDeberta()
    
    # Compute all metrics
    results = compute_all_metrics(
        data, similarity_model, tokenizer, entailment_model, labels,
        use_sar=not args.no_sar,
        use_se=not args.no_se,
        use_rw_gnll=not args.no_rw_gnll,
        num_samples_sar=args.num_samples_sar,
        num_samples_se=args.num_samples_se
    )
    
    # Compute AUROCs
    aurocs = compute_aurocs(results, labels)
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("AUROC RESULTS")
    logger.info("="*80)
    for metric_name, auroc in sorted(aurocs.items(), key=lambda x: x[1] or 0, reverse=True):
        if auroc is not None:
            logger.info(f"  {metric_name:20s}: {auroc:.4f}")
        else:
            logger.info(f"  {metric_name:20s}: N/A")
    
    # Create comparison table
    comparison_df = create_comparison_table(aurocs)
    csv_path = os.path.join(args.output_dir, 'auroc_comparison.csv')
    comparison_df.to_csv(csv_path, index=False)
    logger.info(f"\n✅ Comparison table saved to: {csv_path}")
    
    # Save results to JSON
    results_path = os.path.join(args.output_dir, 'all_metrics_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'aurocs': aurocs,
            'num_examples': len(labels),
            'comparison_table': comparison_df.to_dict('records')
        }, f, indent=2)
    logger.info(f"✅ Results JSON saved to: {results_path}")
    
    # Create visualizations
    cost_perf_path = os.path.join(args.output_dir, 'cost_performance_plot.png')
    create_cost_performance_plot(aurocs, cost_perf_path)
    
    create_roc_curves(results, labels, args.output_dir)
    
    logger.info("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()

