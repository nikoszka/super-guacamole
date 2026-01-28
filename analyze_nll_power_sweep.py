#!/usr/bin/env python3
"""
NLL Power Sweep Analysis.

This script systematically varies the power exponent applied to token NLLs
and plots AUROC vs power for multiple NLL-based methods.

Methods analyzed:
- Pure NLL^p weighting
- NLL^p + Middle Focus
- NLL^p + Negate End
- Uniform (baseline)
"""

import argparse
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from typing import Dict, List, Any, Tuple


# ============================================================================
# WEIGHTING SCHEMES
# ============================================================================

def compute_nll_power_weights(token_log_likelihoods: List[float], power: float = 2.0) -> List[float]:
    """Weight tokens by NLL raised to a power.
    
    Higher power = more amplification of uncertain tokens.
    power=1: linear NLL
    power=2: quadratic (NLL²)
    power=3: cubic
    etc.
    """
    if not token_log_likelihoods:
        return []
    
    nlls = [-log_p for log_p in token_log_likelihoods]
    
    # Ensure positive values (handle edge cases)
    min_nll = min(nlls)
    if min_nll < 0:
        nlls = [nll - min_nll + 0.1 for nll in nlls]
    
    # Apply power
    weights = [nll ** power for nll in nlls]
    
    # Normalize to prevent numerical issues with high powers
    max_w = max(weights) if weights else 1.0
    if max_w > 0:
        weights = [w / max_w for w in weights]
    
    return weights


def compute_middle_peak_weights(num_tokens: int) -> List[float]:
    """Peak in middle: tokens in the middle get more weight."""
    if num_tokens == 1:
        return [1.0]
    weights = []
    mid = (num_tokens - 1) / 2
    for i in range(num_tokens):
        dist = abs(i - mid) / mid if mid > 0 else 0
        weight = 1.0 - dist
        weights.append(weight)
    return weights


def compute_negate_end_weights(num_tokens: int, pct: float = 0.2) -> List[float]:
    """Down-weight last X% of tokens."""
    if num_tokens == 1:
        return [1.0]
    cutoff = int(num_tokens * (1 - pct))
    weights = [1.0 if i < cutoff else 0.01 for i in range(num_tokens)]
    return weights


def compute_uniform_weights(num_tokens: int) -> List[float]:
    """Uniform weighting (baseline = G-NLL)."""
    return [1.0] * num_tokens


def compute_weighted_nll(
    token_log_likelihoods: List[float],
    weights: List[float],
    normalize: bool = True
) -> float:
    """Compute weighted NLL."""
    if len(token_log_likelihoods) != len(weights):
        raise ValueError("Mismatch between token_log_likelihoods and weights")
    
    if not token_log_likelihoods:
        return 0.0
    
    weighted_nlls = [
        weights[t] * (-token_log_likelihoods[t])
        for t in range(len(token_log_likelihoods))
    ]
    numerator = sum(weighted_nlls)
    
    if normalize:
        denominator = sum(weights)
        if denominator == 0:
            return -sum(token_log_likelihoods)
        return numerator / denominator
    else:
        return numerator


# ============================================================================
# DATA LOADING
# ============================================================================

def load_pickle_data(pickle_path: str) -> Dict[str, Any]:
    """Load validation generations from pickle file."""
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")
    
    print(f"Loading data from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} examples")
    return data


def get_correctness_labels(data: Dict[str, Any]) -> Dict[str, int]:
    """Get correctness labels using LLM judge accuracy."""
    labels = {}
    
    for example_id, entry in data.items():
        if 'most_likely_answer' not in entry:
            continue
        
        mla = entry['most_likely_answer']
        accuracy = mla.get('accuracy', None)
        if accuracy is not None:
            labels[example_id] = int(accuracy > 0.5)
    
    return labels


# ============================================================================
# POWER SWEEP ANALYSIS
# ============================================================================

def compute_power_sweep(
    data: Dict[str, Any],
    labels: Dict[str, int],
    powers: List[float],
) -> Dict[str, Dict[float, float]]:
    """Compute AUROC for different powers across multiple methods.
    
    Returns:
        Dictionary mapping method_name -> {power: auroc}
    """
    results = {
        'NLL^p (Pure)': {},
        'NLL^p × Middle': {},
        'NLL^p + Negate End': {},
        'Uniform (G-NLL)': {},
    }
    
    print(f"\nComputing power sweep for powers: {powers}")
    
    for power in tqdm(powers, desc="Power sweep"):
        # Store scores for each method
        scores = {
            'NLL^p (Pure)': {},
            'NLL^p × Middle': {},
            'NLL^p + Negate End': {},
            'Uniform (G-NLL)': {},
        }
        
        for example_id, entry in data.items():
            if example_id not in labels:
                continue
            
            if 'most_likely_answer' not in entry:
                continue
            
            mla = entry['most_likely_answer']
            token_log_likelihoods = mla.get('token_log_likelihoods', [])
            
            if not token_log_likelihoods:
                continue
            
            num_tokens = len(token_log_likelihoods)
            
            try:
                # Pure NLL^p
                nll_power_w = compute_nll_power_weights(token_log_likelihoods, power=power)
                scores['NLL^p (Pure)'][example_id] = compute_weighted_nll(
                    token_log_likelihoods, nll_power_w, normalize=True
                )
                
                # NLL^p × Middle Focus
                middle_w = compute_middle_peak_weights(num_tokens)
                combined_middle_w = [nll_power_w[i] * middle_w[i] for i in range(num_tokens)]
                scores['NLL^p × Middle'][example_id] = compute_weighted_nll(
                    token_log_likelihoods, combined_middle_w, normalize=True
                )
                
                # NLL^p + Negate End
                negate_end_w = compute_negate_end_weights(num_tokens, pct=0.2)
                combined_negate_w = [nll_power_w[i] * negate_end_w[i] for i in range(num_tokens)]
                scores['NLL^p + Negate End'][example_id] = compute_weighted_nll(
                    token_log_likelihoods, combined_negate_w, normalize=True
                )
                
                # Uniform (baseline)
                uniform_w = compute_uniform_weights(num_tokens)
                scores['Uniform (G-NLL)'][example_id] = compute_weighted_nll(
                    token_log_likelihoods, uniform_w, normalize=True
                )
                
            except Exception as e:
                continue
        
        # Compute AUROC for each method at this power
        for method_name, method_scores in scores.items():
            if not method_scores:
                continue
            
            common_ids = set(method_scores.keys()) & set(labels.keys())
            if len(common_ids) < 2:
                continue
            
            y_true = [labels[eid] for eid in common_ids]
            y_scores = [method_scores[eid] for eid in common_ids]
            
            # Negate: higher NLL = more uncertain = predicts incorrect
            y_scores_auc = [-s for s in y_scores]
            
            try:
                auroc = roc_auc_score(y_true, y_scores_auc)
                results[method_name][power] = auroc
            except ValueError:
                results[method_name][power] = None
    
    return results


def plot_power_vs_auroc(
    results: Dict[str, Dict[float, float]],
    output_path: str
):
    """Create the power vs AUROC plot."""
    
    # Set up the figure with a dark, sophisticated theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Color palette - distinctive, vibrant colors
    colors = {
        'NLL^p (Pure)': '#00d4aa',  # Teal/cyan
        'NLL^p × Middle': '#ff6b6b',  # Coral red
        'NLL^p + Negate End': '#ffd93d',  # Gold/yellow
        'Uniform (G-NLL)': '#6c757d',  # Gray (baseline)
    }
    
    markers = {
        'NLL^p (Pure)': 'o',
        'NLL^p × Middle': 's',
        'NLL^p + Negate End': '^',
        'Uniform (G-NLL)': 'D',
    }
    
    # Plot each method
    for method_name, power_aurocs in results.items():
        if not power_aurocs:
            continue
        
        powers = sorted(power_aurocs.keys())
        aurocs = [power_aurocs[p] for p in powers]
        
        # Skip if no valid data
        if all(a is None for a in aurocs):
            continue
        
        # Handle None values
        valid_powers = [p for p, a in zip(powers, aurocs) if a is not None]
        valid_aurocs = [a for a in aurocs if a is not None]
        
        color = colors.get(method_name, '#888888')
        marker = markers.get(method_name, 'o')
        
        # Line style
        linestyle = '--' if method_name == 'Uniform (G-NLL)' else '-'
        linewidth = 2 if method_name == 'Uniform (G-NLL)' else 3
        
        ax.plot(valid_powers, valid_aurocs, 
                color=color, marker=marker, markersize=10,
                linewidth=linewidth, linestyle=linestyle,
                label=method_name, alpha=0.9)
        
        # Fill under the curve for emphasis
        if method_name != 'Uniform (G-NLL)':
            ax.fill_between(valid_powers, 0.5, valid_aurocs, 
                           color=color, alpha=0.1)
    
    # Reference line at random chance
    ax.axhline(y=0.5, color='#444444', linestyle=':', linewidth=2, alpha=0.8)
    ax.text(0.5, 0.505, 'Random (0.5)', fontsize=10, color='#888888', 
            ha='left', va='bottom')
    
    # Find and annotate the best point
    best_auroc = 0
    best_power = 0
    best_method = ''
    for method_name, power_aurocs in results.items():
        for p, a in power_aurocs.items():
            if a is not None and a > best_auroc:
                best_auroc = a
                best_power = p
                best_method = method_name
    
    if best_auroc > 0:
        ax.annotate(f'Best: {best_auroc:.4f}\n(power={best_power})',
                   xy=(best_power, best_auroc),
                   xytext=(best_power + 0.5, best_auroc - 0.05),
                   fontsize=11, color='white', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', 
                            edgecolor=colors.get(best_method, 'white'), linewidth=2))
    
    # Styling
    ax.set_xlabel('Power (p)', fontsize=16, fontweight='bold', color='white')
    ax.set_ylabel('AUROC', fontsize=16, fontweight='bold', color='white')
    ax.set_title('Effect of Token Probability Exponentiation on AUROC\n'
                 'Higher power amplifies uncertainty from low-probability tokens',
                 fontsize=18, fontweight='bold', color='white', pad=20)
    
    # Set x-axis ticks
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    
    # Set y-axis range
    all_aurocs = [a for method_aurocs in results.values() 
                  for a in method_aurocs.values() if a is not None]
    if all_aurocs:
        y_min = min(0.45, min(all_aurocs) - 0.02)
        y_max = max(all_aurocs) + 0.03
        ax.set_ylim(y_min, y_max)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0d0d1a')
    
    # Legend
    legend = ax.legend(loc='lower right', fontsize=12, framealpha=0.9,
                       facecolor='#1a1a2e', edgecolor='#444444')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Add annotation explaining the trend
    ax.text(0.02, 0.98, 
            'Key insight: Exponentiating token probabilities\n'
            '(raising NLL to power p) amplifies signal from\n'
            'low-confidence tokens, improving detection.',
            transform=ax.transAxes, fontsize=10, color='#aaaaaa',
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', 
                     edgecolor='#333333', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print(f"✅ Plot saved to: {output_path}")


def print_results_table(results: Dict[str, Dict[float, float]]):
    """Print a formatted table of results."""
    print("\n" + "="*80)
    print("POWER vs AUROC RESULTS")
    print("="*80)
    
    # Get all powers
    all_powers = set()
    for method_aurocs in results.values():
        all_powers.update(method_aurocs.keys())
    powers = sorted(all_powers)
    
    # Header
    header = f"{'Power':>8}"
    for method in results.keys():
        short_name = method.replace(' (Pure)', '').replace(' (G-NLL)', '')[:15]
        header += f" | {short_name:>15}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for power in powers:
        row = f"{power:>8.1f}"
        for method_name, method_aurocs in results.items():
            auroc = method_aurocs.get(power)
            if auroc is not None:
                row += f" | {auroc:>15.4f}"
            else:
                row += f" | {'N/A':>15}"
        print(row)
    
    print("="*80)
    
    # Best results
    print("\nBest AUROC per method:")
    for method_name, method_aurocs in results.items():
        valid = {p: a for p, a in method_aurocs.items() if a is not None}
        if valid:
            best_power = max(valid, key=valid.get)
            best_auroc = valid[best_power]
            print(f"  {method_name}: {best_auroc:.4f} (power={best_power})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze effect of NLL power on AUROC'
    )
    parser.add_argument(
        '--pickle-path',
        type=str,
        default='src/boldis/uncertainty/wandb/run-20251121_190025-5qvhbs97/files/validation_generations.pkl',
        help='Path to validation_generations.pkl'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/phase6_extended_analysis',
        help='Output directory'
    )
    parser.add_argument(
        '--powers',
        type=str,
        default='1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10',
        help='Comma-separated list of powers to test'
    )
    
    args = parser.parse_args()
    
    # Parse powers
    powers = [float(p) for p in args.powers.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data = load_pickle_data(args.pickle_path)
    labels = get_correctness_labels(data)
    print(f"Got correctness labels for {len(labels)} examples")
    
    # Compute power sweep
    results = compute_power_sweep(data, labels, powers)
    
    # Print results
    print_results_table(results)
    
    # Save results as JSON
    json_path = os.path.join(args.output_dir, 'nll_power_sweep_results.json')
    # Convert to serializable format
    serializable_results = {
        method: {str(p): a for p, a in power_aurocs.items()}
        for method, power_aurocs in results.items()
    }
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n✅ Results saved to: {json_path}")
    
    # Create plot
    output_path = os.path.join(args.output_dir, 'nll_power_sweep.png')
    plot_power_vs_auroc(results, output_path)
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
