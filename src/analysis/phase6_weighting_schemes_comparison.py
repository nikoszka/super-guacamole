"""Phase 6: Token Weighting Schemes Comparison.

This module:
1. Implements multiple token weighting schemes (position-based, SAR-style, etc.)
2. Computes weighted NLL for each scheme
3. Compares AUROC across all weighting schemes
4. Visualizes weight patterns
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
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

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


# ============================================================================
# WEIGHTING SCHEMES
# ============================================================================

def compute_uniform_weights(num_tokens: int) -> List[float]:
    """Uniform weighting (baseline = G-NLL)."""
    return [1.0] * num_tokens


def compute_linear_increasing_weights(num_tokens: int) -> List[float]:
    """Linear increase: later tokens get more weight."""
    if num_tokens == 1:
        return [1.0]
    return [i / (num_tokens - 1) for i in range(num_tokens)]


def compute_linear_decreasing_weights(num_tokens: int) -> List[float]:
    """Linear decrease: earlier tokens get more weight."""
    if num_tokens == 1:
        return [1.0]
    return [(num_tokens - 1 - i) / (num_tokens - 1) for i in range(num_tokens)]


def compute_quadratic_increasing_weights(num_tokens: int) -> List[float]:
    """Quadratic increase: exponentially favor later tokens."""
    if num_tokens == 1:
        return [1.0]
    return [(i / (num_tokens - 1)) ** 2 for i in range(num_tokens)]


def compute_quadratic_decreasing_weights(num_tokens: int) -> List[float]:
    """Quadratic decrease: exponentially favor earlier tokens."""
    if num_tokens == 1:
        return [1.0]
    return [((num_tokens - 1 - i) / (num_tokens - 1)) ** 2 for i in range(num_tokens)]


def compute_middle_peak_weights(num_tokens: int) -> List[float]:
    """Peak in middle: tokens in the middle get more weight."""
    if num_tokens == 1:
        return [1.0]
    weights = []
    mid = (num_tokens - 1) / 2
    for i in range(num_tokens):
        # Distance from middle, normalized
        dist = abs(i - mid) / mid if mid > 0 else 0
        weight = 1.0 - dist
        weights.append(weight)
    return weights


def compute_edges_peak_weights(num_tokens: int) -> List[float]:
    """Peak at edges: first and last tokens get more weight."""
    if num_tokens == 1:
        return [1.0]
    weights = []
    mid = (num_tokens - 1) / 2
    for i in range(num_tokens):
        # Distance from middle, normalized
        dist = abs(i - mid) / mid if mid > 0 else 0
        weight = dist  # Higher weight = further from middle
        weights.append(weight)
    return weights


def compute_exponential_increasing_weights(num_tokens: int, rate: float = 2.0) -> List[float]:
    """Exponential increase: strongly favor later tokens."""
    if num_tokens == 1:
        return [1.0]
    weights = [np.exp(rate * i / (num_tokens - 1)) for i in range(num_tokens)]
    # Normalize to [0, 1] range
    min_w, max_w = min(weights), max(weights)
    if max_w > min_w:
        weights = [(w - min_w) / (max_w - min_w) for w in weights]
    return weights


def compute_exponential_decreasing_weights(num_tokens: int, rate: float = 2.0) -> List[float]:
    """Exponential decrease: strongly favor earlier tokens."""
    if num_tokens == 1:
        return [1.0]
    weights = [np.exp(-rate * i / (num_tokens - 1)) for i in range(num_tokens)]
    # Normalize to [0, 1] range
    min_w, max_w = min(weights), max(weights)
    if max_w > min_w:
        weights = [(w - min_w) / (max_w - min_w) for w in weights]
    return weights


def compute_sar_weights(
    entry: Dict[str, Any],
    similarity_model,
    tokenizer,
    cache: Optional[Dict] = None
) -> List[float]:
    """SAR-style relevance weights using token ablation."""
    question = entry.get('question', '')
    context = entry.get('context', '')
    prompt_x = f"{context} {question}".strip() if context else question
    
    mla = entry['most_likely_answer']
    response = mla.get('response', '')
    
    if not response:
        return []
    
    relevance_weights = compute_token_relevance_weights(
        prompt_x, response, tokenizer, similarity_model,
        cache=cache, show_progress=False
    )
    
    return relevance_weights


# ============================================================================
# ADDITIONAL ADVANCED WEIGHTING SCHEMES
# ============================================================================

def compute_gaussian_weights(num_tokens: int, center_ratio: float = 0.5, 
                             width: float = 0.3) -> List[float]:
    """Gaussian/bell curve centered at specified position.
    
    Refined version of middle_peak with smooth distribution.
    """
    if num_tokens == 1:
        return [1.0]
    
    center = (num_tokens - 1) * center_ratio
    sigma = num_tokens * width
    weights = []
    for i in range(num_tokens):
        # Gaussian: exp(-((x - μ)² / (2σ²)))
        weight = np.exp(-((i - center) ** 2) / (2 * sigma ** 2))
        weights.append(weight)
    return weights


def compute_nll_proportional_weights(token_log_likelihoods: List[float]) -> List[float]:
    """Weight tokens by their own NLL - uncertain tokens get more weight.
    
    Self-referential: tokens with high uncertainty contribute more.
    """
    if not token_log_likelihoods:
        return []
    
    nlls = [-log_p for log_p in token_log_likelihoods]
    # Avoid zero weights
    min_nll = min(nlls)
    if min_nll < 0:
        nlls = [nll - min_nll + 0.1 for nll in nlls]
    return nlls


def compute_confidence_proportional_weights(token_log_likelihoods: List[float]) -> List[float]:
    """Weight tokens by their probability - confident tokens get more weight.
    
    Inverse of NLL-proportional: high-confidence tokens contribute more.
    """
    if not token_log_likelihoods:
        return []
    
    probs = [np.exp(log_p) for log_p in token_log_likelihoods]
    return probs


def compute_surprisal_weights(token_log_likelihoods: List[float]) -> List[float]:
    """Weight by surprisal (information content): -log P(token).
    
    More surprising/unexpected tokens get higher weight.
    """
    if not token_log_likelihoods:
        return []
    
    surprisals = [-log_p for log_p in token_log_likelihoods]
    
    # Normalize to [0, 1] range
    min_s, max_s = min(surprisals), max(surprisals)
    if max_s > min_s:
        weights = [(s - min_s) / (max_s - min_s) + 0.1 for s in surprisals]  # +0.1 to avoid zeros
    else:
        weights = [1.0] * len(surprisals)
    
    return weights


def compute_first_k_weights(num_tokens: int, k: int = 3) -> List[float]:
    """Only weight first k tokens heavily.
    
    Tests if early tokens are most predictive.
    """
    if num_tokens == 1:
        return [1.0]
    
    k = min(k, num_tokens)
    weights = [1.0 if i < k else 0.1 for i in range(num_tokens)]
    return weights


def compute_last_k_weights(num_tokens: int, k: int = 3) -> List[float]:
    """Only weight last k tokens heavily.
    
    Tests if final tokens are most predictive.
    """
    if num_tokens == 1:
        return [1.0]
    
    k = min(k, num_tokens)
    weights = [1.0 if i >= num_tokens - k else 0.1 for i in range(num_tokens)]
    return weights


def compute_first_last_k_weights(num_tokens: int, k: int = 3) -> List[float]:
    """Weight both first and last k tokens heavily.
    
    Tests if extremes are most predictive (combines first-k and last-k).
    """
    if num_tokens == 1:
        return [1.0]
    
    k = min(k, num_tokens // 2)  # Ensure no overlap if sequence too short
    weights = [1.0 if (i < k or i >= num_tokens - k) else 0.1 
               for i in range(num_tokens)]
    return weights


def compute_recency_bias_weights(num_tokens: int, decay: float = 0.95) -> List[float]:
    """Exponential decay favoring recent tokens (attention-like).
    
    Mimics recency bias in human memory / attention mechanisms.
    """
    if num_tokens == 1:
        return [1.0]
    
    weights = [decay ** (num_tokens - 1 - i) for i in range(num_tokens)]
    return weights


def compute_high_uncertainty_only_weights(
    token_log_likelihoods: List[float],
    percentile: float = 75
) -> List[float]:
    """Only weight tokens with NLL above specified percentile.
    
    Focus on most uncertain tokens only.
    """
    if not token_log_likelihoods:
        return []
    
    nlls = [-log_p for log_p in token_log_likelihoods]
    threshold = np.percentile(nlls, percentile)
    weights = [1.0 if nll >= threshold else 0.1 for nll in nlls]
    return weights


def compute_low_uncertainty_only_weights(
    token_log_likelihoods: List[float],
    percentile: float = 25
) -> List[float]:
    """Only weight tokens with NLL below specified percentile.
    
    Focus on most confident tokens only.
    """
    if not token_log_likelihoods:
        return []
    
    nlls = [-log_p for log_p in token_log_likelihoods]
    threshold = np.percentile(nlls, percentile)
    weights = [1.0 if nll <= threshold else 0.1 for nll in nlls]
    return weights


def compute_hybrid_middle_sar_weights(
    num_tokens: int,
    sar_weights: List[float],
    alpha: float = 0.5
) -> List[float]:
    """Hybrid: combine middle_peak with SAR relevance.
    
    Weighted average of position-based (middle) and semantic (SAR) weights.
    alpha controls the mix: 1.0 = all middle, 0.0 = all SAR
    """
    if not sar_weights or len(sar_weights) != num_tokens:
        return compute_middle_peak_weights(num_tokens)
    
    middle_weights = compute_middle_peak_weights(num_tokens)
    hybrid = [alpha * m + (1 - alpha) * s 
              for m, s in zip(middle_weights, sar_weights)]
    return hybrid


def compute_hybrid_gaussian_sar_weights(
    num_tokens: int,
    sar_weights: List[float],
    alpha: float = 0.5
) -> List[float]:
    """Hybrid: combine gaussian with SAR relevance.
    
    Weighted average of smooth gaussian and semantic (SAR) weights.
    """
    if not sar_weights or len(sar_weights) != num_tokens:
        return compute_gaussian_weights(num_tokens)
    
    gaussian_weights = compute_gaussian_weights(num_tokens, center_ratio=0.5, width=0.3)
    hybrid = [alpha * g + (1 - alpha) * s 
              for g, s in zip(gaussian_weights, sar_weights)]
    return hybrid


# ============================================================================
# HIGHER-ORDER NLL POWER SCHEMES
# ============================================================================

def compute_nll_power_weights(token_log_likelihoods: List[float], power: float = 2.0) -> List[float]:
    """Weight tokens by NLL raised to a power.
    
    Higher power = more amplification of uncertain tokens.
    power=1: linear (same as nll_proportional before normalization effect)
    power=2: quadratic (current nll_proportional)
    power=3: cubic
    power=4: quartic
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


def compute_nll_cubic_weights(token_log_likelihoods: List[float]) -> List[float]:
    """NLL³ - cubic amplification of uncertainty."""
    return compute_nll_power_weights(token_log_likelihoods, power=3.0)


def compute_nll_quartic_weights(token_log_likelihoods: List[float]) -> List[float]:
    """NLL⁴ - quartic amplification of uncertainty."""
    return compute_nll_power_weights(token_log_likelihoods, power=4.0)


def compute_nll_quintic_weights(token_log_likelihoods: List[float]) -> List[float]:
    """NLL⁵ - quintic amplification of uncertainty."""
    return compute_nll_power_weights(token_log_likelihoods, power=5.0)


def compute_nll_sqrt_weights(token_log_likelihoods: List[float]) -> List[float]:
    """NLL^0.5 - square root (softer amplification)."""
    return compute_nll_power_weights(token_log_likelihoods, power=0.5)


# ============================================================================
# NEGATE END TOKENS SCHEMES - "Confidence Sinks" Hypothesis
# ============================================================================

def compute_negate_last_k_weights(num_tokens: int, k: int = 3) -> List[float]:
    """Negate (down-weight) last k tokens.
    
    Tests hypothesis: final tokens may be "confidence sinks" where model
    dumps certainty after making mistakes earlier.
    """
    if num_tokens == 1:
        return [1.0]
    
    k = min(k, num_tokens)
    # Normal weight for most tokens, near-zero for last k
    weights = [1.0 if i < num_tokens - k else 0.01 for i in range(num_tokens)]
    return weights


def compute_negate_first_k_weights(num_tokens: int, k: int = 3) -> List[float]:
    """Negate (down-weight) first k tokens.
    
    Control experiment: tests if beginning tokens are less informative.
    """
    if num_tokens == 1:
        return [1.0]
    
    k = min(k, num_tokens)
    weights = [0.01 if i < k else 1.0 for i in range(num_tokens)]
    return weights


def compute_negate_last_pct_weights(num_tokens: int, pct: float = 0.2) -> List[float]:
    """Negate last X% of tokens.
    
    More flexible version - percentage-based.
    """
    if num_tokens == 1:
        return [1.0]
    
    cutoff = int(num_tokens * (1 - pct))
    weights = [1.0 if i < cutoff else 0.01 for i in range(num_tokens)]
    return weights


def compute_negate_first_pct_weights(num_tokens: int, pct: float = 0.2) -> List[float]:
    """Negate first X% of tokens."""
    if num_tokens == 1:
        return [1.0]
    
    cutoff = int(num_tokens * pct)
    weights = [0.01 if i < cutoff else 1.0 for i in range(num_tokens)]
    return weights


# ============================================================================
# UNWEIGHT DIFFERENT PARTS - Position Signal Analysis
# ============================================================================

def compute_unweight_beginning_weights(num_tokens: int, pct: float = 0.33) -> List[float]:
    """Unweight (exclude) beginning third of tokens.
    
    Tests: Is the beginning the strongest hallucination signal?
    If AUROC drops → beginning is important.
    """
    if num_tokens == 1:
        return [1.0]
    
    cutoff = int(num_tokens * pct)
    weights = [0.01 if i < cutoff else 1.0 for i in range(num_tokens)]
    return weights


def compute_unweight_middle_weights(num_tokens: int, pct: float = 0.33) -> List[float]:
    """Unweight (exclude) middle third of tokens.
    
    Tests: Is the middle the strongest hallucination signal?
    If AUROC drops → middle is important.
    """
    if num_tokens == 1:
        return [1.0]
    
    start = int(num_tokens * pct)
    end = int(num_tokens * (1 - pct))
    weights = [0.01 if start <= i < end else 1.0 for i in range(num_tokens)]
    return weights


def compute_unweight_end_weights(num_tokens: int, pct: float = 0.33) -> List[float]:
    """Unweight (exclude) ending third of tokens.
    
    Tests: Is the end the strongest hallucination signal?
    If AUROC drops → end is important.
    """
    if num_tokens == 1:
        return [1.0]
    
    cutoff = int(num_tokens * (1 - pct))
    weights = [1.0 if i < cutoff else 0.01 for i in range(num_tokens)]
    return weights


def compute_only_beginning_weights(num_tokens: int, pct: float = 0.33) -> List[float]:
    """ONLY use beginning third of tokens.
    
    Direct test: How predictive is just the beginning?
    """
    if num_tokens == 1:
        return [1.0]
    
    cutoff = int(num_tokens * pct)
    weights = [1.0 if i < cutoff else 0.01 for i in range(num_tokens)]
    return weights


def compute_only_middle_weights(num_tokens: int, pct: float = 0.33) -> List[float]:
    """ONLY use middle third of tokens.
    
    Direct test: How predictive is just the middle?
    """
    if num_tokens == 1:
        return [1.0]
    
    start = int(num_tokens * pct)
    end = int(num_tokens * (1 - pct))
    weights = [1.0 if start <= i < end else 0.01 for i in range(num_tokens)]
    return weights


def compute_only_end_weights(num_tokens: int, pct: float = 0.33) -> List[float]:
    """ONLY use ending third of tokens.
    
    Direct test: How predictive is just the end?
    """
    if num_tokens == 1:
        return [1.0]
    
    cutoff = int(num_tokens * (1 - pct))
    weights = [0.01 if i < cutoff else 1.0 for i in range(num_tokens)]
    return weights


# ============================================================================
# HYBRID NLL-POWER + POSITION SCHEMES
# ============================================================================

def compute_nll_power_negate_end_weights(
    token_log_likelihoods: List[float],
    power: float = 2.0,
    negate_pct: float = 0.2
) -> List[float]:
    """Combine NLL power weighting with negating end tokens.
    
    Best of both: amplify uncertainty + ignore potential confidence sinks.
    """
    if not token_log_likelihoods:
        return []
    
    num_tokens = len(token_log_likelihoods)
    nll_weights = compute_nll_power_weights(token_log_likelihoods, power=power)
    
    # Apply end negation
    cutoff = int(num_tokens * (1 - negate_pct))
    combined = [nll_weights[i] if i < cutoff else nll_weights[i] * 0.01 
                for i in range(num_tokens)]
    return combined


def compute_nll_power_middle_focus_weights(
    token_log_likelihoods: List[float],
    power: float = 2.0
) -> List[float]:
    """Combine NLL power with middle focus.
    
    Amplify uncertainty but focus on middle tokens.
    """
    if not token_log_likelihoods:
        return []
    
    num_tokens = len(token_log_likelihoods)
    nll_weights = compute_nll_power_weights(token_log_likelihoods, power=power)
    middle_weights = compute_middle_peak_weights(num_tokens)
    
    # Multiply: NLL^power × middle_peak
    combined = [nll_weights[i] * middle_weights[i] for i in range(num_tokens)]
    return combined


# ============================================================================
# WEIGHTED NLL COMPUTATION
# ============================================================================

def compute_weighted_nll(
    token_log_likelihoods: List[float],
    weights: List[float],
    normalize: bool = True
) -> float:
    """Compute weighted NLL.
    
    Args:
        token_log_likelihoods: Log-probabilities for each token
        weights: Weight for each token
        normalize: If True, normalize by sum of weights (like RW-G-NLL)
                  If False, just sum (weighted but not normalized)
    
    Returns:
        Weighted NLL score
    """
    if len(token_log_likelihoods) != len(weights):
        raise ValueError("Mismatch between token_log_likelihoods and weights")
    
    if not token_log_likelihoods:
        return 0.0
    
    # Compute weighted NLL
    weighted_nlls = [
        weights[t] * (-token_log_likelihoods[t])
        for t in range(len(token_log_likelihoods))
    ]
    numerator = sum(weighted_nlls)
    
    if normalize:
        denominator = sum(weights)
        if denominator == 0:
            return -sum(token_log_likelihoods)  # Fallback to G-NLL
        return numerator / denominator
    else:
        return numerator


# ============================================================================
# DATA LOADING AND LABEL EXTRACTION
# ============================================================================

def load_pickle_data(pickle_path: str) -> Dict[str, Any]:
    """Load validation generations from pickle file."""
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")
    
    logger.info(f"Loading data from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded {len(data)} examples")
    return data


def get_correctness_labels(
    data: Dict[str, Any],
    use_rouge: bool = False,
    rouge_threshold: float = 0.3
) -> Dict[str, int]:
    """Get correctness labels for all examples."""
    from rouge_score import rouge_scorer
    
    labels = {}
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) if use_rouge else None
    
    for example_id, entry in data.items():
        if 'most_likely_answer' not in entry:
            continue
        
        mla = entry['most_likely_answer']
        
        if use_rouge and scorer:
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
            accuracy = mla.get('accuracy', None)
            if accuracy is not None:
                labels[example_id] = int(accuracy > 0.5)
    
    return labels


# ============================================================================
# MAIN COMPUTATION
# ============================================================================

def compute_all_weighted_nlls(
    data: Dict[str, Any],
    labels: Dict[str, int],
    similarity_model,
    tokenizer,
    normalize: bool = True
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """Compute weighted NLL for all schemes on all examples.
    
    Returns:
        Tuple of:
        - Dictionary mapping scheme_name to {example_id: weighted_nll}
        - List of example dictionaries with weights for visualization
    """
    results = {
        # Basic position-based
        'uniform': {},
        'linear_increasing': {},
        'linear_decreasing': {},
        'quadratic_increasing': {},
        'quadratic_decreasing': {},
        'exponential_increasing': {},
        'exponential_decreasing': {},
        'middle_peak': {},
        'edges_peak': {},
        # Advanced schemes
        'gaussian_middle': {},
        'nll_proportional': {},
        'confidence_proportional': {},
        'surprisal': {},
        'first_k': {},
        'last_k': {},
        'first_last_k': {},
        'recency_bias': {},
        'high_uncertainty_only': {},
        'low_uncertainty_only': {},
        # Semantic and hybrid
        'sar_relevance': {},
        'hybrid_middle_sar_50': {},
        'hybrid_gaussian_sar_50': {},
        'hybrid_middle_sar_70': {},
        'hybrid_gaussian_sar_70': {},
        # Higher-order NLL powers
        'nll_sqrt': {},           # NLL^0.5
        'nll_cubic': {},          # NLL^3
        'nll_quartic': {},        # NLL^4
        'nll_quintic': {},        # NLL^5
        # Negate end tokens (confidence sinks hypothesis)
        'negate_last_3': {},
        'negate_last_5': {},
        'negate_last_20pct': {},
        'negate_first_3': {},
        'negate_first_20pct': {},
        # Unweight different parts (position signal analysis)
        'unweight_beginning': {},
        'unweight_middle': {},
        'unweight_end': {},
        'only_beginning': {},
        'only_middle': {},
        'only_end': {},
        # Hybrid NLL-power + position
        'nll_squared_negate_end': {},
        'nll_cubic_negate_end': {},
        'nll_squared_middle_focus': {},
        'nll_cubic_middle_focus': {},
    }
    
    sar_cache = {}
    weight_examples = []  # Store some examples for visualization
    
    logger.info("Computing weighted NLL for all schemes...")
    for example_id, entry in tqdm(data.items(), desc="Processing examples"):
        if example_id not in labels:
            continue
        
        if 'most_likely_answer' not in entry:
            continue
        
        mla = entry['most_likely_answer']
        token_log_likelihoods = mla.get('token_log_likelihoods', [])
        
        if not token_log_likelihoods:
            continue
        
        num_tokens = len(token_log_likelihoods)
        
        # Compute all weight schemes
        try:
            # Basic position-based schemes
            uniform_w = compute_uniform_weights(num_tokens)
            linear_inc_w = compute_linear_increasing_weights(num_tokens)
            linear_dec_w = compute_linear_decreasing_weights(num_tokens)
            quad_inc_w = compute_quadratic_increasing_weights(num_tokens)
            quad_dec_w = compute_quadratic_decreasing_weights(num_tokens)
            exp_inc_w = compute_exponential_increasing_weights(num_tokens)
            exp_dec_w = compute_exponential_decreasing_weights(num_tokens)
            middle_w = compute_middle_peak_weights(num_tokens)
            edges_w = compute_edges_peak_weights(num_tokens)
            
            # Advanced position-based schemes
            gaussian_w = compute_gaussian_weights(num_tokens, center_ratio=0.5, width=0.3)
            nll_prop_w = compute_nll_proportional_weights(token_log_likelihoods)
            conf_prop_w = compute_confidence_proportional_weights(token_log_likelihoods)
            surprisal_w = compute_surprisal_weights(token_log_likelihoods)
            first_k_w = compute_first_k_weights(num_tokens, k=3)
            last_k_w = compute_last_k_weights(num_tokens, k=3)
            first_last_k_w = compute_first_last_k_weights(num_tokens, k=3)
            recency_w = compute_recency_bias_weights(num_tokens, decay=0.95)
            high_unc_w = compute_high_uncertainty_only_weights(token_log_likelihoods, percentile=75)
            low_unc_w = compute_low_uncertainty_only_weights(token_log_likelihoods, percentile=25)
            
            # SAR-style relevance weights
            sar_w = compute_sar_weights(entry, similarity_model, tokenizer, cache=sar_cache)
            
            if len(sar_w) != num_tokens:
                logger.warning(f"SAR weight mismatch for {example_id}, skipping SAR-based schemes for this example")
                sar_w = None
            
            # Hybrid schemes (if SAR available)
            if sar_w is not None:
                hybrid_middle_50_w = compute_hybrid_middle_sar_weights(num_tokens, sar_w, alpha=0.5)
                hybrid_gaussian_50_w = compute_hybrid_gaussian_sar_weights(num_tokens, sar_w, alpha=0.5)
                hybrid_middle_70_w = compute_hybrid_middle_sar_weights(num_tokens, sar_w, alpha=0.7)
                hybrid_gaussian_70_w = compute_hybrid_gaussian_sar_weights(num_tokens, sar_w, alpha=0.7)
            else:
                hybrid_middle_50_w = None
                hybrid_gaussian_50_w = None
                hybrid_middle_70_w = None
                hybrid_gaussian_70_w = None
            
            # Compute weighted NLLs for basic schemes
            results['uniform'][example_id] = compute_weighted_nll(
                token_log_likelihoods, uniform_w, normalize=normalize
            )
            results['linear_increasing'][example_id] = compute_weighted_nll(
                token_log_likelihoods, linear_inc_w, normalize=normalize
            )
            results['linear_decreasing'][example_id] = compute_weighted_nll(
                token_log_likelihoods, linear_dec_w, normalize=normalize
            )
            results['quadratic_increasing'][example_id] = compute_weighted_nll(
                token_log_likelihoods, quad_inc_w, normalize=normalize
            )
            results['quadratic_decreasing'][example_id] = compute_weighted_nll(
                token_log_likelihoods, quad_dec_w, normalize=normalize
            )
            results['exponential_increasing'][example_id] = compute_weighted_nll(
                token_log_likelihoods, exp_inc_w, normalize=normalize
            )
            results['exponential_decreasing'][example_id] = compute_weighted_nll(
                token_log_likelihoods, exp_dec_w, normalize=normalize
            )
            results['middle_peak'][example_id] = compute_weighted_nll(
                token_log_likelihoods, middle_w, normalize=normalize
            )
            results['edges_peak'][example_id] = compute_weighted_nll(
                token_log_likelihoods, edges_w, normalize=normalize
            )
            
            # Advanced schemes
            results['gaussian_middle'][example_id] = compute_weighted_nll(
                token_log_likelihoods, gaussian_w, normalize=normalize
            )
            results['nll_proportional'][example_id] = compute_weighted_nll(
                token_log_likelihoods, nll_prop_w, normalize=normalize
            )
            results['confidence_proportional'][example_id] = compute_weighted_nll(
                token_log_likelihoods, conf_prop_w, normalize=normalize
            )
            results['surprisal'][example_id] = compute_weighted_nll(
                token_log_likelihoods, surprisal_w, normalize=normalize
            )
            results['first_k'][example_id] = compute_weighted_nll(
                token_log_likelihoods, first_k_w, normalize=normalize
            )
            results['last_k'][example_id] = compute_weighted_nll(
                token_log_likelihoods, last_k_w, normalize=normalize
            )
            results['first_last_k'][example_id] = compute_weighted_nll(
                token_log_likelihoods, first_last_k_w, normalize=normalize
            )
            results['recency_bias'][example_id] = compute_weighted_nll(
                token_log_likelihoods, recency_w, normalize=normalize
            )
            results['high_uncertainty_only'][example_id] = compute_weighted_nll(
                token_log_likelihoods, high_unc_w, normalize=normalize
            )
            results['low_uncertainty_only'][example_id] = compute_weighted_nll(
                token_log_likelihoods, low_unc_w, normalize=normalize
            )
            
            # SAR and hybrid schemes
            if sar_w is not None:
                results['sar_relevance'][example_id] = compute_weighted_nll(
                    token_log_likelihoods, sar_w, normalize=normalize
                )
                results['hybrid_middle_sar_50'][example_id] = compute_weighted_nll(
                    token_log_likelihoods, hybrid_middle_50_w, normalize=normalize
                )
                results['hybrid_gaussian_sar_50'][example_id] = compute_weighted_nll(
                    token_log_likelihoods, hybrid_gaussian_50_w, normalize=normalize
                )
                results['hybrid_middle_sar_70'][example_id] = compute_weighted_nll(
                    token_log_likelihoods, hybrid_middle_70_w, normalize=normalize
                )
                results['hybrid_gaussian_sar_70'][example_id] = compute_weighted_nll(
                    token_log_likelihoods, hybrid_gaussian_70_w, normalize=normalize
                )
            
            # ================================================================
            # NEW SCHEMES: Higher-order NLL powers
            # ================================================================
            nll_sqrt_w = compute_nll_sqrt_weights(token_log_likelihoods)
            nll_cubic_w = compute_nll_cubic_weights(token_log_likelihoods)
            nll_quartic_w = compute_nll_quartic_weights(token_log_likelihoods)
            nll_quintic_w = compute_nll_quintic_weights(token_log_likelihoods)
            
            results['nll_sqrt'][example_id] = compute_weighted_nll(
                token_log_likelihoods, nll_sqrt_w, normalize=normalize
            )
            results['nll_cubic'][example_id] = compute_weighted_nll(
                token_log_likelihoods, nll_cubic_w, normalize=normalize
            )
            results['nll_quartic'][example_id] = compute_weighted_nll(
                token_log_likelihoods, nll_quartic_w, normalize=normalize
            )
            results['nll_quintic'][example_id] = compute_weighted_nll(
                token_log_likelihoods, nll_quintic_w, normalize=normalize
            )
            
            # ================================================================
            # NEW SCHEMES: Negate end tokens (confidence sinks hypothesis)
            # ================================================================
            negate_last_3_w = compute_negate_last_k_weights(num_tokens, k=3)
            negate_last_5_w = compute_negate_last_k_weights(num_tokens, k=5)
            negate_last_20pct_w = compute_negate_last_pct_weights(num_tokens, pct=0.2)
            negate_first_3_w = compute_negate_first_k_weights(num_tokens, k=3)
            negate_first_20pct_w = compute_negate_first_pct_weights(num_tokens, pct=0.2)
            
            results['negate_last_3'][example_id] = compute_weighted_nll(
                token_log_likelihoods, negate_last_3_w, normalize=normalize
            )
            results['negate_last_5'][example_id] = compute_weighted_nll(
                token_log_likelihoods, negate_last_5_w, normalize=normalize
            )
            results['negate_last_20pct'][example_id] = compute_weighted_nll(
                token_log_likelihoods, negate_last_20pct_w, normalize=normalize
            )
            results['negate_first_3'][example_id] = compute_weighted_nll(
                token_log_likelihoods, negate_first_3_w, normalize=normalize
            )
            results['negate_first_20pct'][example_id] = compute_weighted_nll(
                token_log_likelihoods, negate_first_20pct_w, normalize=normalize
            )
            
            # ================================================================
            # NEW SCHEMES: Unweight different parts (position signal analysis)
            # ================================================================
            unweight_beg_w = compute_unweight_beginning_weights(num_tokens)
            unweight_mid_w = compute_unweight_middle_weights(num_tokens)
            unweight_end_w = compute_unweight_end_weights(num_tokens)
            only_beg_w = compute_only_beginning_weights(num_tokens)
            only_mid_w = compute_only_middle_weights(num_tokens)
            only_end_w = compute_only_end_weights(num_tokens)
            
            results['unweight_beginning'][example_id] = compute_weighted_nll(
                token_log_likelihoods, unweight_beg_w, normalize=normalize
            )
            results['unweight_middle'][example_id] = compute_weighted_nll(
                token_log_likelihoods, unweight_mid_w, normalize=normalize
            )
            results['unweight_end'][example_id] = compute_weighted_nll(
                token_log_likelihoods, unweight_end_w, normalize=normalize
            )
            results['only_beginning'][example_id] = compute_weighted_nll(
                token_log_likelihoods, only_beg_w, normalize=normalize
            )
            results['only_middle'][example_id] = compute_weighted_nll(
                token_log_likelihoods, only_mid_w, normalize=normalize
            )
            results['only_end'][example_id] = compute_weighted_nll(
                token_log_likelihoods, only_end_w, normalize=normalize
            )
            
            # ================================================================
            # NEW SCHEMES: Hybrid NLL-power + position
            # ================================================================
            nll_sq_neg_end_w = compute_nll_power_negate_end_weights(
                token_log_likelihoods, power=2.0, negate_pct=0.2
            )
            nll_cub_neg_end_w = compute_nll_power_negate_end_weights(
                token_log_likelihoods, power=3.0, negate_pct=0.2
            )
            nll_sq_mid_focus_w = compute_nll_power_middle_focus_weights(
                token_log_likelihoods, power=2.0
            )
            nll_cub_mid_focus_w = compute_nll_power_middle_focus_weights(
                token_log_likelihoods, power=3.0
            )
            
            results['nll_squared_negate_end'][example_id] = compute_weighted_nll(
                token_log_likelihoods, nll_sq_neg_end_w, normalize=normalize
            )
            results['nll_cubic_negate_end'][example_id] = compute_weighted_nll(
                token_log_likelihoods, nll_cub_neg_end_w, normalize=normalize
            )
            results['nll_squared_middle_focus'][example_id] = compute_weighted_nll(
                token_log_likelihoods, nll_sq_mid_focus_w, normalize=normalize
            )
            results['nll_cubic_middle_focus'][example_id] = compute_weighted_nll(
                token_log_likelihoods, nll_cub_mid_focus_w, normalize=normalize
            )
            
            # Save first 30 examples for visualization (more data for refined analysis)
            if len(weight_examples) < 30:
                response = mla.get('response', '')
                tokens = mla.get('tokens', [])
                
                weight_examples.append({
                    'example_id': example_id,
                    'response': response,
                    'tokens': tokens if tokens else None,
                    'num_tokens': num_tokens,
                    # Basic schemes
                    'uniform': uniform_w,
                    'linear_increasing': linear_inc_w,
                    'linear_decreasing': linear_dec_w,
                    'quadratic_increasing': quad_inc_w,
                    'exponential_increasing': exp_inc_w,
                    'middle_peak': middle_w,
                    # Advanced schemes
                    'gaussian_middle': gaussian_w,
                    'nll_proportional': nll_prop_w,
                    'surprisal': surprisal_w,
                    'first_k': first_k_w,
                    'last_k': last_k_w,
                    'recency_bias': recency_w,
                    # Higher-order NLL powers
                    'nll_sqrt': nll_sqrt_w,
                    'nll_cubic': nll_cubic_w,
                    'nll_quartic': nll_quartic_w,
                    # Position experiments
                    'only_beginning': only_beg_w,
                    'only_middle': only_mid_w,
                    'only_end': only_end_w,
                    # Semantic schemes (may be None)
                    'sar_relevance': sar_w if sar_w else None,
                    'hybrid_middle_sar_50': hybrid_middle_50_w if hybrid_middle_50_w else None,
                    'hybrid_gaussian_sar_50': hybrid_gaussian_50_w if hybrid_gaussian_50_w else None,
                    # Raw data
                    'token_log_likelihoods': token_log_likelihoods,
                    'is_correct': labels[example_id]
                })
        
        except Exception as e:
            logger.warning(f"Error processing {example_id}: {e}")
            continue
    
    return results, weight_examples


# ============================================================================
# AUROC COMPUTATION
# ============================================================================

def compute_aurocs(
    results: Dict[str, Dict[str, float]],
    labels: Dict[str, int]
) -> Dict[str, float]:
    """Compute AUROC for all weighting schemes."""
    aurocs = {}
    
    for scheme_name, scores in results.items():
        if not scores:
            continue
        
        common_ids = set(scores.keys()) & set(labels.keys())
        if len(common_ids) < 2:
            continue
        
        y_true = [labels[eid] for eid in common_ids]
        y_scores = [scores[eid] for eid in common_ids]
        
        # Higher NLL = more uncertain = predicts incorrect
        # So negate scores for AUROC computation
        y_scores_auc = [-s for s in y_scores]
        
        try:
            auroc = roc_auc_score(y_true, y_scores_auc)
            aurocs[scheme_name] = auroc
            logger.info(f"{scheme_name:25s}: AUROC = {auroc:.4f} (n={len(common_ids)})")
        except ValueError as e:
            logger.warning(f"Error computing AUROC for {scheme_name}: {e}")
            aurocs[scheme_name] = None
    
    return aurocs


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_weight_patterns(
    weight_examples: List[Dict],
    output_dir: str
):
    """Visualize different weighting schemes."""
    
    if not weight_examples:
        logger.warning("No weight examples to visualize")
        return
    
    # 1. Plot weight patterns for different schemes (3x4 grid)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    schemes = [
        'uniform', 'linear_increasing', 'linear_decreasing',
        'middle_peak', 'gaussian_middle', 'edges_peak',
        'first_k', 'last_k', 'recency_bias',
        'nll_proportional', 'surprisal', 'sar_relevance'
    ]
    
    scheme_labels = {
        'uniform': 'Uniform (G-NLL)',
        'linear_increasing': 'Linear Increasing',
        'linear_decreasing': 'Linear Decreasing',
        'quadratic_increasing': 'Quadratic Increasing',
        'exponential_increasing': 'Exponential Increasing',
        'middle_peak': 'Middle Peak',
        'gaussian_middle': 'Gaussian Middle',
        'edges_peak': 'Edges Peak',
        'first_k': 'First-K Tokens',
        'last_k': 'Last-K Tokens',
        'recency_bias': 'Recency Bias',
        'nll_proportional': 'NLL Proportional',
        'surprisal': 'Surprisal',
        'sar_relevance': 'SAR Relevance'
    }
    
    for idx, scheme in enumerate(schemes):
        ax = axes[idx]
        
        # Plot first 5 examples, color by correctness
        correct_plotted = False
        incorrect_plotted = False
        
        for i, ex in enumerate(weight_examples[:5]):
            if scheme in ex and ex[scheme] is not None:
                weights = ex[scheme]
                if not weights:  # Empty list check
                    continue
                positions = np.linspace(0, 1, len(weights))
                
                if ex['is_correct']:
                    color = 'green'
                    alpha = 0.4
                    label = 'Correct' if not correct_plotted else None
                    correct_plotted = True
                else:
                    color = 'red'
                    alpha = 0.6
                    label = 'Incorrect' if not incorrect_plotted else None
                    incorrect_plotted = True
                
                ax.plot(positions, weights, alpha=alpha, color=color, 
                       linewidth=2, label=label)
        
        ax.set_title(scheme_labels[scheme], fontsize=12, fontweight='bold')
        ax.set_xlabel('Token Position (normalized)', fontsize=10)
        ax.set_ylabel('Weight', fontsize=10)
        ax.grid(True, alpha=0.3)
        if correct_plotted or incorrect_plotted:
            ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'weight_patterns_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Weight patterns visualization saved to: {output_path}")
    
    # 2. Detailed SAR weight visualization (2x5 grid for 10 examples)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    
    sar_example_count = 0
    for i, ex in enumerate(weight_examples[:30]):  # Look through more examples to find 10 with SAR
        if sar_example_count >= 10:
            break
            
        if 'sar_relevance' not in ex or ex['sar_relevance'] is None:
            continue
            
        weights = ex['sar_relevance']
        if not weights:  # Empty list check
            continue
            
        ax = axes[sar_example_count]
        sar_example_count += 1
        positions = range(len(weights))
        
        # Color bars by weight magnitude
        colors = ['darkgreen' if w > np.percentile(weights, 66) 
                 else 'orange' if w > np.percentile(weights, 33)
                 else 'darkred' for w in weights]
        
        ax.bar(positions, weights, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        correctness = 'Correct' if ex['is_correct'] else 'Incorrect'
        ax.set_title(f"Example {sar_example_count} ({correctness})",
                    fontsize=11, fontweight='bold',
                    color='green' if ex['is_correct'] else 'red')
        ax.set_xlabel('Token Position', fontsize=9)
        ax.set_ylabel('SAR Relevance', fontsize=9)
        ax.axhline(y=np.median(weights), color='blue', linestyle='--', 
                  alpha=0.7, linewidth=2, label=f'Median={np.median(weights):.3f}')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8, loc='upper left')
        
        # Set consistent y-axis range for better comparison
        ax.set_ylim([0, max(weights) * 1.1])
    
    # Hide unused axes if we found fewer than 10 SAR examples
    for j in range(sar_example_count, 10):
        axes[j].set_visible(False)
    
    plt.suptitle('SAR Relevance Weights: Token-Level Importance Patterns', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'sar_weight_details.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ SAR weight details saved to: {output_path}")


def get_all_scheme_labels() -> Dict[str, str]:
    """Get human-readable labels for all schemes."""
    return {
        # Basic position-based
        'uniform': 'Uniform (G-NLL)',
        'linear_increasing': 'Linear ↑',
        'linear_decreasing': 'Linear ↓',
        'quadratic_increasing': 'Quadratic ↑',
        'quadratic_decreasing': 'Quadratic ↓',
        'exponential_increasing': 'Exponential ↑',
        'exponential_decreasing': 'Exponential ↓',
        'middle_peak': 'Middle Peak',
        'edges_peak': 'Edges Peak',
        # Advanced schemes
        'gaussian_middle': 'Gaussian Middle',
        'nll_proportional': 'NLL² (Proportional)',
        'confidence_proportional': 'Confidence Proportional',
        'surprisal': 'Surprisal',
        'first_k': 'First-K Tokens',
        'last_k': 'Last-K Tokens',
        'first_last_k': 'First+Last-K',
        'recency_bias': 'Recency Bias',
        'high_uncertainty_only': 'High Uncertainty Only',
        'low_uncertainty_only': 'Low Uncertainty Only',
        # Semantic and hybrid
        'sar_relevance': 'SAR Relevance',
        'hybrid_middle_sar_50': 'Hybrid Middle+SAR (50/50)',
        'hybrid_gaussian_sar_50': 'Hybrid Gaussian+SAR (50/50)',
        'hybrid_middle_sar_70': 'Hybrid Middle+SAR (70/30)',
        'hybrid_gaussian_sar_70': 'Hybrid Gaussian+SAR (70/30)',
        # Higher-order NLL powers
        'nll_sqrt': 'NLL^0.5 (√NLL)',
        'nll_cubic': 'NLL³ (Cubic)',
        'nll_quartic': 'NLL⁴ (Quartic)',
        'nll_quintic': 'NLL⁵ (Quintic)',
        # Negate schemes
        'negate_last_3': 'Negate Last 3',
        'negate_last_5': 'Negate Last 5',
        'negate_last_20pct': 'Negate Last 20%',
        'negate_first_3': 'Negate First 3',
        'negate_first_20pct': 'Negate First 20%',
        # Unweight schemes
        'unweight_beginning': 'Unweight Beginning ⅓',
        'unweight_middle': 'Unweight Middle ⅓',
        'unweight_end': 'Unweight End ⅓',
        'only_beginning': 'Only Beginning ⅓',
        'only_middle': 'Only Middle ⅓',
        'only_end': 'Only End ⅓',
        # Hybrid NLL-power + position
        'nll_squared_negate_end': 'NLL² + Negate End',
        'nll_cubic_negate_end': 'NLL³ + Negate End',
        'nll_squared_middle_focus': 'NLL² × Middle Focus',
        'nll_cubic_middle_focus': 'NLL³ × Middle Focus',
    }


def create_auroc_comparison_plot(
    aurocs: Dict[str, float],
    output_dir: str
):
    """Create bar plot comparing AUROCs."""
    # Filter out None values
    valid_aurocs = {k: v for k, v in aurocs.items() if v is not None}
    
    if not valid_aurocs:
        logger.warning("No valid AUROCs to plot")
        return
    
    # Sort by AUROC
    sorted_items = sorted(valid_aurocs.items(), key=lambda x: x[1], reverse=True)
    schemes = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    scheme_labels = get_all_scheme_labels()
    labels = [scheme_labels.get(s, s) for s in schemes]
    
    # Create figure with appropriate height for all schemes
    fig_height = max(10, len(values) * 0.35)
    plt.figure(figsize=(14, fig_height))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(values)))
    bars = plt.barh(labels, values, color=colors, edgecolor='black', linewidth=1)
    
    # Highlight the best performing scheme
    bars[0].set_edgecolor('red')
    bars[0].set_linewidth(3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(val + 0.005, i, f'{val:.4f}', va='center', 
                fontsize=9, fontweight='bold')
    
    plt.xlabel('AUROC', fontsize=14, fontweight='bold')
    plt.title('Token Weighting Schemes: AUROC Comparison\n(Higher is Better)', 
             fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.xlim(min(values) - 0.02, max(values) + 0.06)
    plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random (0.5)')
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'auroc_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ AUROC comparison plot saved to: {output_path}")


def create_nll_power_comparison_plot(
    aurocs: Dict[str, float],
    output_dir: str
):
    """Create focused plot comparing NLL power schemes."""
    nll_schemes = ['nll_sqrt', 'nll_proportional', 'nll_cubic', 'nll_quartic', 'nll_quintic']
    scheme_labels = {
        'nll_sqrt': 'NLL^0.5',
        'nll_proportional': 'NLL² (current best)',
        'nll_cubic': 'NLL³',
        'nll_quartic': 'NLL⁴',
        'nll_quintic': 'NLL⁵',
    }
    
    # Filter and sort
    valid = {k: aurocs.get(k) for k in nll_schemes if aurocs.get(k) is not None}
    if not valid:
        logger.warning("No NLL power schemes found")
        return
    
    # Sort by power (custom order)
    power_order = ['nll_sqrt', 'nll_proportional', 'nll_cubic', 'nll_quartic', 'nll_quintic']
    schemes = [s for s in power_order if s in valid]
    values = [valid[s] for s in schemes]
    labels = [scheme_labels.get(s, s) for s in schemes]
    powers = [0.5, 2, 3, 4, 5][:len(schemes)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    colors = ['#2ecc71' if v == max(values) else '#3498db' for v in values]
    bars = ax1.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax1.set_xlabel('NLL Power Scheme', fontsize=12, fontweight='bold')
    ax1.set_title('NLL Power Comparison: Which Exponent Works Best?', fontsize=14, fontweight='bold')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    
    # Line plot showing trend
    ax2.plot(powers, values, 'bo-', linewidth=2, markersize=10)
    ax2.fill_between(powers, 0.5, values, alpha=0.3)
    ax2.set_xlabel('NLL Power (exponent)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax2.set_title('AUROC vs NLL Power: Amplification Effect', fontsize=14, fontweight='bold')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(powers)
    ax2.legend()
    
    # Mark the best
    best_idx = values.index(max(values))
    ax2.scatter([powers[best_idx]], [max(values)], color='red', s=200, zorder=5, 
               marker='*', label=f'Best: power={powers[best_idx]}')
    ax2.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'nll_power_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ NLL power comparison saved to: {output_path}")


def create_position_analysis_plot(
    aurocs: Dict[str, float],
    output_dir: str
):
    """Create plot analyzing which position (beginning/middle/end) is most important."""
    
    # Position analysis schemes
    position_schemes = {
        'only_beginning': ('Beginning Only', '#e74c3c'),
        'only_middle': ('Middle Only', '#2ecc71'),
        'only_end': ('End Only', '#3498db'),
        'unweight_beginning': ('Without Beginning', '#c0392b'),
        'unweight_middle': ('Without Middle', '#27ae60'),
        'unweight_end': ('Without End', '#2980b9'),
        'uniform': ('Uniform (baseline)', '#95a5a6'),
    }
    
    # Confidence sinks analysis
    negate_schemes = {
        'negate_last_3': ('Negate Last 3', '#9b59b6'),
        'negate_last_5': ('Negate Last 5', '#8e44ad'),
        'negate_last_20pct': ('Negate Last 20%', '#6c3483'),
        'negate_first_3': ('Negate First 3', '#e67e22'),
        'negate_first_20pct': ('Negate First 20%', '#d35400'),
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Only X vs Uniform
    ax1 = axes[0]
    only_schemes = ['only_beginning', 'only_middle', 'only_end', 'uniform']
    valid1 = {k: aurocs.get(k) for k in only_schemes if aurocs.get(k) is not None}
    if valid1:
        labels = [position_schemes.get(k, (k, 'gray'))[0] for k in only_schemes if k in valid1]
        values = [valid1[k] for k in only_schemes if k in valid1]
        colors = [position_schemes.get(k, (k, 'gray'))[1] for k in only_schemes if k in valid1]
        
        bars = ax1.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('AUROC', fontsize=12, fontweight='bold')
        ax1.set_title('Using ONLY Each Position Third\n(Which part has the signal?)', 
                     fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=15)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Plot 2: Without X vs Uniform
    ax2 = axes[1]
    unweight_schemes = ['unweight_beginning', 'unweight_middle', 'unweight_end', 'uniform']
    valid2 = {k: aurocs.get(k) for k in unweight_schemes if aurocs.get(k) is not None}
    if valid2:
        labels = [position_schemes.get(k, (k, 'gray'))[0] for k in unweight_schemes if k in valid2]
        values = [valid2[k] for k in unweight_schemes if k in valid2]
        colors = [position_schemes.get(k, (k, 'gray'))[1] for k in unweight_schemes if k in valid2]
        
        bars = ax2.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('AUROC', fontsize=12, fontweight='bold')
        ax2.set_title('EXCLUDING Each Position Third\n(What happens when we remove it?)', 
                     fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=15)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Plot 3: Confidence Sinks Hypothesis (Negating End Tokens)
    ax3 = axes[2]
    negate_list = list(negate_schemes.keys()) + ['uniform']
    valid3 = {k: aurocs.get(k) for k in negate_list if aurocs.get(k) is not None}
    if valid3:
        labels = []
        values = []
        colors = []
        for k in negate_list:
            if k in valid3:
                if k == 'uniform':
                    labels.append('Uniform (baseline)')
                    colors.append('#95a5a6')
                else:
                    labels.append(negate_schemes[k][0])
                    colors.append(negate_schemes[k][1])
                values.append(valid3[k])
        
        bars = ax3.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.set_ylabel('AUROC', fontsize=12, fontweight='bold')
        ax3.set_title('Confidence Sinks Hypothesis\n(Do end tokens hurt detection?)', 
                     fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=30)
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'position_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Position analysis plot saved to: {output_path}")


def create_refined_nll_visualization(
    weight_examples: List[Dict],
    output_dir: str
):
    """Create refined visualization of NLL patterns across examples."""
    
    if not weight_examples:
        logger.warning("No weight examples for NLL visualization")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: NLL distribution for correct vs incorrect
    ax1 = axes[0, 0]
    correct_nlls = []
    incorrect_nlls = []
    for ex in weight_examples:
        if 'token_log_likelihoods' in ex:
            nlls = [-ll for ll in ex['token_log_likelihoods']]
            if ex['is_correct']:
                correct_nlls.extend(nlls)
            else:
                incorrect_nlls.extend(nlls)
    
    if correct_nlls and incorrect_nlls:
        ax1.hist(correct_nlls, bins=50, alpha=0.6, label=f'Correct (n={len(correct_nlls)})', 
                color='green', density=True)
        ax1.hist(incorrect_nlls, bins=50, alpha=0.6, label=f'Incorrect (n={len(incorrect_nlls)})', 
                color='red', density=True)
        ax1.set_xlabel('NLL Value', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('NLL Distribution: Correct vs Incorrect', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average NLL by position (normalized)
    ax2 = axes[0, 1]
    n_bins = 20
    correct_by_pos = [[] for _ in range(n_bins)]
    incorrect_by_pos = [[] for _ in range(n_bins)]
    
    for ex in weight_examples:
        if 'token_log_likelihoods' in ex:
            nlls = [-ll for ll in ex['token_log_likelihoods']]
            n_tokens = len(nlls)
            for i, nll in enumerate(nlls):
                bin_idx = min(int(i / n_tokens * n_bins), n_bins - 1)
                if ex['is_correct']:
                    correct_by_pos[bin_idx].append(nll)
                else:
                    incorrect_by_pos[bin_idx].append(nll)
    
    positions = np.linspace(0, 1, n_bins)
    correct_means = [np.mean(b) if b else 0 for b in correct_by_pos]
    incorrect_means = [np.mean(b) if b else 0 for b in incorrect_by_pos]
    
    ax2.plot(positions, correct_means, 'g-o', linewidth=2, markersize=6, label='Correct')
    ax2.plot(positions, incorrect_means, 'r-s', linewidth=2, markersize=6, label='Incorrect')
    ax2.fill_between(positions, correct_means, alpha=0.2, color='green')
    ax2.fill_between(positions, incorrect_means, alpha=0.2, color='red')
    ax2.set_xlabel('Normalized Position', fontsize=11)
    ax2.set_ylabel('Mean NLL', fontsize=11)
    ax2.set_title('NLL by Position: Where is Uncertainty?', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: NLL² (proportional) weights visualization
    ax3 = axes[0, 2]
    for i, ex in enumerate(weight_examples[:5]):
        if 'nll_proportional' in ex:
            weights = ex['nll_proportional']
            positions = np.linspace(0, 1, len(weights))
            color = 'green' if ex['is_correct'] else 'red'
            alpha = 0.6 if ex['is_correct'] else 0.8
            ax3.plot(positions, weights, color=color, alpha=alpha, linewidth=1.5)
    
    ax3.set_xlabel('Normalized Position', fontsize=11)
    ax3.set_ylabel('NLL² Weight', fontsize=11)
    ax3.set_title('NLL² Weights: Amplified Uncertainty', fontsize=12, fontweight='bold')
    ax3.legend(['Correct', 'Incorrect'], loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Max NLL comparison (worst token)
    ax4 = axes[1, 0]
    correct_max = []
    incorrect_max = []
    for ex in weight_examples:
        if 'token_log_likelihoods' in ex:
            nlls = [-ll for ll in ex['token_log_likelihoods']]
            max_nll = max(nlls)
            if ex['is_correct']:
                correct_max.append(max_nll)
            else:
                incorrect_max.append(max_nll)
    
    if correct_max and incorrect_max:
        data = [correct_max, incorrect_max]
        bp = ax4.boxplot(data, labels=['Correct', 'Incorrect'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax4.set_ylabel('Max NLL (worst token)', fontsize=11)
        ax4.set_title('Worst Token Uncertainty\n(Higher = More Uncertain)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: NLL variance comparison
    ax5 = axes[1, 1]
    correct_var = []
    incorrect_var = []
    for ex in weight_examples:
        if 'token_log_likelihoods' in ex:
            nlls = [-ll for ll in ex['token_log_likelihoods']]
            var_nll = np.var(nlls)
            if ex['is_correct']:
                correct_var.append(var_nll)
            else:
                incorrect_var.append(var_nll)
    
    if correct_var and incorrect_var:
        data = [correct_var, incorrect_var]
        bp = ax5.boxplot(data, labels=['Correct', 'Incorrect'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax5.set_ylabel('NLL Variance', fontsize=11)
        ax5.set_title('Uncertainty Spread\n(Do incorrect answers have more variable NLL?)', 
                     fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Position of max NLL token
    ax6 = axes[1, 2]
    correct_pos = []
    incorrect_pos = []
    for ex in weight_examples:
        if 'token_log_likelihoods' in ex:
            nlls = [-ll for ll in ex['token_log_likelihoods']]
            max_pos = np.argmax(nlls) / len(nlls)  # Normalized position
            if ex['is_correct']:
                correct_pos.append(max_pos)
            else:
                incorrect_pos.append(max_pos)
    
    if correct_pos and incorrect_pos:
        ax6.hist(correct_pos, bins=10, alpha=0.6, label='Correct', color='green', density=True)
        ax6.hist(incorrect_pos, bins=10, alpha=0.6, label='Incorrect', color='red', density=True)
        ax6.set_xlabel('Normalized Position of Max NLL Token', fontsize=11)
        ax6.set_ylabel('Density', fontsize=11)
        ax6.set_title('Where is the Most Uncertain Token?\n(Beginning/Middle/End)', 
                     fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Refined NLL Analysis: Understanding Token Uncertainty Patterns', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'nll_refined_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Refined NLL visualization saved to: {output_path}")


def create_roc_curves(
    results: Dict[str, Dict[str, float]],
    labels: Dict[str, int],
    output_dir: str,
    smooth: bool = True,
    n_points: int = 200
):
    """Create ROC curves for all weighting schemes with smooth interpolation.
    
    Args:
        results: Scheme results
        labels: Correctness labels
        output_dir: Output directory
        smooth: If True, interpolate for smooth curves instead of step functions
        n_points: Number of points for interpolation
    """
    plt.figure(figsize=(14, 11))
    
    scheme_labels = {
        # Basic position-based
        'uniform': 'Uniform (G-NLL)',
        'linear_increasing': 'Linear ↑',
        'linear_decreasing': 'Linear ↓',
        'quadratic_increasing': 'Quadratic ↑',
        'quadratic_decreasing': 'Quadratic ↓',
        'exponential_increasing': 'Exponential ↑',
        'exponential_decreasing': 'Exponential ↓',
        'middle_peak': 'Middle Peak',
        'edges_peak': 'Edges Peak',
        # Advanced schemes
        'gaussian_middle': 'Gaussian Middle',
        'nll_proportional': 'NLL² (Proportional)',
        'confidence_proportional': 'Confidence Proportional',
        'surprisal': 'Surprisal',
        'first_k': 'First-K Tokens',
        'last_k': 'Last-K Tokens',
        'first_last_k': 'First+Last-K',
        'recency_bias': 'Recency Bias',
        'high_uncertainty_only': 'High Uncertainty Only',
        'low_uncertainty_only': 'Low Uncertainty Only',
        # Semantic and hybrid
        'sar_relevance': 'SAR Relevance',
        'hybrid_middle_sar_50': 'Hybrid Middle+SAR (50/50)',
        'hybrid_gaussian_sar_50': 'Hybrid Gaussian+SAR (50/50)',
        'hybrid_middle_sar_70': 'Hybrid Middle+SAR (70/30)',
        'hybrid_gaussian_sar_70': 'Hybrid Gaussian+SAR (70/30)',
        # Higher-order NLL powers
        'nll_sqrt': 'NLL^0.5 (Square Root)',
        'nll_cubic': 'NLL³ (Cubic)',
        'nll_quartic': 'NLL⁴ (Quartic)',
        'nll_quintic': 'NLL⁵ (Quintic)',
        # Negate schemes
        'negate_last_3': 'Negate Last 3',
        'negate_last_5': 'Negate Last 5',
        'negate_last_20pct': 'Negate Last 20%',
        'negate_first_3': 'Negate First 3',
        'negate_first_20pct': 'Negate First 20%',
        # Unweight schemes
        'unweight_beginning': 'Unweight Beginning',
        'unweight_middle': 'Unweight Middle',
        'unweight_end': 'Unweight End',
        'only_beginning': 'Only Beginning',
        'only_middle': 'Only Middle',
        'only_end': 'Only End',
        # Hybrid NLL-power + position
        'nll_squared_negate_end': 'NLL² + Negate End',
        'nll_cubic_negate_end': 'NLL³ + Negate End',
        'nll_squared_middle_focus': 'NLL² × Middle Focus',
        'nll_cubic_middle_focus': 'NLL³ × Middle Focus',
    }
    
    # Compute AUROCs first to filter top schemes
    aurocs_for_filtering = {}
    for scheme_name, scores in results.items():
        if not scores:
            continue
        common_ids = set(scores.keys()) & set(labels.keys())
        if len(common_ids) < 2:
            continue
        y_true = [labels[eid] for eid in common_ids]
        y_scores = [scores[eid] for eid in common_ids]
        y_scores_auc = [-s for s in y_scores]
        try:
            auroc = roc_auc_score(y_true, y_scores_auc)
            aurocs_for_filtering[scheme_name] = auroc
        except:
            pass
    
    # Get top 12 schemes for readability
    top_schemes = sorted(aurocs_for_filtering.items(), key=lambda x: x[1], reverse=True)[:12]
    top_scheme_names = [s[0] for s in top_schemes]
    
    # Common FPR points for interpolation
    mean_fpr = np.linspace(0, 1, n_points)
    
    for scheme_name, scores in results.items():
        # Only plot top schemes for readability
        if scheme_name not in top_scheme_names:
            continue
            
        if not scores:
            continue
        
        common_ids = set(scores.keys()) & set(labels.keys())
        if len(common_ids) < 2:
            continue
        
        y_true = [labels[eid] for eid in common_ids]
        y_scores = [scores[eid] for eid in common_ids]
        
        # Negate scores for metrics where lower is better
        y_scores_auc = [-s for s in y_scores]
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores_auc)
            auroc = roc_auc_score(y_true, y_scores_auc)
            
            label = f"{scheme_labels.get(scheme_name, scheme_name)} (AUC={auroc:.3f})"
            
            # Make top performer line thicker
            linewidth = 3 if scheme_name == top_scheme_names[0] else 2
            alpha = 1.0 if scheme_name == top_scheme_names[0] else 0.7
            
            if smooth and len(fpr) > 3:
                # Interpolate for smooth curve
                # Use linear interpolation followed by Gaussian smoothing
                interp_func = interp1d(fpr, tpr, kind='linear', bounds_error=False, 
                                       fill_value=(tpr[0], tpr[-1]))
                smooth_tpr = interp_func(mean_fpr)
                
                # Apply light Gaussian smoothing to remove step artifacts
                smooth_tpr = gaussian_filter1d(smooth_tpr, sigma=2)
                
                # Ensure monotonicity (ROC should be non-decreasing)
                smooth_tpr = np.maximum.accumulate(smooth_tpr)
                
                # Clip to valid range
                smooth_tpr = np.clip(smooth_tpr, 0, 1)
                
                plt.plot(mean_fpr, smooth_tpr, label=label, linewidth=linewidth, alpha=alpha)
            else:
                # Fallback to original
                plt.plot(fpr, tpr, label=label, linewidth=linewidth, alpha=alpha)
                
        except Exception as e:
            logger.warning(f"Error creating ROC curve for {scheme_name}: {e}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title('ROC Curves: Token Weighting Schemes (Top 12)\n(Smooth Interpolation)', 
             fontsize=15, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'roc_curves_smooth.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Smooth ROC curves saved to: {output_path}")
    logger.info(f"   (Showing top 12 schemes out of {len(results)} total)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 6: Compare different token weighting schemes'
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
        help='Model name (e.g., Llama-3.2-1B)'
    )
    parser.add_argument(
        '--similarity-model',
        type=str,
        default='cross-encoder/stsb-roberta-large',
        help='Similarity model for SAR weights'
    )
    parser.add_argument(
        '--use-rouge',
        action='store_true',
        help='Use ROUGE for correctness (short answers)'
    )
    parser.add_argument(
        '--rouge-threshold',
        type=float,
        default=0.3,
        help='ROUGE-L threshold'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Do not normalize weighted NLL by sum of weights'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/phase6_weighting_schemes',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data = load_pickle_data(args.pickle_path)
    labels = get_correctness_labels(data, use_rouge=args.use_rouge,
                                   rouge_threshold=args.rouge_threshold)
    logger.info(f"Got correctness labels for {len(labels)} examples")
    
    # Initialize models
    logger.info("Initializing similarity model for SAR weights...")
    similarity_model = initialize_similarity_model(args.similarity_model)
    
    logger.info(f"Loading tokenizer for {args.model_name}...")
    cache_dir = get_hf_cache_dir()
    if 'llama' in args.model_name.lower():
        if 'Llama-3' in args.model_name or 'Llama-3.1' in args.model_name or 'Meta-Llama-3' in args.model_name:
            base = 'meta-llama'
        else:
            base = 'huggyllama'
        tokenizer = AutoTokenizer.from_pretrained(
            f"{base}/{args.model_name}",
            cache_dir=cache_dir
        )
    else:
        raise ValueError(f"Unknown model: {args.model_name}")
    
    # Compute all weighted NLLs
    results, weight_examples = compute_all_weighted_nlls(
        data, labels, similarity_model, tokenizer,
        normalize=not args.no_normalize
    )
    
    # Compute AUROCs
    logger.info("\n" + "="*80)
    logger.info("AUROC RESULTS")
    logger.info("="*80)
    aurocs = compute_aurocs(results, labels)
    
    # Save results
    comparison_df = pd.DataFrame([
        {'Scheme': k, 'AUROC': v}
        for k, v in sorted(aurocs.items(), key=lambda x: x[1] or 0, reverse=True)
    ])
    
    csv_path = os.path.join(args.output_dir, 'weighting_schemes_auroc.csv')
    comparison_df.to_csv(csv_path, index=False)
    logger.info(f"\n✅ Results saved to: {csv_path}")
    
    # Print table
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Save weight examples for further analysis
    examples_path = os.path.join(args.output_dir, 'weight_examples.json')
    with open(examples_path, 'w') as f:
        json.dump(weight_examples, f, indent=2)
    logger.info(f"✅ Weight examples saved to: {examples_path}")
    
    # Save all results
    all_results_path = os.path.join(args.output_dir, 'all_weighted_nlls.json')
    with open(all_results_path, 'w') as f:
        json.dump({
            'aurocs': aurocs,
            'num_examples': len(labels),
            'normalize': not args.no_normalize
        }, f, indent=2)
    logger.info(f"✅ Full results saved to: {all_results_path}")
    
    # Visualizations
    logger.info("\nGenerating visualizations...")
    visualize_weight_patterns(weight_examples, args.output_dir)
    create_auroc_comparison_plot(aurocs, args.output_dir)
    create_roc_curves(results, labels, args.output_dir, smooth=True)
    
    # NEW: Additional focused visualizations
    create_nll_power_comparison_plot(aurocs, args.output_dir)
    create_position_analysis_plot(aurocs, args.output_dir)
    create_refined_nll_visualization(weight_examples, args.output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("✅ Analysis complete!")
    logger.info("="*80)
    logger.info(f"\nOutput directory: {args.output_dir}")
    logger.info("Files created:")
    logger.info("  - weighting_schemes_auroc.csv      (AUROC comparison table)")
    logger.info("  - auroc_comparison.png             (Bar chart - all schemes)")
    logger.info("  - roc_curves_smooth.png            (Smooth ROC curves - top 12)")
    logger.info("  - weight_patterns_comparison.png   (Weight patterns)")
    logger.info("  - sar_weight_details.png           (SAR weight analysis)")
    logger.info("  - nll_power_comparison.png         (NLL power exponent analysis)")
    logger.info("  - position_analysis.png            (Beginning/Middle/End analysis)")
    logger.info("  - nll_refined_analysis.png         (Detailed NLL patterns)")
    logger.info("  - weight_examples.json             (Raw data)")


if __name__ == '__main__':
    main()

