"""Implement SAR (Shifting Attention to Relevance) method.

SAR computes a relevance-weighted NLL across multiple sampled sequences.
For each sample m, it computes token relevance weights R_T(y_t^m) and then
aggregates across samples with relevance weighting.

Formula: SAR = (1/M) * Σ_m [Σ_t R_T(y_t^m) * (-log P(y_t^m))] / [Σ_t R_T(y_t^m)]
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from tqdm import tqdm

from uncertainty_measures.rw_gnll import (
    compute_token_relevance_weights,
    compute_similarity
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_sar_for_entry(
    entry: Dict[str, Any],
    responses: List[Tuple[str, List[float], Any, float]],
    similarity_model,
    tokenizer,
    cache: Optional[Dict[Tuple[str, int], float]] = None,
    show_progress: bool = False
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Compute SAR score for an entry with multiple sampled responses.
    
    Args:
        entry: Dictionary containing 'question', 'context' (optional)
        responses: List of tuples (response, token_log_likelihoods, embedding, acc)
                  from multiple samples (M samples)
        similarity_model: Initialized similarity model
        tokenizer: Tokenizer matching the one used during generation
        cache: Optional cache for similarity computations
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (sar_score, details_dict) where details_dict contains per-sample info
    """
    if not responses:
        logger.warning("No responses provided for SAR computation")
        return 0.0, None
    
    # Construct prompt
    question = entry.get('question', '')
    context = entry.get('context', '')
    if context:
        prompt_x = f"{context} {question}".strip()
    else:
        prompt_x = question
    
    if cache is None:
        cache = {}
    
    M = len(responses)
    sar_scores_per_sample = []
    details = {
        'num_samples': M,
        'per_sample_scores': [],
        'per_sample_relevance_sums': []
    }
    
    # Process each sample
    sample_range = range(M)
    if show_progress:
        sample_range = tqdm(sample_range, desc="Computing SAR", unit="sample")
    
    for m, (response, token_log_likelihoods, _, _) in enumerate(sample_range):
        if not response or not token_log_likelihoods:
            continue
        
        try:
            # Compute relevance weights for this sample
            relevance_weights = compute_token_relevance_weights(
                prompt_x, response, tokenizer, similarity_model,
                cache=cache, show_progress=False
            )
            
            if len(relevance_weights) != len(token_log_likelihoods):
                logger.warning(
                    f"Sample {m}: Token count mismatch: "
                    f"{len(relevance_weights)} relevance weights vs "
                    f"{len(token_log_likelihoods)} token log-likelihoods"
                )
                continue
            
            # Compute weighted NLL for this sample
            # Numerator: Σ_t R_T(y_t^m) * (-log P(y_t^m))
            # Note: token_log_likelihoods are log-probs (negative values)
            # For NLL, we need -L_t, which is -token_log_likelihoods[t]
            weighted_nlls = [
                relevance_weights[t] * (-token_log_likelihoods[t])
                for t in range(len(token_log_likelihoods))
            ]
            numerator = sum(weighted_nlls)
            
            # Denominator: Σ_t R_T(y_t^m)
            denominator = sum(relevance_weights)
            
            # Compute SAR score for this sample
            if denominator == 0:
                logger.warning(f"Sample {m}: Zero denominator (all tokens irrelevant)")
                # Fallback to standard NLL
                sample_sar = -sum(token_log_likelihoods)
            else:
                sample_sar = numerator / denominator
            
            sar_scores_per_sample.append(sample_sar)
            details['per_sample_scores'].append(float(sample_sar))
            details['per_sample_relevance_sums'].append(float(denominator))
            
        except Exception as e:
            logger.warning(f"Error processing sample {m}: {e}")
            continue
    
    if not sar_scores_per_sample:
        logger.warning("No valid samples for SAR computation")
        return 0.0, None
    
    # SAR = (1/M) * Σ_m SAR_score_m
    # Actually, the formula suggests averaging the per-sample SAR scores
    sar_score = np.mean(sar_scores_per_sample)
    
    details['mean_sar'] = float(sar_score)
    details['std_sar'] = float(np.std(sar_scores_per_sample)) if len(sar_scores_per_sample) > 1 else 0.0
    details['num_valid_samples'] = len(sar_scores_per_sample)
    
    return float(sar_score), details


def compute_sar(
    entry: Dict[str, Any],
    similarity_model,
    tokenizer,
    cache: Optional[Dict[Tuple[str, int], float]] = None,
    num_samples: Optional[int] = None,
    show_progress: bool = False
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Compute SAR score for an entry.
    
    This is a convenience function that extracts responses from the entry
    and computes SAR.
    
    Args:
        entry: Dictionary containing 'question', 'context', and 'responses'
        similarity_model: Initialized similarity model
        tokenizer: Tokenizer matching the one used during generation
        cache: Optional cache for similarity computations
        num_samples: Optional limit on number of samples to use (None = use all)
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (sar_score, details_dict)
    """
    if 'responses' not in entry:
        logger.warning("Entry does not contain 'responses' field for multi-sample SAR")
        return 0.0, None
    
    responses = entry['responses']
    
    if num_samples is not None:
        responses = responses[:num_samples]
    
    if not responses:
        logger.warning("No responses available for SAR computation")
        return 0.0, None
    
    return compute_sar_for_entry(
        entry, responses, similarity_model, tokenizer,
        cache=cache, show_progress=show_progress
    )

