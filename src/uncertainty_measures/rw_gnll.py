"""Implement Relevance-Weighted G-NLL (RW-G-NLL) metric.

This module computes RW-G-NLL by weighting token log-likelihoods by semantic relevance,
filtering out noise from "generative inequality."
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


def initialize_similarity_model(model_name: str = 'cross-encoder/stsb-roberta-large'):
    """Initialize semantic similarity model.
    
    Args:
        model_name: Name of the cross-encoder model to load.
                   Default: 'cross-encoder/stsb-roberta-large'
    
    Returns:
        Initialized similarity model ready for inference.
    
    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "sentence-transformers is required for RW-G-NLL. "
            "Install with: pip install sentence-transformers"
        )
    
    logging.info(f"Loading similarity model: {model_name}")
    model = CrossEncoder(model_name)
    logging.info(f"Similarity model loaded successfully")
    return model


def compute_similarity(similarity_model, text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts.
    
    Args:
        similarity_model: Initialized cross-encoder model
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score in [0, 1] range.
        Higher values indicate more similar texts.
    """
    # Cross-encoders typically return scores in different ranges
    # STSB models return scores in [0, 5] range
    # We normalize to [0, 1]
    score = similarity_model.predict([(text1, text2)])[0]
    
    # Normalize to [0, 1] range
    # For STSB models, scores are typically [0, 5]
    # We'll use sigmoid normalization or linear scaling
    # Using linear scaling: normalize from [0, 5] to [0, 1]
    if score < 0:
        # Some models return negative scores, use sigmoid
        normalized_score = 1 / (1 + np.exp(-score))
    else:
        # Assume [0, 5] range for STSB models
        normalized_score = min(1.0, max(0.0, score / 5.0))
    
    return float(normalized_score)


def remove_token_at_position(response_text: str, tokenizer, token_position: int) -> str:
    """Remove token at specified position from response text.
    
    Args:
        response_text: The response text as a string
        tokenizer: Tokenizer to use for encoding/decoding
        token_position: Index of token to remove (0-based)
    
    Returns:
        Response text with the specified token removed.
    
    Raises:
        ValueError: If token_position is out of bounds.
    """
    # Encode the response to get token IDs
    token_ids = tokenizer.encode(response_text, add_special_tokens=False)
    
    # Check bounds
    if token_position < 0 or token_position >= len(token_ids):
        raise ValueError(
            f"Token position {token_position} out of bounds for sequence of length {len(token_ids)}"
        )
    
    # Remove token at position
    token_ids_without_t = token_ids[:token_position] + token_ids[token_position+1:]
    
    # Handle empty sequence
    if len(token_ids_without_t) == 0:
        return ""
    
    # Decode back to text
    response_without_t = tokenizer.decode(token_ids_without_t, skip_special_tokens=True)
    
    return response_without_t


def compute_token_relevance_weights(
    prompt_x: str,
    response_y_greedy: str,
    tokenizer,
    similarity_model,
    cache: Optional[Dict[Tuple[str, int], float]] = None,
    show_progress: bool = False
) -> List[float]:
    """Compute relevance weights for each token in the response.
    
    For each token, computes R_T(y_t) = 1 - g(x ∪ y_greedy, x ∪ y_greedy \ {y_t}),
    where g is the semantic similarity function.
    
    Args:
        prompt_x: The input prompt (context + question)
        response_y_greedy: The greedy-decoded response text
        tokenizer: Tokenizer to use for token removal
        similarity_model: Initialized similarity model
        cache: Optional cache dictionary to store similarity computations
        show_progress: Whether to show progress bar
    
    Returns:
        List of relevance weights, one per token in the response.
    """
    if cache is None:
        cache = {}
    
    # Construct full text: x ∪ y_greedy
    full_text = f"{prompt_x} {response_y_greedy}".strip()
    
    # Tokenize response to get number of tokens
    token_ids = tokenizer.encode(response_y_greedy, add_special_tokens=False)
    num_tokens = len(token_ids)
    
    if num_tokens == 0:
        return []
    
    relevance_weights = []
    
    # Create progress iterator
    token_range = range(num_tokens)
    if show_progress:
        token_range = tqdm(token_range, desc="Computing relevance weights", unit="token")
    
    for t in token_range:
        # Check cache first
        cache_key = (response_y_greedy, t)
        if cache_key in cache:
            similarity = cache[cache_key]
        else:
            # Remove token at position t
            response_without_t = remove_token_at_position(response_y_greedy, tokenizer, t)
            
            # Construct ablated text: x ∪ y_greedy \ {y_t}
            ablated_text = f"{prompt_x} {response_without_t}".strip()
            
            # Compute similarity
            similarity = compute_similarity(similarity_model, full_text, ablated_text)
            
            # Cache the result
            cache[cache_key] = similarity
        
        # Calculate relevance weight: R_T(y_t) = 1 - similarity
        relevance_weight = 1.0 - similarity
        relevance_weights.append(relevance_weight)
    
    return relevance_weights


def compute_rw_gnll(
    entry: Dict[str, Any],
    similarity_model,
    tokenizer,
    cache: Optional[Dict[Tuple[str, int], float]] = None,
    return_relevance_weights: bool = False
) -> Tuple[float, Optional[List[float]]]:
    """Compute Relevance-Weighted G-NLL score for an entry.
    
    Args:
        entry: Dictionary containing 'question', 'context' (optional), 
               and 'most_likely_answer' with 'response' and 'token_log_likelihoods'
        similarity_model: Initialized similarity model
        tokenizer: Tokenizer matching the one used during generation
        cache: Optional cache for similarity computations
        return_relevance_weights: Whether to return relevance weights
    
    Returns:
        Tuple of (rw_gnll_score, relevance_weights) where relevance_weights is None
        unless return_relevance_weights=True.
    
    Raises:
        KeyError: If required fields are missing from entry.
        ValueError: If token counts don't align.
    """
    # Extract data from entry
    question = entry.get('question', '')
    context = entry.get('context', '')
    
    # Construct prompt: context + question (or just question if no context)
    if context:
        prompt_x = f"{context} {question}".strip()
    else:
        prompt_x = question
    
    mla = entry['most_likely_answer']
    y_greedy = mla.get('response', '').strip()
    token_log_likelihoods = mla.get('token_log_likelihoods', [])
    
    if not y_greedy:
        logging.warning("Empty response, returning zero RW-G-NLL")
        return 0.0, None if not return_relevance_weights else []
    
    if not token_log_likelihoods:
        logging.warning("No token log-likelihoods, returning zero RW-G-NLL")
        return 0.0, None if not return_relevance_weights else []
    
    # Compute relevance weights
    relevance_weights = compute_token_relevance_weights(
        prompt_x, y_greedy, tokenizer, similarity_model, cache=cache
    )
    
    # Verify alignment
    if len(relevance_weights) != len(token_log_likelihoods):
        raise ValueError(
            f"Token count mismatch: {len(relevance_weights)} relevance weights "
            f"vs {len(token_log_likelihoods)} token log-likelihoods"
        )
    
    # Calculate numerator: Σ R_T(y_t) · [-L_t]
    # Note: token_log_likelihoods are already log-probs (negative values)
    # For NLL, we need -L_t, which is -token_log_likelihoods[t]
    # So: weighted_nlls = relevance_weights[t] * (-token_log_likelihoods[t])
    weighted_nlls = [
        relevance_weights[t] * (-token_log_likelihoods[t])
        for t in range(len(token_log_likelihoods))
    ]
    numerator = sum(weighted_nlls)
    
    # Calculate denominator: Σ R_T(y_t)
    denominator = sum(relevance_weights)
    
    # Handle edge case: zero denominator
    if denominator == 0:
        logging.warning("Zero denominator (all tokens irrelevant), falling back to standard G-NLL")
        rw_gnll_score = -sum(token_log_likelihoods)  # Standard G-NLL
    else:
        rw_gnll_score = numerator / denominator
    
    if return_relevance_weights:
        return rw_gnll_score, relevance_weights
    else:
        return rw_gnll_score, None

