"""Interactive token-level visualization app (Streamlit).

This app lets you explore:
1. Token-level probabilities and NLLs from validation pickle files
2. Relevance weights from SAR / RW-G-NLL (Phase 2, token_importance_examples.json)

Tokens are rendered with background colors proportional to:
- Probability (confidence)
- NLL (uncertainty)
- Relevance
- Relevance × NLL (RW-G-NLL-style contribution)
"""

import json
import os
import pickle
from typing import List, Dict, Any, Optional

import numpy as np
import streamlit as st  # type: ignore


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_pickle(path: str) -> Dict[str, Any]:
    """Load pickle file with validation generations."""
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_scores(scores: List[float], invert: bool = False) -> List[float]:
    """Normalize scores to [0, 1] for color mapping."""
    arr = np.array(scores, dtype=float)
    if invert:
        arr = -arr
    if arr.size == 0:
        return [0.0] * len(scores)
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v - min_v < 1e-8:
        return [0.5] * len(scores)
    norm = (arr - min_v) / (max_v - min_v)
    return norm.tolist()


def token_span(token: str, score: float, raw_score: float, cmap: str = "red", score_type: str = "NLL", pos_tag: str = None) -> str:
    """Render a single token as an HTML span with background color and score below.

    cmap = "red" -> white to red
    cmap = "blue" -> white to blue
    cmap = "green" -> white to green
    
    Args:
        token: The token text
        score: Normalized score [0, 1] for coloring
        raw_score: The actual score value to display
        cmap: Color map to use
        score_type: Type of score being displayed (for formatting)
        pos_tag: Optional POS tag to display
    """
    alpha = score  # 0..1
    if cmap == "red":
        color = f"rgba(255, 0, 0, {alpha:.2f})"
    elif cmap == "blue":
        color = f"rgba(0, 0, 255, {alpha:.2f})"
    else:  # green for probabilities
        color = f"rgba(0, 200, 0, {alpha:.2f})"
    
    safe_token = token.replace("<", "&lt;").replace(">", "&gt;")
    
    # Format score based on type
    if score_type == "Probability":
        score_str = f"{raw_score:.4f}"
    else:
        score_str = f"{raw_score:.3f}"
    
    # Add POS tag badge if available
    pos_html = ""
    if pos_tag:
        pos_html = f'<div style="font-size:9px; color:#888; margin-bottom:1px;">{pos_tag}</div>'
    
    # Display token on top, score below in smaller font
    return f'''<span style="display:inline-block; text-align:center; margin:2px; padding:3px; border-radius:3px; background-color:{color}; vertical-align: top;">
        {pos_html}
        <div style="font-family:monospace; font-size:14px; white-space:nowrap; font-weight:bold;">{safe_token}</div>
        <div style="font-family:monospace; font-size:10px; color:#555; margin-top:2px;">{score_str}</div>
    </span>'''


def render_token_sequence(tokens: List[str], scores: List[float], cmap: str, score_type: str = "NLL", pos_tags: List[str] = None) -> None:
    """Render a sequence of tokens with colored backgrounds and scores displayed below each token.
    
    Args:
        tokens: List of token strings
        scores: List of raw scores for each token
        cmap: Color map to use ('red', 'blue', or 'green')
        score_type: Type of score being displayed
        pos_tags: List of POS tags for each token (optional)
    """
    norm_scores = normalize_scores(scores, invert=False)
    
    if pos_tags is None:
        pos_tags = [None] * len(tokens)
        
    spans = [token_span(tok, norm_s, raw_s, cmap=cmap, score_type=score_type, pos_tag=pos) 
             for tok, norm_s, raw_s, pos in zip(tokens, norm_scores, scores, pos_tags)]
    html = '<div style="line-height:2.5;">' + "".join(spans) + '</div>'
    st.markdown(html, unsafe_allow_html=True)


def extract_pickle_examples(pickle_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract examples from pickle file with question and answer information."""
    examples = []
    for example_id, entry in pickle_data.items():
        if "most_likely_answer" not in entry:
            continue
        
        mla = entry["most_likely_answer"]
        
        # Extract tokens and log-likelihoods
        tokens = mla.get("tokens", [])
        token_log_likelihoods = mla.get("token_log_likelihoods", [])
        
        if not tokens or not token_log_likelihoods or len(tokens) != len(token_log_likelihoods):
            continue
        
        # Extract question and reference answer
        question = entry.get("question", "N/A")
        reference = entry.get("reference", {})
        correct_answers = reference.get("answers", {}).get("text", [])
        if isinstance(correct_answers, list):
            correct_answer = ", ".join(correct_answers) if correct_answers else "N/A"
        else:
            correct_answer = str(correct_answers)
        
        # Compute NLLs and probabilities
        log_probs = np.array(token_log_likelihoods, dtype=float)
        nlls = -log_probs
        probs = np.exp(log_probs)
        
        examples.append({
            "example_id": example_id,
            "question": question,
            "correct_answer": correct_answer,
            "response": mla.get("response", ""),
            "tokens": tokens,
            "nlls": nlls.tolist(),
            "probs": probs.tolist(),
            "accuracy": mla.get("accuracy", 0.0),
        })
    
    return examples


def main() -> None:
    st.set_page_config(page_title="Token-level Visualization", layout="wide")
    st.title("🔍 Token-level Probability & Uncertainty Visualization")

    st.markdown(
        """
Use this app to explore token-level model behavior:
- **Probabilities** (model confidence per token)
- **NLL** (negative log-likelihood / uncertainty per token)
- **Question & Correct Answer** context
- Support for both **short** and **long** answer formats

Load pickle files from validation runs to visualize how the model generates each token.
"""
    )

    st.sidebar.header("Data Source")
    
    # Dataset type selection
    dataset_type = st.sidebar.radio(
        "Answer Type",
        ["Long Answers", "Short Answers"],
        help="Select the type of answers to visualize"
    )
    
    # Default paths based on selection
    default_paths = {
        "Long Answers": "src/boldis/uncertainty/wandb/run-20251121_092732-wiboofpr/files/validation_generations_judge_corrected.pkl",
        "Short Answers": "src/boldis/uncertainty/wandb/run-20251121_011028-sqykmrn7/files/validation_generations.pkl"
    }
    
    pickle_path = st.sidebar.text_input(
        "Path to file (pickle or JSON)",
        value=default_paths[dataset_type],
        help="Supports .pkl (original) or .json (Phase 1.7 POS analysis output)"
    )
    
    if not pickle_path or not os.path.exists(pickle_path):
        st.warning(f"File not found: {pickle_path}")
        st.info("Please provide a valid path to a validation_generations.pkl or pos_visualization_examples.json file")
        return

    # Load and cache data
    @st.cache_data
    def load_data(path: str):
        if path.endswith(".json"):
            # Load JSON directly (Phase 1.7 format)
            with open(path, "r") as f:
                return json.load(f)
        else:
            # Load pickle and extract
            data = load_pickle(path)
            examples = extract_pickle_examples(data)
            return examples
    
    try:
        examples = load_data(pickle_path)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return
    
    if not examples:
        st.warning("No valid examples found in pickle file.")
        return

    st.sidebar.success(f"✅ Loaded {len(examples)} examples")
    
    # Example selection
    idx = st.sidebar.slider(
        "Select example index", 
        min_value=1, 
        max_value=len(examples), 
        value=1
    )
    example = examples[idx - 1]
    
    # Score type selection
    score_type = st.sidebar.radio(
        "Score to visualize",
        ["Probability", "NLL"],
        help="Probability = model confidence (green), NLL = uncertainty (red)"
    )
    
    # Analysis level selection (Tokens vs Words)
    view_level = st.sidebar.radio(
        "Analysis Level",
        ["Tokens", "Words"],
        help="View individual tokens or aggregated words"
    )
    
    show_pos = st.sidebar.checkbox("Show POS Tags", value=False)
    
    # Main display
    st.markdown("---")
    
    # Display question and correct answer
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📝 Question")
        st.info(example["question"])
    
    with col2:
        st.markdown("### ✅ Correct Answer")
        st.success(example["correct_answer"])
    
    st.markdown("---")
    
    # Display model response
    st.markdown("### 🤖 Model Response")
    accuracy_emoji = "✅" if example["accuracy"] > 0.5 else "❌"
    st.markdown(f"**Accuracy:** {accuracy_emoji} {example['accuracy']:.2f}")
    st.write(example["response"])
    
    st.markdown("---")
    
    # Prepare data based on view level
    if view_level == "Words" and "words" in example:
        # Use word-level data
        words_data = example["words"]
        tokens = [w["word"] for w in words_data]
        pos_tags = [w["pos"] for w in words_data] if show_pos else None
        
        if score_type == "Probability":
            # For words, probability is exp(-sum(NLL))
            # or simpler: sum(NLL) is total uncertainty, exp(-NLL) is prob
            scores = [np.exp(-w["nll"]) for w in words_data]
        else:
            scores = [w["nll"] for w in words_data]
    else:
        # Use token-level data
        tokens = example["tokens"]
        pos_tags = example.get("pos_tags", None) if show_pos else None
        
        if score_type == "Probability":
            scores = example["probs"]
        else:
            scores = example["nlls"]

    # Display visualization
    if score_type == "Probability":
        st.markdown(f"### 🟢 {view_level}-level Probabilities")
        st.markdown("**Green intensity = Higher probability (more confident)**")
        render_token_sequence(tokens, scores, cmap="green", score_type="Probability", pos_tags=pos_tags)
        
        # Statistics
        st.markdown("#### Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Probability", f"{np.mean(scores):.4f}")
        with col2:
            st.metric("Min Probability", f"{np.min(scores):.4f}")
        with col3:
            st.metric("Max Probability", f"{np.max(scores):.4f}")
        with col4:
            st.metric("Std Dev", f"{np.std(scores):.4f}")
    else:
        st.markdown(f"### 🔴 {view_level}-level NLL (Negative Log-Likelihood)")
        st.markdown("**Red intensity = Higher NLL (more uncertain)**")
        render_token_sequence(tokens, scores, cmap="red", score_type="NLL", pos_tags=pos_tags)
        
        # Statistics
        st.markdown("#### Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean NLL", f"{np.mean(scores):.3f}")
        with col2:
            st.metric("Min NLL", f"{np.min(scores):.3f}")
        with col3:
            st.metric("Max NLL", f"{np.max(scores):.3f}")
        with col4:
            st.metric("Std Dev", f"{np.std(scores):.3f}")
    
    # Example ID display
    st.markdown("---")
    st.caption(f"Example ID: `{example['example_id']}`")


if __name__ == "__main__":
    main()


