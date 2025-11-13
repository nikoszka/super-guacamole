"""Interactive token-level visualization app (Streamlit).

This app lets you explore:
1. Raw token-level NLLs (Phase 1.5, sentence_level_nll_examples.json)
2. Relevance weights from SAR / RW-G-NLL (Phase 2, token_importance_examples.json)

Tokens are rendered with background colors proportional to:
- NLL (uncertainty)
- Relevance
- Relevance × NLL (RW-G-NLL-style contribution)
"""

import json
import os
from typing import List, Dict, Any

import numpy as np
import streamlit as st  # type: ignore


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


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


def token_span(token: str, score: float, cmap: str = "red") -> str:
    """Render a single token as an HTML span with background color based on score.

    cmap = "red" -> white to red
    cmap = "blue" -> white to blue
    """
    alpha = score  # 0..1
    if cmap == "red":
        color = f"rgba(255, 0, 0, {alpha:.2f})"
    else:
        color = f"rgba(0, 0, 255, {alpha:.2f})"
    safe_token = token.replace("<", "&lt;").replace(">", "&gt;")
    return f'<span style="background-color:{color}; padding:2px; margin:1px; border-radius:3px;">{safe_token}</span>'


def render_token_sequence(tokens: List[str], scores: List[float], cmap: str) -> None:
    norm_scores = normalize_scores(scores)
    spans = [token_span(tok, s, cmap=cmap) for tok, s in zip(tokens, norm_scores)]
    html = " ".join(spans)
    st.markdown(html, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Token-level Visualization", layout="wide")
    st.title("🔍 Token-level Uncertainty & Relevance Visualization")

    st.markdown(
        """
Use this app to explore:
- **Raw NLL tokens** from Phase 1.5 (`sentence_level_nll_examples.json`)
- **Relevance-weighted tokens** from Phase 2 (`token_importance_examples.json`)

You can view:
- Pure NLL (uncertainty)
- Relevance weights (SAR / RW-G-NLL)
- Relevance × NLL (per-token RW-G-NLL-style contributions)
"""
    )

    st.sidebar.header("Data Sources")
    mode = st.sidebar.selectbox(
        "Select data source",
        ["Raw NLL (Phase 1.5)", "Relevance-weighted (Phase 2)"],
    )

    if mode == "Raw NLL (Phase 1.5)":
        nll_path = st.sidebar.text_input(
            "Path to sentence_level_nll_examples.json",
            value="results/phase1_5/sentence_level_nll_examples.json",
        )
        if not nll_path or not os.path.exists(nll_path):
            st.warning(f"File not found: {nll_path}")
            return

        data = load_json(nll_path)
        if not data:
            st.warning("No examples in JSON file.")
            return

        idx = st.sidebar.slider(
            "Select example index", min_value=1, max_value=len(data), value=1
        )
        example = data[idx - 1]

        tokens = example["tokens"]
        nlls = example["nlls"]

        st.subheader(f"Example {idx}")
        st.write("**Response:**")
        st.write(example["response"])

        st.write("**Token-level NLL (red = higher NLL / more uncertain):**")
        render_token_sequence(tokens, nlls, cmap="red")

    else:
        imp_path = st.sidebar.text_input(
            "Path to token_importance_examples.json",
            value="results/phase2/token_importance_examples.json",
        )
        if not imp_path or not os.path.exists(imp_path):
            st.warning(f"File not found: {imp_path}")
            return

        data = load_json(imp_path)
        if not data:
            st.warning("No examples in JSON file.")
            return

        idx = st.sidebar.slider(
            "Select example index", min_value=1, max_value=len(data), value=1
        )
        example = data[idx - 1]

        tokens = example["tokens"]
        relevance = example["relevance_weights"]
        nlls = example["nlls"]

        st.subheader(f"Example {idx} (ID: {example.get('example_id')})")
        st.write("**Response:**")
        st.write(example["response"])

        view = st.radio(
            "Score to visualize",
            ["Relevance", "NLL", "Relevance × NLL"],
            horizontal=True,
        )

        if view == "Relevance":
            scores = relevance
            desc = "Relevance weights R_T(y_t) (red = more relevant)"
        elif view == "NLL":
            scores = nlls
            desc = "Negative log-likelihood (red = more uncertain)"
        else:
            # Per-token RW-G-NLL-style contribution
            scores = (np.array(relevance) * np.array(nlls)).tolist()
            desc = "Relevance × NLL (red = large contribution to RW-G-NLL)"

        st.write(f"**{desc}:**")
        render_token_sequence(tokens, scores, cmap="red")


if __name__ == "__main__":
    main()


