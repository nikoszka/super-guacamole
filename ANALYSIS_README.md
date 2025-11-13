## Advanced Uncertainty Analysis Guide

This guide explains how to run the new analysis tools we added on top of the G‑NLL / RW‑G‑NLL baseline:

- **Phase 1**: Baseline metrics & token statistics
- **Phase 1.5**: Token-level NLL analysis (no SAR)
- **Phase 1.6**: Prefix-level NLL analysis
- **Phase 2**: Token relevance analysis (SAR-style relevance weights)
- **Phase 5**: Final comparative AUROC & cost-performance analysis
- **Interactive app**: Token-level visualization (NLL, relevance, RW‑G‑NLL-style)

All commands below assume you run them **from the repo root** (the directory containing `src/`).

---

## Prerequisites

- Python environment with the repo’s `requirements.txt` installed.
- Access to the generated pickle files, typically under:
  - `src/nikos/uncertainty/wandb/run-<ID>/files/validation_generations.pkl`
- For Llama models, use a name like `Llama-3.2-1B` and ensure your HF cache is set up.

In all commands below, replace:

- `PATH_TO_SHORT_PKL` with the path to the **short answers** `validation_generations.pkl`
- `PATH_TO_LONG_PKL` with the path to the **long answers** `validation_generations.pkl`

---

## Phase 1: Baseline Metrics & Token Statistics

**Script**: `src/analysis/phase1_baseline_metrics.py`

This script computes:

- Token count stats (min/mean/max/percentiles)
- Baseline metrics per answer:
  - G‑NLL (total NLL)
  - Average NLL
  - Perplexity
  - Average token probability

**Short answers (ROUGE correctness):**

```bash
python -m src.analysis.phase1_baseline_metrics \
  --short-pickle PATH_TO_SHORT_PKL \
  --output-dir results/phase1_short
```

**Long answers (LLM judge correctness):**

```bash
python -m src.analysis.phase1_baseline_metrics \
  --long-pickle PATH_TO_LONG_PKL \
  --output-dir results/phase1_long
```

**Outputs (per `--output-dir`):**

- `token_statistics.json`: token stats (min/mean/max, percentiles, counts)
- `baseline_metrics_short.csv` / `baseline_metrics_long.csv`: per-example metrics
- `baseline_metrics.json`: JSON version of all metrics

---

## Phase 1.5: Token-Level NLL Analysis (No SAR)

**Script**: `src/analysis/phase1_5_token_nll_analysis.py`

This treats **per-token NLL from G‑NLL** as a raw “importance” signal and analyzes:

- NLL vs position
- NLL distributions
- NLL vs POS tags
- Sentence-level token NLLs

**Example (long answers):**

```bash
python -m src.analysis.phase1_5_token_nll_analysis \
  --pickle-path PATH_TO_LONG_PKL \
  --model-name Llama-3.2-1B \
  --sample-size 100 \
  --output-dir results/phase1_5_long
```

Add `--no-pos-tagging` to skip POS analysis if spaCy/NLTK are not available.

**Key outputs:**

- `nll_vs_position.png`: token position vs NLL
- `nll_distribution.png`: distribution of token NLLs
- `nll_by_pos.png`: mean NLL by POS tag
- `position_nll_heatmap.png`: position bin vs mean NLL
- `sentence_level_nll_examples.json`: a few examples with:
  - `response`, `tokens`, `nlls`
- `example_XX_token_nlls.png`: bar plots of per-token NLL for specific answers

---

## Phase 1.6: Prefix-Level NLL Analysis

**Script**: `src/analysis/phase1_6_prefix_nll_analysis.py`

This analyzes how confidence evolves as the answer is generated:

- Per-example **prefix mean NLL** curves
- Aggregated curves for **correct vs incorrect** answers
- AUROC using only:
  - First token NLL
  - Mean NLL of first *k* tokens (configurable)

**Long answers (LLM judge correctness):**

```bash
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path PATH_TO_LONG_PKL \
  --output-dir results/phase1_6_long \
  --max-prefix-len 50 \
  --ks 1 3 5
```

**Short answers (ROUGE correctness):**

```bash
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path PATH_TO_SHORT_PKL \
  --output-dir results/phase1_6_short \
  --use-rouge --rouge-threshold 0.3 \
  --max-prefix-len 50 \
  --ks 1 3 5
```

**Key outputs:**

- `prefix_mean_nll_curves.png`:
  - Mean prefix mean NLL vs prefix length for correct vs incorrect
- `first_token_nll_boxplot.png`:
  - Boxplot of first-token NLL for correct vs incorrect
- `prefix_nll_summary.json`:
  - Aggregated curves
  - Early-token AUROCs (first token, first *k* tokens)

---

## Phase 2: Token Relevance (SAR-Style) Analysis

**Script**: `src/analysis/phase2_token_importance.py`

This computes **token relevance scores** \(R_T\) using the SAR-style ablation:

- For a sample of long answers:
  - Computes relevance weights per token
  - Analyzes correlations with position, NLL, POS tags
  - Prepares data for visualization

**Example (long answers):**

```bash
python -m src.analysis.phase2_token_importance \
  --pickle-path PATH_TO_LONG_PKL \
  --model-name Llama-3.2-1B \
  --similarity-model cross-encoder/stsb-roberta-large \
  --sample-size 50 \
  --output-dir results/phase2_long
```

Add `--no-pos-tagging` to skip POS analysis.

**Key outputs:**

- `relevance_vs_position.png`: relevance vs token position
- `relevance_vs_nll.png`: relevance vs NLL
- `relevance_by_pos.png`: mean relevance by POS tag
- `position_pos_heatmap.png`: position bin vs POS vs relevance
- `token_importance_results.json`: global (token-level, no tokens stored)
- `token_importance_examples.json`: **up to 10 examples** with:
  - `response`, `tokens`, `relevance_weights`, `nlls`, `positions`, `pos_tags`
  - Used by the visualization app
- `analysis_summary.json`: correlations & summary stats

---

## Phase 5: Comparative AUROC & Cost–Performance Analysis

**Script**: `src/analysis/phase5_comparative_analysis.py`

This computes AUROC for:

- Baselines: G‑NLL, Average NLL, Perplexity, Avg token prob
- SOTA baselines: SAR, Semantic Entropy
- Hybrid: RW‑G‑NLL

And produces:

- AUROC comparison table
- Cost vs performance plot (M vs AUROC)
- ROC curves for all metrics

**Example (long answers, using LLM judge accuracy):**

```bash
python -m src.analysis.phase5_comparative_analysis \
  --pickle-path PATH_TO_LONG_PKL \
  --model-name Llama-3.2-1B \
  --similarity-model cross-encoder/stsb-roberta-large \
  --output-dir results/phase5_long
```

Options:

- `--use-rouge` + `--rouge-threshold` for short-answer correctness
- `--no-sar` / `--no-se` / `--no-rw-gnll` to disable individual metrics
- `--num-samples-sar` / `--num-samples-se` to limit number of generations used

**Key outputs:**

- `auroc_comparison.csv`: sorted AUROC table for all metrics
- `all_metrics_results.json`: AUROCs + summary
- `cost_performance_plot.png`: M (or proxy cost) vs AUROC
- `roc_curves.png`: ROC curves overlayed for all metrics

---

## Interactive Token-Level Visualization App

**Script**: `src/analysis/token_visualization_app.py`

This Streamlit app visualizes tokens with colored backgrounds:

- **Mode 1: Raw NLL (Phase 1.5)**  
  - Uses `sentence_level_nll_examples.json`
  - Shows tokens colored by NLL (red = higher NLL / more uncertain)

- **Mode 2: Relevance-weighted (Phase 2)**  
  - Uses `token_importance_examples.json`
  - Lets you choose:
    - Relevance \(R_T\)
    - NLL
    - Relevance × NLL (per-token contribution to RW‑G‑NLL)

### Running the app

From the repo root:

```bash
streamlit run src/analysis/token_visualization_app.py
```

In the browser:

- Select **“Raw NLL (Phase 1.5)”** and set the path to:
  - `results/phase1_5_long/sentence_level_nll_examples.json` (or the short version)
- Or select **“Relevance-weighted (Phase 2)”** and set the path to:
  - `results/phase2_long/token_importance_examples.json`

You’ll see the full answer text with token backgrounds colored according to:

- NLL (uncertainty),
- Relevance (importance),
- Relevance × NLL (RW‑G‑NLL-style contribution) depending on the chosen view.

This app can be reused for:

- Short vs long answers,
- Different runs / models (just point it to different JSONs),
- Comparing raw NLL vs relevance-weighted views side by side.

---

## Suggested Workflow (Putting It All Together)

For a new model + dataset:

1. **Run Phase 1** on short and long pickles to get baseline stats and metrics.
2. **Run Phase 1.5** to understand raw token-level uncertainty patterns.
3. **Run Phase 1.6** to see if early prefixes already separate correct vs incorrect.
4. **Run Phase 2** on long answers to get SAR-style token relevance.
5. **Explore `token_visualization_app.py`** to inspect individual answers (NLL, relevance, relevance × NLL).
6. **Run Phase 5** to compare AUROC and cost–performance for all metrics, including RW‑G‑NLL.


