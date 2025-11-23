## Advanced Uncertainty Analysis Guide

This guide explains how to run the new analysis tools we added on top of the G‑NLL / RW‑G‑NLL baseline:

- **Phase 1**: Baseline metrics & token statistics
- **Phase 1.5**: Token-level NLL analysis (no SAR)
- **Phase 1.6**: Prefix-level NLL analysis
- **Phase 2**: Token relevance analysis (SAR-style relevance weights)
- **Phase 5**: Final comparative AUROC & cost-performance analysis
- **Interactive app**: Token-level visualization (NLL, relevance, RW‑G‑NLL-style)

All commands below assume you run them **from the repo root** (the directory containing `src/`).

### 🆕 Folder Naming Convention

All phase analysis scripts now support **automatic folder naming** based on:
- **Context type**: `short` or `long` (use `--context-type` parameter)
- **WandB run ID**: from your generation run (use `--wandb-run-id` parameter)

**Auto-generated folder pattern**: `results/phase{X}_{context_type}_{wandb_run_id}`

Example: `results/phase1_6_long_wiboofpr`

This helps organize results from multiple runs and makes it easy to track which results correspond to which WandB run.

You can still override this with a custom `--output-dir` if needed.

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

**New Parameters:**
- `--wandb-run-id`: WandB run ID to include in output folder name
- `--context-type`: Context type (short or long) for output folder naming
- `--output-dir`: Custom output directory (optional - if not provided, auto-generated from wandb-run-id and context-type)

**Short answers (ROUGE correctness):**

```bash
python -m src.analysis.phase1_baseline_metrics \
  --short-pickle "C:\Users\nikos\PycharmProjects\nllSAR\src\boldis\uncertainty\wandb\run-20251121_142824-wo7cdccl\files\validation_generations.pkl" \
  --context-type short \
  --wandb-run-id wo7cdccl
```

This will auto-generate output directory: `results/phase1_short_yhxde999`

**Long answers (LLM judge correctness):**

```bash
python -m src.analysis.phase1_baseline_metrics 
  --long-pickle "C:\Users\nikos\PycharmProjects\nllSAR\src\boldis\uncertainty\wandb\run-20251121_092732-wiboofpr\files\validation_generations.pkl" 
  --context-type long \
  --wandb-run-id wiboofpr
```

This will auto-generate output directory: `results/phase1_long_wiboofpr`

**Outputs (per output directory):**

- `token_statistics.json`: token stats (min/mean/max, percentiles, counts)
- `baseline_metrics_short.csv` / `baseline_metrics_long.csv`: per-example metrics
- `baseline_metrics.json`: JSON version of all metrics

---

## Phase 1.5: Token-Level NLL Analysis (No SAR)

**Script**: `src/analysis/phase1_5_token_nll_analysis.py`

This treats **per-token NLL from G‑NLL** as a raw "importance" signal and analyzes:

- NLL vs position
- NLL distributions
- NLL vs POS tags
- Sentence-level token NLLs

**New Parameters:**
- `--wandb-run-id`: WandB run ID to include in output folder name
- `--context-type`: Context type (short or long) for output folder naming
- `--output-dir`: Custom output directory (optional - if not provided, auto-generated)

**Example (long answers):**

```bash
python -m src.analysis.phase1_5_token_nll_analysis \
  --pickle-path "C:\Users\nikos\PycharmProjects\nllSAR\src\boldis\uncertainty\wandb\run-20251121_092732-wiboofpr\files\validation_generations.pkl"  \
  --model-name Llama-3.2-1B \
  --sample-size 100 \
  --context-type long \
  --wandb-run-id wiboofpr
```

This will auto-generate output directory: `results/phase1_5_long_wiboofpr`

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

**New Parameters:**
- `--wandb-run-id`: WandB run ID to include in output folder name
- `--context-type`: Context type (short or long) for output folder naming
- `--output-dir`: Custom output directory (optional - if not provided, auto-generated)

**Long answers (LLM judge correctness):**

```bash
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path "C:\Users\nikos\PycharmProjects\nllSAR\src\boldis\uncertainty\wandb\run-20251121_092732-wiboofpr\files\validation_generations.pkl" \
  --context-type long \
  --wandb-run-id wiboofpr \
  --max-prefix-len 50 \
  --ks 1 3 5
```

This will auto-generate output directory: `results/phase1_6_long_wiboofpr`

**Short answers (ROUGE correctness):**

```bash
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path PATH_TO_SHORT_PKL \
  --context-type short \
  --wandb-run-id yhxde999 \
  --use-rouge --rouge-threshold 0.3 \
  --max-prefix-len 50 \
  --ks 1 3 5
```

This will auto-generate output directory: `results/phase1_6_short_yhxde999`

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

**New Parameters:**
- `--wandb-run-id`: WandB run ID to include in output folder name
- `--context-type`: Context type (short or long) for output folder naming
- `--output-dir`: Custom output directory (optional - if not provided, auto-generated)

**Example (long answers):**

```bash
python -m src.analysis.phase2_token_importance \
  --pickle-path PATH_TO_LONG_PKL \
  --model-name Llama-3.2-1B \
  --similarity-model cross-encoder/stsb-roberta-large \
  --sample-size 50 \
  --context-type long \
  --wandb-run-id wiboofpr
```

This will auto-generate output directory: `results/phase2_long_wiboofpr`

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

**New Parameters:**
- `--wandb-run-id`: WandB run ID to include in output folder name
- `--context-type`: Context type (short or long) for output folder naming
- `--output-dir`: Custom output directory (optional - if not provided, auto-generated)

**Example (long answers, using LLM judge accuracy):**

```bash
python -m src.analysis.phase5_comparative_analysis \
  --pickle-path PATH_TO_LONG_PKL \
  --model-name Llama-3.2-1B \
  --similarity-model cross-encoder/stsb-roberta-large \
  --context-type long \
  --wandb-run-id wiboofpr
```

This will auto-generate output directory: `results/phase5_long_wiboofpr`

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
**Launcher**: `run_token_viz_app.py`  
**Guide**: `TOKEN_VIZ_APP_GUIDE.md`

This **upgraded** Streamlit app now loads directly from validation pickle files and provides:

- **Probability Visualization** (NEW!)
  - Green colormap: darker = more confident
  - Shows token-level model confidence (`exp(log_likelihood)`)
  
- **NLL Visualization**
  - Red colormap: darker = more uncertain
  - Shows negative log-likelihood per token

- **Full Context Display** (NEW!)
  - Original question from dataset
  - Correct answer(s) (ground truth)
  - Model response with accuracy indicator
  
- **Answer Type Support** (NEW!)
  - Toggle between short and long answers
  - Pre-configured paths for both types
  - Custom pickle file support

- **Statistics Panel** (NEW!)
  - Mean, min, max, standard deviation
  - Adapts to score type (probability vs NLL)

### Running the app

From the repo root:

```bash
# Easy launcher (recommended)
python run_token_viz_app.py

# Or directly with streamlit
streamlit run src/analysis/token_visualization_app.py
```

**New Features:**
- 🟢 **Probability visualization** (green colormap) - toggle between probability and NLL
- 📝 **Question & correct answer** displayed alongside model response
- 🔄 **Short & long answer support** with pre-configured paths
- 📊 **Statistics panel** showing mean, min, max, std dev
- ✅ **Accuracy indicators** for each example

In the browser:

1. Select **"Long Answers"** or **"Short Answers"**
   - Pre-filled paths to judge-corrected (long) or standard (short) pickles
   - Or enter custom path to any `validation_generations.pkl` file

2. Browse examples with the slider (1 to N)

3. Toggle between **"Probability"** (green) or **"NLL"** (red) visualization

You'll see:
- The original **question** from the dataset
- The **correct answer(s)** (ground truth)
- The **model's response** with accuracy indicator (✅/❌)
- Token-level visualization with color intensity showing:
  - **Probability**: Darker green = more confident
  - **NLL**: Darker red = more uncertain

**See `TOKEN_VIZ_APP_GUIDE.md` for detailed usage instructions.**

This app can be used for:

- Comparing short vs long answer confidence patterns
- Identifying where the model is uncertain
- Debugging incorrect predictions
- Understanding token-level model behavior
- Analyzing first-token probabilities

---

## Suggested Workflow (Putting It All Together)

For a new model + dataset:

1. **Run Phase 1** on short and long pickles to get baseline stats and metrics.
2. **Run Phase 1.5** to understand raw token-level uncertainty patterns.
3. **Run Phase 1.6** to see if early prefixes already separate correct vs incorrect.
4. **Run Phase 2** on long answers to get SAR-style token relevance.
5. **Explore `token_visualization_app.py`** (launch with `python run_token_viz_app.py`) to:
   - Inspect individual answers with full question/answer context
   - Visualize token-level probabilities (confidence) and NLL (uncertainty)
   - Compare short vs long answer patterns
   - Debug incorrect predictions
6. **Run Phase 5** to compare AUROC and cost–performance for all metrics, including RW‑G‑NLL.


