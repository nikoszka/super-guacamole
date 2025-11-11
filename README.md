# 🔍 Uncertainty Estimation in LLMs: G-NLL Baseline & RW-G-NLL

This repository contains research code for evaluating uncertainty estimation methods in Large Language Models, with a focus on **G-NLL (Greedy Negative Log-Likelihood)** baseline and **RW-G-NLL (Relevance-Weighted G-NLL)**.

---

## 📚 Overview

This work evaluates how well uncertainty metrics can predict answer correctness in LLM outputs. The main contributions are:

- **G-NLL Baseline**: Sum of token log-probabilities from greedy decoding, used as a simple uncertainty metric
- **RW-G-NLL**: An improved metric that re-weights token log-likelihoods by semantic relevance to the prompt, filtering out noise from "generative inequality"
- **Comprehensive Evaluation**: AUROC analysis comparing uncertainty scores against ground truth correctness (via ROUGE for short answers, LLM-as-a-judge for long answers)

---

## 🎯 Key Metrics

### G-NLL (Greedy Negative Log-Likelihood)
The sum of token log-probabilities from the greedy answer:
```
G-NLL = -Σ log P(y_t | x, y_<t)
```
Lower G-NLL indicates higher confidence.

### RW-G-NLL (Relevance-Weighted G-NLL)
Re-weights token log-likelihoods by semantic relevance to the prompt:
```
RW-G-NLL = Σ [R_T(y_t) · (-log P(y_t))] / Σ R_T(y_t)
```
Where `R_T(y_t)` is the semantic relevance of token `y_t` to the prompt `x`, computed using a cross-encoder similarity model. This filters out irrelevant tokens that contribute noise to the uncertainty estimate.

---

## 🛠️ Main Components

### Scripts
- **`run_gnll_baseline.py`**: Full pipeline for generating answers, evaluating with LLM judge, and computing AUROC
- **`compute_gnll_auroc.py`**: Standalone script for computing G-NLL and RW-G-NLL AUROC from pickle files

### Analysis
- **`src/analysis_notebooks/gnll_baseline_analysis.ipynb`**: Comprehensive analysis notebook with:
  - ROC curves and AUROC computation
  - G-NLL distribution analysis (correct vs incorrect)
  - ROUGE score analysis for short answers
  - LLM judge accuracy analysis
  - RW-G-NLL comparison and relevance weights visualization
  - Sample answer inspection

### Core Modules
- **`src/uncertainty_measures/rw_gnll.py`**: RW-G-NLL implementation
- **`src/generate_answers.py`**: Answer generation with greedy decoding
- **`src/compute_uncertainty_measures.py`**: Uncertainty metric computation

---

## 🧪 Evaluation Pipeline

1. **Generate Answers**: Short (concise) and long (detailed) answers using greedy decoding
2. **Evaluate Correctness**:
   - Short answers: ROUGE-L scores (threshold: 0.3) + LLM-as-a-judge
   - Long answers: LLM-as-a-judge only
3. **Compute Uncertainty Metrics**: G-NLL and optionally RW-G-NLL
4. **Calculate AUROC**: Measure how well uncertainty predicts correctness

---

## 📊 Results

The analysis compares:
- **Short vs Long Answers**: How answer length affects uncertainty estimation
- **ROUGE vs LLM Judge**: Different correctness criteria for short answers
- **G-NLL vs RW-G-NLL**: Effectiveness of relevance weighting

Results are exported to JSON and visualized in the analysis notebook.

---

## 🛠️ Tech Stack

- Python 3.10+
- PyTorch / HuggingFace Transformers
- Sentence Transformers (for RW-G-NLL similarity model)
- NumPy, Scikit-learn (for AUROC computation)
- Matplotlib / Seaborn (for visualizations)
- Jupyter Notebooks
- Weights & Biases (for experiment tracking)

---

## 📝 Usage

### Running the Full Pipeline
```bash
python run_gnll_baseline.py
```

### Computing AUROC from Existing Results
```bash
python compute_gnll_auroc.py --pickle_path <path_to_validation_generations.pkl> [--use_rw_gnll]
```

### Analysis Notebook
Open `src/analysis_notebooks/gnll_baseline_analysis.ipynb` and configure paths to your `validation_generations.pkl` files.

---

## 📖 Related Documentation

- `GNLL_BASELINE_README.md`: Detailed guide for G-NLL baseline experiments
- `GREEDY_DECODING_README.md`: Information about greedy decoding setup
- `ENVIRONMENT_SETUP.md`: Environment configuration guide
