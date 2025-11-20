# 🚀 Quick Start: Generate & Analyze

## Step 1: Generate Short Answers (15-30 min)

```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 200 \
  --num_generations 1 \
  --temperature 0.0 \
  --model_max_new_tokens 50 \
  --brief_prompt short \
  --metric squad \
  --use_context False \
  --compute_p_true False \
  --entity nikosteam \
  --project nllSAR_short
```

**Output:** `src/nikos/uncertainty/wandb/run-XXXXX/files/validation_generations.pkl`

---

## Step 2: Generate Long Answers (30-60 min)

```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 200 \
  --num_generations 1 \
  --temperature 0.0 \
  --model_max_new_tokens 200 \
  --brief_prompt detailed \
  --metric llm_llama-3.1-70b \
  --use_context True \
  --compute_p_true False \
  --entity nikosteam \
  --project nllSAR_long
```

**Output:** Another `validation_generations.pkl` in a new wandb run directory

---

## Step 3: Run Analysis Pipeline

### A. Phase 1: Baseline Metrics

**Short:**
```bash
python -m src.analysis.phase1_baseline_metrics \
  --short-pickle PATH_TO_SHORT_PKL \
  --output-dir results/phase1_short
```

**Long:**
```bash
python -m src.analysis.phase1_baseline_metrics \
  --long-pickle PATH_TO_LONG_PKL \
  --output-dir results/phase1_long
```

---

### B. Phase 1.5: Token-Level NLL Analysis

**Long answers:**
```bash
python -m src.analysis.phase1_5_token_nll_analysis \
  --pickle-path PATH_TO_LONG_PKL \
  --model-name Llama-3.2-1B \
  --sample-size 100 \
  --output-dir results/phase1_5_long
```

**Short answers:**
```bash
python -m src.analysis.phase1_5_token_nll_analysis \
  --pickle-path PATH_TO_SHORT_PKL \
  --model-name Llama-3.2-1B \
  --sample-size 100 \
  --output-dir results/phase1_5_short
```

---

### C. Phase 1.6: Prefix-Level NLL

**Long (LLM judge):**
```bash
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path PATH_TO_LONG_PKL \
  --output-dir results/phase1_6_long \
  --max-prefix-len 50 \
  --ks 1 3 5
```

**Short (ROUGE):**
```bash
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path PATH_TO_SHORT_PKL \
  --output-dir results/phase1_6_short \
  --use-rouge \
  --rouge-threshold 0.3 \
  --max-prefix-len 50 \
  --ks 1 3 5
```

---

### D. Phase 2: Token Relevance (SAR-style)

```bash
python -m src.analysis.phase2_token_importance \
  --pickle-path PATH_TO_LONG_PKL \
  --model-name Llama-3.2-1B \
  --similarity-model cross-encoder/stsb-roberta-large \
  --sample-size 50 \
  --output-dir results/phase2_long
```

---

### E. Phase 5: Comparative AUROC Analysis

**Long answers:**
```bash
python -m src.analysis.phase5_comparative_analysis \
  --pickle-path PATH_TO_LONG_PKL \
  --model-name Llama-3.2-1B \
  --similarity-model cross-encoder/stsb-roberta-large \
  --output-dir results/phase5_long
```

**Short answers (ROUGE-based):**
```bash
python -m src.analysis.phase5_comparative_analysis \
  --pickle-path PATH_TO_SHORT_PKL \
  --model-name Llama-3.2-1B \
  --similarity-model cross-encoder/stsb-roberta-large \
  --output-dir results/phase5_short \
  --use-rouge \
  --rouge-threshold 0.3
```

---

## Step 4: Visualize Results

```bash
streamlit run src/analysis/token_visualization_app.py
```

**In the browser:**
- Mode: "Raw NLL (Phase 1.5)" → Load `results/phase1_5_long/sentence_level_nll_examples.json`
- Mode: "Relevance-weighted (Phase 2)" → Load `results/phase2_long/token_importance_examples.json`

---

## 📁 File Locations Cheat Sheet

### After Generation:
```
src/nikos/uncertainty/wandb/
└── run-<TIMESTAMP>-<ID>/
    └── files/
        ├── validation_generations.pkl    ← Use this for analysis
        ├── train_generations.pkl
        └── experiment_details.pkl
```

### After Analysis:
```
results/
├── phase1_short/
│   ├── token_statistics.json
│   ├── baseline_metrics_short.csv
│   └── baseline_metrics.json
├── phase1_5_long/
│   ├── nll_vs_position.png
│   ├── nll_distribution.png
│   └── sentence_level_nll_examples.json
├── phase2_long/
│   ├── relevance_vs_position.png
│   └── token_importance_examples.json
└── phase5_long/
    ├── auroc_comparison.csv
    ├── roc_curves.png
    └── cost_performance_plot.png
```

---

## ✅ Verify Your Pickle

Before running analysis, verify the new pickle structure:

```python
import pickle

# Load your pickle
with open('PATH_TO_YOUR_PICKLE', 'rb') as f:
    data = pickle.load(f)

# Get first example
example = list(data.values())[0]
mla = example['most_likely_answer']

# Verify new fields exist
print("✅ Has token_ids:", 'token_ids' in mla)
print("✅ Has tokens:", 'tokens' in mla)
print("✅ Token count:", len(mla.get('tokens', [])))
print("✅ Log-lik count:", len(mla.get('token_log_likelihoods', [])))
print("✅ Match:", len(mla.get('tokens', [])) == len(mla.get('token_log_likelihoods', [])))

# Show sample
print("\nFirst 5 tokens:", mla.get('tokens', [])[:5])
print("First 5 log-liks:", mla.get('token_log_likelihoods', [])[:5])
```

Expected output:
```
✅ Has token_ids: True
✅ Has tokens: True
✅ Token count: 45
✅ Log-lik count: 45
✅ Match: True

First 5 tokens: [' The', ' Battle', ' of', ' Hast', 'ings']
First 5 log-liks: [-0.234, -0.156, -0.089, -0.445, -0.234]
```

---

## 🎯 What Changed? (Quick Version)

### Before (Old Pickles)
```python
most_likely_answer = {
    'response': "The Battle of Hastings",
    'token_log_likelihoods': [-0.234, -0.156, ...]
}
# ❌ No tokens stored → Analysis re-tokenizes → Mismatches!
```

### After (New Pickles)
```python
most_likely_answer = {
    'response': "The Battle of Hastings",
    'token_ids': [450, 11045, 315, 19826, 826],        # NEW!
    'tokens': [' The', ' Battle', ' of', ' Hast', 'ings'],  # NEW!
    'token_log_likelihoods': [-0.234, -0.156, -0.089, -0.445, -0.234]
}
# ✅ Exact tokens stored → Analysis uses them → Perfect alignment!
```

---

## 🐛 Common Issues

### "Token count mismatch" warnings
→ Using old pickle. Re-generate with updated code.

### "No module named 'transformers'"
→ Install: `pip install transformers accelerate`

### "CUDA out of memory"
→ Reduce `--num_samples` or use smaller model

### "KeyError: 'token_ids'"
→ Old pickle + outdated analysis script. Pull latest code.

---

## 📊 Expected Analysis Coverage

| Phase | Old Pickles | New Pickles |
|-------|-------------|-------------|
| Phase 1 | ✅ Works | ✅ Works |
| Phase 1.5 | ❌ 35% success | ✅ 100% success |
| Phase 1.6 | ✅ Works | ✅ Works |
| Phase 2 | ❌ ~40% success | ✅ 100% success |
| Phase 5 (baselines) | ✅ Works | ✅ Works |
| Phase 5 (SAR/SE) | ❌ Need multi-samples | ✅ Works with temp=1.0 |

---

## 📚 More Documentation

- **Detailed changes**: `CHANGES_SUMMARY.md`
- **Generation settings guide**: `GENERATION_SETTINGS_GUIDE.md`
- **Full analysis pipeline**: `ANALYSIS_README.md`

---

## 💡 Pro Tips

1. **Start small**: Use `--num_samples 50` for testing
2. **Use ROUGE for speed**: `--metric squad` is much faster than LLM judges
3. **Greedy first**: Always start with `--num_generations 1 --temperature 0.0` for baseline
4. **Few-shot helps**: Keep `--num_few_shot 5` (improves quality, unrelated to temperature)
5. **GPU memory**: Monitor with `nvidia-smi` during generation
6. **Intermediate saves**: Wandb auto-saves, check wandb run folder periodically
7. **Analysis order**: Run Phase 1 → 1.5 → 1.6 → 2 → 5 in sequence

---

## 🎉 You're Ready!

1. Generate pickles (Steps 1-2) ✅
2. Run all analysis phases (Step 3) ✅
3. Visualize with Streamlit (Step 4) ✅
4. Enjoy your comprehensive uncertainty analysis! 🎊

