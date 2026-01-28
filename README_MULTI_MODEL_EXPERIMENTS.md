# Multi-Model Family Experiments - Ready to Run 🚀

## Quick Links

- **📖 [Complete Guide](MULTI_MODEL_EXPERIMENTS_GUIDE.md)** - Detailed instructions, troubleshooting, and usage
- **✅ [Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - What was implemented and completion status
- **📊 [Analysis Notebook](src/analysis_notebooks/multi_model_family_comparison.ipynb)** - Comprehensive visualizations

---

## What's Ready

✅ **Code**: Qwen model support added and tested  
✅ **Datasets**: TriviaQA and SQuAD validated  
✅ **Scripts**: 12 experiments ready to run (6 models × 2 datasets)  
✅ **Analysis**: AUROC computation and visualization notebooks  
✅ **Documentation**: Complete guides with troubleshooting  

---

## Quick Start (3 Steps)

### 1️⃣ Validate (5-10 minutes)

```bash
conda activate nllSAR
./run_baseline_validation.sh
```

This runs quick tests with 10 samples to verify everything works.

### 2️⃣ Run Experiments (12-24 hours)

```bash
./run_multi_model_experiments.sh
```

This runs all 12 experiments on your A100 40GB GPU.

**Alternative**: Run by phase
```bash
./run_small_models.sh   # Small models first (4-6 hours)
./run_large_models.sh   # Then large models (8-12 hours)
```

### 3️⃣ Analyze Results

```bash
# Compute AUROC
python compute_multi_model_auroc.py \
    --wandb_dir src/nikos/uncertainty/wandb \
    --output_dir results/multi_model_auroc

# Open analysis notebook
jupyter notebook src/analysis_notebooks/multi_model_family_comparison.ipynb
```

---

## Models to Be Tested

### Llama Family
- `Llama-3.2-1B` (1B, FP16, ~2GB)
- `Llama-3-8B-8bit` (8B, 8-bit, ~8GB)

### Qwen Family 🆕
- `Qwen2.5-1.5B` (1.5B, FP16, ~3GB) ✅ Tested
- `Qwen2.5-7B-8bit` (7B, 8-bit, ~8GB)

### Mistral Family
- `Mistral-7B-v0.3-8bit` (7B, 8-bit, ~8GB)
- `Mistral-7B-Instruct-v0.3-8bit` (7B, 8-bit, ~8GB)

**Total**: 6 models × 2 datasets = 12 experiments

---

## Expected Results

Each experiment generates:
- Answers for 400 questions
- Token-level log-likelihoods
- G-NLL uncertainty scores
- WandB tracking and logging

After analysis, you'll get:
- AUROC comparison across models
- Family performance rankings
- Size effect analysis
- Dataset-specific insights
- 8+ visualizations

---

## Need Help?

1. **Validation fails?** → See troubleshooting in [Complete Guide](MULTI_MODEL_EXPERIMENTS_GUIDE.md#troubleshooting)
2. **Out of memory?** → Scripts include memory cleanup; check GPU usage with `nvidia-smi`
3. **Qwen model issues?** → Verify with `python test_qwen_loading.py`
4. **Dataset issues?** → Verify with `python test_squad_loading.py`

---

## Files Overview

### Run Experiments
- `run_multi_model_experiments.sh` - Run all 12 experiments
- `run_baseline_validation.sh` - Quick validation (10 samples)
- `run_small_models.sh` - Small models only
- `run_large_models.sh` - Large models only

### Analyze Results
- `compute_multi_model_auroc.py` - Compute AUROC metrics
- `src/analysis_notebooks/multi_model_family_comparison.ipynb` - Visualizations

### Test Scripts
- `test_qwen_loading.py` - Validate Qwen implementation
- `test_squad_loading.py` - Validate SQuAD dataset

### Documentation
- `MULTI_MODEL_EXPERIMENTS_GUIDE.md` - Complete guide (detailed)
- `IMPLEMENTATION_SUMMARY.md` - What was built (technical)
- `README_MULTI_MODEL_EXPERIMENTS.md` - This file (quick start)

---

## Research Questions

1. **Model Size**: Do larger models (7B-8B) calibrate uncertainty better than small models (1B-1.5B)?
2. **Model Family**: Which family (Llama, Qwen, Mistral) has the best uncertainty estimation?
3. **Dataset**: Does uncertainty calibration differ between TriviaQA and SQuAD?
4. **Instruction Tuning**: Does Mistral-Instruct improve over base Mistral?

---

## System Requirements

- **GPU**: Single NVIDIA A100 40GB
- **Environment**: nllSAR conda environment
- **Time**: 12-24 hours for full experiments
- **Storage**: ~50GB for model caches

---

## Status

🎉 **READY TO RUN**

All infrastructure is implemented, tested, and documented. Start with validation:

```bash
./run_baseline_validation.sh
```

If that passes, you're ready for the full experiments!

---

**Last Updated**: 2026-01-27  
**Implementation Status**: ✅ Complete  
**Test Status**: ✅ Validated (Qwen loading, SQuAD dataset)
