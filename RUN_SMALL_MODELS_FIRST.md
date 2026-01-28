# Starting with Small Models - Quick Guide

## Perfect Strategy for 3×11GB GPUs 🎯

Starting with small models is smart! They'll definitely fit comfortably in your 11GB GPUs.

---

## Small Models to Test

| Model | Size | Memory | Status on 11GB |
|-------|------|--------|----------------|
| Llama-3.2-1B | 1B | ~2GB | ✅ Very safe |
| Qwen2.5-1.5B | 1.5B | ~3GB | ✅ Very safe |
| Mistral-7B-v0.3-8bit | 7B | ~8-10GB | ⚠️ Should fit, monitor |

**Total**: 3 models × 2 datasets = 6 experiments  
**Time**: ~4-6 hours  
**Risk**: Very low - these will all fit easily

---

## Quick Start (3 Steps)

### 1. Verify GPU Setup (30 seconds)

```bash
# Check you have 3 GPUs
python -c "import torch; print(f'GPUs detected: {torch.cuda.device_count()}')"

# Should show: GPUs detected: 3
```

### 2. Run Quick Validation (5-10 minutes)

```bash
conda activate nllSAR
./run_baseline_validation.sh
```

This tests Llama-3.2-1B (smallest, safest) with just 10 samples on both datasets.

**Watch for**:
- ✅ Model loads successfully
- ✅ Generates answers
- ✅ No CUDA errors
- ✅ Memory usage stays low

### 3. Run All Small Models (4-6 hours)

```bash
./run_small_models.sh
```

This runs all 3 small models on both TriviaQA and SQuAD (6 experiments total).

---

## Monitor While Running

### Watch GPU Memory

Open a second terminal and run:

```bash
watch -n 1 nvidia-smi
```

**What you'll see**:
- Llama-3.2-1B: ~2-3GB on GPU 0 ✅
- Qwen2.5-1.5B: ~3-4GB on GPU 0 ✅
- Mistral-7B-8bit: ~8-10GB on GPU 0 ⚠️ (should fit, but watch this one)

**Expected**: Only GPU 0 will be used (models fit in single GPU)

---

## Expected Timeline

| Model | Dataset | Samples | Est. Time |
|-------|---------|---------|-----------|
| Llama-3.2-1B | TriviaQA | 400 | ~20-30 min |
| Llama-3.2-1B | SQuAD | 400 | ~20-30 min |
| Qwen2.5-1.5B | TriviaQA | 400 | ~25-35 min |
| Qwen2.5-1.5B | SQuAD | 400 | ~25-35 min |
| Mistral-7B-8bit | TriviaQA | 400 | ~40-60 min |
| Mistral-7B-8bit | SQuAD | 400 | ~40-60 min |
| **TOTAL** | | | **~4-6 hours** |

---

## After Small Models Complete

### Analyze Results

```bash
# Compute AUROC for small models
conda run -n nllSAR python compute_multi_model_auroc.py \
    --wandb_dir src/nikos/uncertainty/wandb \
    --output_dir results/small_models_auroc

# Open analysis notebook
jupyter notebook src/analysis_notebooks/multi_model_family_comparison.ipynb
```

### Check Results Before Large Models

**Key questions to answer**:
1. Did all 3 small models complete successfully? ✅
2. Did Mistral-7B-8bit fit in 11GB? ✅/❌
3. What's the AUROC range across models?
4. Are the datasets working correctly?

**Decision point**:
- ✅ If Mistral-7B-8bit worked → Large 7B-8B models will likely work
- ❌ If Mistral-7B-8bit had OOM → Use 4-bit for large models

---

## If Mistral-7B-8bit Doesn't Fit

### Option 1: Switch to 4-bit

Edit `run_small_models.sh` and change:
```bash
"Mistral-7B-v0.3-8bit:Mistral:Small"
# to
"Mistral-7B-v0.3-4bit:Mistral:Small"
```

Then re-run just that model.

### Option 2: Let It Distribute

The model might auto-distribute across multiple GPUs. This is fine, just slightly slower.

---

## Troubleshooting

### Issue: "CUDA out of memory" on Mistral-7B

**Solution**: Switch to 4-bit quantization
```bash
# In run_small_models.sh, change:
"Mistral-7B-v0.3-8bit" → "Mistral-7B-v0.3-4bit"
```

### Issue: Model loads very slowly

**Normal**: First time downloads models (~2-14GB downloads)  
**Future runs**: Much faster (uses cached models)

### Issue: Qwen model fails with "trust_remote_code"

**Should be handled**: Code already has `trust_remote_code=True`  
**If still fails**: Check Python version (needs 3.8+)

---

## After Small Models: Next Steps

### If Everything Worked ✅

You have 3 options:

**Option A: Run Large Models (8-bit)**
```bash
./run_large_models.sh  # Try 8-bit first
```

**Option B: Run Large Models (4-bit, safer)**
Edit `run_large_models.sh` and change all large models to use `-4bit`:
```bash
"Llama-3-8B-4bit"
"Qwen2.5-7B-4bit"  
"Mistral-7B-Instruct-v0.3-4bit"
```

**Option C: Analyze Small Models First**
Take time to analyze small model results before committing to large models (8-12 more hours).

### If Mistral Had Issues ⚠️

**Definitely use 4-bit for large models**:
- Large models are same size as Mistral (7B-8B)
- 4-bit uses ~50% less memory (~4-5GB vs ~8-10GB)
- Fits very comfortably in 11GB
- Only ~1-2% accuracy loss

---

## Summary: Your Immediate Next Steps

```bash
# 1. Check GPUs (30 sec)
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# 2. Quick test (5-10 min)
./run_baseline_validation.sh

# 3. Watch GPU usage in 2nd terminal
watch -n 1 nvidia-smi

# 4. Run small models (4-6 hours)
./run_small_models.sh

# 5. Check WandB progress
# https://wandb.ai/nikosteam/super_guacamole
```

**Logs**: Check `experiment_logs/` if any issues

---

## Why Start with Small Models? ✅

1. **Low Risk**: Guaranteed to fit in 11GB
2. **Quick Feedback**: Get results in hours, not days
3. **Test Pipeline**: Verify everything works before large models
4. **Memory Test**: Mistral-7B-8bit tests upper limit of 11GB
5. **Early Results**: Get some data to analyze while deciding on large models

---

## Expected Success Rate

- Llama-3.2-1B: 100% ✅ (too small to fail)
- Qwen2.5-1.5B: 100% ✅ (too small to fail)
- Mistral-7B-8bit: 95% ✅ (should fit, might be tight)

**Overall**: You should get at least 4/6 experiments done (Llama and Qwen on both datasets).

Good luck! Start with validation, then let it run. Check back in 4-6 hours! 🚀
