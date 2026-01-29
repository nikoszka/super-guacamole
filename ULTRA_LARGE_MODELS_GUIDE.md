# Ultra-Large Models Guide (70B+ Weight Class)

## Overview

This guide covers running **ultra-large models** (70B+) on your **3×11GB GPUs** setup.

**Models**: 3 ultra-large models in the same weight class as Llama-3.1-70B  
**Configuration**: 4-bit quantization + multi-GPU distribution  
**Total**: 3 models × 2 datasets = 6 experiments  
**Estimated Time**: 12-18 hours

---

## Models Included

| Model | Parameters | Quantization | Est. Memory | Distribution |
|-------|------------|--------------|-------------|--------------|
| Llama-3.1-70B-Instruct-4bit | 70B | 4-bit | ~35GB | 3 GPUs |
| Qwen2.5-72B-4bit | 72B | 4-bit | ~36GB | 3 GPUs |
| Mixtral-8x7B-Instruct-v0.1-4bit | 8×7B (56B) | 4-bit | ~24GB | 2-3 GPUs |

All models use **4-bit quantization** (NF4) to fit in your 33GB total VRAM.

---

## Memory Strategy

### Your Setup
- **3× GPUs**: 11GB each = **33GB total**
- **Distribution**: Models will automatically spread across GPUs via `accelerate`
- **Overhead**: ~2-3GB per GPU for system/CUDA

### Expected Distribution
```
Llama-3.1-70B-4bit (~35GB):
  GPU 0: ~11GB (layers 0-20)
  GPU 1: ~11GB (layers 21-40)
  GPU 2: ~11GB (layers 41-60)
  CPU:   ~2GB  (some layers offloaded)

Qwen2.5-72B-4bit (~36GB):
  GPU 0: ~11GB (layers 0-22)
  GPU 1: ~11GB (layers 23-44)
  GPU 2: ~11GB (layers 45-66)
  CPU:   ~3GB  (some layers offloaded)

Mixtral-8x7B-4bit (~24GB):
  GPU 0: ~11GB (experts 0-2)
  GPU 1: ~11GB (experts 3-5)
  GPU 2: ~2GB  (experts 6-7)
```

---

## Quick Start

### 1. Pre-Flight Check

```bash
# Verify 3 GPUs detected
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
# Should show: GPUs: 3

# Check available memory
nvidia-smi
```

### 2. Run Ultra-Large Models

```bash
./run_ultra_large_models.sh
```

**Expected time**: 12-18 hours for all 6 experiments (3 models × 2 datasets)

### 3. Monitor Progress

Open a second terminal:
```bash
watch -n 1 nvidia-smi
```

You should see memory usage across **all 3 GPUs**.

---

## What to Expect

### Loading Time
- **First run**: 20-40 minutes per model (downloads 35-70GB)
- **Subsequent runs**: 5-10 minutes per model (cached)

### Generation Speed
- **Llama-3.1-70B**: ~5-8 tokens/sec (slower due to multi-GPU)
- **Qwen2.5-72B**: ~4-7 tokens/sec
- **Mixtral-8x7B**: ~8-12 tokens/sec (more efficient architecture)

### Per Experiment Time
- **400 samples** at ~50 tokens/answer
- **Llama-3.1-70B**: ~90-120 minutes per dataset
- **Qwen2.5-72B**: ~100-130 minutes per dataset
- **Mixtral-8x7B**: ~60-80 minutes per dataset

**Total**: ~12-18 hours for all 6 experiments

---

## Advanced Configuration

### Option 1: Run Individual Models

```bash
cd src
python generate_answers.py \
    --model_name "Llama-3.1-70B-Instruct-4bit" \
    --dataset "trivia_qa" \
    --num_samples 400 \
    --num_few_shot 5 \
    --temperature 0.0 \
    --num_generations 1 \
    --brief_prompt "short" \
    --enable_brief \
    --brief_always \
    --no-compute_uncertainties \
    --no-compute_p_true \
    --no-get_training_set_generations \
    --use_context \
    --entity "nikosteam" \
    --project "super_guacamole" \
    --experiment_lot "ultra_large_test"
```

### Option 2: Test with Fewer Samples First

Edit `run_ultra_large_models.sh` and change:
```bash
NUM_SAMPLES=50  # Instead of 400
```

This lets you verify everything works before committing to full runs.

---

## Comparison with Smaller Models

| Model Size | Memory (4-bit) | Speed | GPUs Used | Accuracy* |
|------------|----------------|-------|-----------|-----------|
| 1B-1.5B | ~2-3GB | Fast ✅ | 1 | Baseline |
| 7B-8B (8-bit) | ~8-10GB | Medium | 1 | +10-15% |
| 7B-8B (4-bit) | ~4-5GB | Medium | 1 | +8-12% |
| 70B-72B (4-bit) | ~35-36GB | Slow ⚠️ | 3 | +20-30% |

*Relative to 1B baseline

**Trade-off**: 70B models are ~3-5× slower but significantly more accurate.

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Cause**: Model doesn't fit even with 4-bit + 3 GPUs

**Solutions**:
1. **Check no other processes using GPU**:
   ```bash
   nvidia-smi
   # Kill other processes if needed
   ```

2. **Try with CPU offloading** (edit huggingface_models.py):
   ```python
   max_memory = {0: '10GiB', 1: '10GiB', 2: '10GiB', 'cpu': '20GiB'}
   ```

3. **Reduce sample size** temporarily:
   ```bash
   NUM_SAMPLES=50  # Test run
   ```

### Issue: Model loads but generation is extremely slow

**Expected**: 70B models are 3-5× slower than 7B models due to:
- Multi-GPU communication overhead
- 4× more parameters to compute

**Verify it's working**:
```bash
watch -n 1 nvidia-smi
# All 3 GPUs should show high utilization (>80%)
```

If only 1-2 GPUs active → model might not be distributed properly.

### Issue: Model fails to load on Qwen2.5-72B

**Cause**: Qwen models need `trust_remote_code=True` (already in code)

**Verify**: Check you're using the updated `huggingface_models.py` with Qwen support.

### Issue: Mixtral-8x7B uses only 2 GPUs

**Normal**: Mixtral is more memory-efficient (~24GB), may not need all 3 GPUs.

---

## Performance Optimization

### If You Have Time Constraints

**Priority Order** (best value):
1. ✅ **Mixtral-8x7B-4bit**: Fastest, good quality, uses 2 GPUs
2. ✅ **Llama-3.1-70B-4bit**: Balanced, most well-tested
3. ⚠️ **Qwen2.5-72B-4bit**: Slowest, but potentially best quality

### If Memory is Tight

Run models sequentially with manual cleanup:
```bash
# Run one model
./run_ultra_large_models.sh  # Edit to include only 1 model

# After completion, clear cache
python -c "import torch; torch.cuda.empty_cache()"

# Wait 30 seconds, then run next model
```

---

## After Experiments Complete

### Compute AUROC

```bash
python compute_multi_model_auroc.py \
    --wandb_dir src/nikos/uncertainty/wandb \
    --output_dir results/ultra_large_auroc \
    --experiment_pattern "ultra_large"
```

### Compare All Sizes

You'll now have results from:
- Small (1-1.5B)
- Large (7-8B)
- Ultra-Large (70B+)

This lets you analyze the **size-performance trade-off**!

---

## Expected Results

### Accuracy Gains (vs 1B baseline)
- **TriviaQA**: +20-30% (70B models excel at factual recall)
- **SQuAD**: +15-25% (reading comprehension benefits from scale)

### Uncertainty Calibration (G-NLL AUROC)
- **70B models**: Often have better calibration
- **4-bit quantization**: Minimal impact on AUROC (~1-2% difference)

### Key Research Questions
1. **Does size improve uncertainty?** 70B vs 7B calibration
2. **4-bit impact**: How much does quantization hurt?
3. **Multi-GPU overhead**: Speed penalty for distribution
4. **Mixtral architecture**: MoE efficiency vs dense models

---

## Cost Considerations

### Compared to Single Model

| Experiment | GPU Hours | Relative Cost |
|------------|-----------|---------------|
| 1× Small model | 0.5h | 1× |
| 1× Large model | 1.5h | 3× |
| 1× Ultra-large | 6h | 12× |

**Full ultra-large experiments**: ~18 GPU-hours per dataset

### Cloud Cost Estimates (if applicable)
- **3× RTX 3090 rental**: ~$1.50/hour
- **Full 6 experiments**: ~$27 total
- **Per model**: ~$9

---

## When to Use Ultra-Large Models

### ✅ Good Use Cases
- Final benchmark comparison
- Maximum accuracy needed
- Studying scaling laws
- Publication-quality results

### ⚠️ Maybe Not Worth It
- Early exploration (use 7B first)
- Quick iteration (too slow)
- Memory-constrained setup
- Tight deadlines

**Recommendation**: Run small and large models first, analyze results, then decide if ultra-large models are needed.

---

## Summary

**What you get**:
- 3 ultra-large models (70B+ weight class)
- 4-bit quantization for memory efficiency
- Automatic multi-GPU distribution
- 2 datasets (TriviaQA, SQuAD)
- 6 total experiments

**What to expect**:
- 12-18 hours total runtime
- 35-40GB model downloads (first run)
- Multi-GPU usage across all 3 GPUs
- Significant accuracy improvement over smaller models

**Start with**:
```bash
./run_ultra_large_models.sh
```

Monitor with `nvidia-smi` and check back in ~12 hours! 🚀

---

**Pro Tip**: While ultra-large models run (12-18h), you can analyze your small and large model results. By the time ultra-large completes, you'll have comprehensive data across all model sizes!
