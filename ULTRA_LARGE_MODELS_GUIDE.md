# Ultra-Large Models Guide (70B+ Weight Class)

## Overview

This guide covers running **ultra-large models** (70B+) on your **4×11GB GPUs** setup.

**Models**: 3 ultra-large models (70B-123B parameters)  
**Configuration**: 4-bit quantization + multi-GPU distribution + CPU offloading  
**Total**: 3 models × 2 datasets = 6 experiments  
**Estimated Time**: 12-18 hours

---

## Models Included

| Model | Parameters | Quantization | Est. Memory | Distribution |
|-------|------------|--------------|-------------|--------------|
| Llama-3.1-70B-Instruct-4bit | 70B | 4-bit | ~35GB | 4 GPUs |
| Qwen2.5-72B-4bit | 72B | 4-bit | ~36GB | 4 GPUs |
| Mistral-Large-2-4bit | 123B | 4-bit | ~62GB | 4 GPUs + CPU |

All models use **4-bit quantization** (NF4) to fit in your 44GB total VRAM. Mistral-Large-2 also uses CPU offloading.

---

## Memory Strategy

### Your Setup
- **4× GPUs**: 11GB each = **44GB total**
- **Distribution**: Models will automatically spread across GPUs via `accelerate`
- **CPU Offload**: Mistral-Large-2 uses CPU RAM for layers that don't fit in GPU
- **Overhead**: ~500MB per GPU for system/CUDA

### Expected Distribution
```
Llama-3.1-70B-4bit (~35GB):
  GPU 0: ~9GB (layers 0-17)
  GPU 1: ~9GB (layers 18-35)
  GPU 2: ~9GB (layers 36-53)
  GPU 3: ~8GB (layers 54-70)
  CPU:   minimal offload

Qwen2.5-72B-4bit (~36GB):
  GPU 0: ~9GB (layers 0-18)
  GPU 1: ~9GB (layers 19-36)
  GPU 2: ~9GB (layers 37-54)
  GPU 3: ~9GB (layers 55-72)
  CPU:   minimal offload

Mistral-Large-2-4bit (~62GB):
  GPU 0: ~10GB (layers 0-30)
  GPU 1: ~10GB (layers 31-60)
  GPU 2: ~10GB (layers 61-90)
  GPU 3: ~10GB (layers 91-110)
  CPU:   ~22GB (remaining layers)
```

---

## Quick Start

### 1. Pre-Flight Check

```bash
# Verify 4 GPUs detected
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
# Should show: GPUs: 4

# Check available memory
nvidia-smi

# Ensure you have at least 30GB free RAM for CPU offloading (Mistral-Large-2)
free -h
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

You should see memory usage across **all 4 GPUs**. For Mistral-Large-2, you'll also see CPU RAM usage.

---

## What to Expect

### Loading Time
- **First run**: 20-40 minutes per model (downloads 35-70GB)
- **Subsequent runs**: 5-10 minutes per model (cached)

### Generation Speed
- **Llama-3.1-70B**: ~6-10 tokens/sec (4 GPUs faster than 3)
- **Qwen2.5-72B**: ~5-9 tokens/sec
- **Mistral-Large-2**: ~3-5 tokens/sec (slower due to CPU offload + larger size)

### Per Experiment Time
- **400 samples** at ~50 tokens/answer
- **Llama-3.1-70B**: ~80-100 minutes per dataset
- **Qwen2.5-72B**: ~90-110 minutes per dataset
- **Mistral-Large-2**: ~150-200 minutes per dataset

**Total**: ~14-20 hours for all 6 experiments

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
| 7B-8B (fp16) | ~16GB | Medium | 1 | +10-15% |
| 70B-72B (4-bit) | ~35-36GB | Slow ⚠️ | 4 | +20-30% |
| 123B (4-bit) | ~62GB | Very Slow ⚠️⚠️ | 4+CPU | +25-35% |

*Relative to 1B baseline

**Trade-off**: 70B models are ~3-5× slower, 123B models are ~6-8× slower but more accurate.

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Cause**: Model doesn't fit even with 4-bit + 4 GPUs + CPU offload

**Solutions**:
1. **Check no other processes using GPU**:
   ```bash
   nvidia-smi
   # Kill other processes if needed
   ```

2. **Check available CPU RAM** (need 30GB+ for Mistral-Large-2):
   ```bash
   free -h
   # Close other applications if needed
   ```

3. **CPU offloading is already enabled** for Mistral-Large-2 (automatic)

4. **Reduce sample size** temporarily to test:
   ```bash
   NUM_SAMPLES=50  # Test run
   ```

### Issue: Model loads but generation is extremely slow

**Expected**: 70B models are 3-5× slower than 7B models due to:
- Multi-GPU communication overhead
- 10× more parameters to compute

**Verify it's working**:
```bash
watch -n 1 nvidia-smi
# All 4 GPUs should show high utilization (>70%)
```

If only 1-2 GPUs active → model might not be distributed properly.

### Issue: Model fails to load on Qwen2.5-72B

**Cause**: Qwen models need `trust_remote_code=True` (already in code)

**Verify**: Check you're using the updated `huggingface_models.py` with Qwen support.

### Issue: Mistral-Large-2 is extremely slow

**Normal**: 123B model with CPU offloading will be slower than GPU-only models. 
- Expect ~3-5 tokens/sec (vs 6-10 for pure GPU models)
- CPU↔GPU communication adds latency

---

## Performance Optimization

### If You Have Time Constraints

**Priority Order** (best value):
1. ✅ **Llama-3.1-70B-4bit**: Fastest 70B, well-tested, balanced
2. ✅ **Qwen2.5-72B-4bit**: Similar speed to Llama, excellent quality
3. ⚠️ **Mistral-Large-2-4bit**: Slowest (123B + CPU), but highest quality

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
