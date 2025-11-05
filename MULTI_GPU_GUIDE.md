# Multi-GPU Support Guide

This guide explains how to use multiple GPUs with the nllSAR codebase.

## Overview

The codebase **fully supports multi-GPU setups** using HuggingFace's `accelerate` library and `device_map="auto"`. The code automatically detects and utilizes all available GPUs.

## Current Multi-GPU Support

### Automatic Detection
- The code automatically detects the number of available GPUs using `torch.cuda.device_count()`
- For **70B models** (full precision, without quantization), the code distributes the model across all available GPUs
- For **smaller models** (1B, 7B, 8B, 13B), `device_map="auto"` handles distribution automatically
- For **quantized models** (8-bit), `device_map="auto"` also distributes across GPUs

### How It Works

1. **Smaller Models (< 70B) with `device_map="auto"`:**
   - Automatically splits model layers across available GPUs
   - Each GPU gets a portion of the model
   - Inference happens across GPUs automatically

2. **70B Models (Full Precision) - Explicit Multi-GPU:**
   - Uses `accelerate.infer_auto_device_map()` to create an optimal distribution
   - Splits model layers evenly across all GPUs
   - Automatically detects number of GPUs and configures accordingly

3. **Quantized Models:**
   - Uses `device_map="auto"` which handles multi-GPU automatically
   - Can specify per-GPU memory limits

## Using 4 GPUs

### For Your 4-GPU Cluster Setup:

**1. Automatic Detection:**
The code now automatically detects all 4 GPUs and distributes models accordingly:

```python
# When loading a 70B model (full precision), it will:
# - Detect 4 GPUs
# - Split the model across all 4 GPUs
# - Each GPU gets approximately 1/4 of the model
```

**2. Verify GPU Detection:**
```bash
python -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}')"
```

**3. Check GPU Usage During Execution:**
```bash
# In another terminal, monitor GPU usage:
watch -n 1 nvidia-smi
```

### Example: Running with 4 GPUs

**For Generation (Llama-3.2-1B):**
```bash
# Small model, but will use all GPUs if beneficial
python run_greedy_decoding.py
# device_map="auto" will decide the best distribution
```

**For 70B Judge (Full Precision):**
```bash
# Without quantization, will split across all 4 GPUs
python recompute_accuracy_with_judge.py <run_id> llm_llama-3.1-70b
# The model will be automatically distributed across 4 GPUs
```

**For 70B Judge (8-bit Quantization):**
```bash
# With quantization, still uses all GPUs but with less memory per GPU
python recompute_accuracy_with_judge.py <run_id> llm_llama-3.1-70b-8bit
# device_map="auto" distributes across all 4 GPUs
```

## Memory Distribution with 4 GPUs

### 70B Model (Full Precision, FP16):
- **Total Memory:** ~140 GB
- **Per GPU (4 GPUs):** ~35 GB per GPU
- **Recommended:** 4× A100 40GB or 4× A100 80GB

### 70B Model (8-bit Quantization):
- **Total Memory:** ~70 GB
- **Per GPU (4 GPUs):** ~17.5 GB per GPU
- **Recommended:** 4× A100 40GB (comfortable)

### 70B Model (4-bit Quantization):
- **Total Memory:** ~35 GB
- **Per GPU (4 GPUs):** ~9 GB per GPU
- **Can fit on:** 4× RTX 4090 (24GB each) or 4× A100 40GB

## Benefits of 4 GPUs

1. **Larger Models:** Can run 70B models in full precision without quantization
2. **Faster Inference:** Parallel processing across 4 GPUs
3. **Better Memory Utilization:** Distributes memory load evenly
4. **Flexibility:** Can run multiple experiments simultaneously

## Configuration

### Default Behavior
- **Automatically uses all available GPUs**
- No configuration needed - just ensure GPUs are visible to PyTorch

### Environment Variables (Optional)
```bash
# Restrict to specific GPUs (if needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1       # Use only first 2 GPUs
```

### SLURM Integration
If using SLURM on your cluster:
```bash
# Request 4 GPUs
sbatch --gres=gpu:a100:4 run_greedy_decoding.sh

# Or in your SLURM script:
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
```

## Performance Expectations

### Speed Improvements with 4 GPUs:
- **70B Model:** ~2-3x faster inference compared to single GPU
- **Smaller Models:** May not benefit much (overhead of communication)
- **Batch Processing:** Can handle larger batches with 4 GPUs

### Actual Performance:
- Depends on GPU interconnect (NVLink, PCIe)
- Models with NVLink will see better performance
- PCIe-only setups may have communication bottlenecks

## Troubleshooting

### Issue: Only 1 GPU Detected
```bash
# Check GPU visibility
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"

# If only 1 GPU visible:
# 1. Check CUDA_VISIBLE_DEVICES environment variable
# 2. Check SLURM GPU allocation
# 3. Verify all GPUs are accessible
```

### Issue: Out of Memory on One GPU
- The model distribution should be automatic, but if one GPU gets more memory:
  - Check actual GPU memory: `nvidia-smi`
  - The code distributes evenly, but actual usage may vary
  - Consider using quantization: add `-8bit` suffix to model name

### Issue: Slow Performance
- Multi-GPU has overhead for communication
- For small models (< 8B), single GPU may be faster
- Check GPU utilization: `nvidia-smi` should show all GPUs active
- Verify NVLink is enabled (if available): `nvidia-smi topo -m`

## Best Practices for 4 GPUs

1. **Use Full Precision for 70B Models:**
   - With 4× A100 40GB, you can run 70B models in FP16 without quantization
   - Better accuracy than quantized models

2. **Sequential Workflow:**
   - Generate with small model (single GPU is fine)
   - Then evaluate with 70B model (uses all 4 GPUs)

3. **Monitor GPU Usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```
   - Should see all 4 GPUs with memory usage
   - All GPUs should show compute activity

4. **Batch Processing:**
   - With 4 GPUs, you can process larger batches
   - Increases throughput

## Example: Full Workflow with 4 GPUs

```bash
# Step 1: Generate answers (small model, can use 1 GPU)
python run_greedy_decoding.py
# This uses 1 GPU efficiently

# Step 2: Evaluate with 70B judge (uses all 4 GPUs automatically)
python recompute_accuracy_with_judge.py <run_id> llm_llama-3.1-70b
# Automatically distributes across 4 GPUs

# Monitor usage:
watch -n 1 nvidia-smi
# Should see all 4 GPUs active during evaluation
```

## Code Changes Made

The code has been updated to:
1. **Automatically detect number of GPUs** using `torch.cuda.device_count()`
2. **Configure max_memory for all GPUs** dynamically
3. **Log GPU distribution** for debugging
4. **Support any number of GPUs** (2, 4, 8, etc.)

## Summary

✅ **Multi-GPU is fully supported and automatic**
✅ **Works with 4 GPUs out of the box**
✅ **No configuration needed - just run your scripts**
✅ **Best for 70B models in full precision**
✅ **Automatic load balancing across GPUs**

Your 4-GPU cluster will automatically be utilized when running large models (especially 70B models). The code detects all available GPUs and distributes the model accordingly.

