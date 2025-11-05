# GPU Requirements Guide

This document outlines GPU memory requirements for running all experiments and models in this codebase.

## Model Memory Requirements

### Generation Models (Answer Generation)

| Model | Size | FP16 VRAM | 8-bit VRAM | 4-bit VRAM | Notes |
|-------|------|-----------|------------|------------|-------|
| Llama-3.2-1B | 1B | ~2 GB | ~1.5 GB | ~1 GB | Current default |
| Llama-3-8B | 8B | ~16 GB | ~8 GB | ~4 GB | Alternative generation model |
| Llama-2-7b-chat | 7B | ~14 GB | ~7 GB | ~3.5 GB | Older generation model |

### Judge Models (LLM-as-Judge Evaluation)

| Model | Size | FP16 VRAM | 8-bit VRAM | 4-bit VRAM | Notes |
|-------|------|-----------|------------|------------|-------|
| Llama-3-8B-Instruct | 8B | ~16 GB | ~8 GB | ~4 GB | Recommended for 8GB GPU |
| Llama-3.1-70B-Instruct | 70B | ~140 GB | ~70 GB | ~35 GB | Requires large GPU setup |
| Llama-2-70b-chat-hf | 70B | ~140 GB | ~70 GB | ~35 GB | Older 70B model |
| GPT-4 (via API) | N/A | 0 GB | 0 GB | 0 GB | No local GPU needed |

## Scenario-Based Requirements

### Scenario 1: Generation Only (No Judge Model)
**Use case:** Generate answers first, evaluate later

- **Minimum:** 4 GB GPU (Llama-3.2-1B)
- **Recommended:** 8 GB GPU (can use larger generation models)
- **Comfortable:** 16 GB GPU (flexibility for larger models)

### Scenario 2: Generation + Judge Simultaneously
**Use case:** Real-time evaluation during generation

#### With 8B Judge Model:
- **Generation (1B) + Judge (8B, 8-bit):** ~10 GB total
- **Generation (8B) + Judge (8B, 8-bit):** ~16 GB total
- **Recommended:** 24 GB GPU for comfortable operation

#### With 70B Judge Model:
- **Generation (1B) + Judge (70B, 8-bit):** ~72 GB total
- **Generation (1B) + Judge (70B, 4-bit):** ~37 GB total
- **Recommended:** 
  - 2× A100 40GB (80GB total) with 8-bit quantization
  - 1× A100 80GB with 4-bit quantization
  - 2× H100 80GB (160GB total) for full precision

### Scenario 3: 70B Judge Only (Recommended Workflow)
**Use case:** Generate answers first, then evaluate with judge

- **Judge (70B, 8-bit):** ~70 GB
- **Judge (70B, 4-bit):** ~35 GB
- **Judge (70B, FP16, multi-GPU):** ~140 GB (split across GPUs)

**Recommended Setup:**
- **Option A:** 2× NVIDIA A100 40GB (80GB total) - 8-bit quantization
- **Option B:** 1× NVIDIA A100 80GB - 8-bit or 4-bit quantization
- **Option C:** 2× NVIDIA H100 80GB (160GB total) - Full precision
- **Option D:** 4× NVIDIA A100 40GB (160GB total) - Full precision

## Recommended GPU Setups

### Budget-Friendly (8GB GPU)
✅ **Can run:**
- Generation: Llama-3.2-1B
- Judge: Llama-3-8B (8-bit quantization)
- **Workflow:** Generate first, then evaluate separately

❌ **Cannot run:**
- 70B models without significant compromises
- Large batch sizes

### Mid-Range (24GB GPU - e.g., RTX 4090, A4000)
✅ **Can run:**
- Generation: Up to Llama-3-8B
- Judge: Llama-3-8B or Llama-3-70B (4-bit quantization)
- Both simultaneously with 8B judge

❌ **Limited:**
- 70B models with 8-bit quantization (tight fit)
- Full precision 70B models

### High-End (40GB GPU - e.g., A100 40GB)
✅ **Can run:**
- Generation: Any model up to 8B
- Judge: Llama-3-70B (8-bit quantization) comfortably
- Both simultaneously with smaller models

**Recommended:** 2× A100 40GB for 70B models

### Enterprise (80GB+ GPU - e.g., A100 80GB, H100 80GB)
✅ **Can run:**
- Everything comfortably
- 70B models with full precision (FP16)
- Multiple models simultaneously
- Large batch sizes

**Recommended:** 
- 1× A100 80GB for 70B with quantization
- 2× A100 80GB or 2× H100 80GB for 70B full precision

## Memory Optimization Strategies

### 1. Sequential Loading (Recommended)
- Generate answers first (load generation model)
- Unload generation model
- Load judge model for evaluation
- **Memory saving:** Only one model in memory at a time

### 2. Quantization
- **8-bit quantization:** ~50% memory reduction
- **4-bit quantization:** ~75% memory reduction
- **Trade-off:** Slight accuracy loss (usually <1%)

### 3. Model Offloading
- Use CPU offloading for less critical models
- Use disk offloading (slower but works)

### 4. Batch Processing
- Process evaluations in smaller batches
- Clear cache between batches

## Current Code Configuration

### Generation Model (run_greedy_decoding.py)
- **Default:** Llama-3.2-1B (~2 GB)
- **Memory efficient:** Already optimized

### Judge Models (recompute_accuracy_with_judge.py)
- **8B models:** Automatic quantization if needed
- **70B models:** Automatic 8-bit quantization enabled
- **Multi-GPU:** Supported via accelerate library

## Practical Recommendations

### For Comfortable Operation (All Models + 70B Judge):

**Minimum Setup:**
- **2× NVIDIA A100 40GB** (80GB total)
- Run 70B judge with 8-bit quantization
- Can run generation and judge sequentially

**Recommended Setup:**
- **2× NVIDIA A100 80GB** (160GB total)
- Run 70B judge with full precision (FP16)
- Can run generation and judge simultaneously
- Supports large batch sizes

**Ideal Setup:**
- **4× NVIDIA H100 80GB** (320GB total)
- Maximum flexibility
- Can run multiple experiments in parallel
- Future-proof for larger models

### For Current Workflow (Generate First, Judge Later):

**Minimum:** 1× NVIDIA A100 40GB (40GB)
- Generation: Llama-3.2-1B (~2 GB)
- Judge: Llama-3-70B with 8-bit quantization (~70 GB)

**Recommended:** 1× NVIDIA A100 80GB (80GB)
- More comfortable headroom
- Can use 4-bit quantization for even more efficiency
- Supports larger generation models

## Cost Estimates (Approximate)

### Cloud GPU Rental (per hour):
- **A100 40GB:** ~$1.50-2.00/hour
- **A100 80GB:** ~$2.50-3.50/hour
- **H100 80GB:** ~$8.00-12.00/hour

### For 1000 examples:
- **Generation:** ~10-30 minutes (depending on model)
- **Judge evaluation:** ~30-60 minutes (depending on judge model)
- **Total cost:** $1-5 (depending on GPU type)

## Summary

**For comfortable operation with 70B models:**
- **Minimum:** 2× A100 40GB (80GB total) - with quantization
- **Recommended:** 2× A100 80GB (160GB total) - full precision
- **Ideal:** 4× H100 80GB (320GB total) - maximum flexibility

**For current workflow (sequential):**
- **Minimum:** 1× A100 40GB (40GB) - with 8-bit quantization
- **Recommended:** 1× A100 80GB (80GB) - comfortable operation

The current codebase is optimized to work with sequential loading (generate first, judge later), which minimizes memory requirements.


