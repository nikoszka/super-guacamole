# Setup Guide for 3×11GB GPUs

## Hardware Configuration
- **3× GPUs with 11GB VRAM each** (e.g., RTX 2080 Ti, RTX 3060)
- **Total VRAM**: 33GB (vs 40GB A100)
- **Status**: ✅ Should work with minor adjustments

---

## Memory Planning

### Per-Model Memory Requirements

| Model | Size | Quantization | Expected VRAM | Fits in 11GB? |
|-------|------|--------------|---------------|---------------|
| **Small Models** |
| Llama-3.2-1B | 1B | FP16 | ~2GB | ✅ Easy |
| Qwen2.5-1.5B | 1.5B | FP16 | ~3GB | ✅ Easy |
| Mistral-7B-v0.3-8bit | 7B | 8-bit | ~8-10GB | ⚠️ Tight |
| **Large Models** |
| Llama-3-8B-8bit | 8B | 8-bit | ~8-10GB | ⚠️ Tight |
| Qwen2.5-7B-8bit | 7B | 8-bit | ~8-10GB | ⚠️ Tight |
| Mistral-7B-Instruct-v0.3-8bit | 7B | 8-bit | ~8-10GB | ⚠️ Tight |

**Key Insight**: 8-bit quantized 7B-8B models should fit in 11GB, but it's tight. The code already has multi-GPU support via accelerate.

---

## Automatic Multi-GPU Support

The existing code in `src/models/huggingface_models.py` already handles multi-GPU:

```python
# Automatic GPU detection and memory allocation
max_memory_dict = get_gpu_memory_dict()  # Detects all GPUs
self.model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',  # Automatic distribution across GPUs
    max_memory=max_memory_dict,  # Uses all available GPUs
    ...
)
```

**What this means**:
- ✅ Small models: Will load on GPU 0 (plenty of room)
- ✅ Large models: Will automatically distribute if needed
- ✅ Accelerate handles layer distribution transparently

---

## Recommended Configuration Changes

### Option 1: Use 4-bit Quantization (Safer)

For extra memory safety on 7B-8B models, use 4-bit instead of 8-bit:

**Edit the model names in scripts**:

```bash
# Original (8-bit)
"Llama-3-8B-8bit"
"Qwen2.5-7B-8bit"
"Mistral-7B-v0.3-8bit"

# Safer (4-bit) - uses ~4-5GB instead of ~8-10GB
"Llama-3-8B-4bit"
"Qwen2.5-7B-4bit"
"Mistral-7B-v0.3-4bit"
```

**Trade-off**: 4-bit uses ~50% less memory with only ~1-2% accuracy loss.

### Option 2: Keep 8-bit (Should Work)

The 8-bit models should fit in 11GB, but monitor GPU usage:

```bash
# Monitor GPU memory during experiments
watch -n 1 nvidia-smi
```

If you see OOM errors, switch to 4-bit for specific models.

---

## Verification Steps

### 1. Check GPU Detection

```python
import torch
print(f"GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")
    print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
```

Expected output:
```
GPUs available: 3
GPU 0: <Your GPU Name>
  Total memory: 11.0 GB
GPU 1: <Your GPU Name>
  Total memory: 11.0 GB
GPU 2: <Your GPU Name>
  Total memory: 11.0 GB
```

### 2. Test with Smallest Model First

```bash
# Test Llama-3.2-1B (smallest, safest)
cd src
conda run -n nllSAR python generate_answers.py \
    --model_name "Llama-3.2-1B" \
    --dataset "trivia_qa" \
    --num_samples 10 \
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
    --experiment_lot "3gpu_test"
```

Watch `nvidia-smi` to see which GPU it uses and how much memory.

### 3. Test with Large Model (8-bit)

```bash
# Test Llama-3-8B-8bit (largest, tightest fit)
# Same command but change model_name to "Llama-3-8B-8bit"
```

**If you see OOM error**: Switch to 4-bit for that model.

---

## Modified Experiment Scripts for 3×11GB

### Safe Configuration (4-bit for large models)

Create a modified version: `run_multi_model_experiments_3gpu.sh`

```bash
#!/bin/bash

set -e

ENTITY="${WANDB_ENTITY:-nikosteam}"
PROJECT="${WANDB_PROJECT:-super_guacamole}"
NUM_SAMPLES=400
NUM_FEW_SHOT=5
TEMPERATURE=0.0
BRIEF_PROMPT="${BRIEF_PROMPT:-short}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

DATASETS=("trivia_qa" "squad")

# Modified for 3×11GB GPUs - use 4-bit for large models
MODELS=(
    "Llama-3.2-1B:Llama:Small"
    "Llama-3-8B-4bit:Llama:Large"           # Changed to 4-bit
    "Qwen2.5-1.5B:Qwen:Small"
    "Qwen2.5-7B-4bit:Qwen:Large"            # Changed to 4-bit
    "Mistral-7B-v0.3-4bit:Mistral:Small"    # Changed to 4-bit
    "Mistral-7B-Instruct-v0.3-4bit:Mistral:Large"  # Changed to 4-bit
)

echo "================================================================================"
echo "Multi-Model Family Experiments (3×11GB GPU Configuration)"
echo "================================================================================"
echo "Using 4-bit quantization for 7B-8B models for memory safety"
echo "================================================================================"

# Rest of the script same as run_multi_model_experiments.sh...
for model_info in "${MODELS[@]}"; do
    MODEL=$(echo "$model_info" | cut -d: -f1)
    FAMILY=$(echo "$model_info" | cut -d: -f2)
    SIZE=$(echo "$model_info" | cut -d: -f3)
    
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "Running: $MODEL on $DATASET"
        
        cd src
        python generate_answers.py \
            --model_name "$MODEL" \
            --dataset "$DATASET" \
            --num_samples "$NUM_SAMPLES" \
            --num_few_shot "$NUM_FEW_SHOT" \
            --temperature "$TEMPERATURE" \
            --num_generations 1 \
            --brief_prompt "$BRIEF_PROMPT" \
            --enable_brief \
            --brief_always \
            --no-compute_uncertainties \
            --no-compute_p_true \
            --no-get_training_set_generations \
            --use_context \
            --entity "$ENTITY" \
            --project "$PROJECT" \
            --experiment_lot "3gpu_${FAMILY}_${SIZE}_${DATASET}_${TIMESTAMP}"
        cd ..
        
        echo "✅ Completed: $MODEL on $DATASET"
        sleep 15
    done
done

echo ""
echo "================================================================================"
echo "✅ All experiments completed on 3×11GB GPU setup!"
echo "================================================================================"
```

---

## Monitoring During Experiments

### Watch GPU Usage

```bash
watch -n 1 nvidia-smi
```

Look for:
- **Memory usage**: Should stay under 10GB per GPU for 8-bit, under 6GB for 4-bit
- **GPU utilization**: Should be high (>90%) during generation
- **Multi-GPU**: If a model is distributed, you'll see memory on multiple GPUs

### If You See OOM Errors

1. **Immediate fix**: Switch that specific model to 4-bit
2. **Verify**: Check no other processes are using GPU memory
3. **Clean cache**: Run `torch.cuda.empty_cache()` between runs (already in scripts)

---

## Performance Expectations

### 3×11GB vs 1×40GB

| Aspect | 3×11GB | 1×40GB |
|--------|--------|--------|
| **Small models** (1-1.5B) | Same speed ✅ | Same speed ✅ |
| **Large models** (7-8B, single GPU) | Same speed ✅ | Same speed ✅ |
| **Large models** (distributed) | Slightly slower ⚠️ | Not needed |
| **Memory safety** | Need 4-bit for comfort | 8-bit comfortable |

**Verdict**: 3×11GB should work fine, especially with 4-bit quantization for large models.

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1**: Use 4-bit quantization
```bash
--model_name "Llama-3-8B-4bit"  # instead of -8bit
```

**Solution 2**: Force multi-GPU distribution
```python
# In huggingface_models.py, for specific model
max_memory = {0: '10GiB', 1: '10GiB', 2: '10GiB'}
```

**Solution 3**: Close other GPU processes
```bash
# Check what's using GPU
nvidia-smi
# Kill other processes if needed
```

### Issue: Model loads on only 1 GPU

**This is normal** if the model fits in 11GB! Accelerate only distributes if necessary.

**Verification**: If it fits in 11GB and runs without errors, you're good ✅

### Issue: Multi-GPU slower than expected

**Expected**: Small communication overhead between GPUs for distributed models (~5-10% slower)

**If much slower**: Check if model is unnecessarily distributed (should fit in 1 GPU)

---

## Quick Start for 3×11GB Setup

```bash
# 1. Verify GPU detection
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# 2. Test smallest model (2GB, safe)
./run_baseline_validation.sh

# 3. Test large model (try 8-bit first)
cd src
python generate_answers.py --model_name "Llama-3-8B-8bit" --dataset "trivia_qa" --num_samples 10 ...

# 4. If OOM, switch to 4-bit and run full experiments
# Edit run_multi_model_experiments.sh to use 4-bit for large models
./run_multi_model_experiments.sh
```

---

## Recommendations Summary

### Conservative (Safest):
✅ Use 4-bit quantization for all 7B-8B models  
✅ Memory usage: ~4-6GB per model  
✅ Fits comfortably in 11GB with room to spare  
✅ Accuracy loss: ~1-2%  

### Balanced (Recommended):
✅ Try 8-bit first (scripts already configured)  
✅ Monitor GPU memory with `nvidia-smi`  
✅ Switch to 4-bit only if you hit OOM  
✅ Should work for most models  

### Aggressive (If you're confident):
⚠️ Keep all 8-bit as-is  
⚠️ Watch memory closely  
⚠️ Be ready to restart with 4-bit if needed  

---

## Conclusion

Your 3×11GB setup should work well! The code already has multi-GPU support built in. 

**Key actions**:
1. ✅ Start with validation: `./run_baseline_validation.sh`
2. ✅ Watch GPU memory during first few experiments
3. ✅ Switch to 4-bit if you hit any OOM errors

The total 33GB is sufficient, and individual models should fit in 11GB with 8-bit (or definitely with 4-bit).

Good luck with your experiments! 🚀
