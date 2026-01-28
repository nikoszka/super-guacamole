# Multi-Model Family Experiments Guide

## Overview

This guide covers the complete implementation for comparing **3 model families** (Llama, Qwen, Mistral) with **2 models per family** on **2 datasets** (TriviaQA, SQuAD).

**Total Experiments**: 6 models × 2 datasets = 12 experiment runs  
**Estimated Time**: 12-24 hours (depending on GPU speed)  
**GPU Requirement**: Single A100 40GB (optimized with quantization)

---

## What Was Implemented

### 1. ✅ Code Changes

#### Added Qwen Model Support
**File**: `src/models/huggingface_models.py`

- Added Qwen model loading (lines 387-430)
- Support for 8-bit and 4-bit quantization
- Trust remote code handling (required for Qwen)
- Token limit configuration (4096 for consistency)
- Integration with existing prediction pipeline

**Test**: Run `python test_qwen_loading.py` to verify Qwen model loading works.

#### Dataset Support
- **TriviaQA**: Already supported ✅
- **SQuAD**: Already supported ✅ (validated with `test_squad_loading.py`)

### 2. ✅ Experiment Scripts

#### Master Script
**File**: `run_multi_model_experiments.sh`
- Runs all 12 experiments sequentially
- Handles memory cleanup between runs
- Error handling and progress tracking
- Configurable via environment variables

#### Quick Validation
**File**: `run_baseline_validation.sh`
- Quick test with 10 samples (instead of 400)
- Validates pipeline before full runs
- Tests Llama-3.2-1B on both datasets

#### Phase Scripts
- `run_small_models.sh`: Small models (1B-1.5B)
- `run_large_models.sh`: Large models (7B-8B, 8-bit)

### 3. ✅ Analysis Tools

#### AUROC Computation
**File**: `compute_multi_model_auroc.py`
- Finds all experiment results automatically
- Computes G-NLL AUROC for each experiment
- Optional RW-G-NLL computation
- Exports to CSV and JSON
- Summary statistics

#### Analysis Notebook
**File**: `src/analysis_notebooks/multi_model_family_comparison.ipynb`
- Comprehensive visualizations:
  - AUROC comparison matrix (heatmap)
  - Model family comparison
  - Size effect analysis
  - Dataset effect (TriviaQA vs SQuAD)
  - Accuracy vs uncertainty correlation
  - Radar charts for performance profiles
- Summary statistics and key findings
- Export capabilities

### 4. ✅ Test Scripts
- `test_qwen_loading.py`: Validates Qwen model implementation
- `test_squad_loading.py`: Validates SQuAD dataset loading

---

## Model Configuration

### Llama Family
| Model | Size | Quantization | VRAM | Status |
|-------|------|--------------|------|--------|
| Llama-3.2-1B | 1B | FP16 | ~2GB | ✅ Tested |
| Llama-3-8B-8bit | 8B | 8-bit | ~8GB | Ready |

### Qwen Family (NEW)
| Model | Size | Quantization | VRAM | Status |
|-------|------|--------------|------|--------|
| Qwen2.5-1.5B | 1.5B | FP16 | ~3GB | ✅ Tested |
| Qwen2.5-7B-8bit | 7B | 8-bit | ~8GB | Ready |

### Mistral Family
| Model | Size | Quantization | VRAM | Status |
|-------|------|--------------|------|--------|
| Mistral-7B-v0.3-8bit | 7B | 8-bit | ~8GB | Ready |
| Mistral-7B-Instruct-v0.3-8bit | 7B | 8-bit | ~8GB | Ready |

---

## Running Experiments

### Prerequisites

1. **Environment Setup**:
   ```bash
   conda activate nllSAR
   ```

2. **Set Environment Variables** (optional):
   ```bash
   export WANDB_ENTITY="nikosteam"
   export WANDB_PROJECT="super_guacamole"
   export BRIEF_PROMPT="short"  # or "detailed" for long answers
   ```

3. **GPU Access**: Ensure A100 40GB GPU is available
   ```bash
   nvidia-smi
   ```

### Quick Validation (Recommended First Step)

Before running full experiments, validate the pipeline:

```bash
./run_baseline_validation.sh
```

This runs 2 quick tests (10 samples each):
- Llama-3.2-1B on TriviaQA
- Llama-3.2-1B on SQuAD

**Time**: ~5-10 minutes  
**Purpose**: Verify everything works before committing to 12+ hours

### Full Experiment Run

#### Option 1: Run Everything (12-24 hours)
```bash
./run_multi_model_experiments.sh
```

#### Option 2: Run by Phase

**Phase 1: Small Models** (~4-6 hours)
```bash
./run_small_models.sh
```

**Phase 2: Large Models** (~8-12 hours)
```bash
./run_large_models.sh
```

#### Option 3: Individual Experiments

```bash
cd src
conda run -n nllSAR python generate_answers.py \
    --model_name "Qwen2.5-1.5B" \
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
    --experiment_lot "qwen_trivia_test"
```

### Monitoring Progress

1. **GPU Usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Logs**: Check `experiment_logs/` directory for detailed logs

3. **WandB**: Monitor runs at `https://wandb.ai/nikosteam/super_guacamole`

---

## Computing AUROC Metrics

After experiments complete:

### Basic AUROC (G-NLL only)
```bash
conda run -n nllSAR python compute_multi_model_auroc.py \
    --wandb_dir src/nikos/uncertainty/wandb \
    --output_dir results/multi_model_auroc
```

### With RW-G-NLL
```bash
conda run -n nllSAR python compute_multi_model_auroc.py \
    --wandb_dir src/nikos/uncertainty/wandb \
    --output_dir results/multi_model_auroc \
    --use_rw_gnll
```

**Outputs**:
- `results/multi_model_auroc/auroc_results.csv`
- `results/multi_model_auroc/auroc_results.json`
- Console: Summary statistics

---

## Analysis

### Jupyter Notebook Analysis

1. **Start Jupyter**:
   ```bash
   conda run -n nllSAR jupyter notebook
   ```

2. **Open**: `src/analysis_notebooks/multi_model_family_comparison.ipynb`

3. **Run all cells** to generate:
   - AUROC comparison heatmap
   - Model family comparisons
   - Size effect analysis
   - Dataset effect analysis
   - Accuracy vs uncertainty plots
   - Radar charts
   - Summary statistics

**Outputs**: All visualizations saved to `results/multi_model_auroc/`

---

## Expected Results

### Metrics to Compare

1. **G-NLL AUROC**: How well uncertainty predicts correctness (0.5-1.0, higher is better)
2. **Accuracy**: Percentage of correct answers (0.0-1.0, higher is better)
3. **RW-G-NLL AUROC**: Relevance-weighted uncertainty (optional)
4. **Generation Speed**: Tokens per second
5. **Memory Usage**: Peak VRAM during generation

### Research Questions

1. **Model size effect**: Do larger models (7B-8B) have better uncertainty calibration than small models (1B-1.5B)?
2. **Model family effect**: Which family (Llama, Qwen, Mistral) has best uncertainty estimation?
3. **Dataset effect**: Does uncertainty calibration differ between TriviaQA (fact recall) and SQuAD (reading comprehension)?
4. **Instruction tuning effect**: Does Mistral-Instruct improve over Mistral-Base?

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
**Symptoms**: `CUDA out of memory` error

**Solutions**:
- Ensure no other models are loaded
- Add more aggressive memory cleanup in scripts (increase `sleep` delays)
- Use 4-bit quantization for larger models:
  ```bash
  --model_name "Llama-3-8B-4bit"  # instead of -8bit
  ```

#### 2. Qwen Model Trust Remote Code Error
**Symptoms**: `Trust remote code is not enabled`

**Solution**: Already handled in code. If issues persist:
```python
# In huggingface_models.py, ensure trust_remote_code=True is set
self.tokenizer = AutoTokenizer.from_pretrained(
    model_id, trust_remote_code=True, ...
)
```

#### 3. Dataset Loading Slow
**Symptoms**: First run takes long to load dataset

**Solution**: Normal - HuggingFace caches datasets after first load. Subsequent runs are faster.

#### 4. WandB Authentication
**Symptoms**: `wandb: ERROR Authentication failed`

**Solution**:
```bash
wandb login
# Or set API key:
export WANDB_API_KEY="your_key_here"
```

---

## File Structure

```
taz/
├── src/
│   ├── models/
│   │   └── huggingface_models.py          # ✨ Updated with Qwen support
│   ├── data/
│   │   └── data_utils.py                  # SQuAD already supported
│   ├── analysis_notebooks/
│   │   └── multi_model_family_comparison.ipynb  # ✨ New analysis notebook
│   └── ...
├── results/
│   └── multi_model_auroc/                 # Generated AUROC results
├── experiment_logs/                       # Generated experiment logs
├── run_multi_model_experiments.sh         # ✨ Master experiment runner
├── run_baseline_validation.sh             # ✨ Quick validation
├── run_small_models.sh                    # ✨ Small models runner
├── run_large_models.sh                    # ✨ Large models runner
├── compute_multi_model_auroc.py           # ✨ AUROC computation
├── test_qwen_loading.py                   # ✨ Qwen model test
├── test_squad_loading.py                  # ✨ SQuAD dataset test
└── MULTI_MODEL_EXPERIMENTS_GUIDE.md       # ✨ This guide
```

---

## Next Steps

### Immediate
1. ✅ Run quick validation: `./run_baseline_validation.sh`
2. ✅ If validation passes, run full experiments: `./run_multi_model_experiments.sh`
3. ✅ Monitor progress via WandB and logs
4. ✅ Compute AUROC after completion
5. ✅ Analyze results in Jupyter notebook

### Extended (Optional)
1. Add 70B+ models with 4-bit quantization:
   - `Llama-3.1-70B-Instruct-4bit` (~35GB)
   - `Qwen2.5-72B-4bit` (~36GB)
2. Add more datasets:
   - Natural Questions (nq)
   - BioASQ (biomedical QA)
3. Compute RW-G-NLL for all experiments
4. Statistical significance testing between families
5. Qualitative error analysis

---

## Implementation Summary

### ✅ Completed Tasks

1. **Code Implementation**
   - Added Qwen model family support
   - Tested Qwen loading (verified working)
   - Validated SQuAD integration (verified working)

2. **Experiment Infrastructure**
   - Master experiment runner (12 experiments)
   - Phase-specific runners (small/large models)
   - Quick validation script
   - Comprehensive error handling

3. **Analysis Pipeline**
   - AUROC computation script with aggregation
   - Jupyter notebook with 8+ visualizations
   - Export capabilities (CSV, JSON, PNG)
   - Summary statistics

4. **Documentation**
   - This comprehensive guide
   - Inline code comments
   - Test scripts with validation
   - Troubleshooting section

### Ready to Run

All infrastructure is in place. The experiments are **ready to run** with a single command:

```bash
./run_multi_model_experiments.sh
```

Estimated completion: 12-24 hours on A100 40GB GPU.

---

## Contact & Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs in `experiment_logs/`
3. Check WandB runs for detailed metrics
4. Verify test scripts pass (`test_qwen_loading.py`, `test_squad_loading.py`)

---

**Last Updated**: 2026-01-27  
**Status**: ✅ Ready for Production Use  
**Hardware Requirements**: Single A100 40GB GPU  
**Software Requirements**: nllSAR conda environment
