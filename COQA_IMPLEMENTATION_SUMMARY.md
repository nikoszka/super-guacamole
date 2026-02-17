# CoQA Implementation Summary

## Overview

Complete implementation for running all models (small → large → ultra-large) on the **CoQA dataset** (Conversational Question Answering Challenge).

---

## Changes Made

### 1. **Added CoQA Dataset Support** ✅

**File**: `src/data/data_utils.py`

Added complete CoQA dataset loading with conversational structure flattening:

```python
elif dataset_name == "coqa":
    # Load CoQA from HuggingFace (stanfordnlp/coqa)
    # Flatten conversational structure (story + Q&A pairs)
    # Format: Each Q&A pair becomes individual sample with shared context
```

**Key Features**:
- Loads from `stanfordnlp/coqa` on HuggingFace
- Flattens conversations into individual Q&A pairs
- Preserves story context for each question
- Maintains conversational IDs (e.g., `story_id_q0`, `story_id_q1`)

### 2. **Added CoQA Dataset Handling** ✅

**File**: `src/generate_answers.py`

Added automatic configuration for CoQA:

```python
elif args.dataset == 'coqa':
    # Force use_context=True (conversations require context)
```

CoQA questions are conversational and require the story context to be meaningful.

### 3. **Created Comprehensive Run Script** ✅

**File**: `run_CoQA_all.sh`

Complete script that runs all 9 models sequentially:

**Phase 1: Small Models (3 models)**
- Llama-3.2-1B
- Qwen2.5-1.5B
- Mistral-7B-v0.3-8bit

**Phase 2: Large Models (3 models)**
- Llama-3.1-8B
- Qwen3-8B
- Mistral-7B-Instruct-v0.3

**Phase 3: Ultra-Large Models (3 models)**
- Llama-3.1-70B-Instruct-4bit
- Qwen2.5-72B-4bit
- Mistral-Large-Instruct-2407-4bit

**Features**:
- Sequential execution with memory cleanup between models
- Progress indicators and timestamps
- Automatic GPU scaling (1 GPU → 4 GPUs as needed)
- CPU offloading for 123B model
- Estimated ~24-30 hours for all experiments

### 4. **Created Comprehensive Guide** ✅

**File**: `COQA_EXPERIMENTS_GUIDE.md`

Complete 400+ line guide covering:
- ✅ What is CoQA and why it's different
- ✅ All 9 models with specifications
- ✅ Quick start instructions
- ✅ Performance expectations
- ✅ Timeline and cost estimates
- ✅ Troubleshooting guide
- ✅ Analysis recommendations
- ✅ Comparison with TriviaQA/SQuAD

### 5. **Created Test Script** ✅

**File**: `test_coqa_loading.py`

Verification script that:
- ✅ Tests CoQA dataset loading
- ✅ Validates format and structure
- ✅ Shows sample conversations
- ✅ Confirms dataset is ready for experiments

---

## File Structure

```
super-guacamole/
├── run_CoQA_all.sh                    ← Main script (executable)
├── test_coqa_loading.py               ← Test dataset loading
├── COQA_EXPERIMENTS_GUIDE.md          ← Complete guide
├── COQA_IMPLEMENTATION_SUMMARY.md     ← This file
└── src/
    ├── data/
    │   └── data_utils.py              ← CoQA loading added
    └── generate_answers.py            ← CoQA handling added
```

---

## How to Use

### 1. Quick Test (Recommended First)

Verify CoQA dataset loads correctly:

```bash
python test_coqa_loading.py
```

Expected output:
```
================================================================================
Testing CoQA Dataset Loading
================================================================================

1. Loading CoQA dataset...
✅ Successfully loaded CoQA dataset
   - Training samples: 108,647
   - Validation samples: 7,983

2. Inspecting sample format...
   Sample keys: dict_keys(['id', 'question', 'context', 'answers'])
   
[... more output ...]

✅ ALL TESTS PASSED - CoQA dataset is ready to use!
```

### 2. Test Run (50 samples, ~3 hours)

Edit `run_CoQA_all.sh` and change:
```bash
NUM_SAMPLES=50  # Instead of 400
```

Then run:
```bash
./run_CoQA_all.sh
```

### 3. Full Run (400 samples, ~30 hours)

```bash
./run_CoQA_all.sh
```

### 4. Monitor Progress

```bash
# Terminal 1: Run experiments
./run_CoQA_all.sh

# Terminal 2: Monitor GPUs
watch -n 1 nvidia-smi

# Browser: Check WandB
# https://wandb.ai/[YOUR_ENTITY]/super_guacamole
```

---

## Dataset Details

### CoQA Statistics

- **Total Questions**: 127,000+ questions
- **Conversations**: 8,000+ conversations
- **Training Samples**: 108,647 Q&A pairs (after flattening)
- **Validation Samples**: 7,983 Q&A pairs (after flattening)
- **Avg Questions/Conversation**: ~15 questions
- **Domains**: Literature, Wikipedia, children's stories, exams, news, etc.

### Sample Format After Processing

```python
{
    'id': 'story_123_q0',
    'question': 'Who went to the park?',
    'context': 'Sarah went to the park. She brought her dog Max...',
    'answers': {
        'text': ['Sarah'],
        'answer_start': [0]
    }
}
```

---

## Model Configuration

### Small Models (Phase 1)

| Model | Params | Quant | GPU | Time/Model |
|-------|--------|-------|-----|------------|
| Llama-3.2-1B | 1B | None | 1 | ~2h |
| Qwen2.5-1.5B | 1.5B | None | 1 | ~2h |
| Mistral-7B-v0.3 | 7B | 8-bit | 1 | ~3h |

### Large Models (Phase 2)

| Model | Params | Quant | GPU | Time/Model |
|-------|--------|-------|-----|------------|
| Llama-3.1-8B | 8B | None | 1 | ~3h |
| Qwen3-8B | 8B | None | 1 | ~3h |
| Mistral-7B-Instruct | 7B | None | 1 | ~3h |

### Ultra-Large Models (Phase 3)

| Model | Params | Quant | GPU | CPU | Time/Model |
|-------|--------|-------|-----|-----|------------|
| Llama-3.1-70B | 70B | 4-bit | 4 | No | ~3h |
| Qwen2.5-72B | 72B | 4-bit | 4 | No | ~3.5h |
| Mistral-Large-2 | 123B | 4-bit | 4 | Yes | ~4h |

---

## Expected Results

### Accuracy Estimates (CoQA)

| Model Category | Exact Match | F1 Score |
|---------------|-------------|----------|
| Small (1-7B) | 45-55% | 55-65% |
| Large (7-8B) | 65-72% | 72-80% |
| Ultra (70B+) | 75-85% | 82-90% |

**Note**: CoQA's conversational nature means:
- Larger models may show **bigger gains** than on TriviaQA/SQuAD
- Coreference resolution improves significantly with scale
- Context window utilization matters more

### Performance Comparison

Compared to TriviaQA/SQuAD, expect:
- **Higher variance**: Conversational questions more ambiguous
- **Larger scale effects**: 70B models may excel more on CoQA
- **Different calibration**: Uncertainty patterns may differ

---

## Timeline

### Full Run (400 samples per model)

```
Phase 1: Small Models         → 6-8 hours
Phase 2: Large Models         → 8-10 hours  
Phase 3: Ultra-Large Models   → 10-12 hours
----------------------------------------
Total:                        → 24-30 hours
```

### Test Run (50 samples per model)

```
Phase 1: Small Models         → ~1 hour
Phase 2: Large Models         → ~1 hour
Phase 3: Ultra-Large Models   → ~1 hour
----------------------------------------
Total:                        → ~3 hours
```

---

## Resource Requirements

### GPU Setup
- **Minimum**: 1× 11GB GPU (for small/large models)
- **Optimal**: 4× 11GB GPUs (44GB total, for all models)
- **Phase 1-2**: Uses 1 GPU
- **Phase 3**: Uses all 4 GPUs + CPU offload

### System Memory
- **Minimum**: 16GB RAM
- **Recommended**: 32GB+ RAM
- **For Mistral-Large-2**: 30GB free RAM required (CPU offload)

### Disk Space
- **CoQA Dataset**: ~50MB
- **Model Downloads** (first run):
  - Small models: ~8GB total
  - Large models: ~45GB total
  - Ultra-large models: ~200GB total
- **Results**: ~1GB per model

---

## Next Steps After Completion

### 1. Analyze Results

```bash
# Compute AUROC metrics
python compute_multi_model_auroc.py \
    --wandb_dir src/nikos/uncertainty/wandb \
    --output_dir results/coqa_auroc \
    --experiment_pattern "coqa_"
```

### 2. Compare Across Datasets

You now have:
- ✅ TriviaQA results (factual QA)
- ✅ SQuAD results (reading comprehension)
- ✅ CoQA results (conversational QA)

**Research Questions**:
1. Which dataset is hardest for which model size?
2. Do models rank differently across datasets?
3. How does uncertainty calibration differ?
4. Which model family excels at conversations vs facts?

### 3. Publication-Ready Analysis

With 3 datasets × 9 models = 27 experiments:
- Comprehensive scaling analysis
- Cross-dataset generalization study
- Uncertainty calibration research
- Model family comparison

---

## Troubleshooting

### Common Issues

**1. Dataset fails to download**
```bash
# Manually download first
cd src
python -c "from datasets import load_dataset; load_dataset('stanfordnlp/coqa')"
```

**2. Out of memory on small models**
```bash
# Check for other processes
nvidia-smi
# Kill if needed, then retry
```

**3. Ultra-large models slow**
- **Normal**: 70B+ models are slow
- **Verify**: Check `nvidia-smi` shows 4 GPUs active
- **Patience**: 3-4 hours per model is expected

**4. Mistral-Large-2 uses lots of RAM**
- **Normal**: 123B model needs CPU offload
- **Monitor**: `watch -n 1 free -h`
- **Solution**: Close other applications

---

## Summary

✅ **Complete CoQA support implemented**  
✅ **All 9 models configured for CoQA**  
✅ **Automatic scaling from 1B to 123B**  
✅ **Comprehensive guide and documentation**  
✅ **Test script for verification**  

**Ready to run**:
```bash
# Test first
python test_coqa_loading.py

# Then run all experiments
./run_CoQA_all.sh
```

**Total time**: ~24-30 hours for complete evaluation across all model sizes! 🚀

---

## Files Changed/Created

1. ✅ `src/data/data_utils.py` - Added CoQA loading
2. ✅ `src/generate_answers.py` - Added CoQA handling  
3. ✅ `run_CoQA_all.sh` - Main execution script
4. ✅ `COQA_EXPERIMENTS_GUIDE.md` - Complete guide
5. ✅ `test_coqa_loading.py` - Verification script
6. ✅ `COQA_IMPLEMENTATION_SUMMARY.md` - This summary

**All ready to use!** 🎉
