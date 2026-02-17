# CoQA Experiments Guide - Complete Model Suite

## Overview

This guide covers running **all models** (small → large → ultra-large) on the **CoQA dataset** (Conversational Question Answering Challenge).

**Dataset**: CoQA - 127K questions from 8K conversations  
**Total Models**: 9 models across 3 size categories  
**Estimated Time**: 24-30 hours for all experiments  
**GPU Setup**: Optimized for 4×11GB GPUs (44GB total)

---

## What is CoQA?

**CoQA (Conversational Question Answering)** is a dataset that tests models' ability to:
- ✅ Understand text passages (stories, articles, dialogues)
- ✅ Answer questions that depend on conversational history
- ✅ Handle follow-up questions that reference previous Q&A
- ✅ Generate free-form text answers

### Key Differences from TriviaQA/SQuAD

| Feature | TriviaQA | SQuAD | CoQA |
|---------|----------|-------|------|
| **Question Type** | Isolated factual | Isolated reading comp | Conversational |
| **Context Dependency** | Low | Medium | High |
| **Follow-up Questions** | ❌ No | ❌ No | ✅ Yes |
| **Answer Type** | Short facts | Text spans | Free-form text |
| **Avg Questions/Context** | 1 | 5 | 15+ |

### Why CoQA is Challenging

1. **Conversational Reasoning**: Questions like "What happened next?" require understanding previous Q&A pairs
2. **Coreference Resolution**: "Who is he?" requires tracking entities across conversation
3. **Implicit Context**: Later questions assume knowledge from earlier in the conversation
4. **Free-form Answers**: Not constrained to exact text spans

**Example CoQA Conversation**:
```
Story: "Sarah went to the park. She brought her dog Max..."

Q1: Who went to the park?
A1: Sarah

Q2: What did she bring?          ← "she" refers to Sarah
A2: Her dog Max

Q3: What is his name?             ← "his" refers to the dog
A3: Max
```

---

## Models Included

### Phase 1: Small Models (1B-7B)
| Model | Parameters | Quantization | Speed | Memory |
|-------|------------|--------------|-------|--------|
| Llama-3.2-1B | 1B | None | Very Fast ⚡⚡⚡ | ~2GB |
| Qwen2.5-1.5B | 1.5B | None | Very Fast ⚡⚡⚡ | ~3GB |
| Mistral-7B-v0.3 | 7B | 8-bit | Fast ⚡⚡ | ~7GB |

**Expected Time**: ~6-8 hours for all 3 models

### Phase 2: Large Models (7-8B)
| Model | Parameters | Quantization | Speed | Memory |
|-------|------------|--------------|-------|--------|
| Llama-3.1-8B | 8B | None (fp16) | Medium ⚡ | ~16GB |
| Qwen3-8B | 8B | None (fp16) | Medium ⚡ | ~16GB |
| Mistral-7B-Instruct-v0.3 | 7B | None (fp16) | Medium ⚡ | ~14GB |

**Expected Time**: ~8-10 hours for all 3 models

### Phase 3: Ultra-Large Models (70B+)
| Model | Parameters | Quantization | Speed | Memory |
|-------|------------|--------------|-------|--------|
| Llama-3.1-70B-Instruct | 70B | 4-bit | Slow ⏱️ | ~35GB (4 GPUs) |
| Qwen2.5-72B | 72B | 4-bit | Slow ⏱️ | ~36GB (4 GPUs) |
| Mistral-Large-2 | 123B | 4-bit | Very Slow ⏱️⏱️ | ~62GB (4 GPUs + CPU) |

**Expected Time**: ~10-12 hours for all 3 models

---

## Quick Start

### Prerequisites

1. **GPU Check**:
   ```bash
   python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
   # Should show: GPUs: 4
   ```

2. **Memory Check**:
   ```bash
   nvidia-smi  # Check GPU memory
   free -h     # Ensure 30GB+ free RAM for CPU offloading
   ```

3. **Dataset Check**:
   ```bash
   cd src
   python -c "from data.data_utils import load_ds; load_ds('coqa', 42)"
   # Should download CoQA dataset (~50MB)
   ```

### Run All Experiments

**Option 1: Full Run (Recommended)**
```bash
./run_CoQA_all.sh
```
This will run all 9 models sequentially over ~24-30 hours.

**Option 2: Run Individual Phases**

If you want to run phases separately:

```bash
# Edit run_CoQA_all.sh and comment out phases you don't want
# For example, to run only small models:
# - Comment out PHASE 2 and PHASE 3 sections
./run_CoQA_all.sh
```

**Option 3: Test Run (Recommended First)**

Test with fewer samples first:
```bash
# Edit run_CoQA_all.sh
# Change: NUM_SAMPLES=400
# To:     NUM_SAMPLES=50

./run_CoQA_all.sh
```
This will complete in ~2-3 hours and verify everything works.

### Monitor Progress

Open a second terminal:
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check WandB dashboard
# https://wandb.ai/[YOUR_ENTITY]/super_guacamole
```

---

## Expected Results

### Accuracy Comparison (Expected Trends)

Based on CoQA's conversational nature:

| Model Size | Exact Match (EM) | F1 Score | Notes |
|------------|------------------|----------|-------|
| 1-1.5B | 45-55% | 55-65% | Struggles with coreference |
| 7-8B (quantized) | 60-70% | 70-78% | Better context tracking |
| 7-8B (fp16) | 65-72% | 72-80% | Strong baseline |
| 70-72B (4-bit) | 75-82% | 82-88% | Good conversational reasoning |
| 123B (4-bit) | 78-85% | 85-90% | Best coreference resolution |

**Key Insight**: CoQA's conversational nature means:
- Larger models have **bigger gains** than on TriviaQA/SQuAD
- Coreference resolution improves significantly with scale
- Context window utilization matters more

### Uncertainty Calibration

**Expected**: CoQA may show **different calibration** than factual QA because:
- Conversational questions have more ambiguity
- Multiple valid answer phrasings exist
- Context dependency adds uncertainty

This makes CoQA valuable for studying **calibration under conversational reasoning**.

---

## Performance Timeline

### Full Run (400 samples per model)

```
Hour 0-2:   Llama-3.2-1B         [Small Phase]
Hour 2-4:   Qwen2.5-1.5B         [Small Phase]
Hour 4-8:   Mistral-7B-v0.3      [Small Phase]

Hour 8-11:  Llama-3.1-8B         [Large Phase]
Hour 11-14: Qwen3-8B             [Large Phase]
Hour 14-18: Mistral-7B-Instruct  [Large Phase]

Hour 18-21: Llama-3.1-70B        [Ultra-Large Phase]
Hour 21-25: Qwen2.5-72B          [Ultra-Large Phase]
Hour 25-30: Mistral-Large-2      [Ultra-Large Phase]

Total: ~30 hours
```

### Test Run (50 samples per model)

```
Hour 0-1:   All 3 Small Models
Hour 1-2:   All 3 Large Models  
Hour 2-3:   All 3 Ultra-Large Models

Total: ~3 hours
```

---

## Comparison with Other Datasets

After completing CoQA experiments, you'll have comprehensive results across:

### Dataset Characteristics

| Dataset | Type | Avg Context | Questions | Best For |
|---------|------|-------------|-----------|----------|
| **TriviaQA** | Factual QA | None/Web | Isolated | Factual recall |
| **SQuAD** | Reading Comp | Paragraph | Isolated | Span extraction |
| **CoQA** | Conversational | Story | Sequential | Reasoning + Context |

### Research Questions You Can Answer

1. **How does conversational reasoning scale?**
   - Compare small vs large vs ultra-large on CoQA
   - Measure gap between model sizes

2. **Does model size help with coreference?**
   - Analyze questions with pronouns ("he", "she", "it")
   - Compare accuracy on early vs late conversation turns

3. **Uncertainty calibration differences**
   - Is CoQA harder to calibrate than factual QA?
   - Do larger models show better calibration on conversational tasks?

4. **Cross-dataset generalization**
   - Do models that excel at TriviaQA also excel at CoQA?
   - Which model family (Llama/Qwen/Mistral) handles conversations best?

---

## Troubleshooting

### Issue: CoQA dataset fails to download

**Solution**:
```bash
# Manually trigger download first
cd src
python -c "
from datasets import load_dataset
dataset = load_dataset('stanfordnlp/coqa')
print('CoQA dataset downloaded successfully!')
"
```

### Issue: "CUDA out of memory" on small models

**Unexpected** - Small models should fit easily. Check:
```bash
nvidia-smi  # Are other processes using GPU?
# Kill other processes if needed
```

### Issue: Ultra-large models are extremely slow

**Expected** - This is normal for 70B+ models:
- Llama-3.1-70B: ~6-10 tokens/sec
- Qwen2.5-72B: ~5-9 tokens/sec
- Mistral-Large-2: ~3-5 tokens/sec (CPU offload)

**Verify it's working**:
```bash
watch -n 1 nvidia-smi
# All 4 GPUs should show 70-90% utilization
```

### Issue: Mistral-Large-2 uses lots of RAM

**Expected** - 123B model requires CPU offloading:
- Will use up to 30GB of system RAM
- Monitor with: `watch -n 1 free -h`
- Close other applications if needed

---

## After Experiments Complete

### 1. View Results on WandB

```bash
# Results will be in your WandB project
# https://wandb.ai/[YOUR_ENTITY]/super_guacamole
# Filter by experiment_lot: "coqa_*"
```

### 2. Compute AUROC Metrics

```bash
python compute_multi_model_auroc.py \
    --wandb_dir src/nikos/uncertainty/wandb \
    --output_dir results/coqa_auroc \
    --experiment_pattern "coqa_"
```

### 3. Compare Across Datasets

Create comparison plots:
```python
# Compare TriviaQA vs SQuAD vs CoQA
# - Which dataset is hardest?
# - How do models rank differently?
# - Does conversational reasoning change uncertainty calibration?
```

---

## Cost/Time Considerations

### GPU Hours

| Phase | Models | Hours | GPU-Hours |
|-------|--------|-------|-----------|
| Small | 3 | ~8h | 8h (1 GPU) |
| Large | 3 | ~10h | 10h (1 GPU) |
| Ultra-Large | 3 | ~12h | 48h (4 GPUs) |
| **Total** | **9** | **~30h** | **66 GPU-hours** |

### Cloud Cost Estimates (if applicable)

- **4× RTX 3090 setup**: ~$1.50/hour
- **Full CoQA experiments**: ~$45 total
- **Per phase**: Small ~$12, Large ~$15, Ultra-Large ~$18

---

## Scientific Value

### Why Run CoQA Experiments?

1. **Conversational Reasoning**: CoQA tests abilities not covered by TriviaQA/SQuAD
2. **Scale Analysis**: See if larger models have disproportionate gains
3. **Calibration Research**: Study uncertainty in conversational vs factual tasks
4. **Model Comparison**: Comprehensive comparison across model families

### Expected Insights

1. **Larger models may show bigger gains** on CoQA than on factual QA
2. **Uncertainty calibration** may differ for conversational tasks
3. **Model families** may show different strengths (e.g., Qwen strong at multi-turn)
4. **Context utilization** becomes more critical than in isolated QA

---

## Summary

**What you get**:
- 9 models tested on CoQA dataset
- Complete size spectrum (1B → 123B)
- 3 model families (Llama, Qwen, Mistral)
- Conversational reasoning evaluation
- Uncertainty calibration on complex task

**What to expect**:
- 24-30 hours total runtime (test with 50 samples first)
- Automatic model scaling (1 GPU → 4 GPUs as needed)
- CPU offloading for 123B model (automatic)
- Comprehensive results for cross-dataset analysis

**Start with**:
```bash
./run_CoQA_all.sh
```

Monitor with `nvidia-smi` and check back in ~30 hours! 🚀

---

## Next Steps After CoQA

You'll now have results from **3 datasets × 9 models = 27 experiments total**:

1. **Compare across datasets**: TriviaQA (factual) vs SQuAD (reading) vs CoQA (conversational)
2. **Analyze scaling laws**: How does performance improve with model size?
3. **Study uncertainty**: Which tasks are easier to calibrate?
4. **Model strengths**: Which family excels at which task type?

This comprehensive evaluation will give you rich insights into:
- Model capabilities across task types
- Uncertainty calibration properties
- Scaling behavior for different reasoning types
- Family-specific strengths and weaknesses

**Good luck with your experiments!** 🎉
