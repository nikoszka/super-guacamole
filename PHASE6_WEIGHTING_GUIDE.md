# Phase 6: Token Weighting Schemes Analysis

## Overview

This analysis compares different token weighting schemes for uncertainty quantification, addressing your supervisor's request to:

1. ✅ Try weights that increase/decrease linearly with token position
2. ✅ Compute importance weights with SAR method
3. ✅ Visualize SAR weights to see structural patterns
4. ✅ Compare effectiveness using AUROC

---

## 🎯 What This Does

The script implements **10 different token weighting schemes**:

### Position-Based Weights
1. **Uniform** - All tokens weighted equally (baseline = G-NLL)
2. **Linear Increasing** - Weight increases linearly from first to last token
3. **Linear Decreasing** - Weight decreases linearly from first to last token
4. **Quadratic Increasing** - Weight increases quadratically (strongly favor later tokens)
5. **Quadratic Decreasing** - Weight decreases quadratically (strongly favor earlier tokens)
6. **Exponential Increasing** - Exponential growth favoring later tokens
7. **Exponential Decreasing** - Exponential decay favoring earlier tokens
8. **Middle Peak** - Tokens in the middle get higher weight
9. **Edges Peak** - First and last tokens get higher weight

### Semantic-Based Weights
10. **SAR Relevance** - Token ablation to measure semantic importance

---

## 🚀 How to Run

### Using Your Existing Long Answer Data

```bash
# For long answers (LLM judge)
python -m src.analysis.phase6_weighting_schemes_comparison \
  --pickle-path src/boldis/uncertainty/wandb/run-20251121_190025-5qvhbs97/files/validation_generations.pkl \
  --model-name Llama-3.2-1B \
  --similarity-model cross-encoder/stsb-roberta-large \
  --output-dir results/phase6_weighting_long_5qvhbs97
```

### Using Short Answer Data

```bash
# For short answers (ROUGE-based)
python -m src.analysis.phase6_weighting_schemes_comparison \
  --pickle-path src/boldis/uncertainty/wandb/run-20251121_142824-wo7cdccl/files/validation_generations.pkl \
  --model-name Llama-3.2-1B \
  --similarity-model cross-encoder/stsb-roberta-large \
  --use-rouge \
  --rouge-threshold 0.3 \
  --output-dir results/phase6_weighting_short_wo7cdccl
```

---

## 📊 Expected Outputs

The script creates 6 files in your output directory:

### 1. `weighting_schemes_auroc.csv`
**Main results table** - Share this with your supervisor!

Example:
```
Scheme                    AUROC
sar_relevance            0.6234
linear_decreasing        0.5987
uniform                  0.5845
...
```

### 2. `auroc_comparison.png`
Bar chart showing which weighting scheme works best

### 3. `roc_curves.png`
ROC curves overlaying all schemes for direct comparison

### 4. `weight_patterns_comparison.png`
8-panel visualization showing how each scheme weights tokens across position

### 5. `sar_weight_details.png`
10 examples showing SAR relevance weight patterns
- Green bars = high importance tokens
- Red bars = low importance tokens
- Shows structural patterns in token importance

### 6. `weight_examples.json`
Raw data for 10 examples with all weights computed

---

## 💡 Interpreting Results

### AUROC Interpretation
- **> 0.60**: Strong predictor of correctness
- **0.55-0.60**: Moderate predictor
- **0.50-0.55**: Weak predictor
- **~0.50**: No better than random

### Key Questions to Answer
1. **Which position-based scheme works best?**
   - Does linear increasing/decreasing beat uniform?
   - Are earlier or later tokens more predictive?

2. **Does SAR relevance beat simple position-based schemes?**
   - If yes → semantic importance matters!
   - If no → position is sufficient proxy

3. **What patterns appear in SAR weights?**
   - Do certain token positions consistently get high weights?
   - Are there differences between correct vs incorrect answers?

---

## 🔬 Advanced Options

### Without Normalization
By default, weighted NLL is normalized by sum of weights (like RW-G-NLL).
To use raw weighted sum instead:

```bash
python -m src.analysis.phase6_weighting_schemes_comparison \
  --pickle-path YOUR_PICKLE \
  --model-name Llama-3.2-1B \
  --no-normalize \
  --output-dir results/phase6_unnormalized
```

### Different Similarity Model
SAR weights use a cross-encoder by default. To try a different model:

```bash
python -m src.analysis.phase6_weighting_schemes_comparison \
  --pickle-path YOUR_PICKLE \
  --model-name Llama-3.2-1B \
  --similarity-model sentence-transformers/all-MiniLM-L6-v2 \
  --output-dir results/phase6_different_sim
```

---

## 📝 Example Workflow

```bash
# Step 1: Run the analysis
python -m src.analysis.phase6_weighting_schemes_comparison \
  --pickle-path src/boldis/uncertainty/wandb/run-20251121_190025-5qvhbs97/files/validation_generations.pkl \
  --model-name Llama-3.2-1B \
  --output-dir results/phase6_long

# Step 2: Check results
cat results/phase6_long/weighting_schemes_auroc.csv

# Step 3: View visualizations
open results/phase6_long/auroc_comparison.png
open results/phase6_long/sar_weight_details.png

# Step 4: Share with supervisor
# Send the CSV file and key plots
```

---

## 🎯 What to Report to Supervisor

### Recommended Report Structure

**1. Summary Table**
- Share `weighting_schemes_auroc.csv`
- Highlight the best performing scheme

**2. Key Findings**
- "Linear increasing weights achieved AUROC of X.XXX, improving over uniform baseline by Y%"
- "SAR relevance weighting achieved AUROC of X.XXX, showing semantic importance [does/doesn't] improve over position-based schemes"

**3. SAR Weight Patterns (from `sar_weight_details.png`)**
- "Observed that [describe pattern]: e.g., 'first 3 tokens consistently receive low weights', 'middle tokens show highest relevance', etc."
- "Correct answers show [pattern] while incorrect answers show [pattern]"

**4. Recommendation**
- "For this task, [scheme] provides the best uncertainty estimation with AUROC of X.XXX"
- "Future work could explore [suggestions based on patterns observed]"

---

## 🐛 Troubleshooting

### "No correctness labels found"
Make sure your pickle has either:
- LLM judge accuracy scores in `most_likely_answer['accuracy']`, OR
- Use `--use-rouge` flag for short answers

### "SAR weight mismatch"
This happens if token alignment fails for some examples. The script will skip those examples and continue.

### "CUDA out of memory"
The similarity model needs GPU. If you run out of memory:
- Use a smaller similarity model: `--similarity-model sentence-transformers/all-MiniLM-L6-v2`
- Process fewer examples (edit the script to sample)

### Script runs slowly
SAR weight computation requires semantic similarity for every token. Expected time:
- ~5-10 min for 100 examples
- ~20-30 min for 400 examples

---

## 📚 Technical Details

### Weighted NLL Formula

For each scheme, we compute:

```
Weighted_NLL = [Σ w_t × (-log P(y_t))] / [Σ w_t]
```

Where:
- `w_t` = weight for token t (scheme-dependent)
- `log P(y_t)` = log-probability of token t
- Normalization by Σ w_t ensures fair comparison

### SAR Weight Computation

For each token position t:
1. Remove token t from response
2. Compute semantic similarity between:
   - Full: `prompt + complete_response`
   - Ablated: `prompt + response_without_token_t`
3. Relevance weight: `w_t = 1 - similarity`

Higher relevance = removing token causes more semantic change

---

## 🎉 Success Checklist

- [ ] Script runs without errors
- [ ] CSV file contains AUROC for all 10 schemes
- [ ] All 5 PNG visualizations created
- [ ] SAR weights show clear patterns in visualization
- [ ] Best performing scheme identified
- [ ] Results ready to share with supervisor

---

## Next Steps After Analysis

Based on your findings, you might want to:

1. **If SAR performs best**: Investigate which tokens get highest weights and why
2. **If linear works well**: Consider hybrid schemes (e.g., linear + SAR)
3. **If position-based beats SAR**: Simpler is better! Report computational savings
4. **Compare across datasets**: Run on both short and long answers to see if patterns hold

---

Good luck with your analysis! 🚀

