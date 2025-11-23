# Phase 1.6: Prefix-Level NLL Analysis - Explained

## What is Phase 1.6?

Phase 1.6 analyzes how the model's **uncertainty evolves as it generates an answer token by token**. The key question: Can early tokens (e.g., first 1, 3, or 5 tokens) already predict whether the final answer will be correct or incorrect?

### What it does:

1. **Computes prefix mean NLL curves**: For each answer, calculates the average NLL of the first k tokens (k = 1, 2, 3, ..., up to `max_prefix_len`)

2. **Compares correct vs incorrect answers**: Aggregates these curves separately to see if there are patterns

3. **Early-token AUROC**: Evaluates predictive power using only the first few tokens' NLL

---

## Your Results: run-20251121_092732-wiboofpr

### 📊 Before Judge Correction (WRONG LABELS)

**Results from**: `results/phase1_6_long_wiboofpr/`

- **Class distribution**: 5 correct vs 395 incorrect (1.2% accuracy) ❌
- **AUROC (first token)**: 0.534 (barely better than random)
- **AUROC trend**: Decreases with more tokens (0.410 for 3 tokens, 0.341 for 5 tokens)
- **Pattern**: Correct answers had HIGHER NLL than incorrect ones (inverted!)

**Problem**: The original accuracy labels were severely wrong, leading to meaningless analysis.

---

### ✅ After Judge Correction (CORRECT LABELS)

**Results from**: `results/phase1_6_long_wiboofpr_judge_corrected/`

- **Class distribution**: 392 correct vs 8 incorrect (98.0% accuracy) ✅
- **AUROC (first token)**: 0.506 (slightly better than random)
- **AUROC trend**: Slightly increases with more tokens (0.521 for 3 tokens, 0.532 for 5 tokens)
- **Pattern**: Correct answers have slightly higher first-token NLL (0.66 vs 0.55)

### How to Interpret These Results:

#### 1. **High Accuracy = Limited Discriminability**

With 98% accuracy, the model is performing very well. The 8 incorrect answers are rare outliers, making it hard to find consistent patterns.

#### 2. **AUROC ≈ 0.5 is Expected**

AUROC around 0.5 means NLL-based uncertainty doesn't strongly separate correct from incorrect answers. This can happen when:
- The model is confident and correct most of the time
- Errors are random/unpredictable rather than systematic
- The few errors don't follow a clear uncertainty pattern

#### 3. **First-Token NLL**

Correct answers: **0.66** (higher uncertainty)  
Incorrect answers: **0.55** (lower uncertainty)

This counter-intuitive finding suggests the model is sometimes **confidently wrong** on its few errors.

---

## Visualizations

### 1. `prefix_mean_nll_curves.png`

Shows how mean NLL evolves as more tokens are generated:
- **Blue/Green line**: Correct answers (392 examples)
- **Red/Orange line**: Incorrect answers (8 examples)

With so few incorrect examples, the incorrect curve is noisy and less reliable.

### 2. `first_token_nll_boxplot.png`

Boxplot comparing first-token NLL distribution:
- Most correct answers have low first-token NLL
- A few outliers have high uncertainty
- Incorrect answers show varied behavior (but only 8 samples)

---

## Key Takeaways

### ✅ Good News

1. **Model performs well**: 98% accuracy on long-form generation
2. **Judge correction worked**: Labels now make sense
3. **Analysis is valid**: Results align with a high-performing model

### ⚠️ Limitations

1. **Severe class imbalance**: Only 8 incorrect answers make it hard to draw strong conclusions
2. **Early tokens don't help much**: AUROC ≈ 0.5 means prefix NLL isn't a strong predictor
3. **Need more data**: With more incorrect examples, you could better understand error patterns

### 💡 Implications for Research

- **G-NLL alone may not be enough**: For high-accuracy models, raw NLL doesn't strongly indicate errors
- **Consider other metrics**: SAR (token relevance) or Semantic Entropy might provide better discrimination
- **Selective prediction**: With AUROC ≈ 0.5, you can't reliably filter incorrect answers using early-token NLL

---

## How to Use the Judge-Corrected Pickle

### For Future Analysis

Always use the judge-corrected pickle for accurate results:

```bash
python -m src.analysis.<any_phase_script> \
  --pickle-path "src/boldis/uncertainty/wandb/run-20251121_092732-wiboofpr/files/validation_generations_judge_corrected.pkl" \
  --output-dir results/<phase>_judge_corrected
```

### Creating Judge-Corrected Pickles

If you run `recompute_accuracy_with_judge.py` in the future, use this script:

```bash
python update_pickle_with_judge_from_uncertainty.py \
  --original-pickle "<path_to_validation_generations.pkl>" \
  --uncertainty-measures "<judge_run_wandb_dir>/files/uncertainty_measures.pkl" \
  --output "<optional_output_path>"
```

The script reads `validation_is_false` from `uncertainty_measures.pkl` and updates the accuracy labels in the pickle file.

---

## Technical Details

### How Labels are Stored

**Original pickle** (`validation_generations.pkl`):
```python
{
    'example_id': {
        'most_likely_answer': {
            'accuracy': 0.0 or 1.0,  # Original metric (e.g., ROUGE, F1)
            ...
        },
        ...
    }
}
```

**Judge run** (`uncertainty_measures.pkl`):
```python
{
    'validation_is_false': [0.0, 1.0, 0.0, ...],  # 0.0 = correct, 1.0 = incorrect
    ...
}
```

**Judge-corrected pickle** (`validation_generations_judge_corrected.pkl`):
```python
{
    'example_id': {
        'most_likely_answer': {
            'accuracy': 1.0 or 0.0,  # Updated from judge
            ...
        },
        ...
    }
}
```

---

## Files Created

✅ **`update_pickle_with_judge_from_uncertainty.py`**  
   Script to merge judge scores from `uncertainty_measures.pkl` into validation pickle

✅ **`validation_generations_judge_corrected.pkl`**  
   Judge-corrected pickle file ready for analysis

✅ **`results/phase1_6_long_wiboofpr_judge_corrected/`**  
   Phase 1.6 results with correct labels

---

## Next Steps

1. **Run other phases** with the judge-corrected pickle (Phase 1.5, Phase 2, Phase 5)
2. **Compare metrics**: See if SAR or Semantic Entropy provide better AUROC than raw G-NLL
3. **Investigate the 8 errors**: Manually inspect why the model got these wrong
4. **Consider more challenging data**: If you want to study uncertainty, you might need examples where the model struggles more

---

**Summary**: Phase 1.6 now shows realistic results for a high-performing model. Early-token NLL doesn't strongly predict errors, which is valuable information suggesting you need more sophisticated uncertainty metrics for selective prediction.

