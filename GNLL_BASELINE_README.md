# G-NLL Baseline Evaluation

This guide explains how to generate short and long answers, evaluate them, and compute AUROC for the G-NLL baseline method.

## Overview

- **G-NLL**: Sum of token log-probabilities from the greedy answer (negative log-likelihood)
- **Short answers**: Brief, concise answers (evaluated with ROUGE and LLM-as-a-judge)
- **Long answers**: Detailed, well-structured answers (evaluated with LLM-as-a-judge only)

## Quick Start

### Step 1: Generate Short Answers

```bash
cd src
python generate_answers.py \
    --model_name Llama-3.2-1B \
    --dataset trivia_qa \
    --num_samples 400 \
    --num_few_shot 5 \
    --temperature 0.0 \
    --num_generations 1 \
    --brief_prompt short \
    --enable_brief \
    --brief_always \
    --no-compute_uncertainties \
    --no-compute_p_true \
    --no-get_training_set_generations \
    --use_context \
    --entity nikosteam \
    --project super_guacamole \
    --experiment_lot gnll_short_$(date +%Y%m%d_%H%M%S)
```

**Note the wandb run ID from the output!**

### Step 2: Evaluate Short Answers with LLM Judge

```bash
cd src
python compute_uncertainty_measures.py \
    --eval_wandb_runid <SHORT_RUN_ID> \
    --metric llm_llama-3-8b \
    --recompute_accuracy \
    --no-compute_predictive_entropy \
    --no-compute_p_ik \
    --no-compute_context_entails_response \
    --no-analyze_run \
    --entity nikosteam \
    --project super_guacamole \
    --restore_entity_eval nikosteam
```

### Step 3: Generate Long Answers

```bash
cd src
python generate_answers.py \
    --model_name Llama-3.2-1B \
    --dataset trivia_qa \
    --num_samples 400 \
    --num_few_shot 5 \
    --temperature 0.0 \
    --num_generations 1 \
    --brief_prompt detailed \
    --enable_brief \
    --brief_always \
    --no-compute_uncertainties \
    --no-compute_p_true \
    --no-get_training_set_generations \
    --use_context \
    --entity nikosteam \
    --project super_guacamole \
    --experiment_lot gnll_long_$(date +%Y%m%d_%H%M%S)
```

**Note the wandb run ID from the output!**

### Step 4: Evaluate Long Answers with LLM Judge

```bash
cd src
python compute_uncertainty_measures.py \
    --eval_wandb_runid <LONG_RUN_ID> \
    --metric llm_llama-3-8b \
    --recompute_accuracy \
    --no-compute_predictive_entropy \
    --no-compute_p_ik \
    --no-compute_context_entails_response \
    --no-analyze_run \
    --entity nikosteam \
    --project super_guacamole \
    --restore_entity_eval nikosteam
```

### Step 5: Compute G-NLL AUROC

After generation and evaluation, compute AUROC using the standalone script:

#### For Short Answers (ROUGE-based):

```bash
python compute_gnll_auroc.py \
    src/nikos/uncertainty/wandb/run-<SHORT_RUN_ID>/files/validation_generations.pkl \
    --rouge \
    --rouge-threshold 0.3 \
    --output short_rouge_results.json
```

#### For Short Answers (LLM Judge-based):

```bash
python compute_gnll_auroc.py \
    src/nikos/uncertainty/wandb/run-<SHORT_RUN_ID>/files/validation_generations.pkl \
    --output short_judge_results.json
```

#### For Long Answers (LLM Judge-based):

```bash
python compute_gnll_auroc.py \
    src/nikos/uncertainty/wandb/run-<LONG_RUN_ID>/files/validation_generations.pkl \
    --output long_judge_results.json
```

## Automated Script

Alternatively, use the automated script (requires manual input of wandb run IDs):

```bash
python run_gnll_baseline.py
```

## Expected Output

The `compute_gnll_auroc.py` script will output:

```
RESULTS
================================================================================
  G-NLL AUROC: 0.XXXX
  Accuracy: 0.XXXX (XXX/XXX)
  Number of examples: XXX
  Mean G-NLL: X.XXXX
  Std G-NLL: X.XXXX
  Min G-NLL: X.XXXX
  Max G-NLL: X.XXXX
================================================================================
```

## Understanding the Results

- **G-NLL AUROC**: Area Under ROC Curve for G-NLL as an uncertainty measure
  - Higher values (closer to 1.0) indicate better ability to distinguish correct from incorrect answers
  - Values > 0.5 indicate the measure is useful
  - Values < 0.5 indicate the measure is worse than random
  
- **Accuracy**: Fraction of answers that are correct (according to the evaluation method)

- **G-NLL values**: 
  - Higher G-NLL = lower probability = more uncertain
  - We expect incorrect answers to have higher G-NLL than correct answers

## Notes

1. **Short answers** use the `'short'` brief prompt which asks for "brief, concise answers"
2. **Long answers** use the `'detailed'` brief prompt which asks for "detailed, well-structured answers"
3. Both use **greedy decoding** (temperature=0.0) with a single generation
4. The G-NLL is computed as the negative sum of token log-likelihoods from the greedy answer
5. For short answers, you can evaluate with both ROUGE (automatic) and LLM judge
6. For long answers, only LLM judge is used (ROUGE is less suitable for longer answers)

## Troubleshooting

- If AUROC computation fails with "all labels are the same", check that your evaluation method is working correctly
- Make sure the validation_generations.pkl file exists in the wandb run directory
- Ensure the LLM judge evaluation has completed before computing AUROC with judge-based correctness

