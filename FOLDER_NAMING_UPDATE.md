# Phase Scripts Folder Naming Update

## Overview

All phase analysis scripts have been updated to support **automatic folder naming** based on WandB run IDs and context types (short/long). This makes it easier to organize and track results from multiple experimental runs.

## Changes Made

### Modified Scripts

1. `src/analysis/phase1_baseline_metrics.py`
2. `src/analysis/phase1_5_token_nll_analysis.py`
3. `src/analysis/phase1_6_prefix_nll_analysis.py`
4. `src/analysis/phase2_token_importance.py`
5. `src/analysis/phase5_comparative_analysis.py`

### New Parameters

All phase scripts now accept these additional parameters:

- `--wandb-run-id`: WandB run ID to include in output folder name
- `--context-type`: Context type (`short` or `long`) for output folder naming
- `--output-dir`: Custom output directory (optional - if not provided, auto-generated)

### Folder Naming Convention

**Pattern**: `results/phase{X}_{context_type}_{wandb_run_id}`

**Examples**:
- `results/phase1_short_yhxde999`
- `results/phase1_long_wiboofpr`
- `results/phase1_6_long_wiboofpr`
- `results/phase2_long_wiboofpr`
- `results/phase5_long_wiboofpr`

### Backwards Compatibility

- If `--output-dir` is explicitly provided, it takes precedence (old behavior)
- If `--wandb-run-id` and/or `--context-type` are not provided, they are omitted from the folder name
- Default folder names (without parameters) remain: `results/phase1`, `results/phase1_5`, etc.

## Usage Examples

### Before (Old Way)

```bash
python -m src.analysis.phase1_baseline_metrics \
  --long-pickle "path/to/validation_generations.pkl" \
  --output-dir results/phase1_long
```

### After (New Way - Recommended)

```bash
python -m src.analysis.phase1_baseline_metrics \
  --long-pickle "path/to/validation_generations.pkl" \
  --context-type long \
  --wandb-run-id wiboofpr
```

Output directory auto-generated: `results/phase1_long_wiboofpr`

### All Phases Example

For a complete analysis of a long-answer run with WandB ID `wiboofpr`:

```bash
# Phase 1: Baseline metrics
python -m src.analysis.phase1_baseline_metrics \
  --long-pickle "path/to/validation_generations.pkl" \
  --context-type long \
  --wandb-run-id wiboofpr

# Phase 1.5: Token-level NLL analysis
python -m src.analysis.phase1_5_token_nll_analysis \
  --pickle-path "path/to/validation_generations.pkl" \
  --model-name Llama-3.2-1B \
  --sample-size 100 \
  --context-type long \
  --wandb-run-id wiboofpr

# Phase 1.6: Prefix-level NLL analysis
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path "path/to/validation_generations.pkl" \
  --context-type long \
  --wandb-run-id wiboofpr \
  --max-prefix-len 50 \
  --ks 1 3 5

# Phase 2: Token relevance analysis
python -m src.analysis.phase2_token_importance \
  --pickle-path "path/to/validation_generations.pkl" \
  --model-name Llama-3.2-1B \
  --similarity-model cross-encoder/stsb-roberta-large \
  --sample-size 50 \
  --context-type long \
  --wandb-run-id wiboofpr

# Phase 5: Comparative AUROC analysis
python -m src.analysis.phase5_comparative_analysis \
  --pickle-path "path/to/validation_generations.pkl" \
  --model-name Llama-3.2-1B \
  --similarity-model cross-encoder/stsb-roberta-large \
  --context-type long \
  --wandb-run-id wiboofpr
```

All results will be organized in respective folders:
- `results/phase1_long_wiboofpr/`
- `results/phase1_5_long_wiboofpr/`
- `results/phase1_6_long_wiboofpr/`
- `results/phase2_long_wiboofpr/`
- `results/phase5_long_wiboofpr/`

## Benefits

1. **Better Organization**: Results from different runs are clearly separated
2. **Traceability**: Easy to match results to WandB runs
3. **Context Clarity**: Immediately see if results are for short or long answers
4. **Automation**: No need to manually construct folder names
5. **Backwards Compatible**: Old scripts/workflows still work with explicit `--output-dir`

## Migration Guide

### For Existing Scripts/Workflows

**Option 1**: Keep using explicit `--output-dir` (no changes needed)

**Option 2**: Replace `--output-dir` with `--context-type` and `--wandb-run-id`:

```diff
- --output-dir results/phase1_long_wiboofpr
+ --context-type long --wandb-run-id wiboofpr
```

### Finding Your WandB Run ID

The WandB run ID is part of the folder name where your pickles are stored:

```
src/boldis/uncertainty/wandb/run-20251121_092732-wiboofpr/files/
                                                  ^^^^^^^^^
                                                  This is your run ID
```

Or check your WandB dashboard at the end of the run URL.

## Implementation Details

Each script now includes this logic:

```python
# Auto-generate output directory if not provided
if args.output_dir is None:
    dir_parts = ['results', 'phase{X}']
    if args.context_type:
        dir_parts.append(args.context_type)
    if args.wandb_run_id:
        dir_parts.append(args.wandb_run_id)
    args.output_dir = '_'.join(dir_parts)
    logger.info(f"Auto-generated output directory: {args.output_dir}")
```

This ensures:
- Flexibility: Only add components that are provided
- Clarity: Log the generated directory name
- Override: Explicit `--output-dir` always takes precedence

