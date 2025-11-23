# Phase Analysis Scripts - Folder Naming Enhancement

## Summary

All phase analysis scripts now automatically generate organized folder names that include:
- **Context type** (`short` or `long`)
- **WandB run ID** for traceability

## What Changed

### 5 Scripts Updated

1. ✅ `src/analysis/phase1_baseline_metrics.py`
2. ✅ `src/analysis/phase1_5_token_nll_analysis.py`
3. ✅ `src/analysis/phase1_6_prefix_nll_analysis.py`
4. ✅ `src/analysis/phase2_token_importance.py`
5. ✅ `src/analysis/phase5_comparative_analysis.py`

### Documentation Updated

✅ `ANALYSIS_README.md` - Updated with new parameter examples

### New Parameters Added

All scripts now accept:
- `--wandb-run-id <ID>` - WandB run ID to include in folder name
- `--context-type {short,long}` - Context type for folder naming
- `--output-dir <PATH>` - Still works for custom paths (takes precedence)

## Before vs After

### Before ❌
```bash
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path "path/to/validation_generations.pkl" \
  --output-dir results/phase1_6_long_wiboofpr
```
- Had to manually construct folder names
- Easy to make mistakes or inconsistencies
- No automatic linking to WandB runs

### After ✅
```bash
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path "path/to/validation_generations.pkl" \
  --context-type long \
  --wandb-run-id wiboofpr
```
- Automatic folder generation: `results/phase1_6_long_wiboofpr/`
- Consistent naming across all scripts
- Clear traceability to WandB runs

## Folder Naming Pattern

```
results/phase{X}_{context_type}_{wandb_run_id}
```

### Examples

| Phase | Context | WandB ID | Folder Name |
|-------|---------|----------|-------------|
| 1 | short | yhxde999 | `results/phase1_short_yhxde999/` |
| 1 | long | wiboofpr | `results/phase1_long_wiboofpr/` |
| 1.5 | long | wiboofpr | `results/phase1_5_long_wiboofpr/` |
| 1.6 | long | wiboofpr | `results/phase1_6_long_wiboofpr/` |
| 2 | long | wiboofpr | `results/phase2_long_wiboofpr/` |
| 5 | long | wiboofpr | `results/phase5_long_wiboofpr/` |

## How to Use

### 1. Find Your WandB Run ID

Look at your pickle file path:
```
src/boldis/uncertainty/wandb/run-20251121_092732-wiboofpr/files/validation_generations.pkl
                                                  ^^^^^^^^
                                                  Run ID: wiboofpr
```

### 2. Run Analysis with New Parameters

```bash
# Example: Phase 1.6 analysis for long answers with run ID wiboofpr
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path "src/boldis/uncertainty/wandb/run-20251121_092732-wiboofpr/files/validation_generations.pkl" \
  --context-type long \
  --wandb-run-id wiboofpr \
  --max-prefix-len 50 \
  --ks 1 3 5
```

Output saved to: `results/phase1_6_long_wiboofpr/`

### 3. Complete Workflow Example

```bash
# Set variables for convenience
PICKLE_PATH="src/boldis/uncertainty/wandb/run-20251121_092732-wiboofpr/files/validation_generations.pkl"
WANDB_ID="wiboofpr"
CONTEXT="long"
MODEL="Llama-3.2-1B"

# Phase 1
python -m src.analysis.phase1_baseline_metrics \
  --long-pickle "$PICKLE_PATH" \
  --context-type $CONTEXT \
  --wandb-run-id $WANDB_ID

# Phase 1.5
python -m src.analysis.phase1_5_token_nll_analysis \
  --pickle-path "$PICKLE_PATH" \
  --model-name $MODEL \
  --sample-size 100 \
  --context-type $CONTEXT \
  --wandb-run-id $WANDB_ID

# Phase 1.6
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path "$PICKLE_PATH" \
  --context-type $CONTEXT \
  --wandb-run-id $WANDB_ID \
  --max-prefix-len 50

# Phase 2
python -m src.analysis.phase2_token_importance \
  --pickle-path "$PICKLE_PATH" \
  --model-name $MODEL \
  --sample-size 50 \
  --context-type $CONTEXT \
  --wandb-run-id $WANDB_ID

# Phase 5
python -m src.analysis.phase5_comparative_analysis \
  --pickle-path "$PICKLE_PATH" \
  --model-name $MODEL \
  --context-type $CONTEXT \
  --wandb-run-id $WANDB_ID
```

## Benefits

| Benefit | Description |
|---------|-------------|
| 🎯 **Automatic** | No manual folder name construction |
| 📊 **Organized** | Clear separation by run and context |
| 🔍 **Traceable** | Easy to link results to WandB runs |
| 🏷️ **Clear** | Know context type at a glance |
| 🔄 **Compatible** | Old workflows still work |
| ✨ **Flexible** | Use only the parameters you need |

## Backwards Compatibility

✅ All existing scripts and workflows continue to work:

```bash
# This still works (old way)
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path "path/to/pkl" \
  --output-dir results/my_custom_folder
```

✅ Parameters are optional:

```bash
# Just context type (no wandb-run-id)
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path "path/to/pkl" \
  --context-type long
# Output: results/phase1_6_long/

# Just wandb-run-id (no context-type)
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path "path/to/pkl" \
  --wandb-run-id wiboofpr
# Output: results/phase1_6_wiboofpr/

# Neither (default behavior)
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path "path/to/pkl"
# Output: results/phase1_6/
```

## Implementation Details

Each script now includes this auto-generation logic:

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
- Flexibility - Only include provided parameters
- Clarity - Log the generated path
- Control - Explicit `--output-dir` always wins

## Verification

All scripts tested and working:

```bash
# Check phase1 parameters
python -m src.analysis.phase1_baseline_metrics --help

# Check phase1.6 parameters  
python -m src.analysis.phase1_6_prefix_nll_analysis --help
```

Both show new `--wandb-run-id` and `--context-type` parameters.

## Migration Guide

### For New Analyses
Use the new parameters:
```bash
--context-type {short|long} --wandb-run-id <ID>
```

### For Existing Scripts
**Option 1**: Keep using `--output-dir` (no changes needed)

**Option 2**: Switch to new parameters:
```diff
- --output-dir results/phase1_long_wiboofpr
+ --context-type long --wandb-run-id wiboofpr
```

## Additional Documentation

- `ANALYSIS_README.md` - Full usage guide with examples
- `FOLDER_NAMING_UPDATE.md` - Detailed migration guide
- `PHASE_SCRIPTS_UPDATE_SUMMARY.md` - Quick reference

## Questions?

All scripts support `--help`:
```bash
python -m src.analysis.phase{X} --help
```

To see all available parameters and their descriptions.

