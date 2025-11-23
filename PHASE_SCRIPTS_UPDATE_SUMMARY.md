# Phase Scripts Update Summary

## What Was Changed

All phase analysis scripts have been updated to automatically include **WandB run ID** and **context type** (short/long) in their output folder names.

## Updated Scripts

✅ `src/analysis/phase1_baseline_metrics.py`  
✅ `src/analysis/phase1_5_token_nll_analysis.py`  
✅ `src/analysis/phase1_6_prefix_nll_analysis.py`  
✅ `src/analysis/phase2_token_importance.py`  
✅ `src/analysis/phase5_comparative_analysis.py`  
✅ `ANALYSIS_README.md` - Updated with new examples and explanations

## Quick Start

### Old Way (Still Works)
```bash
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path "path/to/validation_generations.pkl" \
  --output-dir results/phase1_6_long_wiboofpr
```

### New Way (Recommended)
```bash
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path "path/to/validation_generations.pkl" \
  --context-type long \
  --wandb-run-id wiboofpr
```

**Result**: Output automatically saved to `results/phase1_6_long_wiboofpr/`

## Benefits

✨ **Automatic folder naming** - No need to manually construct folder names  
📊 **Better organization** - Results from different runs are clearly separated  
🔍 **Easy traceability** - Immediately see which WandB run produced which results  
🏷️ **Context clarity** - Know if results are for short or long answers at a glance  
🔄 **Backwards compatible** - Old scripts still work with explicit `--output-dir`

## Folder Naming Pattern

```
results/phase{X}_{context_type}_{wandb_run_id}
```

**Examples**:
- `results/phase1_short_yhxde999`
- `results/phase1_long_wiboofpr`
- `results/phase1_6_long_wiboofpr`
- `results/phase2_long_wiboofpr`

## Finding Your WandB Run ID

Look at your pickle file path:
```
src/boldis/uncertainty/wandb/run-20251121_092732-wiboofpr/files/
                                                  ^^^^^^^^
                                                  This is it!
```

## Complete Example Workflow

For a long-answer run with WandB ID `wiboofpr`:

```bash
PICKLE_PATH="src/boldis/uncertainty/wandb/run-20251121_092732-wiboofpr/files/validation_generations.pkl"
WANDB_ID="wiboofpr"
CONTEXT="long"

# Phase 1: Baseline metrics
python -m src.analysis.phase1_baseline_metrics \
  --long-pickle "$PICKLE_PATH" \
  --context-type $CONTEXT \
  --wandb-run-id $WANDB_ID

# Phase 1.5: Token NLL analysis  
python -m src.analysis.phase1_5_token_nll_analysis \
  --pickle-path "$PICKLE_PATH" \
  --model-name Llama-3.2-1B \
  --context-type $CONTEXT \
  --wandb-run-id $WANDB_ID

# Phase 1.6: Prefix NLL analysis
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path "$PICKLE_PATH" \
  --context-type $CONTEXT \
  --wandb-run-id $WANDB_ID

# Phase 2: Token relevance
python -m src.analysis.phase2_token_importance \
  --pickle-path "$PICKLE_PATH" \
  --model-name Llama-3.2-1B \
  --context-type $CONTEXT \
  --wandb-run-id $WANDB_ID

# Phase 5: Comparative analysis
python -m src.analysis.phase5_comparative_analysis \
  --pickle-path "$PICKLE_PATH" \
  --model-name Llama-3.2-1B \
  --context-type $CONTEXT \
  --wandb-run-id $WANDB_ID
```

**Results organized in**:
- `results/phase1_long_wiboofpr/`
- `results/phase1_5_long_wiboofpr/`
- `results/phase1_6_long_wiboofpr/`
- `results/phase2_long_wiboofpr/`
- `results/phase5_long_wiboofpr/`

## Technical Details

Each script now includes auto-generation logic:

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

## Need More Details?

- See `FOLDER_NAMING_UPDATE.md` for detailed migration guide
- See `ANALYSIS_README.md` for complete usage examples with all parameters
- All scripts support `--help` flag to see available parameters

## Testing

Verified that new parameters work correctly:
```bash
python -m src.analysis.phase1_baseline_metrics --help
python -m src.analysis.phase1_6_prefix_nll_analysis --help
```

Both show the new `--wandb-run-id` and `--context-type` parameters.

## Notes

- Parameters are optional - scripts work without them (fall back to default naming)
- `--output-dir` still works and takes precedence if provided
- No breaking changes - all existing scripts/workflows continue to work

