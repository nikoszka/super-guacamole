# Stop Sequence Fix for Detailed Answers

## 🐛 The Problem

When generating answers with `brief_prompt: detailed` and `model_max_new_tokens: 200`, the model was producing **very short answers** (only 2-4 tokens like "wadsworth", "ringway") instead of detailed, multi-sentence responses.

### Root Cause

The issue was in the **stop sequences** defined in `src/models/base_model.py`:

```python
STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', 'Question:', 'Context:', 'Answer:']
```

The model would:
1. Generate a short answer like "wadsworth"
2. Naturally add newlines at the end: "wadsworth\n\n"
3. Hit the `'\n\n'` stop sequence and **immediately stop generation**
4. Never get a chance to generate the detailed, multi-sentence response requested

This is why even with:
- `model_max_new_tokens: 200` (plenty of tokens available)
- `brief_prompt: detailed` (instruction to be detailed)
- Temperature `0.0` (greedy, deterministic)

The model was still stopping after just a few tokens.

## ✅ The Solution

Created **separate stop sequences for detailed vs. short answers**:

### For Short Answers (default)
```python
STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', 'Question:', 'Context:', 'Answer:']
```
- Includes `'\n\n'` to stop after brief responses
- Appropriate for `brief_prompt: short`

### For Detailed Answers (new)
```python
STOP_SEQUENCES_DETAILED = ['\n\n\n\n', 'Question:', 'Context:', 'Answer:']
```
- **Removed** `'\n\n'` and `'\n\n\n'` to allow multi-paragraph responses
- **Kept** `'\n\n\n\n'` (4 newlines) for extreme separation cases
- **Kept** `'Question:'`, `'Context:'`, `'Answer:'` to prevent generating new Q&A pairs
- **EOS token** still added automatically by the model

## 📝 Changes Made

### 1. `src/models/base_model.py`
Added new constant for detailed mode:
```python
STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', 'Question:', 'Context:', 'Answer:']
STOP_SEQUENCES_DETAILED = ['\n\n\n\n', 'Question:', 'Context:', 'Answer:']  # Relaxed for detailed answers
```

### 2. `src/models/huggingface_models.py`
- Imported `STOP_SEQUENCES_DETAILED`
- Updated `__init__` to handle `stop_sequences='detailed'`:
```python
if stop_sequences == 'default':
    stop_sequences = STOP_SEQUENCES
elif stop_sequences == 'detailed':
    stop_sequences = STOP_SEQUENCES_DETAILED
```

### 3. `src/utils/utils.py`
Updated `init_model()` to select stop sequences based on `brief_prompt`:
```python
def init_model(args):
    mn = args.model_name
    if 'llama' in mn.lower() or 'falcon' in mn or 'mistral' in mn.lower():
        # Use relaxed stop sequences for detailed answers to allow multi-sentence responses
        # For detailed mode, we remove '\n\n' and '\n\n\n' to prevent premature stopping
        if args.brief_prompt == 'detailed':
            stop_sequences = 'detailed'
        else:
            stop_sequences = 'default'
        
        model = HuggingfaceModel(
            mn, stop_sequences=stop_sequences,
            max_new_tokens=args.model_max_new_tokens)
    else:
        raise ValueError(f'Unknown model_name `{mn}`.')
    return model
```

## 🎯 Impact

### Before Fix
```json
{
  "response": "wadsworth",
  "tokens": [" w", "ad", "sworth"],
  "nlls": [1.358, 0.0001, 0.00002]
}
```
Only 3 tokens generated, despite `model_max_new_tokens: 200` and `brief_prompt: detailed`

### After Fix (Expected)
```json
{
  "response": "Wadsworth is the name of the butler character in the 1985 film 'Clue', which was based on the popular board game. The character was portrayed by actor Tim Curry, who delivered a memorable performance in this cult classic mystery comedy.",
  "tokens": [" W", "ad", "sworth", " is", " the", ...],
  "nlls": [1.2, 0.15, 0.03, 0.8, 0.4, ...]
}
```
Full detailed answer with multiple sentences, utilizing the available token budget

## 🚀 Usage

No changes needed to your generation commands! The fix automatically applies based on `--brief_prompt`:

### Short Answers
```bash
python run_generate_short_answers.py
# Uses default stop sequences (includes '\n\n')
```

### Detailed/Long Answers  
```bash
python run_generate_long_answers.py
# Uses relaxed stop sequences (removes '\n\n')
```

## 🔍 How to Verify

After regenerating your pickles with this fix:

1. **Check token counts**:
   ```bash
   python src/analysis/phase1_baseline_metrics.py \
       --pickle_path "path/to/validation_generations.pkl" \
       --output_dir "results/phase1_long"
   ```
   You should see much higher mean/max token counts for detailed answers.

2. **Inspect examples**:
   ```bash
   cat results/phase1_5_long/sentence_level_nll_examples.json
   ```
   The responses should now be full sentences/paragraphs, not just 2-4 word answers.

3. **Visual inspection**:
   Open the pickle file and spot-check a few `most_likely_answer['response']` entries. They should be detailed and complete.

## 📊 Expected Metrics

### Short Answers (`brief_prompt: short`)
- Mean tokens: 5-10
- Max tokens: 15-20
- Style: Brief phrases, single words

### Detailed Answers (`brief_prompt: detailed`)
- Mean tokens: 30-80
- Max tokens: 100-150
- Style: Complete sentences, multi-sentence paragraphs

## 🔧 Technical Notes

1. **No impact on short answers**: The default stop sequences are unchanged for `brief_prompt: short`
2. **Backward compatible**: Existing short answer pickles work without regeneration
3. **Model-agnostic**: Works with any LLM (Llama, Falcon, Mistral, etc.)
4. **Temperature-agnostic**: Works for both greedy (0.0) and stochastic (1.0) generation

## ⚠️ Important

After applying this fix, you **must regenerate** your detailed answer pickles to benefit from the change. Old pickles were generated with the restrictive stop sequences and will still contain short answers.

```bash
# Regenerate detailed answers
python run_generate_long_answers.py
```

---

**Fix applied**: November 20, 2024  
**Affects**: Detailed answer generation only  
**Requires**: Pickle regeneration for detailed answers

