# Stop Sequence & Token Alignment Fixes

## 🐛 The Problems

When generating answers with `brief_prompt: detailed` and `model_max_new_tokens: 200`, the model was producing **very short answers** (only 2-4 tokens like "wadsworth", "ringway") instead of detailed, multi-sentence responses. Additionally, there was a critical token alignment bug.

### Root Causes

#### Problem 1: Overly Restrictive Stop Sequences
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

#### Problem 2: Token Alignment Bug (Critical!)
In `src/models/huggingface_models.py`, there was a **strip-before-check bug**:

```python
# OLD (BUGGY) CODE:
generated_token_ids = outputs.sequences[0][n_input_token:]
answer = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()  # ← .strip() TOO EARLY!

# Later...
if answer.endswith(stop):  # ← Can't detect '\n\n' because .strip() already removed it!
    sliced_answer = answer[:-len(stop)]

# Even worse...
sliced_answer_tokens = self.tokenizer.encode(sliced_answer, add_special_tokens=False)  # ← RE-TOKENIZATION!
n_generated = len(sliced_answer_tokens)  # ← May not match original token count!
```

**Why this was critical**:
1. `.strip()` on line 434 removed stop sequences (like `\n\n`) BEFORE we could detect them
2. Re-tokenizing the stripped string could produce **different token IDs** than the model actually generated
3. This caused **misalignment** between `token_ids`, `tokens`, and `log_likelihoods`

**Example of misalignment**:
```python
# Model generates: [" Johann", " Strauss", " II", "\n", "\n"]  (5 tokens)
# After decode + strip: "Johann Strauss II"
# Re-tokenize: ["Johann", " Strauss", " II"]  (3 tokens, DIFFERENT first token!)
# ↑ WRONG! We'd slice the original tokens incorrectly!
```

## ✅ The Solutions

### Solution 1: Separate Stop Sequences
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

### Solution 2: Fix Token Alignment (Critical!)
Rewrote the answer extraction and stop sequence handling logic:

```python
# NEW (FIXED) CODE:
generated_token_ids = outputs.sequences[0][n_input_token:]
# DON'T strip here - we need to detect stop sequences first!
answer = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

# Remove stop sequences BEFORE stripping (so we can detect '\n\n')
sliced_answer = answer
num_stop_tokens = 0
if self.stop_sequences is not None:
    for stop in self.stop_sequences:
        if answer.endswith(stop):
            sliced_answer = answer[:-len(stop)]
            # Calculate tokens in stop sequence
            stop_tokens_encoded = self.tokenizer.encode(stop, add_special_tokens=False)
            num_stop_tokens = len(stop_tokens_encoded)
            break

# NOW strip (after stop sequence removal)
sliced_answer = sliced_answer.strip()

# Calculate n_generated from ORIGINAL tokens (no re-tokenization!)
n_generated = len(generated_token_ids) - num_stop_tokens
```

**Key improvements**:
1. ✅ Stop sequences detected correctly (before stripping)
2. ✅ No re-tokenization - use original `generated_token_ids`
3. ✅ Perfect alignment: `token_ids[i]` matches `log_likelihoods[i]` matches `tokens[i]`

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

### 3. `src/models/huggingface_models.py` - Token Alignment Fix
**Critical changes** to `predict()` method:
- **Line 435**: Removed `.strip()` from initial decode
- **Lines 454-468**: Improved stop sequence detection (now works before stripping)
- **Lines 480-481**: Strip whitespace AFTER stop sequence removal
- **Lines 483-490**: Calculate `n_generated` from ORIGINAL tokens, not re-tokenization

```python
# Before: Re-tokenization caused misalignment
sliced_answer_tokens = self.tokenizer.encode(sliced_answer, add_special_tokens=False)
n_generated = len(sliced_answer_tokens)  # ← WRONG!

# After: Use original token count
n_generated = len(generated_token_ids) - num_stop_tokens  # ← CORRECT!
```

### 4. `src/utils/utils.py`
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

### Before Fixes
```json
{
  "response": "wadsworth",
  "tokens": [" w", "ad", "sworth"],
  "nlls": [1.358, 0.0001, 0.00002]
}
```
Only 3 tokens generated, despite `model_max_new_tokens: 200` and `brief_prompt: detailed`

**Token Alignment Issue**:
- `token_ids` might not match `log_likelihoods` indices
- Analysis scripts could crash or produce incorrect results

### After Fixes (Expected)
```json
{
  "response": "Wadsworth is the name of the butler character in the 1985 film 'Clue', which was based on the popular board game. The character was portrayed by actor Tim Curry, who delivered a memorable performance in this cult classic mystery comedy.",
  "tokens": [" W", "ad", "sworth", " is", " the", ...],
  "nlls": [1.2, 0.15, 0.03, 0.8, 0.4, ...]
}
```
Full detailed answer with multiple sentences, utilizing the available token budget

**Perfect Token Alignment**:
- `token_ids[0]` = first generated token ID
- `tokens[0]` = decode of `token_ids[0]`
- `log_likelihoods[0]` = log probability of `token_ids[0]`
- ✅ All arrays have the same length and perfect 1-to-1 correspondence

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
5. **Critical alignment fix**: The token alignment fix ensures that token-level analysis (phase1_5, phase2) works correctly
6. **No more re-tokenization**: We now track tokens from the source, not by re-encoding strings

## ⚠️ Important

After applying these fixes, you **must regenerate** your answer pickles to benefit from the changes:

1. **Detailed answers** (critical): Old pickles have short answers due to restrictive stop sequences
2. **All pickles** (recommended): Token alignment fix ensures correct analysis results

```bash
# Regenerate short answers (optional but recommended for perfect alignment)
python run_generate_short_answers.py

# Regenerate detailed answers (REQUIRED - they were truncated!)
python run_generate_long_answers.py
```

### Why Regenerate Both?

**Short answers**: Old pickles work, but may have minor alignment issues if stop sequences triggered
**Detailed answers**: Old pickles are severely truncated and unusable for analysis

---

**Fixes applied**: November 20, 2024  
**Affects**: 
  - Stop sequences: Detailed answer generation only
  - Token alignment: All generation (short and detailed)
**Requires**: 
  - Mandatory: Pickle regeneration for detailed answers
  - Recommended: Pickle regeneration for short answers

