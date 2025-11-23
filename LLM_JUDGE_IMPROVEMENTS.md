# LLM Judge Improvements

## Overview

This document explains the improvements made to the LLM-as-judge evaluation system to make it more accurate and robust.

## The Problem

The original judge prompt had several issues:

### 1. **Example of Incorrect Evaluation**

```
Question: "Which female, former TV newsreader, was the first presenter of the TV programme, 'Top Gear'?"
Correct Answer: "angela rippon"
Model Response: "Who" is a female former TV newsreader, born in 1968, who was the first presenter of the TV programme, "Top Gear."
Judge Result: ✅ CORRECT (1.00) - WRONG!
```

The model response is clearly nonsensical (just repeating "Who" without answering), but the 8B judge marked it as correct, likely due to keyword matching.

### 2. **Original Prompt Issues**

The original prompt in `src/utils/utils.py` (`model_based_metric` function):

```python
prompt = f'We are assessing the quality of answers to the following question: {question}\n'
prompt += f"The expected answer is: {correct_answer}.\n"
prompt += f"The proposed answer is: {predicted_answer}\n"
prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
prompt += " Respond only with yes or no.\nResponse:"
```

**Problems:**
- ❌ No instruction formatting for instruct models (Llama-3.1-70B-Instruct)
- ❌ Too vague: "mean the same" is ambiguous
- ❌ No explicit criteria for evaluation
- ❌ Susceptible to keyword matching bias
- ❌ No guidance on handling nonsensical or incomplete answers

## The Solution

### Improved Prompt with Chat Formatting

For instruct models (detected by "instruct", "chat", or "llama-3" in model name), we now use proper chat formatting with system and user messages:

**System Message:**
```
You are an expert evaluator assessing whether a proposed answer correctly answers a given question.

Your task:
1. Compare the proposed answer to the expected answer(s)
2. Determine if they convey the same core information
3. Ignore minor differences in wording, but be strict about factual accuracy
4. If the proposed answer is nonsensical, incomplete, or factually incorrect, respond "no"
5. Only respond "yes" if the proposed answer correctly and clearly answers the question

Respond with ONLY "yes" or "no".
```

**User Message:**
```
Question: {question}
Expected answer: {correct_answer}
Proposed answer: {predicted_answer}

Does the proposed answer correctly answer the question with the same meaning as the expected answer?
```

### Key Improvements

1. ✅ **Proper instruction formatting** - Uses `tokenizer.apply_chat_template()` for instruct models
2. ✅ **Explicit evaluation criteria** - Clear 5-step process
3. ✅ **Strict factual accuracy** - Emphasizes factual correctness over keyword matching
4. ✅ **Handles nonsensical answers** - Explicitly instructs to reject incomplete/nonsensical responses
5. ✅ **Backward compatible** - Falls back to original prompt for non-instruct models

## How It Works

The improved `model_based_metric` function in `src/utils/utils.py`:

1. **Detects instruct models:**
   ```python
   use_chat_format = (
       'instruct' in model.model_name.lower() or 
       'chat' in model.model_name.lower() or
       'llama-3' in model.model_name.lower()
   )
   ```

2. **Applies chat template if available:**
   ```python
   if use_chat_format and hasattr(model, 'tokenizer'):
       prompt = model.tokenizer.apply_chat_template(messages, ...)
   ```

3. **Falls back gracefully** if chat template fails or for non-instruct models

## Testing the Improvements

### Option 1: Test with Llama-3.1-70B (Recommended)

Once the 70B model loading issue is resolved, run:

```bash
python recompute_accuracy_with_judge.py 5qvhbs97 llm_llama-3.1-70b
```

This will:
- Use the improved chat-formatted prompt
- Provide more accurate evaluations
- Be stricter about nonsensical answers

### Option 2: Test with Llama-3.1-8B (Faster, but less accurate)

```bash
python recompute_accuracy_with_judge.py 5qvhbs97 llm_llama-3.1-8b
```

The 8B model will also benefit from the improved prompt, though it may still make some errors due to its smaller size.

### Option 3: Use GPT-4 (Most Accurate)

If you have OpenAI API access:

```bash
python recompute_accuracy_with_judge.py 5qvhbs97 llm_gpt-4
```

GPT-4 is the most reliable judge and will use the improved prompt structure.

## Expected Improvements

With the improved prompt, you should see:

1. **Fewer false positives** - Nonsensical answers like the "Who" example should now be correctly marked as incorrect
2. **Better semantic understanding** - The judge should focus on meaning rather than keyword matching
3. **More consistent evaluations** - Clearer criteria lead to more consistent results
4. **Better use of instruct models** - Proper chat formatting leverages the model's instruction-following capabilities

## Comparing Results

After running with the improved prompt, you can compare:

1. **Old accuracy** (from original run with 8B judge):
   - Check `results_phase1_long_5qvhbs97/baseline_metrics.json`

2. **New accuracy** (from recomputation with improved prompt):
   - Check the new WandB run's metrics
   - Look for the `uncertainty_measures.pkl` file in the new run

3. **Manual inspection**:
   - Use the analysis notebook to inspect specific examples
   - Look for cases where the old judge was too lenient

## Code Changes

### Modified File: `src/utils/utils.py`

**Function:** `model_based_metric(predicted_answer, example, model)`

**Changes:**
- Added chat format detection for instruct models
- Implemented system/user message structure
- Added explicit evaluation criteria
- Maintained backward compatibility with original prompt

## Future Improvements

Potential enhancements:

1. **Few-shot examples** - Add example evaluations to the prompt
2. **Chain-of-thought** - Ask the judge to explain its reasoning before answering
3. **Confidence scores** - Extract confidence instead of just yes/no
4. **Multi-step evaluation** - First check if answer is complete, then check correctness
5. **Custom criteria per dataset** - Different evaluation criteria for different question types

## Troubleshooting

### If the judge still makes errors:

1. **Check the model size** - Larger models (70B) are more reliable than smaller ones (8B)
2. **Inspect the prompts** - Add logging to see the actual prompts being sent
3. **Try temperature 0** - Already set to 0.01 for deterministic outputs
4. **Use GPT-4** - If accuracy is critical, GPT-4 is the most reliable option

### If chat template fails:

The code automatically falls back to the original prompt format, so there should be no breaking changes.

## Summary

The improved LLM judge prompt:
- Uses proper instruction formatting for instruct models
- Provides clear evaluation criteria
- Is stricter about factual accuracy
- Handles nonsensical answers better
- Maintains backward compatibility

This should significantly improve the accuracy of LLM-based evaluation, especially for edge cases like nonsensical or incomplete answers.

