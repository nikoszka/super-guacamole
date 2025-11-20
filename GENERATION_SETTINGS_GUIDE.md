# Generation Settings Guide for Analysis

This guide provides recommended settings for generating pickle files that work optimally with the analysis pipeline.

## 🎯 Quick Start: Recommended Settings

### For SHORT Answers (ROUGE-based evaluation)

```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 400 \
  --num_generations 10 \
  --temperature 1.0 \
  --model_max_new_tokens 50 \
  --num_few_shot 5 \
  --brief_prompt short \
  --enable_brief \
  --metric squad \
  --use_context False \
  --answerable_only True \
  --get_training_set_generations True \
  --compute_p_true False \
  --entity YOUR_WANDB_ENTITY \
  --project nllSAR_short_answers
```

### For LONG Answers (LLM judge evaluation)

```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 400 \
  --num_generations 10 \
  --temperature 1.0 \
  --model_max_new_tokens 200 \
  --num_few_shot 5 \
  --brief_prompt detailed \
  --enable_brief \
  --metric llm_gpt-4 \
  --use_context True \
  --answerable_only True \
  --get_training_set_generations True \
  --compute_p_true False \
  --entity YOUR_WANDB_ENTITY \
  --project nllSAR_long_answers
```

---

## 📊 Parameter Explanations

### Core Model Settings

| Parameter | Short Answers | Long Answers | Explanation |
|-----------|---------------|--------------|-------------|
| `--model_name` | `Llama-3.2-1B` | `Llama-3.2-1B` | Model to use. Also try: `Llama-3.1-8B`, `Llama-2-7b-chat` |
| `--model_max_new_tokens` | **50** | **200** | Max tokens per answer. Short: 30-70, Long: 150-300 |
| `--temperature` | **1.0** | **1.0** | Temperature for sampling (greedy=0.0 is i=0 generation) |

**Why these values?**
- **50 tokens** for short: Enough for factoid answers (e.g., "The Battle of Hastings occurred in 1066")
- **200 tokens** for long: Allows detailed explanations with context
- **Temperature 1.0**: Creates diversity in high-temp generations for SAR/SE analysis

---

### Dataset & Sampling

| Parameter | Recommended | Explanation |
|-----------|-------------|-------------|
| `--dataset` | `trivia_qa`, `squad`, `bioasq` | QA dataset to use |
| `--num_samples` | **400** (min: 100, max: 1000) | Number of questions to process |
| `--num_generations` | **10** (min: 5, max: 20) | High-temp samples for SAR/SE |
| `--num_few_shot` | **5** | Few-shot examples in prompt |
| `--use_context` | Short: `False`, Long: `True` | Include context passages |
| `--answerable_only` | **True** | Skip unanswerable questions |

**Why these values?**
- **400 samples**: Good balance between statistical power and computation time
- **10 generations**: Sufficient for SAR (needs multiple samples) and SE analysis
- **Few-shot 5**: Provides enough in-context examples without excessive prompt length

---

### Answer Length Control

| Parameter | Short | Long | Explanation |
|-----------|-------|------|-------------|
| `--brief_prompt` | `short` | `detailed` | Controls answer style |
| `--enable_brief` | `True` | `True` | Enables brief instructions |
| `--brief_always` | `False` | `False` | Forces brief for all generations |

**Available `--brief_prompt` options:**
- `short`: "Give a short answer" → Encourages 5-20 token responses
- `sentence`: "Give a one-sentence answer" → 10-30 tokens
- `detailed`: "Provide a detailed answer" → 50-200+ tokens
- `default`: No specific length instruction
- `chat`: Conversational style

---

### Accuracy Evaluation

| Parameter | Short Answers | Long Answers | Explanation |
|-----------|---------------|--------------|-------------|
| `--metric` | `squad` | `llm_gpt-4` or `llm_llama-3.1-70b` | Accuracy metric |
| `--compute_accuracy_at_all_temps` | `True` | `True` | Compute accuracy for all generations |

**Metric choices:**
- **`squad`**: Fast ROUGE-based evaluation (good for short factoid answers)
- **`llm_gpt-4`**: Best quality LLM judge (requires OpenAI API key)
- **`llm_llama-3.1-70b`**: Open-source alternative LLM judge
- **`llm_llama-3-8b`**: Faster but less accurate LLM judge

---

### Analysis-Specific Settings

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| `--compute_p_true` | `False` | Not needed for G-NLL/RW-G-NLL analysis; saves time |
| `--get_training_set_generations` | `True` | Generates both train & validation splits |
| `--compute_uncertainties` | `True` | Triggers uncertainty computation automatically |

---

## 🚀 Live Test: Optimal Settings for Rich Analysis

For a comprehensive analysis demo, use these settings:

```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 200 \
  --num_generations 15 \
  --temperature 1.0 \
  --model_max_new_tokens 150 \
  --num_few_shot 5 \
  --brief_prompt detailed \
  --enable_brief \
  --metric llm_llama-3.1-70b \
  --use_context True \
  --answerable_only True \
  --get_training_set_generations False \
  --compute_p_true False \
  --compute_accuracy_at_all_temps True \
  --entity YOUR_WANDB_ENTITY \
  --project nllSAR_demo
```

**Why these settings?**
- ✅ **200 samples**: Fast enough for testing (~1-2 hours on GPU), sufficient for analysis
- ✅ **150 max tokens**: Middle ground between short/long, good token distribution
- ✅ **15 generations**: Rich sampling for SAR/SE with good diversity
- ✅ **detailed prompt**: Generates interesting multi-token answers with varied NLL patterns
- ✅ **llm_llama-3.1-70b judge**: Good accuracy without needing OpenAI API

---

## 📈 Analysis Coverage by Settings

| Analysis Phase | Needs Multiple Generations? | Min Tokens per Answer | Recommended Settings |
|----------------|------------------------------|------------------------|----------------------|
| **Phase 1** (Baseline) | No (uses i=0 only) | 5+ | Any settings work |
| **Phase 1.5** (Token NLL) | No | 10+ | `max_new_tokens >= 50` |
| **Phase 1.6** (Prefix NLL) | No | 10+ | `max_new_tokens >= 30` |
| **Phase 2** (Token Relevance) | No | 15+ | `max_new_tokens >= 50` |
| **Phase 5** (SAR/SE/RW-G-NLL) | **Yes** (for SAR/SE) | 10+ | `num_generations >= 10` |

---

## 🔧 What Changed in Your Code

We added **exact token tracking** to prevent mismatches:

### New Fields in Pickle Files:
- `most_likely_answer['token_ids']`: Token IDs generated by the model
- `most_likely_answer['tokens']`: String representation of each token
- High-temp generations also now stored as dicts with token info

### Backwards Compatibility:
- Analysis scripts **automatically fall back** to re-tokenization for old pickles
- New pickles guarantee **exact alignment** between tokens and log-likelihoods

---

## 🎯 Example Commands for Different Scenarios

### Scenario 1: Quick Test (15 minutes)
```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 50 \
  --num_generations 5 \
  --model_max_new_tokens 100 \
  --brief_prompt detailed \
  --metric squad \
  --compute_p_true False
```

### Scenario 2: Full Analysis Run (2-3 hours)
```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 400 \
  --num_generations 15 \
  --model_max_new_tokens 200 \
  --brief_prompt detailed \
  --metric llm_llama-3.1-70b \
  --use_context True
```

### Scenario 3: Short Answers Only (ROUGE evaluation)
```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 400 \
  --num_generations 10 \
  --model_max_new_tokens 40 \
  --brief_prompt short \
  --metric squad \
  --use_context False
```

---

## 📁 Output Files Location

After running generation:
```
src/nikos/uncertainty/wandb/run-<TIMESTAMP>-<ID>/files/
├── validation_generations.pkl    # Your main analysis file
├── train_generations.pkl          # (if --get_training_set_generations=True)
├── uncertainty_measures.pkl       # Computed uncertainties
└── experiment_details.pkl         # Run configuration
```

Use `validation_generations.pkl` for all analysis phases!

---

## ✅ Verification Checklist

After generation, verify your pickle has everything:

```python
import pickle

with open('validation_generations.pkl', 'rb') as f:
    data = pickle.load(f)

first_example = list(data.values())[0]
mla = first_example['most_likely_answer']

# Check for new fields
assert 'token_ids' in mla, "❌ Missing token_ids - old generation code?"
assert 'tokens' in mla, "❌ Missing tokens - old generation code?"
assert len(mla['tokens']) == len(mla['token_log_likelihoods']), "❌ Token count mismatch!"

print(f"✅ Pickle verified!")
print(f"✅ Example has {len(mla['tokens'])} tokens")
print(f"✅ First 5 tokens: {mla['tokens'][:5]}")
```

---

## 🎯 Tips for Better Analysis

1. **Token Distribution**: Aim for mean token length 20-100 for interesting patterns
2. **Accuracy Balance**: Target ~60-80% accuracy for good correct/incorrect separation
3. **High-Temp Diversity**: Use `temperature=1.0` and `num_generations >= 10` for SAR/SE
4. **Avoid Edge Cases**: Skip very short (<5 tokens) or interrupted (max_tokens) generations
5. **GPU Memory**: If OOM, reduce `num_samples` or use smaller model

---

## 📞 Need Help?

- Check ANALYSIS_README.md for running analysis phases
- See README.md for general setup
- Verify pickle structure with the verification script above

