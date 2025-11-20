# Generation Settings Guide for Analysis

This guide provides recommended settings for generating pickle files that work optimally with the analysis pipeline.

## 🎯 Quick Start: Recommended Settings

### For SHORT Answers (ROUGE-based evaluation)

```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 400 \
  --num_generations 1 \
  --temperature 0.0 \
  --model_max_new_tokens 50 \
  --num_few_shot 5 \
  --brief_prompt short \
  --enable_brief \
  --metric squad \
  --use_context False \
  --answerable_only True \
  --get_training_set_generations True \
  --compute_p_true False \
  --entity nikosteam \
  --project nllSAR_short_answers
```

### For LONG Answers (LLM judge evaluation)

```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 400 \
  --num_generations 1 \
  --temperature 0.0 \
  --model_max_new_tokens 200 \
  --num_few_shot 5 \
  --brief_prompt detailed \
  --enable_brief \
  --metric llm_gpt-4 \
  --use_context True \
  --answerable_only True \
  --get_training_set_generations True \
  --compute_p_true False \
  --entity nikosteam \
  --project nllSAR_long_answers
```

---

## 📊 Parameter Explanations

### Core Model Settings

| Parameter | Short Answers | Long Answers | Explanation |
|-----------|---------------|--------------|-------------|
| `--model_name` | `Llama-3.2-1B` | `Llama-3.2-1B` | Model to use. Also try: `Llama-3.1-8B`, `Llama-2-7b-chat` |
| `--model_max_new_tokens` | **50** | **200** | Max tokens per answer. Short: 30-70, Long: 150-300 |
| `--temperature` | **0.0** | **0.0** | Temperature for sampling (0.0 = greedy/deterministic) |
| `--num_generations` | **1** | **1** | Number of samples (1 = greedy only, 10+ for SAR/SE) |

**Why these values?**
- **50 tokens** for short: Enough for factoid answers (e.g., "The Battle of Hastings occurred in 1066")
- **200 tokens** for long: Allows detailed explanations with context
- **Temperature 0.0**: Pure greedy decoding for baseline/token-level analysis
- **1 generation**: Focuses on greedy baseline (increase to 10+ only if you need SAR/SE)

---

### Dataset & Sampling

| Parameter | Recommended | Explanation |
|-----------|-------------|-------------|
| `--dataset` | `trivia_qa`, `squad`, `bioasq` | QA dataset to use |
| `--num_samples` | **400** (min: 100, max: 1000) | Number of questions to process |
| `--num_generations` | **1** (baseline) or **10+** (SAR/SE) | Number of samples per question |
| `--num_few_shot` | **5** | Few-shot examples in prompt |
| `--use_context` | Short: `False`, Long: `True` | Include context passages |
| `--answerable_only` | **True** | Skip unanswerable questions |

**Why these values?**
- **400 samples**: Good balance between statistical power and computation time
- **1 generation**: For baseline/token analysis (Phases 1-2). Use 10+ only for SAR/SE (Phase 5)
- **Few-shot 5**: Provides enough in-context examples without excessive prompt length (unrelated to temperature!)

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
  --num_generations 1 \
  --temperature 0.0 \
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
  --entity nikosteam \
  --project nllSAR_demo
```

**Why these settings?**
- ✅ **200 samples**: Fast enough for testing (~30-60 min on GPU), sufficient for analysis
- ✅ **150 max tokens**: Middle ground between short/long, good token distribution
- ✅ **1 generation + temp 0.0**: Pure greedy decoding for baseline/token analysis
- ✅ **detailed prompt**: Generates interesting multi-token answers with varied NLL patterns
- ✅ **llm_llama-3.1-70b judge**: Good accuracy without needing OpenAI API

**Note:** For SAR/SE analysis (Phase 5), you'll need multiple generations. See "Temperature & Sampling Strategy" below.

---

## 📈 Analysis Coverage by Settings

| Analysis Phase | Needs Multiple Generations? | Min Tokens per Answer | Recommended Settings |
|----------------|------------------------------|------------------------|----------------------|
| **Phase 1** (Baseline) | No (uses i=0 only) | 5+ | `num_gen=1, temp=0.0` |
| **Phase 1.5** (Token NLL) | No | 10+ | `num_gen=1, temp=0.0, max_tokens>=50` |
| **Phase 1.6** (Prefix NLL) | No | 10+ | `num_gen=1, temp=0.0, max_tokens>=30` |
| **Phase 2** (Token Relevance / RW-G-NLL) | No | 15+ | `num_gen=1, temp=0.0, max_tokens>=50` |
| **Phase 5 (partial)** (Baselines + RW-G-NLL) | No | 10+ | `num_gen=1, temp=0.0` |
| **Phase 5 (full)** (+ SAR/SE) | **Yes** | 10+ | `num_gen=10+, temp=1.0` |

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

### Scenario 1: Quick Test (10-15 minutes)
```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 50 \
  --num_generations 1 \
  --temperature 0.0 \
  --model_max_new_tokens 100 \
  --brief_prompt detailed \
  --metric squad \
  --compute_p_true False \
  --entity nikosteam
```
**Runs:** Phase 1, 1.5, 1.6, 2, 5 (partial - no SAR/SE)

### Scenario 2: Full Baseline Analysis (30-60 min)
```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 400 \
  --num_generations 1 \
  --temperature 0.0 \
  --model_max_new_tokens 150 \
  --brief_prompt detailed \
  --metric llm_llama-3.1-70b \
  --use_context True \
  --entity nikosteam
```
**Runs:** Phase 1, 1.5, 1.6, 2, 5 (baselines + RW-G-NLL only)

### Scenario 3: Short Answers (ROUGE, 20-40 min)
```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 400 \
  --num_generations 1 \
  --temperature 0.0 \
  --model_max_new_tokens 40 \
  --brief_prompt short \
  --metric squad \
  --use_context False \
  --entity nikosteam
```
**Runs:** All phases except SAR/SE

### Scenario 4: High-Temp for SAR/SE (2-3 hours)
```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 400 \
  --num_generations 10 \
  --temperature 1.0 \
  --model_max_new_tokens 150 \
  --brief_prompt detailed \
  --metric llm_llama-3.1-70b \
  --use_context True \
  --entity nikosteam
```
**Runs:** Phase 5 (SAR & SE analysis)  
**Note:** i=0 will be stochastic, not pure greedy!

---

---

## 🌡️ Temperature & Sampling Strategy

### Understanding Temperature and Generations

**Temperature** controls randomness:
- `--temperature 0.0` → **Greedy** (deterministic, always picks highest probability token)
- `--temperature 1.0` → **Stochastic** (samples from full probability distribution)

**Number of Generations** (`--num_generations`):
- Sets how many answers to generate per question
- `i=0` is the baseline answer (uses `--temperature`)
- `i>0` are additional samples (also use `--temperature`)

### ⚠️ Important Limitation

**The current code uses the same temperature for ALL generations** (including i=0). This means:

```python
# All generations use args.temperature
for i in range(num_generations):
    answer = model.predict(prompt, temperature=args.temperature)
```

### 📋 Recommended Strategies

#### **Strategy 1: Pure Greedy (Recommended for token analysis)**
```bash
--num_generations 1 \
--temperature 0.0
```
- ✅ Best for Phases 1, 1.5, 1.6, 2 (baseline & token-level analysis)
- ✅ Deterministic, reproducible results
- ✅ Focused on the model's most confident answer
- ❌ Cannot compute SAR/SE (need multiple samples)

#### **Strategy 2: Multiple Greedy Runs (Not useful)**
```bash
--num_generations 10 \
--temperature 0.0
```
- ❌ All 10 generations will be **identical**
- ❌ Wastes compute time
- ❌ SAR/SE will fail (no diversity)

#### **Strategy 3: Two Separate Runs (For full analysis)**
```bash
# Run 1: Greedy baseline
--num_generations 1 --temperature 0.0

# Run 2: High-temp for SAR/SE
--num_generations 10 --temperature 1.0
```
- ✅ Clean separation of greedy vs stochastic
- ✅ Can run all analysis phases
- ⚠️ Requires two separate generation runs

#### **Strategy 4: Low Non-Zero Temperature (Compromise)**
```bash
--num_generations 10 \
--temperature 0.1
```
- ⚠️ i=0 is **mostly greedy** but slightly stochastic
- ✅ i>0 have some diversity for SAR/SE
- ⚠️ Not pure greedy baseline
- ⚠️ May affect baseline metrics slightly

### 💡 Our Recommendation

**For your first run (token-level analysis focus):**
```bash
--num_generations 1 \
--temperature 0.0 \
--num_samples 400
```

This gives you:
- ✅ Phase 1: Baseline metrics (G-NLL, Avg NLL, Perplexity) ✅
- ✅ Phase 1.5: Token-level NLL analysis ✅
- ✅ Phase 1.6: Prefix-level NLL analysis ✅
- ✅ Phase 2: Token relevance (RW-G-NLL) ✅
- ❌ Phase 5: SAR & SE (need multiple samples) ❌

**Later, if you want SAR/SE analysis:**
Run a second generation with `--num_generations 10 --temperature 1.0`

---

## 🏷️ WandB Entity & Project Settings

### What is `--entity`?
Your **wandb username or team name**. Based on your previous runs, you're using: **`nikosteam`**

### Setting Your Entity

**Option 1: Environment Variable (Recommended)**
```powershell
# PowerShell (Windows)
$env:WANDB_SEM_UNC_ENTITY = "nikosteam"

# Then omit --entity from commands (auto-filled)
```

**Option 2: Command Argument**
```bash
--entity nikosteam
```

**Option 3: Let wandb auto-detect** (uses your logged-in username)

### Project Names
- `--project nllSAR_short` → Short answer experiments
- `--project nllSAR_long` → Long answer experiments  
- `--project nllSAR_demo` → Test/demo runs

Your runs appear at: `wandb.ai/nikosteam/PROJECT_NAME/runs/...`

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
3. **Start with Greedy**: Use `--num_generations 1 --temperature 0.0` for initial analysis
4. **High-Temp for SAR/SE**: Only use `temperature=1.0` + `num_generations >= 10` if you need SAR/SE
5. **Few-Shot Helps**: Always use `--num_few_shot 5` (improves answer quality, unrelated to temperature)
6. **Avoid Edge Cases**: Skip very short (<5 tokens) or interrupted (max_tokens) generations
7. **GPU Memory**: If OOM, reduce `num_samples` or use smaller model

---

## 📞 Need Help?

- Check ANALYSIS_README.md for running analysis phases
- See README.md for general setup
- Verify pickle structure with the verification script above

