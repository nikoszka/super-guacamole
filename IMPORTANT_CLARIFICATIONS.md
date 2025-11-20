# ⚠️ Important Clarifications: Temperature & Entity

## 🔥 Temperature Settings (CORRECTED)

### ❌ Previous Recommendation (WRONG)
```bash
--num_generations 10 \
--temperature 1.0
```
**Problem:** This makes ALL generations (including i=0 baseline) stochastic!

### ✅ Corrected Recommendation
```bash
# For baseline & token-level analysis (Phases 1-2)
--num_generations 1 \
--temperature 0.0
```

---

## 🌡️ Why This Matters

### Current Code Behavior
The code uses **the same temperature for ALL generations**:

```python
for i in range(num_generations):
    temperature = args.temperature  # ← Same for i=0, i=1, i=2, ...
    answer = model.predict(prompt, temperature)
```

**This means:**
- `--temperature 0.0` → All generations are greedy (identical)
- `--temperature 1.0` → All generations are stochastic (including i=0!)

### What You Actually Want

For token-level analysis (Phase 1, 1.5, 1.6, 2):
- ✅ Pure **greedy baseline** (i=0 with temp=0.0)
- ✅ Deterministic, reproducible
- ✅ Represents model's most confident answer

**Use:**
```bash
--num_generations 1 --temperature 0.0
```

---

## 🎓 Few-Shot vs Temperature (Clarification)

### ❓ Your Question:
> "Why are we doing few shots? This is supposed to be a greedy run too, can you tell me why the few shots in the parameters?"

### ✅ Answer:
**Few-shot prompting and greedy decoding are INDEPENDENT!**

| Concept | What it controls | Setting |
|---------|------------------|---------|
| **Few-shot prompting** | Prompt **format/quality** | `--num_few_shot 5` |
| **Temperature** | **Sampling randomness** | `--temperature 0.0` |

### Few-Shot Prompt Example

```
Question: What is the capital of France?
Answer: Paris

Question: Who wrote Romeo and Juliet?
Answer: William Shakespeare

... [3 more examples] ...

Question: What caused the Great Depression?  ← YOUR ACTUAL QUESTION
Answer: [MODEL GENERATES HERE WITH TEMP=0.0]
```

**Why use few-shot?**
- Helps model understand the **task format**
- Improves **answer quality**
- Standard practice in LLM prompting
- **Completely unrelated to greedy vs stochastic**

You can (and should!) use few-shot even with greedy decoding:
```bash
--num_few_shot 5 \      # ← Few-shot for quality
--temperature 0.0 \     # ← Greedy for determinism
--num_generations 1     # ← Single answer
```

---

## 🏷️ WandB Entity (ANSWERED)

### ❓ Your Question:
> "What is this --entity wandb entity? What should I add here? Or will it be filled?"

### ✅ Answer:
Your wandb entity is: **`nikosteam`**

**What it is:**
- Your wandb **username or team name**
- Used to organize your experiment runs
- Appears in your wandb URL: `wandb.ai/nikosteam/PROJECT/runs/...`

**How to set it:**

#### Option 1: Command Argument (Simple)
```bash
python -m src.generate_answers \
  --entity nikosteam \
  ...
```

#### Option 2: Environment Variable (Persistent)
```powershell
# PowerShell (Windows)
$env:WANDB_SEM_UNC_ENTITY = "nikosteam"

# Then omit --entity from commands
```

#### Option 3: Let wandb auto-detect
If you don't specify, wandb uses your logged-in username automatically.

---

## 📋 Corrected Quick Commands

### For Your First Run (Token Analysis)
```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 400 \
  --num_generations 1 \
  --temperature 0.0 \
  --model_max_new_tokens 150 \
  --num_few_shot 5 \
  --brief_prompt detailed \
  --metric llm_llama-3.1-70b \
  --use_context True \
  --entity nikosteam \
  --compute_p_true False
```

**This enables:**
- ✅ Phase 1: Baseline metrics ✅
- ✅ Phase 1.5: Token-level NLL ✅
- ✅ Phase 1.6: Prefix-level NLL ✅
- ✅ Phase 2: Token relevance (RW-G-NLL) ✅
- ✅ Phase 5: Baseline comparison (G-NLL, Avg NLL, RW-G-NLL) ✅
- ❌ Phase 5: SAR & SE (need multiple samples) ❌

### Later: For SAR/SE Analysis (Optional)
```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 400 \
  --num_generations 10 \
  --temperature 1.0 \
  --model_max_new_tokens 150 \
  --num_few_shot 5 \
  --brief_prompt detailed \
  --metric llm_llama-3.1-70b \
  --use_context True \
  --entity nikosteam
```

**Note:** i=0 will be stochastic (not pure greedy)!

---

## 🎯 Key Takeaways

1. **Few-shot** (`--num_few_shot 5`) improves answer quality → Always use it!
2. **Temperature** (`--temperature 0.0`) controls randomness → Use 0.0 for greedy!
3. **Entity** (`--entity nikosteam`) is your wandb username → Use your actual entity!
4. **Start simple**: `--num_generations 1 --temperature 0.0` for baseline analysis
5. **Run twice** if you need both greedy baseline AND SAR/SE analysis

---

## 📊 What Analysis Needs What Settings

| Analysis Goal | num_generations | temperature | Why |
|---------------|-----------------|-------------|-----|
| Baseline metrics (G-NLL) | 1 | 0.0 | Pure greedy baseline |
| Token-level NLL | 1 | 0.0 | Deterministic tokens |
| RW-G-NLL | 1 | 0.0 | Single greedy answer |
| SAR & SE | 10+ | 1.0 | Need diverse samples |
| All baselines + RW-G-NLL | 1 | 0.0 | ✅ Start here |
| Full comparison (+ SAR/SE) | 10+ | 1.0 | Run separately |

---

## ✅ Updated Documentation

All guides have been corrected:
- ✅ `GENERATION_SETTINGS_GUIDE.md` - Fixed temperature & entity
- ✅ `QUICK_START.md` - Fixed temperature & entity
- ✅ `IMPORTANT_CLARIFICATIONS.md` - This file!

**You're all set to generate!** 🚀

