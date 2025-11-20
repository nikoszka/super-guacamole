# Token Alignment Fix - Quick Summary

## 🐛 The Bug

**Strip-before-check bug** in `src/models/huggingface_models.py`:
- Line 434 was doing `.strip()` immediately after decoding
- This removed stop sequences (like `\n\n`) before we could detect them
- Then re-tokenization produced different tokens than the model actually generated
- Result: **Misalignment** between `token_ids`, `tokens`, and `log_likelihoods`

## ✅ The Fix

Changed the order of operations:
1. Decode (NO strip)
2. Detect and remove stop sequences
3. Strip whitespace
4. Calculate `n_generated` from ORIGINAL tokens (no re-tokenization)

## 📊 Impact

**Before**:
```python
# Tokens generated: [" w", "ad", "sworth", "\n", "\n"]
# After strip + re-tokenize: ["w", "ad", "sworth"]  ← WRONG first token!
```

**After**:
```python
# Tokens generated: [" w", "ad", "sworth", "\n", "\n"]
# n_generated = 5 - 2 = 3
# Return tokens[0:3] = [" w", "ad", "sworth"]  ← CORRECT!
```

## 🚀 Action Required

**Must regenerate detailed answer pickles** (they were severely truncated):
```bash
python run_generate_long_answers.py
```

**Recommended to regenerate short answer pickles** (for perfect alignment):
```bash
python run_generate_short_answers.py
```

## ✨ Benefits

- ✅ Perfect 1-to-1 alignment: `token_ids[i]` ↔ `tokens[i]` ↔ `log_likelihoods[i]`
- ✅ Stop sequences detected correctly
- ✅ Detailed answers can now generate full responses
- ✅ Token-level analysis (phase1_5, phase2) will work correctly

---

**Date**: November 20, 2024  
**Priority**: CRITICAL - Must regenerate pickles before analysis

