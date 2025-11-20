# 📚 nllSAR Documentation Index

Complete guide to all documentation available in the nllSAR project.

---

## 🎯 Documentation Map

### For New Users

**Start here if you're new to the project:**

1. **[README.md](README.md)** - Project overview
   - What is nllSAR?
   - Key features and capabilities
   - High-level architecture

2. **[QUICK_START.md](QUICK_START.md)** - Get started in minutes
   - Step-by-step generation guide
   - Analysis pipeline walkthrough
   - File locations and verification

3. **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)** - Set up your environment
   - Installation instructions
   - Dependencies
   - Environment variables

4. **[GPU_REQUIREMENTS.md](GPU_REQUIREMENTS.md)** - GPU setup
   - Hardware requirements
   - CUDA configuration
   - Multi-GPU setup

---

### For Users Running Experiments

**Essential guides for conducting experiments:**

1. **[GENERATION_SETTINGS_GUIDE.md](GENERATION_SETTINGS_GUIDE.md)** - Generation parameters
   - Temperature settings
   - Token limits
   - Stop sequences
   - Prompt strategies

2. **[ANALYSIS_README.md](ANALYSIS_README.md)** - Analysis pipeline
   - Phase 1: Baseline metrics
   - Phase 1.5: Token-level NLL
   - Phase 1.6: Prefix NLL
   - Phase 2: Token importance
   - Phase 5: AUROC comparison

3. **[GNLL_BASELINE_README.md](GNLL_BASELINE_README.md)** - G-NLL baseline
   - What is G-NLL?
   - How to run baseline experiments
   - Interpreting results

4. **[GREEDY_DECODING_README.md](GREEDY_DECODING_README.md)** - Greedy decoding
   - Greedy vs. sampling
   - Deterministic generation
   - Use cases

5. **[MODEL_CACHE_GUIDE.md](MODEL_CACHE_GUIDE.md)** - Model caching
   - Cache directory configuration
   - Managing disk space
   - Troubleshooting

6. **[MULTI_GPU_GUIDE.md](MULTI_GPU_GUIDE.md)** - Multi-GPU usage
   - Automatic GPU distribution
   - Memory management
   - Device mapping

---

### For Developers

**Essential for contributing or extending the codebase:**

1. **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** ⭐ START HERE
   - Development workflow
   - Code style guidelines
   - Testing strategy
   - Common development tasks
   - Debugging tips
   - Performance profiling

2. **[CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md)** ⭐ COMPREHENSIVE
   - Complete code overview
   - All modules explained
   - Data structures
   - Configuration
   - API usage examples

3. **[API_REFERENCE.md](API_REFERENCE.md)** ⭐ DETAILED
   - Complete API documentation
   - Function signatures
   - Parameters and return types
   - Usage examples
   - Type definitions

4. **[ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md)** ⭐ IN-DEPTH
   - Design principles
   - Architectural patterns
   - Implementation details
   - Performance optimization
   - Testing strategy

5. **[FILE_STRUCTURE.md](FILE_STRUCTURE.md)** ⭐ REFERENCE
   - Complete file tree
   - What each file does
   - Where to find things
   - Naming conventions
   - File organization

---

### For Understanding Changes

**Track recent changes and fixes:**

1. **[CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)** - Recent changes
   - Major updates
   - Feature additions
   - Bug fixes

2. **[ALIGNMENT_FIX_SUMMARY.md](ALIGNMENT_FIX_SUMMARY.md)** - Token alignment fix
   - Problem description
   - Solution implemented
   - Impact on analysis

3. **[ANSWER_EXTRACTION_FIX.md](ANSWER_EXTRACTION_FIX.md)** - Answer extraction
   - Token-based extraction
   - Why it's more reliable
   - Migration guide

4. **[QUICK_FIX_SUMMARY.md](QUICK_FIX_SUMMARY.md)** - Quick fixes
   - Small bug fixes
   - Patches
   - Workarounds

5. **[SESSION_FIXES_SUMMARY.md](SESSION_FIXES_SUMMARY.md)** - Session fixes
   - Session-specific issues
   - Resolutions

6. **[STOP_SEQUENCE_FIX.md](STOP_SEQUENCE_FIX.md)** - Stop sequence handling
   - Stop sequence issues
   - Improved handling
   - Token counting

---

### For Specific Topics

**Deep dives into specific areas:**

1. **[CLOUD_PLATFORMS.md](CLOUD_PLATFORMS.md)** - Cloud deployment
   - AWS setup
   - Google Cloud setup
   - Azure setup
   - Cost optimization

2. **[IMPORTANT_CLARIFICATIONS.md](IMPORTANT_CLARIFICATIONS.md)** - Key clarifications
   - Common misconceptions
   - Important distinctions
   - Best practices

---

## 📖 Documentation by Role

### I am a... Researcher

**Goal: Run experiments and analyze results**

**Read in this order:**
1. [README.md](README.md) - Understand the project
2. [QUICK_START.md](QUICK_START.md) - Run your first experiment
3. [GENERATION_SETTINGS_GUIDE.md](GENERATION_SETTINGS_GUIDE.md) - Configure experiments
4. [ANALYSIS_README.md](ANALYSIS_README.md) - Analyze results
5. [GNLL_BASELINE_README.md](GNLL_BASELINE_README.md) - Understand G-NLL baseline

**Keep handy:**
- [GPU_REQUIREMENTS.md](GPU_REQUIREMENTS.md)
- [MODEL_CACHE_GUIDE.md](MODEL_CACHE_GUIDE.md)
- [MULTI_GPU_GUIDE.md](MULTI_GPU_GUIDE.md)

---

### I am a... Developer

**Goal: Contribute code or extend functionality**

**Read in this order:**
1. [README.md](README.md) - Project overview
2. [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Development workflow
3. [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) - Architecture details
4. [FILE_STRUCTURE.md](FILE_STRUCTURE.md) - File organization
5. [API_REFERENCE.md](API_REFERENCE.md) - API details
6. [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) - Code reference

**Keep handy:**
- [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) - Recent changes
- Testing guidelines in [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)

---

### I am a... System Administrator

**Goal: Deploy and maintain the system**

**Read in this order:**
1. [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) - Environment setup
2. [GPU_REQUIREMENTS.md](GPU_REQUIREMENTS.md) - Hardware requirements
3. [MULTI_GPU_GUIDE.md](MULTI_GPU_GUIDE.md) - Multi-GPU configuration
4. [MODEL_CACHE_GUIDE.md](MODEL_CACHE_GUIDE.md) - Cache management
5. [CLOUD_PLATFORMS.md](CLOUD_PLATFORMS.md) - Cloud deployment

**Keep handy:**
- Debugging section in [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)

---

### I am a... Data Scientist

**Goal: Understand uncertainty metrics and analysis**

**Read in this order:**
1. [README.md](README.md) - Project overview
2. [GNLL_BASELINE_README.md](GNLL_BASELINE_README.md) - G-NLL baseline
3. [ANALYSIS_README.md](ANALYSIS_README.md) - Analysis pipeline
4. [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) - Uncertainty measures section
5. [API_REFERENCE.md](API_REFERENCE.md) - Uncertainty API

**Keep handy:**
- Analysis notebooks in `src/analysis_notebooks/`
- Visualization app: `src/analysis/token_visualization_app.py`

---

## 🔍 Documentation by Task

### "I want to..."

#### ...understand what nllSAR does
→ [README.md](README.md)
→ [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) (Overview section)

#### ...run my first experiment
→ [QUICK_START.md](QUICK_START.md)
→ [GENERATION_SETTINGS_GUIDE.md](GENERATION_SETTINGS_GUIDE.md)

#### ...analyze results
→ [ANALYSIS_README.md](ANALYSIS_README.md)
→ [QUICK_START.md](QUICK_START.md) (Analysis section)

#### ...add a new model
→ [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) (Adding New Model section)
→ [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) (Models API section)
→ [API_REFERENCE.md](API_REFERENCE.md) (Models API section)

#### ...add a new uncertainty metric
→ [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) (Adding New Uncertainty Metric section)
→ [API_REFERENCE.md](API_REFERENCE.md) (Uncertainty Measures API section)

#### ...add a new dataset
→ [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) (Adding New Dataset section)
→ [API_REFERENCE.md](API_REFERENCE.md) (Data Loading API section)

#### ...understand the architecture
→ [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md)
→ [FILE_STRUCTURE.md](FILE_STRUCTURE.md)

#### ...debug an issue
→ [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) (Debugging Tips section)
→ [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) (Known Issues)

#### ...optimize performance
→ [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) (Performance Profiling section)
→ [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) (Performance Optimization section)

#### ...deploy to cloud
→ [CLOUD_PLATFORMS.md](CLOUD_PLATFORMS.md)
→ [GPU_REQUIREMENTS.md](GPU_REQUIREMENTS.md)

#### ...contribute code
→ [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) (Contributing Guidelines section)
→ [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) (Design Principles section)

#### ...understand recent changes
→ [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)
→ Related fix summaries (ALIGNMENT_FIX_SUMMARY.md, etc.)

---

## 📊 Documentation Statistics

### Core Documentation (Must Read)
- **README.md** - Project overview
- **QUICK_START.md** - Getting started guide
- **CODE_DOCUMENTATION.md** - Comprehensive code docs (45+ pages)
- **API_REFERENCE.md** - Complete API reference (35+ pages)
- **DEVELOPER_GUIDE.md** - Developer guide (30+ pages)

### Reference Documentation
- **ARCHITECTURE_GUIDE.md** - Architecture details (25+ pages)
- **FILE_STRUCTURE.md** - File organization (20+ pages)
- **ANALYSIS_README.md** - Analysis pipeline (15+ pages)
- **GENERATION_SETTINGS_GUIDE.md** - Generation settings (15+ pages)

### Specialized Guides
- **ENVIRONMENT_SETUP.md** - Environment setup
- **GPU_REQUIREMENTS.md** - GPU setup
- **MODEL_CACHE_GUIDE.md** - Cache management
- **MULTI_GPU_GUIDE.md** - Multi-GPU usage
- **CLOUD_PLATFORMS.md** - Cloud deployment
- **GNLL_BASELINE_README.md** - G-NLL baseline
- **GREEDY_DECODING_README.md** - Greedy decoding

### Change Logs
- **CHANGES_SUMMARY.md** - Recent changes
- **ALIGNMENT_FIX_SUMMARY.md** - Token alignment fix
- **ANSWER_EXTRACTION_FIX.md** - Answer extraction
- **QUICK_FIX_SUMMARY.md** - Quick fixes
- **SESSION_FIXES_SUMMARY.md** - Session fixes
- **STOP_SEQUENCE_FIX.md** - Stop sequence handling

**Total:** 20+ documentation files, 200+ pages

---

## 🎓 Learning Path

### Beginner Path (2-4 hours)
1. Read [README.md](README.md) (15 min)
2. Follow [QUICK_START.md](QUICK_START.md) (1 hour)
3. Review [GENERATION_SETTINGS_GUIDE.md](GENERATION_SETTINGS_GUIDE.md) (30 min)
4. Explore [ANALYSIS_README.md](ANALYSIS_README.md) (30 min)
5. Try running experiments (1-2 hours)

### Intermediate Path (1-2 days)
1. Complete Beginner Path
2. Study [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) (2-3 hours)
3. Read [GNLL_BASELINE_README.md](GNLL_BASELINE_README.md) (1 hour)
4. Review [API_REFERENCE.md](API_REFERENCE.md) (2 hours)
5. Explore analysis notebooks (2-3 hours)
6. Run multiple experiments (4-8 hours)

### Advanced Path (1 week)
1. Complete Intermediate Path
2. Study [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) (4 hours)
3. Read [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) (3 hours)
4. Review [FILE_STRUCTURE.md](FILE_STRUCTURE.md) (2 hours)
5. Study source code with documentation (8-16 hours)
6. Implement a new feature (16-32 hours)

### Expert Path (Ongoing)
1. Complete Advanced Path
2. Contribute to codebase regularly
3. Review all change logs
4. Stay updated with latest research
5. Mentor others

---

## 📝 Documentation Conventions

### File Naming
- All caps with underscores for importance: `README.md`, `QUICK_START.md`
- Descriptive names: `GENERATION_SETTINGS_GUIDE.md`
- Suffix indicates type: `*_GUIDE.md`, `*_README.md`, `*_SUMMARY.md`

### Sections
- **📚, 🚀, 🎯, etc.** - Emoji headers for visual navigation
- **Bold** - Important terms and key concepts
- `Code` - Code, filenames, and commands
- > Quotes - Important notes and warnings

### Code Examples
- Always include imports
- Show expected output
- Indicate language: ```python, ```bash
- Comment complex parts

### Cross-References
- Use relative links: `[text](file.md)`
- Link to specific sections: `[text](file.md#section)`
- Reference other docs frequently

---

## 🔗 External Resources

### Official Documentation
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Weights & Biases](https://docs.wandb.ai/)
- [Sentence Transformers](https://www.sbert.net/)

### Related Papers
- Semantic Uncertainty (Kuhn et al.)
- Token-level Uncertainty
- Language Model Calibration

### Community
- GitHub Issues
- GitHub Discussions
- Research Papers using nllSAR

---

## 🆘 Getting Help

### Documentation Not Clear?
1. Check [IMPORTANT_CLARIFICATIONS.md](IMPORTANT_CLARIFICATIONS.md)
2. Search across all documentation (Ctrl+F in IDE)
3. Check GitHub Issues for similar questions
4. Create new GitHub Issue with "documentation" label

### Feature Not Working?
1. Check [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) for known issues
2. Review relevant fix summaries
3. Check debugging tips in [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
4. Create GitHub Issue with error logs

### Want to Contribute?
1. Read [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
2. Check GitHub Issues for "good first issue" label
3. Follow contributing guidelines
4. Submit PR with clear description

---

## 📅 Documentation Maintenance

### Last Updated
- **Core Docs:** November 2025
- **API Reference:** November 2025
- **Architecture Guide:** November 2025
- **Developer Guide:** November 2025

### Update Frequency
- **Core Docs:** With each major release
- **API Reference:** When API changes
- **Change Logs:** With each significant change
- **Guides:** As needed based on user feedback

### How to Update Documentation
1. Edit relevant .md file
2. Update "Last Updated" date
3. Update version if applicable
4. Submit PR with documentation updates
5. Tag with "documentation" label

---

## 🎉 Quick Reference Card

### Essential Commands

**Generate Answers:**
```bash
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 200 \
  --temperature 0.0 \
  --model_max_new_tokens 50
```

**Run Analysis:**
```bash
python -m src.analysis.phase1_baseline_metrics \
  --long-pickle path/to/pickle.pkl \
  --output-dir results/phase1
```

**Visualize Results:**
```bash
streamlit run src/analysis/token_visualization_app.py
```

**Run Tests:**
```bash
pytest tests/ -v
```

### Essential Files
- **Pickle Output:** `src/[user]/uncertainty/wandb/run-*/files/validation_generations.pkl`
- **Analysis Results:** `results/phase*/`
- **Configuration:** `.env` or environment variables
- **Logs:** `wandb/` directories

### Essential Links
- [Main README](README.md)
- [Quick Start](QUICK_START.md)
- [Developer Guide](DEVELOPER_GUIDE.md)
- [API Reference](API_REFERENCE.md)

---

**Last Updated:** November 2025
**Documentation Version:** 2.0
**Total Documentation:** 20+ files, 200+ pages

---


