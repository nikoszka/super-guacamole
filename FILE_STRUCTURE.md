# 📁 nllSAR File Structure

Complete guide to the file and directory structure of the nllSAR project.

---

## Repository Root

```
nllSAR/
├── src/                           # Source code (main package)
├── config/                        # Configuration files
├── experiments/                   # Experiment scripts
├── results/                       # Analysis outputs
├── scripts/                       # Utility scripts
├── tests/                         # Test suite
├── docs/                          # Additional documentation (optional)
├── .github/                       # GitHub workflows (CI/CD)
├── requirements.txt               # Python dependencies
├── nllsar.yml                     # Conda environment
├── setup.py                       # Package setup
├── .gitignore                     # Git ignore patterns
├── README.md                      # Project overview
├── CODE_DOCUMENTATION.md          # Comprehensive code documentation
├── API_REFERENCE.md               # API reference
├── ARCHITECTURE_GUIDE.md          # Architecture details
├── DEVELOPER_GUIDE.md             # Developer guide
├── FILE_STRUCTURE.md              # This file
├── QUICK_START.md                 # Quick start guide
├── GENERATION_SETTINGS_GUIDE.md   # Generation parameters guide
├── ANALYSIS_README.md             # Analysis pipeline guide
├── CHANGES_SUMMARY.md             # Recent changes
└── [other documentation files]    # Domain-specific guides
```

---

## Source Code (`src/`)

### Main Package Structure

```
src/
├── __init__.py                    # Package initialization
├── generate_answers.py            # Main answer generation script
├── compute_uncertainty_measures.py # Uncertainty computation
├── analyze_results.py             # Results analysis utilities
├── models/                        # Model implementations
├── uncertainty_measures/          # Uncertainty metrics
├── data/                          # Data loading utilities
├── analysis/                      # Analysis scripts
├── utils/                         # Helper utilities
└── analysis_notebooks/            # Jupyter notebooks for analysis
```

---

### Models Module (`src/models/`)

```
src/models/
├── __init__.py                    # Models package init
├── base_model.py                  # Abstract base class for all models
│                                  # - BaseModel (ABC)
│                                  # - STOP_SEQUENCES constants
│                                  # - predict() and get_p_true() interface
│
└── huggingface_models.py          # HuggingFace model implementation
                                   # - HuggingfaceModel class
                                   # - get_hf_cache_dir()
                                   # - get_gpu_memory_dict()
                                   # - StoppingCriteriaSub class
                                   # - remove_split_layer()
                                   # Supports: Llama, Mistral, Falcon
```

**Key Files:**

#### `base_model.py`
- **Purpose:** Define interface for all language models
- **Key Components:**
  - `BaseModel` abstract class
  - Stop sequence constants
- **Dependencies:** None (abstract)
- **Usage:** Subclass to implement new models

#### `huggingface_models.py`
- **Purpose:** HuggingFace Transformers integration
- **Key Components:**
  - `HuggingfaceModel` class (629 lines)
  - Multi-GPU support
  - Token-based extraction
  - Stop sequence handling
  - Cache management
- **Dependencies:** transformers, torch, accelerate, huggingface_hub
- **Usage:** Primary model class for generation

---

### Uncertainty Measures Module (`src/uncertainty_measures/`)

```
src/uncertainty_measures/
├── __init__.py                    # Package init
├── rw_gnll.py                     # Relevance-Weighted G-NLL
│                                  # - initialize_similarity_model()
│                                  # - compute_similarity()
│                                  # - remove_token_at_position()
│                                  # - compute_token_relevance_weights()
│                                  # - compute_rw_gnll()
│
├── sar.py                         # Shifting Attention to Relevance
│                                  # - compute_sar_for_entry()
│                                  # - compute_sar()
│
├── p_true.py                      # P(True) baseline
│                                  # - get_p_true_from_pred()
│
├── p_ik.py                        # P_IK baseline
│                                  # - compute_p_ik()
│
└── semantic_entropy.py            # Semantic entropy
                                   # - compute_semantic_entropy()
```

**Key Files:**

#### `rw_gnll.py` (265 lines)
- **Purpose:** Relevance-weighted uncertainty estimation
- **Algorithm:** Weight token log-likelihoods by semantic relevance
- **Key Functions:**
  - `initialize_similarity_model()` - Load cross-encoder
  - `compute_token_relevance_weights()` - Compute R_T(y_t) for each token
  - `compute_rw_gnll()` - Main RW-G-NLL computation
- **Dependencies:** sentence-transformers, numpy
- **Usage:** For single-sample uncertainty with relevance weighting

#### `sar.py` (188 lines)
- **Purpose:** Multi-sample uncertainty with relevance weighting
- **Algorithm:** Average RW-G-NLL across multiple samples
- **Key Functions:**
  - `compute_sar_for_entry()` - Compute SAR across samples
  - `compute_sar()` - Convenience wrapper
- **Dependencies:** rw_gnll, numpy
- **Usage:** For multi-sample uncertainty (requires num_generations > 1)

#### `p_true.py`
- **Purpose:** P(True) baseline uncertainty
- **Algorithm:** Log-probability of generating 'True' answer
- **Usage:** Simple baseline for comparison

#### `semantic_entropy.py`
- **Purpose:** Semantic entropy over multiple samples
- **Algorithm:** Cluster similar responses and compute entropy
- **Usage:** Multi-sample uncertainty based on semantic diversity

---

### Data Module (`src/data/`)

```
src/data/
├── __init__.py                    # Package init
└── data_utils.py                  # Dataset loading utilities
                                   # - load_ds() - Main loading function
                                   # Supports:
                                   #   - squad (SQuAD v2.0)
                                   #   - trivia_qa (TriviaQA)
                                   #   - svamp (Math problems)
                                   #   - nq (Natural Questions)
                                   #   - bioasq (BioASQ)
```

**Key Files:**

#### `data_utils.py` (104 lines)
- **Purpose:** Load and normalize datasets
- **Key Function:** `load_ds(dataset_name, seed, add_options)`
- **Output Format:**
  ```python
  {
      'question': str,
      'answers': {'text': List[str]},
      'context': str,
      'id': str
  }
  ```
- **Dependencies:** datasets, json
- **Usage:** Primary data loading interface

---

### Analysis Module (`src/analysis/`)

```
src/analysis/
├── __init__.py                         # Package init
├── phase1_baseline_metrics.py          # Baseline statistics
│                                       # - Accuracy, NLL, token stats
│                                       # Output: JSON, CSV
│
├── phase1_5_token_nll_analysis.py      # Token-level NLL analysis
│                                       # - NLL by position
│                                       # - NLL distributions
│                                       # Output: PNG plots, JSON examples
│
├── phase1_6_prefix_nll_analysis.py     # Prefix-based NLL
│                                       # - Early stopping analysis
│                                       # - AUROC by prefix length
│                                       # Output: AUROC curves
│
├── phase2_token_importance.py          # Token relevance analysis
│                                       # - Relevance weights
│                                       # - RW-G-NLL computation
│                                       # Output: Relevance plots, JSON
│
├── phase5_comparative_analysis.py      # AUROC comparison
│                                       # - Compare all uncertainty metrics
│                                       # - ROC curves
│                                       # Output: CSV, PNG
│
└── token_visualization_app.py          # Streamlit visualization app
                                        # - Interactive token highlighting
                                        # - Load JSON from analysis phases
```

**Key Files:**

#### `phase1_baseline_metrics.py`
- **Purpose:** Compute baseline metrics
- **Outputs:**
  - `baseline_metrics.json` - Accuracy, NLL stats
  - `token_statistics.json` - Token count distributions
  - `baseline_metrics.csv` - CSV export
- **Usage:** First step in analysis pipeline

#### `phase1_5_token_nll_analysis.py` (186 lines)
- **Purpose:** Detailed token-level NLL analysis
- **Outputs:**
  - `nll_vs_position.png` - NLL by token position
  - `nll_distribution.png` - Distribution histograms
  - `position_nll_heatmap.png` - Heatmap visualization
  - `sentence_level_nll_examples.json` - Detailed examples (for Streamlit)
  - `example_{01-10}_token_nlls.png` - Individual plots
- **Usage:** Understand token-level patterns

#### `phase2_token_importance.py`
- **Purpose:** Compute token relevance weights
- **Algorithm:** SAR-style relevance computation
- **Outputs:**
  - `relevance_vs_position.png` - Relevance by position
  - `token_importance_examples.json` - Examples with weights (for Streamlit)
- **Usage:** Analyze which tokens are most relevant

#### `phase5_comparative_analysis.py`
- **Purpose:** Compare AUROC of all uncertainty metrics
- **Metrics Compared:**
  - G-NLL (baseline)
  - RW-G-NLL
  - SAR (if multi-sample)
  - Semantic Entropy (if multi-sample)
  - Length (baseline)
- **Outputs:**
  - `auroc_comparison.csv` - AUROC scores
  - `roc_curves.png` - ROC curve plots
  - `cost_performance_plot.png` - Cost-benefit analysis
- **Usage:** Final analysis step

#### `token_visualization_app.py`
- **Purpose:** Interactive visualization with Streamlit
- **Features:**
  - Load JSON from phase 1.5 or phase 2
  - Interactive token highlighting
  - Color-coded by NLL or relevance
- **Usage:** `streamlit run src/analysis/token_visualization_app.py`

---

### Utilities Module (`src/utils/`)

```
src/utils/
├── __init__.py                    # Package init
├── utils.py                       # General utilities
│                                  # - setup_logger()
│                                  # - init_model()
│                                  # - get_metric()
│                                  # - construct_fewshot_prompt_from_indices()
│                                  # - get_make_prompt()
│                                  # - split_dataset()
│                                  # - BRIEF_PROMPTS constant
│
├── eval_utils.py                  # Evaluation utilities
│                                  # - compute_rouge_scores()
│                                  # - compute_squad_metrics()
│                                  # - llm_judge_evaluate()
│
└── openai.py                      # OpenAI API utilities
                                   # - call_openai_api()
                                   # - parse_llm_judge_response()
```

**Key Files:**

#### `utils.py`
- **Purpose:** General utility functions
- **Key Functions:**
  - Model initialization
  - Prompt construction
  - Dataset splitting
  - Logging setup
- **Usage:** Used throughout codebase

#### `eval_utils.py`
- **Purpose:** Answer evaluation metrics
- **Key Functions:**
  - ROUGE-L computation
  - SQuAD F1 and EM
  - LLM-as-a-judge
- **Usage:** Evaluate answer correctness

#### `openai.py`
- **Purpose:** OpenAI API integration
- **Key Functions:**
  - API calls with retry logic
  - Response parsing
- **Usage:** For LLM judge evaluation

---

### Analysis Notebooks (`src/analysis_notebooks/`)

```
src/analysis_notebooks/
├── first_analysis_uncertainties.ipynb   # Initial uncertainty analysis
├── gnll_baseline_analysis.ipynb         # G-NLL baseline exploration
├── gnll_baseline_analysis_results.json  # Cached results
├── analysis_llm_as_judge.ipynb          # LLM judge analysis
├── llm_judge_results.pkl                # LLM judge cached results
└── llm_judge_summary.json               # LLM judge summary
```

**Purpose:** Exploratory analysis and visualization in Jupyter notebooks.

**Key Notebooks:**

- `first_analysis_uncertainties.ipynb` - Initial exploration of uncertainty metrics
- `gnll_baseline_analysis.ipynb` - Comprehensive G-NLL analysis with plots
- `analysis_llm_as_judge.ipynb` - Analyze LLM judge performance

---

### Main Scripts (`src/`)

#### `generate_answers.py` (302+ lines)
- **Purpose:** Main answer generation pipeline
- **Key Function:** `main(args)`
- **Pipeline:**
  1. Load dataset
  2. Construct few-shot prompt
  3. Initialize model
  4. Generate answers with token info
  5. Evaluate answers
  6. Save to pickle
  7. Upload to Wandb
- **Output:** `validation_generations.pkl`
- **Usage:** Primary entry point for generation

#### `compute_uncertainty_measures.py`
- **Purpose:** Compute uncertainty measures from pickle
- **Key Function:** `main(args)`
- **Supported Metrics:**
  - G-NLL
  - RW-G-NLL
  - SAR
  - Semantic Entropy
  - P(True)
- **Usage:** Post-generation uncertainty computation

#### `analyze_results.py`
- **Purpose:** General results analysis utilities
- **Functions:** Helper functions for analysis scripts
- **Usage:** Imported by analysis scripts

---

## Top-Level Scripts

```
├── run_generate_short_answers.py       # Short answer generation wrapper
├── run_generate_long_answers.py        # Long answer generation wrapper
├── run_gnll_baseline.py                # Full G-NLL baseline pipeline
├── run_greedy_decoding.py              # Greedy decoding experiments
├── run_greedy_decoding_short.py        # Short answer greedy decoding
├── generate_gnll_answers.py            # G-NLL answer generation
├── compute_gnll_auroc.py               # Standalone AUROC computation
└── recompute_accuracy_with_judge.py    # Recompute with LLM judge
```

**Key Scripts:**

#### `run_generate_short_answers.py`
- **Purpose:** Convenient wrapper for short answer generation
- **Default Settings:**
  - `max_new_tokens=50`
  - `brief_prompt=short`
  - `metric=squad`
- **Usage:** Quick short answer experiments

#### `run_generate_long_answers.py`
- **Purpose:** Convenient wrapper for long answer generation
- **Default Settings:**
  - `max_new_tokens=200`
  - `brief_prompt=detailed`
  - `metric=llm_llama-3.1-70b`
- **Usage:** Quick long answer experiments

#### `run_gnll_baseline.py`
- **Purpose:** Full pipeline for G-NLL baseline
- **Steps:**
  1. Generate answers (short and long)
  2. Evaluate with LLM judge
  3. Compute G-NLL and RW-G-NLL
  4. Calculate AUROC
  5. Generate plots
- **Usage:** Complete G-NLL baseline evaluation

#### `compute_gnll_auroc.py`
- **Purpose:** Standalone AUROC computation from existing pickle
- **Usage:**
  ```bash
  python compute_gnll_auroc.py \
    --pickle_path path/to/pickle.pkl \
    --use_rw_gnll
  ```

---

## Configuration (`config/`)

```
config/
├── default_config.yaml            # Default configuration
├── model_configs/                 # Model-specific configs
│   ├── llama_config.yaml
│   ├── mistral_config.yaml
│   └── falcon_config.yaml
└── dataset_configs/               # Dataset-specific configs
    ├── squad_config.yaml
    ├── trivia_qa_config.yaml
    └── bioasq_config.yaml
```

**Purpose:** YAML configuration files for reproducible experiments.

**Usage:**
```bash
python -m src.generate_answers --config config/llama_config.yaml
```

---

## Experiments (`experiments/`)

```
experiments/
├── experiment_001_baseline.sh     # Baseline experiments
├── experiment_002_rw_gnll.sh      # RW-G-NLL experiments
├── experiment_003_sar.sh          # SAR experiments
└── [more experiment scripts]
```

**Purpose:** Shell scripts for running full experiments.

**Example:**
```bash
#!/bin/bash
# experiment_001_baseline.sh

# Generate short answers
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 200 \
  --temperature 0.0 \
  --model_max_new_tokens 50 \
  --brief_prompt short \
  --metric squad \
  --project baseline_short

# Analyze results
python -m src.analysis.phase1_baseline_metrics \
  --short-pickle path/to/pickle.pkl \
  --output-dir results/baseline_short
```

---

## Results (`results/`)

```
results/
├── phase1_short/                  # Phase 1 short answer results
│   ├── baseline_metrics.json
│   ├── token_statistics.json
│   └── baseline_metrics_short.csv
│
├── phase1_long/                   # Phase 1 long answer results
│   ├── baseline_metrics.json
│   ├── token_statistics.json
│   └── baseline_metrics_long.csv
│
├── phase1_5_short/                # Phase 1.5 short answer results
│   ├── nll_vs_position.png
│   ├── nll_distribution.png
│   ├── position_nll_heatmap.png
│   ├── sentence_level_nll_examples.json
│   ├── token_nll_results.json
│   └── example_{01-10}_token_nlls.png
│
├── phase1_5_long/                 # Phase 1.5 long answer results
│   └── [same structure as phase1_5_short]
│
├── phase2_short/                  # Phase 2 short answer results
│   ├── relevance_vs_position.png
│   └── token_importance_examples.json
│
├── phase2_long/                   # Phase 2 long answer results
│   └── [same structure as phase2_short]
│
└── phase5_long/                   # Phase 5 comparison results
    ├── auroc_comparison.csv
    ├── roc_curves.png
    ├── cost_performance_plot.png
    └── analysis_summary.json
```

**Purpose:** Analysis outputs organized by phase and answer type.

---

## Scripts (`scripts/`)

```
scripts/
├── setup_environment.sh           # Environment setup script
├── download_models.sh             # Download model weights
├── clean_cache.sh                 # Clean model cache
├── run_tests.sh                   # Run test suite
├── format_code.sh                 # Run code formatters
└── setup_hf_token.ps1             # HuggingFace token setup (Windows)
```

**Purpose:** Utility scripts for environment setup and maintenance.

---

## Tests (`tests/`)

```
tests/
├── __init__.py                    # Test package init
├── conftest.py                    # Pytest configuration and fixtures
├── test_models.py                 # Model tests
├── test_uncertainty_measures.py   # Uncertainty metric tests
├── test_data_utils.py             # Data loading tests
├── test_analysis.py               # Analysis script tests
├── test_integration.py            # End-to-end integration tests
└── fixtures/                      # Test data and fixtures
    ├── sample_pickle.pkl          # Sample generation results
    ├── sample_dataset.json        # Sample dataset
    └── sample_config.yaml         # Sample configuration
```

**Purpose:** Comprehensive test suite for all modules.

**Usage:**
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Documentation Files

### Core Documentation

```
├── README.md                      # Project overview and quick start
├── CODE_DOCUMENTATION.md          # Comprehensive code documentation
├── API_REFERENCE.md               # Complete API reference
├── ARCHITECTURE_GUIDE.md          # Architecture and design patterns
├── DEVELOPER_GUIDE.md             # Developer guide
└── FILE_STRUCTURE.md              # This file
```

### User Guides

```
├── QUICK_START.md                 # Quick start guide
├── GENERATION_SETTINGS_GUIDE.md   # Generation parameter guide
├── ANALYSIS_README.md             # Analysis pipeline guide
├── ENVIRONMENT_SETUP.md           # Environment setup
├── GPU_REQUIREMENTS.md            # GPU setup guide
├── MULTI_GPU_GUIDE.md             # Multi-GPU usage
└── MODEL_CACHE_GUIDE.md           # Model cache management
```

### Domain-Specific Guides

```
├── GNLL_BASELINE_README.md        # G-NLL baseline guide
├── GREEDY_DECODING_README.md      # Greedy decoding guide
├── CLOUD_PLATFORMS.md             # Cloud platform setup
└── IMPORTANT_CLARIFICATIONS.md    # Important clarifications
```

### Change Logs

```
├── CHANGES_SUMMARY.md             # Recent changes summary
├── ALIGNMENT_FIX_SUMMARY.md       # Token alignment fix
├── ANSWER_EXTRACTION_FIX.md       # Answer extraction fix
├── QUICK_FIX_SUMMARY.md           # Quick fixes
├── SESSION_FIXES_SUMMARY.md       # Session-specific fixes
└── STOP_SEQUENCE_FIX.md           # Stop sequence handling fix
```

---

## Hidden Files and Directories

```
├── .gitignore                     # Git ignore patterns
├── .github/                       # GitHub-specific files
│   ├── workflows/                 # CI/CD workflows
│   │   ├── tests.yml              # Run tests on push
│   │   ├── lint.yml               # Linting checks
│   │   └── docs.yml               # Documentation building
│   ├── ISSUE_TEMPLATE.md          # Issue template
│   └── PULL_REQUEST_TEMPLATE.md   # PR template
│
├── .vscode/                       # VS Code settings
│   ├── settings.json              # Editor settings
│   ├── launch.json                # Debug configurations
│   └── extensions.json            # Recommended extensions
│
├── .env                           # Environment variables (gitignored)
└── .pytest_cache/                 # Pytest cache (gitignored)
```

---

## Generated/Cached Directories (Gitignored)

```
├── __pycache__/                   # Python bytecode cache
├── .pytest_cache/                 # Pytest cache
├── .ipynb_checkpoints/            # Jupyter notebook checkpoints
├── wandb/                         # Weights & Biases logs
├── .cache/                        # General cache
├── models/                        # Downloaded model weights
└── htmlcov/                       # Coverage reports
```

---

## Data Directories

```
src/boldis/uncertainty/wandb/      # User 'boldis' wandb runs
src/nikos/uncertainty/wandb/       # User 'nikos' wandb runs
  └── run-<timestamp>-<id>/        # Individual run directory
      ├── files/
      │   ├── validation_generations.pkl    # Main output
      │   ├── train_generations.pkl
      │   ├── experiment_details.pkl
      │   ├── config.yaml
      │   └── wandb-summary.json
      └── logs/
          └── debug.log
```

**Purpose:** Weights & Biases experiment tracking data.

**Usage:** After generation, find pickle files here.

---

## File Naming Conventions

### Python Files
- **Modules:** lowercase with underscores (`data_utils.py`)
- **Classes:** PascalCase (`HuggingfaceModel`)
- **Functions:** lowercase with underscores (`compute_rw_gnll()`)
- **Constants:** UPPERCASE with underscores (`STOP_SEQUENCES`)

### Output Files
- **Metrics:** `*_metrics.json`, `*_metrics.csv`
- **Plots:** `*_vs_*.png`, `*_distribution.png`, `*_heatmap.png`
- **Examples:** `*_examples.json`, `example_01_*.png`
- **Results:** `*_results.json`, `*_analysis.json`

### Documentation Files
- **Guides:** `*_GUIDE.md`, `*_README.md`
- **Summaries:** `*_SUMMARY.md`
- **References:** `*_REFERENCE.md`

---

## File Size Expectations

### Code Files
- Small: < 200 lines (utilities, simple classes)
- Medium: 200-600 lines (complex classes, analysis scripts)
- Large: > 600 lines (main generation script, comprehensive models)

### Data Files
- Pickles: 10MB - 500MB (depending on num_samples)
- JSON: 1KB - 50MB (analysis results)
- PNG: 50KB - 5MB (plots)

### Model Files (in cache)
- 1B models: ~5GB
- 7B models: ~15GB
- 13B models: ~25GB
- 70B models: ~140GB

---

## Important Files to NOT Modify Directly

```
# Auto-generated files
wandb/                             # Managed by Weights & Biases
__pycache__/                       # Python bytecode
*.pyc                              # Compiled Python
.pytest_cache/                     # Pytest cache

# Downloaded models
models/                            # Model weights

# User-specific
.env                               # Environment variables
.vscode/                           # Editor settings (can be shared but user-specific)
```

---

## Quick Reference: Where to Find Things

### "I want to..."

**...generate answers:**
- Script: `src/generate_answers.py`
- Wrappers: `run_generate_short_answers.py`, `run_generate_long_answers.py`

**...compute uncertainty metrics:**
- RW-G-NLL: `src/uncertainty_measures/rw_gnll.py`
- SAR: `src/uncertainty_measures/sar.py`
- Compute from pickle: `src/compute_uncertainty_measures.py`

**...analyze results:**
- Phase 1: `src/analysis/phase1_baseline_metrics.py`
- Phase 1.5: `src/analysis/phase1_5_token_nll_analysis.py`
- Phase 2: `src/analysis/phase2_token_importance.py`
- Phase 5: `src/analysis/phase5_comparative_analysis.py`

**...visualize results:**
- Streamlit app: `src/analysis/token_visualization_app.py`
- Jupyter notebooks: `src/analysis_notebooks/`

**...add a new model:**
- Create: `src/models/my_model.py`
- Register: `src/utils/utils.py` (in `init_model()`)
- Test: `tests/test_my_model.py`

**...add a new dataset:**
- Implement: `src/data/data_utils.py` (in `load_ds()`)
- Test: `tests/test_data_utils.py`

**...add a new uncertainty metric:**
- Create: `src/uncertainty_measures/my_metric.py`
- Add to analysis: `src/analysis/phase5_comparative_analysis.py`
- Test: `tests/test_my_metric.py`

**...understand the architecture:**
- Read: `ARCHITECTURE_GUIDE.md`
- API details: `API_REFERENCE.md`
- Code overview: `CODE_DOCUMENTATION.md`

**...contribute to the project:**
- Read: `DEVELOPER_GUIDE.md`
- Submit PR: See `.github/PULL_REQUEST_TEMPLATE.md`

---

## File Creation Timeline

### Initial Setup (v1.0)
1. Core models (`base_model.py`, `huggingface_models.py`)
2. Data loading (`data_utils.py`)
3. Basic generation (`generate_answers.py`)
4. Uncertainty measures (`p_true.py`, `p_ik.py`)

### Token Alignment Update (v1.5)
1. Enhanced `huggingface_models.py` (token_ids, tokens storage)
2. Updated analysis scripts to use stored tokens
3. Documentation updates (`CHANGES_SUMMARY.md`, `ANSWER_EXTRACTION_FIX.md`)

### RW-G-NLL Addition (v1.8)
1. `rw_gnll.py` implementation
2. `phase2_token_importance.py` analysis
3. `token_visualization_app.py` Streamlit app

### SAR Implementation (v1.9)
1. `sar.py` implementation
2. Multi-sample support in generation
3. Updated Phase 5 analysis

### Comprehensive Documentation (v2.0)
1. `CODE_DOCUMENTATION.md`
2. `API_REFERENCE.md`
3. `ARCHITECTURE_GUIDE.md`
4. `DEVELOPER_GUIDE.md`
5. `FILE_STRUCTURE.md` (this file)

---

**Last Updated:** November 2025
**Version:** 2.0


