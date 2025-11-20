# 📖 nllSAR Code Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Core Modules](#core-modules)
4. [Data Flow](#data-flow)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Testing and Analysis](#testing-and-analysis)

---

## Project Overview

**nllSAR** (Negative Log-Likelihood - Shifting Attention to Relevance) is a research codebase for evaluating uncertainty estimation methods in Large Language Models (LLMs). The project focuses on:

- **G-NLL (Greedy Negative Log-Likelihood)**: A baseline uncertainty metric using token log-probabilities from greedy decoding
- **RW-G-NLL (Relevance-Weighted G-NLL)**: An improved metric that weights token log-likelihoods by semantic relevance
- **SAR (Shifting Attention to Relevance)**: Multi-sample uncertainty estimation with relevance weighting
- **Comprehensive Evaluation**: AUROC analysis comparing uncertainty scores against ground truth correctness

### Key Features
- Support for multiple LLM architectures (Llama, Mistral, Falcon)
- Multi-GPU inference with automatic memory management
- Token-level and sequence-level uncertainty analysis
- Interactive visualization with Streamlit
- Integration with Weights & Biases for experiment tracking

---

## Architecture

### High-Level Structure

```
nllSAR/
├── src/
│   ├── models/              # LLM wrapper classes
│   ├── uncertainty_measures/ # Uncertainty computation methods
│   ├── data/                # Dataset loading utilities
│   ├── analysis/            # Analysis scripts and visualization
│   ├── utils/               # Helper utilities
│   └── generate_answers.py  # Main answer generation pipeline
├── config/                  # Configuration files
├── experiments/             # Experiment scripts
├── results/                 # Analysis outputs
├── scripts/                 # Utility scripts
└── tests/                   # Unit tests
```

### Component Interaction

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface                            │
│  (CLI Args, Config Files, Interactive Scripts)              │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼─────┐          ┌─────▼──────┐
    │ Generate │          │  Analysis  │
    │ Answers  │          │  Pipeline  │
    └────┬─────┘          └─────┬──────┘
         │                      │
    ┌────▼─────────────────────▼─────┐
    │    Data Layer (data_utils.py)  │
    │  - load_ds()                    │
    │  - Dataset preprocessing        │
    └────────────┬────────────────────┘
                 │
         ┌───────┴────────┐
         │                │
    ┌────▼─────┐    ┌────▼──────────┐
    │  Model   │    │  Uncertainty  │
    │  Layer   │    │  Measures     │
    └────┬─────┘    └────┬──────────┘
         │               │
    ┌────▼───────────────▼────┐
    │   Results / Artifacts    │
    │  (Pickle, JSON, Plots)   │
    └──────────────────────────┘
```

---

## Core Modules

### 1. Models Module (`src/models/`)

#### `base_model.py`
Defines the abstract base class for all LLM implementations.

**Key Components:**
```python
class BaseModel(ABC):
    """Abstract base class for language models."""
    
    @abstractmethod
    def predict(self, input_data, temperature):
        """Generate predictions with token-level outputs."""
        pass
    
    @abstractmethod
    def get_p_true(self, input_data):
        """Get probability of answering 'True' (for p_true baseline)."""
        pass
```

**Stop Sequences:**
- `STOP_SEQUENCES`: Default stop markers for short answers
- `STOP_SEQUENCES_DETAILED`: Relaxed stop markers for long answers

#### `huggingface_models.py`
HuggingFace model wrapper with advanced features.

**Class: `HuggingfaceModel`**

**Initialization:**
```python
def __init__(self, model_name, stop_sequences=None, max_new_tokens=None, cache_dir=None):
    """
    Args:
        model_name: Model identifier (e.g., "Llama-3.2-1B", "Mistral-7B-v0.1")
        stop_sequences: List of strings to stop generation
        max_new_tokens: Maximum tokens to generate
        cache_dir: Custom cache directory for model weights
    """
```

**Supported Models:**
- **Llama Family**: Llama-2, Llama-3, Llama-3.1, Llama-3.2 (1B, 7B, 8B, 13B, 70B)
- **Mistral Family**: Mistral-7B, Mixtral-8x7B
- **Falcon**: Falcon-7B, Falcon-40B

**Key Methods:**

1. **`predict(input_data, temperature, return_full=False)`**
   ```python
   """Generate answer with detailed token information.
   
   Returns:
       Tuple of (answer, log_likelihoods, embedding, token_ids, tokens)
       - answer: str - Generated text (stop sequences removed)
       - log_likelihoods: List[float] - Log-prob for each token
       - embedding: torch.Tensor - Last token embedding
       - token_ids: List[int] - Token IDs
       - tokens: List[str] - Token strings
   """
   ```

2. **`get_p_true(input_data)`**
   ```python
   """Get log-probability of generating 'A' (True).
   
   Used for p_true baseline uncertainty estimation.
   
   Returns:
       float: -NLL of generating 'A'
   """
   ```

**Special Features:**

- **Multi-GPU Support**: Automatic distribution across GPUs
  ```python
  def get_gpu_memory_dict():
      """Auto-detect GPU memory and create max_memory dict."""
  ```

- **Cache Management**: 
  ```python
  def get_hf_cache_dir():
      """Respects HF_MODELS_CACHE, TRANSFORMERS_CACHE, HF_HOME environment variables."""
  ```

- **Token Extraction**: Reliable token-based extraction (not string matching)
  ```python
  # Extract only generated tokens
  generated_token_ids = outputs.sequences[0][n_input_token:]
  answer = self.tokenizer.decode(generated_token_ids)
  ```

- **Stop Sequence Handling**: Accurate stop token counting
  ```python
  # Calculate how many tokens were in the stop sequence
  stop_tokens_encoded = self.tokenizer.encode(stop, add_special_tokens=False)
  num_stop_tokens = len(stop_tokens_encoded)
  n_generated = len(generated_token_ids) - num_stop_tokens
  ```

**Quantization Support:**
- 8-bit quantization: Add `-8bit` suffix to model name
- 4-bit quantization: Add `-4bit` suffix (Mistral only)

**Example:**
```python
# Initialize model
model = HuggingfaceModel(
    model_name="Llama-3.2-1B",
    stop_sequences="default",
    max_new_tokens=50
)

# Generate answer
answer, log_liks, embedding, token_ids, tokens = model.predict(
    input_data="Question: What is the capital of France?\nAnswer:",
    temperature=0.0
)
```

---

### 2. Uncertainty Measures Module (`src/uncertainty_measures/`)

#### `rw_gnll.py` - Relevance-Weighted G-NLL

**Core Algorithm:**
```
RW-G-NLL = Σ [R_T(y_t) · (-log P(y_t))] / Σ R_T(y_t)
```
Where `R_T(y_t)` measures semantic relevance of token `y_t` to the prompt.

**Key Functions:**

1. **`initialize_similarity_model(model_name)`**
   ```python
   """Initialize cross-encoder for semantic similarity.
   
   Args:
       model_name: Default 'cross-encoder/stsb-roberta-large'
   
   Returns:
       CrossEncoder model ready for inference
   """
   ```

2. **`compute_similarity(similarity_model, text1, text2)`**
   ```python
   """Compute semantic similarity score.
   
   Returns:
       float: Normalized similarity in [0, 1]
   """
   ```

3. **`remove_token_at_position(response_text, tokenizer, token_position)`**
   ```python
   """Remove token at position t from response.
   
   Used to compute ablated response: y \ {y_t}
   
   Returns:
       str: Response with token removed
   """
   ```

4. **`compute_token_relevance_weights(prompt_x, response_y_greedy, tokenizer, similarity_model, cache=None)`**
   ```python
   """Compute relevance weight for each token.
   
   For each token t:
       R_T(y_t) = 1 - similarity(x ∪ y, x ∪ (y \ {y_t}))
   
   Args:
       prompt_x: Input prompt
       response_y_greedy: Generated response
       tokenizer: Tokenizer for token manipulation
       similarity_model: Initialized similarity model
       cache: Optional dict for caching similarities
   
   Returns:
       List[float]: Relevance weight per token
   """
   ```

5. **`compute_rw_gnll(entry, similarity_model, tokenizer, cache=None, return_relevance_weights=False)`**
   ```python
   """Compute RW-G-NLL for an entry.
   
   Args:
       entry: Dict with 'question', 'context', 'most_likely_answer'
       similarity_model: Initialized similarity model
       tokenizer: Tokenizer for processing
       cache: Optional cache
       return_relevance_weights: Whether to return weights
   
   Returns:
       Tuple[float, Optional[List[float]]]: (rw_gnll_score, relevance_weights)
   """
   ```

**Usage Example:**
```python
from uncertainty_measures.rw_gnll import initialize_similarity_model, compute_rw_gnll
from transformers import AutoTokenizer

# Initialize models
similarity_model = initialize_similarity_model('cross-encoder/stsb-roberta-large')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')

# Compute RW-G-NLL
entry = {
    'question': 'What is the capital of France?',
    'context': '',
    'most_likely_answer': {
        'response': 'Paris',
        'token_log_likelihoods': [-0.234, -0.156]
    }
}

rw_gnll_score, relevance_weights = compute_rw_gnll(
    entry, 
    similarity_model, 
    tokenizer,
    return_relevance_weights=True
)
```

#### `sar.py` - Shifting Attention to Relevance

**Core Algorithm:**
```
SAR = (1/M) * Σ_m [Σ_t R_T(y_t^m) * (-log P(y_t^m))] / [Σ_t R_T(y_t^m)]
```
Multi-sample extension of RW-G-NLL.

**Key Functions:**

1. **`compute_sar_for_entry(entry, responses, similarity_model, tokenizer, cache=None)`**
   ```python
   """Compute SAR score across multiple sampled responses.
   
   Args:
       entry: Dict with 'question', 'context'
       responses: List of sampled responses with log-likelihoods
       similarity_model: Similarity model
       tokenizer: Tokenizer
       cache: Optional cache
   
   Returns:
       Tuple[float, Dict]: (sar_score, details_dict)
   """
   ```

2. **`compute_sar(entry, similarity_model, tokenizer, cache=None, num_samples=None)`**
   ```python
   """Convenience function to compute SAR from entry['responses'].
   
   Returns:
       Tuple[float, Dict]: (sar_score, details_dict)
   """
   ```

**Details Dict Structure:**
```python
{
    'num_samples': int,
    'per_sample_scores': List[float],
    'per_sample_relevance_sums': List[float],
    'mean_sar': float,
    'std_sar': float,
    'num_valid_samples': int
}
```

#### Other Uncertainty Measures

- **`p_true.py`**: P(True) baseline
- **`p_ik.py`**: P_IK baseline  
- **`semantic_entropy.py`**: Semantic entropy computation

---

### 3. Data Module (`src/data/`)

#### `data_utils.py`

**Function: `load_ds(dataset_name, seed, add_options=None)`**

**Supported Datasets:**

1. **SQuAD v2.0** (`squad`)
   ```python
   # Questions with context passages
   # Includes unanswerable questions
   ```

2. **TriviaQA** (`trivia_qa`)
   ```python
   # Open-domain questions
   # Formatted in SQuAD style
   ```

3. **SVAMP** (`svamp`)
   ```python
   # Math word problems
   # Equation and answer format
   ```

4. **Natural Questions** (`nq`)
   ```python
   # Wikipedia-based questions
   # Multiple possible answers
   ```

5. **BioASQ** (`bioasq`)
   ```python
   # Biomedical questions
   # Expert-annotated answers
   ```

**Return Format:**
```python
train_dataset, validation_dataset = load_ds('trivia_qa', seed=42)

# Each entry:
{
    'question': str,
    'answers': {'text': List[str]},
    'context': str (optional),
    'id': str
}
```

---

### 4. Analysis Module (`src/analysis/`)

#### `phase1_baseline_metrics.py`
**Purpose:** Compute baseline statistics and metrics.

**Outputs:**
- `baseline_metrics.json`: Accuracy, NLL statistics
- `token_statistics.json`: Token count distributions
- `baseline_metrics_{short/long}.csv`: CSV export

**Usage:**
```bash
python -m src.analysis.phase1_baseline_metrics \
  --long-pickle path/to/validation_generations.pkl \
  --output-dir results/phase1_long
```

#### `phase1_5_token_nll_analysis.py`
**Purpose:** Token-level NLL analysis and visualization.

**Outputs:**
- `nll_vs_position.png`: NLL by token position
- `nll_distribution.png`: Distribution histograms
- `position_nll_heatmap.png`: Heatmap visualization
- `sentence_level_nll_examples.json`: Detailed examples
- `example_{01-10}_token_nlls.png`: Individual example plots

**Key Metrics:**
- Token-level NLL values
- Positional trends
- Correct vs. incorrect distributions

**Usage:**
```bash
python -m src.analysis.phase1_5_token_nll_analysis \
  --pickle-path path/to/pickle.pkl \
  --model-name Llama-3.2-1B \
  --sample-size 100 \
  --output-dir results/phase1_5_long
```

#### `phase1_6_prefix_nll_analysis.py`
**Purpose:** Prefix-based NLL analysis for early stopping.

**Algorithm:**
```
For each prefix length k:
  Compute NLL of first k tokens
  Compare against answer correctness
  Calculate AUROC
```

**Outputs:**
- AUROC curves for different k values
- Optimal k selection analysis

**Usage:**
```bash
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path path/to/pickle.pkl \
  --output-dir results/phase1_6_long \
  --max-prefix-len 50 \
  --ks 1 3 5 10 20
```

#### `phase2_token_importance.py`
**Purpose:** Compute token relevance weights (SAR-style analysis).

**Outputs:**
- `relevance_vs_position.png`: Relevance by position
- `token_importance_examples.json`: Detailed examples with weights

**Usage:**
```bash
python -m src.analysis.phase2_token_importance \
  --pickle-path path/to/pickle.pkl \
  --model-name Llama-3.2-1B \
  --similarity-model cross-encoder/stsb-roberta-large \
  --sample-size 50 \
  --output-dir results/phase2_long
```

#### `phase5_comparative_analysis.py`
**Purpose:** AUROC comparison of uncertainty metrics.

**Compared Metrics:**
- G-NLL (baseline)
- RW-G-NLL
- SAR (if multi-sample)
- Semantic Entropy
- Length-based baselines

**Outputs:**
- `auroc_comparison.csv`: AUROC scores
- `roc_curves.png`: ROC curve plots
- `cost_performance_plot.png`: Cost-benefit analysis

**Usage:**
```bash
python -m src.analysis.phase5_comparative_analysis \
  --pickle-path path/to/pickle.pkl \
  --model-name Llama-3.2-1B \
  --similarity-model cross-encoder/stsb-roberta-large \
  --output-dir results/phase5_long
```

#### `token_visualization_app.py`
**Purpose:** Interactive Streamlit app for visualizing token-level analysis.

**Features:**
- Load JSON results from analysis phases
- Interactive token highlighting by NLL/relevance
- Side-by-side comparison
- Export selected examples

**Usage:**
```bash
streamlit run src/analysis/token_visualization_app.py
```

**Modes:**
1. **Raw NLL (Phase 1.5)**: Load `sentence_level_nll_examples.json`
2. **Relevance-weighted (Phase 2)**: Load `token_importance_examples.json`

---

### 5. Generation Module (`src/generate_answers.py`)

**Main Function: `main(args)`**

**Pipeline Steps:**
1. **Setup**
   - Initialize Weights & Biases
   - Load dataset
   - Construct few-shot prompt
   
2. **Model Initialization**
   - Load LLM
   - Configure generation parameters
   
3. **Answer Generation**
   - Generate answers with token-level info
   - Compute log-likelihoods
   - Extract embeddings
   
4. **Evaluation**
   - Compute accuracy (ROUGE / LLM judge)
   - Save results to pickle
   
5. **Uncertainty Computation** (optional)
   - Compute uncertainty measures
   - Save uncertainty metrics

**Key Arguments:**
```python
# Model configuration
--model_name: str              # e.g., "Llama-3.2-1B"
--model_max_new_tokens: int    # Max tokens to generate

# Dataset configuration
--dataset: str                 # e.g., "trivia_qa"
--num_samples: int             # Number of validation examples
--use_context: bool            # Include context in prompt

# Generation configuration
--temperature: float           # 0.0 for greedy, >0 for sampling
--num_generations: int         # Number of samples per question

# Evaluation configuration
--metric: str                  # "squad", "rouge", "llm_llama-3.1-70b"
--brief_prompt: str            # "short" or "detailed"

# Experiment tracking
--entity: str                  # Wandb entity
--project: str                 # Wandb project name
```

**Output Format (Pickle):**
```python
{
    'question_id': {
        'question': str,
        'answers': List[str],
        'context': str,
        'most_likely_answer': {
            'response': str,
            'token_log_likelihoods': List[float],
            'token_ids': List[int],           # NEW!
            'tokens': List[str],              # NEW!
            'embedding': np.ndarray,
            'accuracy': float
        },
        'responses': [                        # Multi-sample
            {
                'response': str,
                'token_log_likelihoods': List[float],
                'token_ids': List[int],
                'tokens': List[str],
                'embedding': np.ndarray,
                'accuracy': float
            },
            ...
        ]
    },
    ...
}
```

**Example Usage:**
```bash
# Short answer generation (greedy)
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 200 \
  --num_generations 1 \
  --temperature 0.0 \
  --model_max_new_tokens 50 \
  --brief_prompt short \
  --metric squad \
  --entity myteam \
  --project nllSAR_short

# Long answer generation (with LLM judge)
python -m src.generate_answers \
  --model_name Llama-3.2-1B \
  --dataset trivia_qa \
  --num_samples 200 \
  --num_generations 1 \
  --temperature 0.0 \
  --model_max_new_tokens 200 \
  --brief_prompt detailed \
  --metric llm_llama-3.1-70b \
  --use_context True \
  --entity myteam \
  --project nllSAR_long
```

---

### 6. Utilities Module (`src/utils/`)

#### `utils.py`
General utility functions.

**Key Functions:**

1. **`setup_logger()`**
   ```python
   """Configure logging with INFO level and timestamps."""
   ```

2. **`init_model(args)`**
   ```python
   """Initialize model from args.
   
   Returns:
       HuggingfaceModel instance
   """
   ```

3. **`get_metric(metric_name)`**
   ```python
   """Get evaluation metric function.
   
   Supported:
       - 'squad': F1 and Exact Match
       - 'rouge': ROUGE-L
       - 'llm_*': LLM-as-a-judge
   """
   ```

4. **`construct_fewshot_prompt_from_indices(dataset, indices, brief, arg, make_prompt)`**
   ```python
   """Construct few-shot prompt from examples.
   
   Returns:
       str: Formatted few-shot prompt
   """
   ```

5. **`get_make_prompt(args)`**
   ```python
   """Get prompt construction function based on dataset.
   
   Returns:
       Callable: Function to format single example
   """
   ```

**Prompt Templates:**
```python
BRIEF_PROMPTS = {
    'short': "Answer concisely in 1-5 words.",
    'detailed': "Provide a detailed explanation."
}
```

#### `eval_utils.py`
Evaluation utilities.

**Key Functions:**

1. **`compute_rouge_scores(prediction, reference)`**
   ```python
   """Compute ROUGE-L scores.
   
   Returns:
       Dict with 'rouge-l' scores
   """
   ```

2. **`compute_squad_metrics(prediction, references)`**
   ```python
   """Compute SQuAD F1 and EM.
   
   Returns:
       Dict with 'f1' and 'exact_match'
   """
   ```

3. **`llm_judge_evaluate(question, context, prediction, reference, model_name)`**
   ```python
   """Use LLM to judge answer correctness.
   
   Returns:
       float: Correctness score [0, 1]
   """
   ```

#### `openai.py`
OpenAI API utilities for LLM judge.

**Functions:**
- `call_openai_api()`: Call OpenAI API with retry logic
- `parse_llm_judge_response()`: Parse judge output

---

## Data Flow

### 1. Answer Generation Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Dataset Loading (data_utils.load_ds)                      │
│    - Load train/validation splits                            │
│    - Normalize format                                         │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Few-Shot Prompt Construction (utils.construct_fewshot...)│
│    - Sample few-shot examples                                │
│    - Format with brief instructions                          │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. Model Initialization (HuggingfaceModel.__init__)         │
│    - Load model weights                                       │
│    - Setup tokenizer                                          │
│    - Configure multi-GPU                                      │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. Generation Loop (for each validation example)             │
│    ┌────────────────────────────────────────────────────┐   │
│    │ a. Format input with few-shot prompt               │   │
│    │ b. Call model.predict(input, temperature)          │   │
│    │    - Generate tokens                                │   │
│    │    - Compute log-likelihoods                        │   │
│    │    - Extract embeddings                             │   │
│    │    - Track token IDs and strings                    │   │
│    │ c. Evaluate answer (ROUGE / LLM judge)             │   │
│    │ d. Store results with token-level info             │   │
│    └────────────────────────────────────────────────────┘   │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. Save Results                                               │
│    - Pickle: validation_generations.pkl                       │
│    - Wandb: Upload to cloud                                   │
└──────────────────────────────────────────────────────────────┘
```

### 2. Analysis Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│ Load Pickle (validation_generations.pkl)                     │
└───────────────────────┬──────────────────────────────────────┘
                        │
         ┌──────────────┼──────────────┬──────────────┐
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────┐ ┌──────────────┐ ┌──────────┐ ┌────────────┐
│  Phase 1    │ │  Phase 1.5   │ │ Phase 2  │ │  Phase 5   │
│  Baseline   │ │  Token NLL   │ │ Token    │ │ AUROC      │
│  Metrics    │ │  Analysis    │ │ Relevance│ │ Comparison │
└─────────────┘ └──────────────┘ └──────────┘ └────────────┘
     │                │              │              │
     │                │              │              │
     ▼                ▼              ▼              ▼
┌─────────────┐ ┌──────────────┐ ┌──────────┐ ┌────────────┐
│ baseline_   │ │ nll_vs_      │ │ relevance│ │ auroc_     │
│ metrics.json│ │ position.png │ │ weights  │ │ comparison │
│ token_stats │ │ sentence_    │ │ token_   │ │ roc_curves │
│ .json       │ │ examples.json│ │ importance│ │ .png       │
└─────────────┘ └──────────────┘ └──────────┘ └────────────┘
```

### 3. Token Alignment System

**The Critical Innovation:**

Old pickles stored only `response` and `token_log_likelihoods`, causing misalignment when re-tokenizing. New pickles store the **exact tokens** used during generation.

```
Generation Time                    Analysis Time
──────────────────                 ─────────────

model.generate()                   Load pickle
    │                                  │
    ▼                                  ▼
token_ids = [450, 11045, ...]      token_ids = entry['most_likely_answer']['token_ids']
tokens = [' The', ' Battle', ...]  tokens = entry['most_likely_answer']['tokens']
log_liks = [-0.234, -0.156, ...]   log_liks = entry['most_likely_answer']['token_log_likelihoods']
    │                                  │
    ▼                                  ▼
Store ALL THREE in pickle          Use stored tokens directly
                                   NO RE-TOKENIZATION!
```

**Why This Matters:**
```python
# Without stored tokens (OLD WAY - BROKEN):
response = "The Battle of Hastings"
re_tokenized = tokenizer.encode(response)  # May differ from generation!
# Result: [450, 11046, 315, 19826]  ← Different token 11046!

# With stored tokens (NEW WAY - CORRECT):
token_ids = entry['most_likely_answer']['token_ids']  
# Result: [450, 11045, 315, 19826]  ← Exact match!
```

---

## API Reference

### HuggingfaceModel

```python
class HuggingfaceModel(BaseModel):
    """HuggingFace LLM wrapper with token-level outputs."""
    
    def __init__(
        self,
        model_name: str,
        stop_sequences: Optional[Union[str, List[str]]] = None,
        max_new_tokens: Optional[int] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize model.
        
        Args:
            model_name: Model identifier or path
            stop_sequences: 'default', 'detailed', or list of strings
            max_new_tokens: Maximum tokens to generate (required)
            cache_dir: Optional custom cache directory
        """
    
    def predict(
        self,
        input_data: str,
        temperature: float,
        return_full: bool = False
    ) -> Union[str, Tuple[str, List[float], torch.Tensor, List[int], List[str]]]:
        """Generate answer with token-level information.
        
        Args:
            input_data: Input prompt
            temperature: Sampling temperature (0.0 for greedy)
            return_full: If True, return only full answer string
        
        Returns:
            If return_full=False:
                Tuple of (answer, log_likelihoods, embedding, token_ids, tokens)
            If return_full=True:
                str: Full answer including input
        """
    
    def get_p_true(self, input_data: str) -> float:
        """Get log-probability of generating 'A' (True).
        
        Args:
            input_data: Input prompt (will append ' A')
        
        Returns:
            float: Negative loss (log-probability)
        """
```

### RW-G-NLL Functions

```python
def initialize_similarity_model(
    model_name: str = 'cross-encoder/stsb-roberta-large'
) -> CrossEncoder:
    """Initialize semantic similarity model."""

def compute_similarity(
    similarity_model: CrossEncoder,
    text1: str,
    text2: str
) -> float:
    """Compute normalized similarity score [0, 1]."""

def remove_token_at_position(
    response_text: str,
    tokenizer,
    token_position: int
) -> str:
    """Remove token at specified position."""

def compute_token_relevance_weights(
    prompt_x: str,
    response_y_greedy: str,
    tokenizer,
    similarity_model,
    cache: Optional[Dict[Tuple[str, int], float]] = None,
    show_progress: bool = False
) -> List[float]:
    """Compute relevance weight for each token.
    
    Returns:
        List[float]: Relevance weights (one per token)
    """

def compute_rw_gnll(
    entry: Dict[str, Any],
    similarity_model,
    tokenizer,
    cache: Optional[Dict[Tuple[str, int], float]] = None,
    return_relevance_weights: bool = False
) -> Tuple[float, Optional[List[float]]]:
    """Compute RW-G-NLL score.
    
    Returns:
        Tuple of (rw_gnll_score, relevance_weights)
    """
```

### SAR Functions

```python
def compute_sar_for_entry(
    entry: Dict[str, Any],
    responses: List[Dict[str, Any]],
    similarity_model,
    tokenizer,
    cache: Optional[Dict[Tuple[str, int], float]] = None,
    show_progress: bool = False
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Compute SAR score across multiple samples.
    
    Returns:
        Tuple of (sar_score, details_dict)
    """

def compute_sar(
    entry: Dict[str, Any],
    similarity_model,
    tokenizer,
    cache: Optional[Dict[Tuple[str, int], float]] = None,
    num_samples: Optional[int] = None,
    show_progress: bool = False
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Convenience function to compute SAR from entry['responses']."""
```

### Data Loading

```python
def load_ds(
    dataset_name: str,
    seed: int,
    add_options: Optional[bool] = None
) -> Tuple[Union[List, datasets.Dataset], Union[List, datasets.Dataset]]:
    """Load dataset.
    
    Args:
        dataset_name: One of ['squad', 'trivia_qa', 'svamp', 'nq', 'bioasq']
        seed: Random seed for train/test split
        add_options: Whether to include MC options (if available)
    
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
```

---

## Configuration

### Environment Variables

```bash
# HuggingFace cache (priority order)
export HF_MODELS_CACHE="/path/to/cache"        # Direct model cache path (highest priority)
export TRANSFORMERS_CACHE="/path/to/cache"     # Alternative direct cache path
export HF_HOME="/path/to/hf_home"              # Base HF directory (models in hub/)

# HuggingFace token
export HF_TOKEN="hf_your_token_here"

# User directory for wandb
export USER="your_username"                     # Or USERNAME on Windows
export SCRATCH_DIR="/path/to/scratch"          # For wandb logs

# OpenAI (for LLM judge)
export OPENAI_API_KEY="sk-your_key_here"
```

### Command-Line Arguments

**Generation Script (`src/generate_answers.py`):**

```bash
# Required
--model_name <name>           # Model to use
--dataset <name>              # Dataset to evaluate
--num_samples <int>           # Number of validation examples
--model_max_new_tokens <int>  # Max tokens to generate

# Generation
--temperature <float>         # Default: 0.0 (greedy)
--num_generations <int>       # Default: 1 (multi-sample for SAR)
--num_few_shot <int>          # Default: 5

# Prompt
--brief_prompt <type>         # 'short' or 'detailed'
--use_context <bool>          # Include context passages

# Evaluation
--metric <name>               # 'squad', 'rouge', 'llm_llama-3.1-70b'

# Experiment tracking
--entity <name>               # Wandb entity
--project <name>              # Wandb project
--debug                       # Use debug project

# Optional
--compute_p_true <bool>       # Compute p_true baseline (default: False)
--random_seed <int>           # Default: 10
```

**Analysis Scripts:**

```bash
# Phase 1: Baseline Metrics
python -m src.analysis.phase1_baseline_metrics \
  --long-pickle <path>       # Or --short-pickle
  --output-dir <path>

# Phase 1.5: Token NLL
python -m src.analysis.phase1_5_token_nll_analysis \
  --pickle-path <path> \
  --model-name <name> \
  --sample-size <int> \
  --output-dir <path>

# Phase 1.6: Prefix NLL
python -m src.analysis.phase1_6_prefix_nll_analysis \
  --pickle-path <path> \
  --output-dir <path> \
  --max-prefix-len <int> \
  --ks <int> <int> ...       # List of k values
  [--use-rouge]              # Use ROUGE instead of LLM judge
  [--rouge-threshold <float>] # Default: 0.3

# Phase 2: Token Importance
python -m src.analysis.phase2_token_importance \
  --pickle-path <path> \
  --model-name <name> \
  --similarity-model <name> \
  --sample-size <int> \
  --output-dir <path>

# Phase 5: AUROC Comparison
python -m src.analysis.phase5_comparative_analysis \
  --pickle-path <path> \
  --model-name <name> \
  --similarity-model <name> \
  --output-dir <path> \
  [--use-rouge] \
  [--rouge-threshold <float>]
```

### Configuration Files

**`nllsar.yml`** - Conda environment:
```yaml
name: nllsar
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch
  - transformers
  - datasets
  - wandb
  # ... more dependencies
```

**`requirements.txt`** - Python packages:
```
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
datasets>=2.12.0
sentence-transformers>=2.2.0
wandb>=0.15.0
streamlit>=1.25.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
rouge-score>=0.1.2
openai>=0.27.0
```

---

## Testing and Analysis

### Verification Scripts

**1. Verify Pickle Structure**

```python
import pickle

# Load pickle
with open('path/to/validation_generations.pkl', 'rb') as f:
    data = pickle.load(f)

# Get first example
example = list(data.values())[0]
mla = example['most_likely_answer']

# Check required fields
print("✅ Has token_ids:", 'token_ids' in mla)
print("✅ Has tokens:", 'tokens' in mla)
print("✅ Has token_log_likelihoods:", 'token_log_likelihoods' in mla)

# Check alignment
print("✅ Token count:", len(mla.get('tokens', [])))
print("✅ Log-lik count:", len(mla.get('token_log_likelihoods', [])))
print("✅ Match:", len(mla.get('tokens', [])) == len(mla.get('token_log_likelihoods', [])))

# Show sample
print("\nFirst 5 tokens:", mla.get('tokens', [])[:5])
print("First 5 log-liks:", mla.get('token_log_likelihoods', [])[:5])
```

**2. Test Model Loading**

```python
from models.huggingface_models import HuggingfaceModel

# Test initialization
model = HuggingfaceModel(
    model_name="Llama-3.2-1B",
    stop_sequences="default",
    max_new_tokens=50
)

# Test generation
answer, log_liks, embedding, token_ids, tokens = model.predict(
    input_data="Question: What is 2+2?\nAnswer:",
    temperature=0.0
)

print("Answer:", answer)
print("Num tokens:", len(tokens))
print("Tokens:", tokens)
print("Log-likelihoods:", log_liks)
```

**3. Test RW-G-NLL Computation**

```python
from uncertainty_measures.rw_gnll import initialize_similarity_model, compute_rw_gnll
from transformers import AutoTokenizer

# Initialize
similarity_model = initialize_similarity_model()
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')

# Test entry
entry = {
    'question': 'What is the capital of France?',
    'context': '',
    'most_likely_answer': {
        'response': 'Paris',
        'token_log_likelihoods': [-0.234, -0.156],
        'token_ids': [40,Pairs],
        'tokens': ['Par', 'is']
    }
}

# Compute
rw_gnll, weights = compute_rw_gnll(
    entry, similarity_model, tokenizer,
    return_relevance_weights=True
)

print("RW-G-NLL:", rw_gnll)
print("Relevance weights:", weights)
```

### Common Issues and Solutions

**Issue 1: Token count mismatch**
```
WARNING: Token count mismatch: 45 relevance weights vs 47 token log-likelihoods
```
**Solution:** Using old pickle without `token_ids` and `tokens`. Re-generate with updated code.

**Issue 2: CUDA out of memory**
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Reduce `--num_samples`
- Use smaller model (1B instead of 7B)
- Add `-8bit` suffix for quantization
- Set `CUDA_VISIBLE_DEVICES` to use specific GPUs

**Issue 3: HuggingFace token errors**
```
401 Client Error: Unauthorized
```
**Solution:**
```bash
# Set token
export HF_TOKEN="hf_your_token_here"

# Or use script
./setup_hf_token.ps1
```

**Issue 4: Stop sequence not removed**
```
ValueError: Error: Stop words not removed successfully!
```
**Solution:** This is expected for some models (Falcon). The code handles it gracefully.

---

## Performance Considerations

### GPU Memory Usage

**Model Size Guidelines:**
- **1B models**: 1 GPU (8GB VRAM)
- **7B models**: 1-2 GPUs (16-24GB VRAM)
- **13B models**: 2-4 GPUs (24-48GB VRAM)
- **70B models**: 4-8 GPUs (80GB+ total VRAM)

**Memory Optimization Tips:**
1. Use 8-bit quantization: Add `-8bit` to model name
2. Clear cache frequently: `torch.cuda.empty_cache()`
3. Use smaller batch sizes
4. Enable gradient checkpointing (for training)

### Speed Optimization

**Generation Speed:**
- Greedy decoding: ~1-2 tokens/sec for 7B model
- Sampling: Slightly slower
- Multi-sample: Linear scaling with `num_generations`

**Analysis Speed:**
- Phase 1: Fast (~1 min for 200 examples)
- Phase 1.5: Medium (~5 min for 100 examples)
- Phase 2: Slow (~30 min for 50 examples, depends on similarity model)
- Phase 5: Medium (~10 min for 200 examples)

**Caching:**
- RW-G-NLL uses caching for similarity computations
- Cache is per-session (not persisted)
- Can speed up by 10-100x for repeated tokens

---

## Extension Points

### Adding New Models

1. **Inherit from `BaseModel`:**
```python
class MyCustomModel(BaseModel):
    def predict(self, input_data, temperature):
        # Implement generation
        # Must return: (answer, log_liks, embedding, token_ids, tokens)
        pass
    
    def get_p_true(self, input_data):
        # Implement p_true computation
        pass
```

2. **Update `utils.init_model()`:**
```python
if args.model_name.startswith('my_custom_'):
    return MyCustomModel(...)
```

### Adding New Datasets

1. **Update `data_utils.load_ds()`:**
```python
elif dataset_name == 'my_dataset':
    dataset = datasets.load_dataset('my/dataset')
    train_dataset = dataset["train"]
    validation_dataset = dataset["test"]
    
    # Normalize format
    reformat = lambda x: {
        'question': x['q'],
        'answers': {'text': x['a']},
        'context': x.get('c', ''),
        'id': x['id']
    }
    
    train_dataset = [reformat(d) for d in train_dataset]
    validation_dataset = [reformat(d) for d in validation_dataset]
```

2. **Update dataset-specific logic in `generate_answers.py`.**

### Adding New Uncertainty Metrics

1. **Create new file in `src/uncertainty_measures/`:**
```python
# src/uncertainty_measures/my_metric.py

def compute_my_metric(entry, **kwargs):
    """Compute my custom uncertainty metric.
    
    Args:
        entry: Dict with question, answer, token info
    
    Returns:
        float: Uncertainty score
    """
    # Your implementation
    pass
```

2. **Add to `compute_uncertainty_measures.py`:**
```python
if 'my_metric' in args.uncertainty_measures:
    my_metric_scores = [compute_my_metric(entry) for entry in data.values()]
    results['my_metric'] = my_metric_scores
```

3. **Add to Phase 5 analysis for AUROC comparison.**

---

## References

### Key Papers
1. **G-NLL Baseline**: Token log-likelihood summation for uncertainty
2. **RW-G-NLL**: Relevance-weighted uncertainty estimation
3. **SAR**: Shifting Attention to Relevance (multi-sample)
4. **Semantic Entropy**: Kuhn et al., "Semantic Uncertainty"

### Documentation Files
- `README.md`: Project overview
- `QUICK_START.md`: Getting started guide
- `GENERATION_SETTINGS_GUIDE.md`: Generation parameter reference
- `ANALYSIS_README.md`: Analysis pipeline details
- `CHANGES_SUMMARY.md`: Recent changes and fixes
- `ENVIRONMENT_SETUP.md`: Environment configuration
- `GPU_REQUIREMENTS.md`: GPU setup guide
- `MULTI_GPU_GUIDE.md`: Multi-GPU usage
- `MODEL_CACHE_GUIDE.md`: Cache management
- `GNLL_BASELINE_README.md`: G-NLL baseline guide
- `GREEDY_DECODING_README.md`: Greedy decoding info

---

## License and Citation

*[Add license information and citation details here]*

---

## Contact

*[Add contact information here]*

---

**Last Updated:** November 2025
**Version:** 2.0
**Maintainer:** nllSAR Team

