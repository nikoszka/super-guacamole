# 🏗️ nllSAR Architecture Guide

## Overview

This document provides an in-depth view of the nllSAR architecture, design patterns, and implementation details.

---

## Design Principles

### 1. Modularity
- **Separation of Concerns**: Models, uncertainty measures, data loading, and analysis are independent modules
- **Plug-and-Play**: Easy to add new models, datasets, or uncertainty metrics
- **Loose Coupling**: Modules communicate through well-defined interfaces

### 2. Reproducibility
- **Exact Token Tracking**: Store token IDs and strings to ensure perfect alignment
- **Deterministic Generation**: Greedy decoding (temp=0.0) produces reproducible results
- **Experiment Tracking**: Weights & Biases integration for full experiment history

### 3. Scalability
- **Multi-GPU Support**: Automatic distribution across available GPUs
- **Memory Management**: Aggressive cache clearing and quantization options
- **Batch Processing**: Efficient processing of large validation sets

### 4. Extensibility
- **Abstract Base Classes**: Easy to subclass for new model types
- **Configuration-Driven**: Command-line args and environment variables for flexibility
- **Modular Analysis**: Independent analysis phases that can be run separately

---

## Core Architectural Patterns

### 1. Abstract Factory Pattern (Models)

```python
# Base interface
class BaseModel(ABC):
    @abstractmethod
    def predict(self, input_data, temperature):
        pass

# Concrete implementations
class HuggingfaceModel(BaseModel):
    def predict(self, input_data, temperature):
        # Implementation for HF models
        pass

# Factory function
def init_model(args):
    if 'llama' in args.model_name.lower():
        return HuggingfaceModel(...)
    elif 'gpt' in args.model_name.lower():
        return OpenAIModel(...)
    # ... more model types
```

**Benefits:**
- Easy to add new model types without changing generation code
- Consistent interface for all models
- Decouples model selection from usage

### 2. Strategy Pattern (Uncertainty Measures)

```python
# Different strategies for computing uncertainty
def compute_gnll(entry):
    """Standard G-NLL: sum of token log-likelihoods."""
    return -sum(entry['most_likely_answer']['token_log_likelihoods'])

def compute_rw_gnll(entry, similarity_model, tokenizer):
    """Relevance-weighted G-NLL."""
    # Compute relevance weights
    weights = compute_token_relevance_weights(...)
    # Weight log-likelihoods
    return weighted_sum(weights, log_likelihoods)

def compute_sar(entry, similarity_model, tokenizer):
    """Multi-sample SAR."""
    # Average over multiple samples
    return mean([compute_rw_gnll(r, ...) for r in entry['responses']])

# Usage: select strategy at runtime
if args.uncertainty_method == 'gnll':
    uncertainty = compute_gnll(entry)
elif args.uncertainty_method == 'rw_gnll':
    uncertainty = compute_rw_gnll(entry, sim_model, tokenizer)
# ...
```

**Benefits:**
- Easy to compare different uncertainty methods
- Same interface for all uncertainty measures
- Can add new methods without modifying existing code

### 3. Pipeline Pattern (Generation & Analysis)

```python
# Generation pipeline
def generate_answers_pipeline(args):
    # Stage 1: Setup
    dataset = load_ds(args.dataset)
    model = init_model(args)
    prompt = construct_fewshot_prompt(dataset, args)
    
    # Stage 2: Generation
    results = {}
    for example in dataset:
        input_text = format_input(example, prompt)
        answer, log_liks, embedding, token_ids, tokens = model.predict(input_text, args.temperature)
        results[example['id']] = {
            'response': answer,
            'token_log_likelihoods': log_liks,
            'token_ids': token_ids,
            'tokens': tokens,
            'embedding': embedding
        }
    
    # Stage 3: Evaluation
    for id, result in results.items():
        accuracy = evaluate(result['response'], example['answers'])
        result['accuracy'] = accuracy
    
    # Stage 4: Save
    save_pickle(results, output_path)
    upload_to_wandb(results)
    
    return results
```

**Benefits:**
- Clear sequence of operations
- Easy to debug and monitor progress
- Can checkpoint between stages

### 4. Repository Pattern (Data Access)

```python
# Abstract data access
def load_ds(dataset_name, seed, add_options=None):
    """Load dataset from various sources."""
    if dataset_name == "squad":
        return _load_squad()
    elif dataset_name == "trivia_qa":
        return _load_trivia_qa()
    # ... more datasets

# Normalized output format
def _load_squad():
    dataset = datasets.load_dataset("squad_v2")
    # Normalize to common format
    return {
        'question': str,
        'answers': {'text': List[str]},
        'context': str,
        'id': str
    }
```

**Benefits:**
- Abstracts away dataset-specific details
- Provides consistent interface for all datasets
- Easy to add new datasets

---

## Key Data Structures

### 1. Generation Result (Pickle Format)

```python
# validation_generations.pkl structure
{
    'question_id': {
        # Input data
        'question': str,
        'answers': List[str],
        'context': str,
        
        # Greedy answer (always present)
        'most_likely_answer': {
            'response': str,                    # Generated text
            'token_log_likelihoods': List[float],  # Log P(token|context)
            'token_ids': List[int],             # Token IDs (NEW!)
            'tokens': List[str],                # Token strings (NEW!)
            'embedding': np.ndarray,            # Last token embedding
            'accuracy': float                   # Correctness score [0, 1]
        },
        
        # Multiple samples (if num_generations > 1)
        'responses': [
            {
                'response': str,
                'token_log_likelihoods': List[float],
                'token_ids': List[int],
                'tokens': List[str],
                'embedding': np.ndarray,
                'accuracy': float
            },
            # ... more samples
        ]
    },
    # ... more questions
}
```

**Key Design Decisions:**

1. **Store Token IDs and Strings**: Prevents re-tokenization mismatches
   - Old way: Store only `response`, re-tokenize during analysis → misalignment
   - New way: Store exact tokens used during generation → perfect alignment

2. **Separate `most_likely_answer` and `responses`**: 
   - Greedy answer always in `most_likely_answer`
   - Multiple samples in `responses`
   - Simplifies single-sample analysis

3. **Embeddings Included**: 
   - Enables representation-based uncertainty methods
   - Can compute semantic similarity

### 2. Analysis Output (JSON Format)

**Phase 1: Baseline Metrics**
```json
{
    "accuracy": {
        "mean": 0.75,
        "std": 0.02,
        "per_example": [0.8, 0.7, ...]
    },
    "gnll": {
        "mean": 3.45,
        "std": 1.23,
        "per_example": [2.1, 4.3, ...]
    },
    "token_statistics": {
        "mean_length": 15.3,
        "std_length": 5.2,
        "min_length": 3,
        "max_length": 45
    }
}
```

**Phase 1.5: Token-Level NLL**
```json
{
    "examples": [
        {
            "question": "What is...",
            "response": "The answer is...",
            "tokens": [" The", " answer", " is", "..."],
            "token_nlls": [0.234, 0.156, 0.089, ...],
            "positions": [0, 1, 2, ...],
            "accuracy": 1.0
        },
        // ... more examples
    ],
    "statistics": {
        "mean_nll_per_position": [0.5, 0.4, 0.3, ...],
        "correct_vs_incorrect": {
            "correct_mean": 0.3,
            "incorrect_mean": 0.5
        }
    }
}
```

**Phase 2: Token Importance**
```json
{
    "examples": [
        {
            "question": "What is...",
            "response": "The answer is...",
            "tokens": [" The", " answer", " is", "..."],
            "token_nlls": [0.234, 0.156, 0.089, ...],
            "relevance_weights": [0.2, 0.8, 0.9, ...],
            "weighted_nlls": [0.047, 0.125, 0.080, ...],
            "rw_gnll": 0.252,
            "gnll": 0.479
        },
        // ... more examples
    ]
}
```

**Phase 5: AUROC Comparison**
```json
{
    "auroc_scores": {
        "gnll": 0.72,
        "rw_gnll": 0.78,
        "sar": 0.81,
        "semantic_entropy": 0.75,
        "length": 0.55
    },
    "roc_curves": {
        "gnll": {
            "fpr": [0.0, 0.1, 0.2, ...],
            "tpr": [0.0, 0.3, 0.5, ...],
            "thresholds": [inf, 5.0, 3.0, ...]
        },
        // ... more curves
    }
}
```

---

## Critical Implementation Details

### 1. Token Alignment System

**Problem:** Re-tokenization can produce different tokens than generation.

**Example:**
```python
# During generation
input_ids = tokenizer.encode("Question: What is 2+2?")
# model.generate() produces tokens: [450, 11045, 315]
# tokenizer.decode([450, 11045, 315]) = " The answer is"

# During analysis (OLD WAY - BROKEN)
response = " The answer is"
re_tokenized = tokenizer.encode(response)  
# Might produce: [451, 11045, 315]  ← Different first token!
# Causes: len(tokens) != len(log_likelihoods)
```

**Solution:** Store exact tokens during generation.

```python
# During generation (NEW WAY)
generated_token_ids = outputs.sequences[0][n_input_token:]
tokens = [tokenizer.decode([tid]) for tid in generated_token_ids]

# Store in pickle
result = {
    'response': tokenizer.decode(generated_token_ids),
    'token_ids': generated_token_ids.tolist(),
    'tokens': tokens,
    'token_log_likelihoods': log_likelihoods
}

# During analysis (NEW WAY - CORRECT)
tokens = entry['most_likely_answer']['tokens']  # Use stored tokens
log_liks = entry['most_likely_answer']['token_log_likelihoods']
# Guaranteed: len(tokens) == len(log_liks)
```

### 2. Stop Sequence Handling

**Challenge:** Stop sequences can be multi-token and need accurate removal.

```python
# Example: stop_sequence = "\n\n" might be 1 or 2 tokens depending on context

# WRONG approach (string-based)
if answer.endswith("\n\n"):
    answer = answer[:-2]  # Remove 2 characters
    n_generated = len(generated_token_ids) - 1  # Assume 1 token? WRONG!

# CORRECT approach (token-based)
if answer.endswith(stop_sequence):
    answer = answer[:-len(stop_sequence)]
    # Count actual tokens in stop sequence
    stop_tokens = tokenizer.encode(stop_sequence, add_special_tokens=False)
    num_stop_tokens = len(stop_tokens)
    n_generated = len(generated_token_ids) - num_stop_tokens
```

**Why it matters:** Affects:
- Token count for embeddings extraction
- Log-likelihood summation
- Alignment with stored tokens

### 3. Multi-GPU Memory Management

**Challenge:** Large models need distribution across multiple GPUs.

```python
# Automatic GPU detection
def get_gpu_memory_dict():
    num_gpus = torch.cuda.device_count()
    max_memory = {}
    for i in range(num_gpus):
        total_memory = torch.cuda.get_device_properties(i).total_memory
        # Reserve 500MB for overhead
        available_memory = max(1, int(total_memory / (1024**3) - 0.5))
        max_memory[i] = f'{available_memory}GiB'
    return max_memory

# Use with accelerate
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically distribute across GPUs
    max_memory=get_gpu_memory_dict()
)
```

**Special handling for 70B models:**
```python
if '70b' in model_name.lower():
    # Custom device map to prevent layer splitting
    path = snapshot_download(repo_id=model_name, ...)
    config = AutoConfig.from_pretrained(model_name)
    
    with accelerate.init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    
    model.tie_weights()
    
    # Infer device map
    device_map = accelerate.infer_auto_device_map(
        model.model,
        max_memory={i: max_mem for i in range(num_gpus)},
        dtype='float16'
    )
    
    # Remove split layers (prevents single layer across multiple GPUs)
    device_map = remove_split_layer(device_map)
    
    # Load with custom map
    model = accelerate.load_checkpoint_and_dispatch(
        model, path, device_map=device_map, dtype='float16'
    )
```

### 4. Relevance Weight Caching

**Challenge:** Computing similarity for each token is expensive.

```python
def compute_token_relevance_weights(prompt_x, response_y, tokenizer, similarity_model, cache=None):
    if cache is None:
        cache = {}
    
    relevance_weights = []
    for t in range(num_tokens):
        # Check cache first
        cache_key = (response_y, t)
        if cache_key in cache:
            similarity = cache[cache_key]
        else:
            # Expensive computation
            response_without_t = remove_token_at_position(response_y, tokenizer, t)
            full_text = f"{prompt_x} {response_y}"
            ablated_text = f"{prompt_x} {response_without_t}"
            similarity = compute_similarity(similarity_model, full_text, ablated_text)
            
            # Cache for future use
            cache[cache_key] = similarity
        
        relevance_weight = 1.0 - similarity
        relevance_weights.append(relevance_weight)
    
    return relevance_weights
```

**Cache hit rate:**
- First example: 0% (all cache misses)
- Subsequent examples: 20-40% (common tokens like "the", "is")
- Same example re-processed: 100% (all cache hits)

**Memory usage:**
- Each cache entry: ~8 bytes (float)
- 100 examples × 50 tokens × 8 bytes = ~40KB (negligible)

---

## Error Handling Strategies

### 1. Graceful Degradation

```python
# If token count mismatch, fall back to re-tokenization with warning
if len(tokens) != len(log_likelihoods):
    logging.warning("Token count mismatch, attempting re-tokenization")
    tokens = tokenizer.encode(response, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in tokens]
    
    # Still mismatched? Skip this example
    if len(tokens) != len(log_likelihoods):
        logging.error("Cannot align tokens, skipping example")
        return None
```

### 2. Retry Logic

```python
# OpenAI API calls with exponential backoff
def call_openai_api(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(...)
            return response
        except openai.error.RateLimitError:
            wait_time = 2 ** attempt  # Exponential backoff
            logging.warning(f"Rate limited, waiting {wait_time}s")
            time.sleep(wait_time)
        except openai.error.APIError as e:
            logging.error(f"API error: {e}")
            if attempt == max_retries - 1:
                raise
    
    raise Exception("Max retries exceeded")
```

### 3. Validation

```python
# Validate pickle structure before analysis
def validate_pickle(data):
    required_fields = ['question', 'answers', 'most_likely_answer']
    mla_fields = ['response', 'token_log_likelihoods', 'token_ids', 'tokens']
    
    for id, entry in data.items():
        # Check top-level fields
        for field in required_fields:
            if field not in entry:
                raise ValueError(f"Missing field '{field}' in entry {id}")
        
        # Check most_likely_answer fields
        mla = entry['most_likely_answer']
        for field in mla_fields:
            if field not in mla:
                raise ValueError(f"Missing field '{field}' in most_likely_answer of entry {id}")
        
        # Check alignment
        if len(mla['tokens']) != len(mla['token_log_likelihoods']):
            raise ValueError(f"Token count mismatch in entry {id}")
    
    logging.info("✅ Pickle validation passed")
    return True
```

---

## Performance Optimization

### 1. GPU Memory Optimization

**Techniques:**
1. **Clear unused tensors:**
   ```python
   del inputs
   torch.cuda.empty_cache()
   ```

2. **Move to CPU immediately:**
   ```python
   embedding = last_layer[:, -1, :].cpu()  # Move to CPU
   del last_layer  # Free GPU memory
   ```

3. **Use quantization:**
   ```python
   # 8-bit quantization reduces memory by ~75%
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       quantization_config=BitsAndBytesConfig(load_in_8bit=True)
   )
   ```

4. **Batch size = 1:**
   ```python
   # Process one example at a time for minimal memory
   for example in tqdm(dataset):
       result = model.predict(example, temperature)
       # Process and save immediately
   ```

### 2. Computation Optimization

**Techniques:**
1. **Vectorization:**
   ```python
   # SLOW: Loop over tokens
   weighted_nlls = []
   for t in range(len(tokens)):
       weighted_nlls.append(weights[t] * (-log_liks[t]))
   
   # FAST: Vectorized
   import numpy as np
   weights = np.array(weights)
   log_liks = np.array(log_liks)
   weighted_nlls = weights * (-log_liks)
   ```

2. **Caching:**
   ```python
   # Cache similarity computations
   cache = {}  # Persist across examples
   
   for entry in dataset:
       rw_gnll, _ = compute_rw_gnll(entry, sim_model, tokenizer, cache=cache)
   ```

3. **Parallel processing (when possible):**
   ```python
   # Process multiple examples in parallel
   from multiprocessing import Pool
   
   def process_example(entry):
       return compute_rw_gnll(entry, sim_model, tokenizer)
   
   with Pool(processes=4) as pool:
       results = pool.map(process_example, dataset)
   ```

### 3. I/O Optimization

**Techniques:**
1. **Stream large files:**
   ```python
   # Don't load entire pickle into memory if not needed
   with open('large_file.pkl', 'rb') as f:
       for entry_id in tqdm(range(num_entries)):
           entry = pickle.load(f)  # Load one at a time
           process(entry)
   ```

2. **Compress outputs:**
   ```python
   import gzip
   import pickle
   
   with gzip.open('results.pkl.gz', 'wb') as f:
       pickle.dump(results, f)
   ```

---

## Testing Strategy

### 1. Unit Tests

```python
# tests/test_models.py
def test_huggingface_model_initialization():
    model = HuggingfaceModel(
        model_name="Llama-3.2-1B",
        stop_sequences="default",
        max_new_tokens=50
    )
    assert model is not None
    assert model.tokenizer is not None

def test_huggingface_model_predict():
    model = HuggingfaceModel(...)
    answer, log_liks, emb, token_ids, tokens = model.predict(
        "Question: What is 2+2?\nAnswer:",
        temperature=0.0
    )
    assert isinstance(answer, str)
    assert len(log_liks) == len(tokens)
    assert len(token_ids) == len(tokens)

# tests/test_uncertainty_measures.py
def test_rw_gnll_computation():
    entry = {
        'question': 'Test question',
        'context': '',
        'most_likely_answer': {
            'response': 'Test answer',
            'token_log_likelihoods': [-0.5, -0.3],
            'token_ids': [1234, 5678],
            'tokens': ['Test', ' answer']
        }
    }
    
    rw_gnll, weights = compute_rw_gnll(
        entry, sim_model, tokenizer,
        return_relevance_weights=True
    )
    
    assert isinstance(rw_gnll, float)
    assert len(weights) == 2
    assert all(0 <= w <= 1 for w in weights)
```

### 2. Integration Tests

```python
# tests/test_integration.py
def test_end_to_end_generation():
    """Test full generation pipeline."""
    # Setup
    args = SimpleNamespace(
        model_name="Llama-3.2-1B",
        dataset="trivia_qa",
        num_samples=5,
        temperature=0.0,
        # ... more args
    )
    
    # Run generation
    results = generate_answers_pipeline(args)
    
    # Verify results
    assert len(results) == 5
    for id, entry in results.items():
        assert 'most_likely_answer' in entry
        mla = entry['most_likely_answer']
        assert len(mla['tokens']) == len(mla['token_log_likelihoods'])

def test_end_to_end_analysis():
    """Test full analysis pipeline."""
    # Load test pickle
    with open('test_data/validation_generations.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Run analysis
    phase1_results = run_phase1_analysis(data)
    phase15_results = run_phase15_analysis(data)
    
    # Verify outputs
    assert 'accuracy' in phase1_results
    assert 'examples' in phase15_results
```

### 3. Performance Tests

```python
# tests/test_performance.py
def test_generation_speed():
    """Ensure generation meets performance targets."""
    import time
    
    model = HuggingfaceModel(...)
    start = time.time()
    
    for _ in range(10):
        model.predict("Question: Test?\nAnswer:", temperature=0.0)
    
    elapsed = time.time() - start
    avg_time = elapsed / 10
    
    # Should generate in < 5 seconds per example
    assert avg_time < 5.0

def test_memory_usage():
    """Ensure memory usage is within limits."""
    import psutil
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024**3  # GB
    
    model = HuggingfaceModel(...)
    for _ in range(100):
        model.predict("Question: Test?\nAnswer:", temperature=0.0)
    
    final_memory = process.memory_info().rss / 1024**3  # GB
    memory_increase = final_memory - initial_memory
    
    # Should not leak more than 1GB
    assert memory_increase < 1.0
```

---

## Future Improvements

### 1. Planned Features
- [ ] Support for more model architectures (GPT-Neo, OPT, BLOOM)
- [ ] Distributed training support
- [ ] Real-time inference API
- [ ] Web dashboard for results visualization
- [ ] Automated hyperparameter tuning

### 2. Performance Enhancements
- [ ] Model quantization (4-bit, GPTQ)
- [ ] Flash Attention integration
- [ ] Batched generation for speed
- [ ] Persistent caching for similarity computations

### 3. Analysis Improvements
- [ ] More uncertainty metrics (epistemic vs. aleatoric)
- [ ] Confidence intervals for AUROC
- [ ] Cross-dataset evaluation
- [ ] Calibration analysis

---

## Appendix

### A. Glossary

- **G-NLL**: Greedy Negative Log-Likelihood - sum of token log-probabilities
- **RW-G-NLL**: Relevance-Weighted G-NLL - weighted by semantic relevance
- **SAR**: Shifting Attention to Relevance - multi-sample uncertainty with relevance weighting
- **AUROC**: Area Under the ROC Curve - metric for uncertainty quality
- **Token Alignment**: Ensuring token count matches across generation and analysis
- **Stop Sequence**: Text markers that halt generation (e.g., "\n\n")
- **Few-Shot Prompt**: Examples included in prompt to guide model

### B. Common Abbreviations

- **NLL**: Negative Log-Likelihood
- **MLA**: Most Likely Answer
- **EM**: Exact Match
- **F1**: F1 Score
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation
- **LLM**: Large Language Model
- **HF**: HuggingFace
- **GPU**: Graphics Processing Unit
- **VRAM**: Video RAM (GPU memory)

---

**Last Updated:** November 2025


