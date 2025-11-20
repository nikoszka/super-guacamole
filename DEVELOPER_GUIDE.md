# 👨‍💻 nllSAR Developer Guide

A practical guide for developers contributing to or extending the nllSAR codebase.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Code Style Guidelines](#code-style-guidelines)
4. [Testing Guidelines](#testing-guidelines)
5. [Common Development Tasks](#common-development-tasks)
6. [Debugging Tips](#debugging-tips)
7. [Performance Profiling](#performance-profiling)
8. [Contributing Guidelines](#contributing-guidelines)

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- Git for version control
- 50GB+ free disk space (for models and data)

### Development Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-org/nllSAR.git
cd nllSAR
```

2. **Create conda environment:**
```bash
conda env create -f nllsar.yml
conda activate nllsar
```

3. **Install in development mode:**
```bash
pip install -e .
```

4. **Set up environment variables:**
```bash
# Create .env file
cat > .env << EOF
HF_TOKEN=hf_your_token_here
HF_MODELS_CACHE=/path/to/model/cache
OPENAI_API_KEY=sk_your_key_here  # If using LLM judge
export USER=your_username
EOF

# Load environment
source .env
```

5. **Verify installation:**
```python
python -c "
from models.huggingface_models import HuggingfaceModel
from uncertainty_measures.rw_gnll import initialize_similarity_model
print('✅ Installation successful!')
"
```

### IDE Setup

**VS Code (Recommended):**
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true,
    "editor.rulers": [100],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

**PyCharm:**
- Mark `src/` as Sources Root
- Enable Black formatter
- Configure pytest as test runner
- Set line length to 100

---

## Development Workflow

### Branch Strategy

```
main
  ├─ develop (active development)
  │   ├─ feature/add-new-model
  │   ├─ feature/improve-analysis
  │   └─ bugfix/token-alignment
  └─ release/v2.0
```

### Typical Development Cycle

1. **Create feature branch:**
```bash
git checkout develop
git pull origin develop
git checkout -b feature/my-new-feature
```

2. **Make changes and test:**
```bash
# Make your changes
vim src/models/my_new_model.py

# Run tests
pytest tests/test_my_new_model.py -v

# Run linting
pylint src/models/my_new_model.py
black src/models/my_new_model.py
```

3. **Commit with descriptive messages:**
```bash
git add src/models/my_new_model.py
git commit -m "feat: Add support for GPT-Neo models

- Implement GPTNeoModel class
- Add tests for GPT-Neo generation
- Update documentation
"
```

4. **Push and create PR:**
```bash
git push origin feature/my-new-feature
# Create pull request on GitHub
```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(models): Add GPT-Neo model support
fix(analysis): Correct token alignment in Phase 1.5
docs(api): Update RW-G-NLL API documentation
perf(generation): Optimize GPU memory usage
```

---

## Code Style Guidelines

### General Principles

1. **Readability over cleverness**
2. **Explicit is better than implicit**
3. **Minimize dependencies between modules**
4. **Write self-documenting code**

### Python Style

Follow [PEP 8](https://pep8.org/) with these specifics:

**Line Length:**
```python
# Max 100 characters (not 80)
# OK:
result = compute_rw_gnll(entry, similarity_model, tokenizer, cache=cache)

# Better (if too long):
result = compute_rw_gnll(
    entry, 
    similarity_model, 
    tokenizer, 
    cache=cache
)
```

**Imports:**
```python
# Standard library
import os
import logging
from typing import List, Dict, Optional

# Third-party
import torch
import numpy as np
from transformers import AutoTokenizer

# Local
from models.base_model import BaseModel
from uncertainty_measures.rw_gnll import compute_rw_gnll
```

**Docstrings:**
```python
def compute_uncertainty(
    entry: Dict[str, Any],
    method: str = 'gnll',
    **kwargs
) -> float:
    """Compute uncertainty score for an entry.
    
    Args:
        entry: Dictionary with question, answer, and token info
        method: Uncertainty method ('gnll', 'rw_gnll', 'sar')
        **kwargs: Method-specific arguments
    
    Returns:
        float: Uncertainty score (higher = more uncertain)
    
    Raises:
        ValueError: If method not recognized
        KeyError: If entry missing required fields
    
    Example:
        >>> entry = {'question': '...', 'most_likely_answer': {...}}
        >>> score = compute_uncertainty(entry, method='gnll')
        >>> print(f"Uncertainty: {score:.3f}")
    """
```

**Type Hints:**
```python
# Use type hints for function signatures
def process_batch(
    examples: List[Dict[str, Any]],
    model: BaseModel,
    temperature: float = 0.0
) -> List[Tuple[str, List[float]]]:
    """Process batch of examples."""
    pass

# Use Optional for nullable types
def load_cache(
    cache_path: Optional[str] = None
) -> Dict[Tuple[str, int], float]:
    """Load cache from file if provided."""
    pass
```

**Error Handling:**
```python
# Be specific with exceptions
try:
    result = compute_metric(data)
except KeyError as e:
    logging.error(f"Missing required field: {e}")
    raise
except ValueError as e:
    logging.warning(f"Invalid value, using default: {e}")
    result = default_value

# Use context managers
with open('file.pkl', 'rb') as f:
    data = pickle.load(f)
# File automatically closed
```

**Logging:**
```python
# Use appropriate log levels
logging.debug("Token IDs: %s", token_ids)  # Verbose details
logging.info("Processing %d examples", len(dataset))  # General info
logging.warning("Token count mismatch: %d vs %d", n1, n2)  # Potential issues
logging.error("Failed to load model: %s", error)  # Actual errors
logging.critical("Out of GPU memory, aborting")  # Critical failures

# Use lazy formatting
logging.info("Result: %s", result)  # Good
logging.info(f"Result: {result}")  # Bad (formats even if not logged)
```

---

## Testing Guidelines

### Test Structure

```
tests/
├── test_models.py              # Model tests
├── test_uncertainty_measures.py # Uncertainty metric tests
├── test_data_utils.py          # Data loading tests
├── test_integration.py         # End-to-end tests
├── fixtures/                   # Test data
│   ├── sample_pickle.pkl
│   └── sample_dataset.json
└── conftest.py                 # Pytest configuration
```

### Writing Unit Tests

```python
# tests/test_uncertainty_measures.py
import pytest
import numpy as np
from uncertainty_measures.rw_gnll import compute_rw_gnll

class TestRWGNLL:
    """Tests for RW-G-NLL computation."""
    
    @pytest.fixture
    def sample_entry(self):
        """Fixture providing sample entry."""
        return {
            'question': 'What is the capital of France?',
            'context': '',
            'most_likely_answer': {
                'response': 'Paris',
                'token_log_likelihoods': [-0.234, -0.156],
                'token_ids': [40, 80285],
                'tokens': ['Par', 'is']
            }
        }
    
    @pytest.fixture
    def similarity_model(self):
        """Fixture providing similarity model."""
        from uncertainty_measures.rw_gnll import initialize_similarity_model
        return initialize_similarity_model()
    
    def test_compute_rw_gnll_basic(self, sample_entry, similarity_model, tokenizer):
        """Test basic RW-G-NLL computation."""
        rw_gnll, weights = compute_rw_gnll(
            sample_entry, 
            similarity_model, 
            tokenizer,
            return_relevance_weights=True
        )
        
        assert isinstance(rw_gnll, float)
        assert rw_gnll >= 0  # NLL should be non-negative
        assert len(weights) == 2
        assert all(0 <= w <= 1 for w in weights)
    
    def test_compute_rw_gnll_empty_response(self, similarity_model, tokenizer):
        """Test RW-G-NLL with empty response."""
        entry = {
            'question': 'Test?',
            'context': '',
            'most_likely_answer': {
                'response': '',
                'token_log_likelihoods': [],
                'token_ids': [],
                'tokens': []
            }
        }
        
        rw_gnll, _ = compute_rw_gnll(entry, similarity_model, tokenizer)
        assert rw_gnll == 0.0
    
    def test_compute_rw_gnll_alignment_error(self, similarity_model, tokenizer):
        """Test RW-G-NLL with misaligned tokens."""
        entry = {
            'question': 'Test?',
            'context': '',
            'most_likely_answer': {
                'response': 'Test answer',
                'token_log_likelihoods': [-0.1, -0.2],
                'token_ids': [123, 456, 789],  # Mismatch!
                'tokens': ['Test', ' answer', ' extra']
            }
        }
        
        with pytest.raises(ValueError, match="Token count mismatch"):
            compute_rw_gnll(entry, similarity_model, tokenizer)
    
    @pytest.mark.parametrize("num_tokens,expected_range", [
        (1, (0, 2)),
        (5, (0, 10)),
        (20, (0, 50))
    ])
    def test_compute_rw_gnll_various_lengths(
        self, num_tokens, expected_range, similarity_model, tokenizer
    ):
        """Test RW-G-NLL with various response lengths."""
        entry = {
            'question': 'Test?',
            'context': '',
            'most_likely_answer': {
                'response': ' '.join(['word'] * num_tokens),
                'token_log_likelihoods': [-0.5] * num_tokens,
                'token_ids': list(range(num_tokens)),
                'tokens': ['word'] * num_tokens
            }
        }
        
        rw_gnll, _ = compute_rw_gnll(entry, similarity_model, tokenizer)
        assert expected_range[0] <= rw_gnll <= expected_range[1]
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_uncertainty_measures.py -v

# Run specific test class
pytest tests/test_uncertainty_measures.py::TestRWGNLL -v

# Run specific test method
pytest tests/test_uncertainty_measures.py::TestRWGNLL::test_compute_rw_gnll_basic -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest tests/ -m "not slow"

# Run in parallel (faster)
pytest tests/ -n auto
```

### Test Markers

```python
# Mark slow tests
@pytest.mark.slow
def test_full_generation_pipeline():
    """This test takes 5+ minutes."""
    pass

# Mark GPU tests
@pytest.mark.gpu
def test_multi_gpu_loading():
    """Requires multiple GPUs."""
    pass

# Mark integration tests
@pytest.mark.integration
def test_end_to_end_workflow():
    """Full end-to-end test."""
    pass

# Run marked tests:
pytest tests/ -m slow           # Only slow tests
pytest tests/ -m "not slow"     # Skip slow tests
pytest tests/ -m "gpu and not slow"  # GPU tests, but not slow ones
```

---

## Common Development Tasks

### 1. Adding a New Model

**Step 1: Create model class**
```python
# src/models/my_model.py
from models.base_model import BaseModel

class MyModel(BaseModel):
    """My custom model implementation."""
    
    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        # Initialize your model
        self.model = load_my_model(model_name)
        self.tokenizer = load_my_tokenizer(model_name)
        self.stop_sequences = stop_sequences
        self.max_new_tokens = max_new_tokens
    
    def predict(self, input_data, temperature, return_full=False):
        # Implement generation
        # MUST return: (answer, log_liks, embedding, token_ids, tokens)
        
        # Tokenize input
        input_ids = self.tokenizer.encode(input_data)
        
        # Generate
        output = self.model.generate(input_ids, temperature=temperature)
        
        # Extract components
        generated_ids = output.token_ids
        tokens = [self.tokenizer.decode([tid]) for tid in generated_ids]
        log_liks = output.log_probabilities
        embedding = output.last_hidden_state[-1]
        answer = self.tokenizer.decode(generated_ids)
        
        return answer, log_liks, embedding, generated_ids, tokens
    
    def get_p_true(self, input_data):
        # Implement p_true
        input_with_a = input_data + ' A'
        return self.model.get_log_prob(input_with_a)
```

**Step 2: Register in utils**
```python
# src/utils/utils.py
def init_model(args):
    if 'my_model' in args.model_name.lower():
        from models.my_model import MyModel
        return MyModel(
            model_name=args.model_name,
            stop_sequences=args.stop_sequences,
            max_new_tokens=args.model_max_new_tokens
        )
    # ... existing models
```

**Step 3: Add tests**
```python
# tests/test_my_model.py
def test_my_model_initialization():
    model = MyModel("my_model_1b", max_new_tokens=50)
    assert model is not None

def test_my_model_generation():
    model = MyModel("my_model_1b", max_new_tokens=50)
    answer, log_liks, emb, token_ids, tokens = model.predict(
        "Test input", temperature=0.0
    )
    assert len(tokens) == len(log_liks)
    assert len(token_ids) == len(tokens)
```

**Step 4: Update documentation**
```markdown
# README.md
## Supported Models
- Llama (1B, 7B, 13B, 70B)
- Mistral (7B, 8x7B)
- **My Model (1B, 7B)** ← Add this
```

### 2. Adding a New Uncertainty Metric

**Step 1: Implement metric**
```python
# src/uncertainty_measures/my_metric.py
def compute_my_metric(
    entry: Dict[str, Any],
    **kwargs
) -> float:
    """Compute my custom uncertainty metric.
    
    Args:
        entry: Entry with question, answer, token info
        **kwargs: Additional parameters
    
    Returns:
        float: Uncertainty score
    """
    mla = entry['most_likely_answer']
    tokens = mla['tokens']
    log_liks = mla['token_log_likelihoods']
    
    # Your metric computation
    # Example: weighted variance
    variance = np.var(log_liks)
    mean = np.mean(log_liks)
    
    return variance / (abs(mean) + 1e-10)
```

**Step 2: Add to phase 5 analysis**
```python
# src/analysis/phase5_comparative_analysis.py
def compute_auroc_comparison(data, model_name, similarity_model_name, output_dir):
    # ... existing metrics
    
    # Add your metric
    from uncertainty_measures.my_metric import compute_my_metric
    
    my_metric_scores = []
    for id, entry in tqdm(data.items(), desc="Computing My Metric"):
        score = compute_my_metric(entry)
        my_metric_scores.append(score)
    
    # Compute AUROC
    y_true = [entry['most_likely_answer']['accuracy'] for entry in data.values()]
    auroc = roc_auc_score(y_true, my_metric_scores)
    
    results['my_metric'] = auroc
    # ... save results
```

**Step 3: Add tests**
```python
# tests/test_my_metric.py
def test_my_metric_basic():
    entry = {
        'most_likely_answer': {
            'tokens': ['a', 'b', 'c'],
            'token_log_likelihoods': [-0.1, -0.2, -0.3]
        }
    }
    
    score = compute_my_metric(entry)
    assert isinstance(score, float)
    assert score >= 0
```

### 3. Adding a New Dataset

**Step 1: Implement loader**
```python
# src/data/data_utils.py
def load_ds(dataset_name, seed, add_options=None):
    # ... existing datasets
    
    elif dataset_name == 'my_dataset':
        # Load from HuggingFace or local
        dataset = datasets.load_dataset('path/to/my_dataset')
        
        # Split train/val
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']
        
        # Normalize format
        def reformat(x):
            return {
                'question': x['my_question_field'],
                'answers': {'text': x['my_answer_field']},
                'context': x.get('my_context_field', ''),
                'id': x['my_id_field']
            }
        
        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]
        
        return train_dataset, validation_dataset
```

**Step 2: Add dataset-specific logic**
```python
# src/generate_answers.py
def main(args):
    # Dataset-specific handling
    if args.dataset == 'my_dataset':
        # Specific settings for my dataset
        if not args.use_special_format:
            logging.info('Forcing use_special_format=True for my_dataset')
            args.use_special_format = True
```

**Step 3: Add documentation**
```markdown
# DATA_README.md (create if doesn't exist)
## My Dataset
- Description: ...
- Size: X train, Y validation
- Format: Question + context → answer
- Usage: `--dataset my_dataset`
```

### 4. Improving Performance

**Profile first:**
```python
# Use line_profiler
@profile
def slow_function():
    # Your code
    pass

# Run:
kernprof -l -v script.py
```

**Common optimizations:**

1. **Vectorize operations:**
```python
# Slow
result = []
for i in range(len(values)):
    result.append(weights[i] * values[i])

# Fast
import numpy as np
result = np.array(weights) * np.array(values)
```

2. **Use caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_computation(x):
    # Cached automatically
    return complex_operation(x)
```

3. **Batch processing:**
```python
# Process in batches to reduce overhead
for batch in chunks(data, batch_size=32):
    results = model.predict_batch(batch)
```

4. **GPU memory management:**
```python
# Move to CPU immediately
result = computation().cpu()
del intermediate_tensors
torch.cuda.empty_cache()
```

---

## Debugging Tips

### 1. Enable Debug Logging

```python
# At start of script
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific module
logging.getLogger('models.huggingface_models').setLevel(logging.DEBUG)
```

### 2. Interactive Debugging

```python
# Insert breakpoint
import pdb; pdb.set_trace()

# Or use ipdb for better experience
import ipdb; ipdb.set_trace()

# Common commands:
# n - next line
# s - step into function
# c - continue
# l - list code
# p variable - print variable
# pp variable - pretty print
```

### 3. Inspect Pickle Files

```python
import pickle
import pprint

with open('validation_generations.pkl', 'rb') as f:
    data = pickle.load(f)

# Check structure
first_id = list(data.keys())[0]
pprint.pprint(data[first_id])

# Check all keys
for id in list(data.keys())[:5]:
    print(f"\n{id}:")
    print(f"  Keys: {data[id].keys()}")
    print(f"  Tokens: {len(data[id]['most_likely_answer'].get('tokens', []))}")
```

### 4. GPU Debugging

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Python profiling
python -m torch.utils.bottleneck script.py

# Memory profiling
python -m memory_profiler script.py
```

### 5. Common Issues & Solutions

**Issue: Token count mismatch**
```python
# Check if pickle has new format
entry = data[list(data.keys())[0]]
mla = entry['most_likely_answer']
print("Has token_ids:", 'token_ids' in mla)
print("Has tokens:", 'tokens' in mla)

# If missing, need to regenerate pickle
```

**Issue: CUDA out of memory**
```python
# Check memory usage
import torch
print(torch.cuda.memory_allocated() / 1024**3, "GB")
print(torch.cuda.memory_reserved() / 1024**3, "GB")

# Clear cache
torch.cuda.empty_cache()

# Reduce batch size or use quantization
model = HuggingfaceModel("Llama-3.2-1B-8bit", ...)
```

**Issue: Slow similarity computation**
```python
# Use cache
cache = {}
for entry in dataset:
    rw_gnll = compute_rw_gnll(entry, sim_model, tokenizer, cache=cache)
    
# Check cache hit rate
print(f"Cache size: {len(cache)}")
```

---

## Performance Profiling

### CPU Profiling

```bash
# Install
pip install line_profiler

# Add @profile decorator
# No import needed
@profile
def my_function():
    # ...code...
    pass

# Run
kernprof -l -v script.py
```

### GPU Profiling

```python
# PyTorch profiler
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Your code
    model.predict(input_data, temperature=0.0)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Memory Profiling

```bash
# Install
pip install memory_profiler

# Run
python -m memory_profiler script.py

# Or use decorator
@profile
def my_function():
    # ...code...
    pass
```

---

## Contributing Guidelines

### Before Submitting PR

1. **Run all tests:**
```bash
pytest tests/ -v
```

2. **Run linting:**
```bash
pylint src/
black src/
```

3. **Update documentation:**
- Update docstrings
- Update relevant .md files
- Add examples if applicable

4. **Add changelog entry:**
```markdown
# CHANGELOG.md
## [Unreleased]
### Added
- New feature X (#123)
### Fixed
- Bug in Y (#124)
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] Added new tests for new functionality
- [ ] Tested on GPU

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Changelog updated
```

---

## Resources

### Documentation
- [README.md](README.md) - Project overview
- [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) - Comprehensive code docs
- [API_REFERENCE.md](API_REFERENCE.md) - API reference
- [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) - Architecture details

### External Resources
- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Weights & Biases Docs](https://docs.wandb.ai/)

### Community
- GitHub Issues: Report bugs and request features
- GitHub Discussions: Ask questions and share ideas

---

**Last Updated:** November 2025

