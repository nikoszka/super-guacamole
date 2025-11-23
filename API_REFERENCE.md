# 📚 nllSAR API Reference

Complete API documentation for developers working with the nllSAR codebase.

---

## Table of Contents

1. [Models API](#models-api)
2. [Uncertainty Measures API](#uncertainty-measures-api)
3. [Data Loading API](#data-loading-api)
4. [Utilities API](#utilities-api)
5. [Analysis API](#analysis-api)

---

## Models API

### `models.base_model`

#### Class: `BaseModel`

Abstract base class for all language models.

```python
class BaseModel(ABC):
    """Abstract interface for language models."""
    
    stop_sequences: List[Text]
    
    @abstractmethod
    def predict(self, input_data: str, temperature: float) -> Tuple:
        """Generate prediction with token-level outputs.
        
        Args:
            input_data: Input prompt text
            temperature: Sampling temperature (0.0 = greedy)
        
        Returns:
            Tuple of (answer, log_likelihoods, embedding, token_ids, tokens)
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass
    
    @abstractmethod
    def get_p_true(self, input_data: str) -> float:
        """Get log-probability of generating 'True'.
        
        Args:
            input_data: Input prompt (will append ' A')
        
        Returns:
            float: Negative loss (log-probability)
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass
```

#### Constants

```python
STOP_SEQUENCES: List[str] = [
    '\n\n\n\n', 
    '\n\n\n', 
    '\n\n', 
    'Question:', 
    'Context:', 
    'Answer:'
]

STOP_SEQUENCES_DETAILED: List[str] = [
    '\n\n\n\n', 
    'Question:', 
    'Context:', 
    'Answer:'
]
```

---

### `models.huggingface_models`

#### Function: `get_hf_cache_dir()`

```python
def get_hf_cache_dir() -> Optional[str]:
    """Get HuggingFace model cache directory.
    
    Checks environment variables in priority order:
    1. HF_MODELS_CACHE (direct model cache path)
    2. TRANSFORMERS_CACHE (direct model cache path)
    3. HF_HOME (base directory, models in hub/)
    4. Default: ~/.cache/huggingface/hub
    
    Returns:
        str: Path to cache directory, or None for default
    
    Side Effects:
        Logs the cache directory being used
    """
```

#### Function: `get_gpu_memory_dict()`

```python
def get_gpu_memory_dict() -> Optional[Dict[int, str]]:
    """Get max_memory dictionary for all available GPUs.
    
    Automatically detects GPU memory capacity and creates a max_memory
    dictionary for use with device_map="auto".
    
    Returns:
        dict: Mapping of GPU index to available memory string
              Example: {0: '10GiB', 1: '10GiB'}
              Returns None if CUDA is not available
    
    Example:
        >>> get_gpu_memory_dict()
        {0: '15GiB', 1: '15GiB'}
    """
```

#### Class: `StoppingCriteriaSub`

```python
class StoppingCriteriaSub(StoppingCriteria):
    """Stop generation when matching text or tokens.
    
    Attributes:
        stops: List of stop strings or token sequences
        tokenizer: Tokenizer for decoding
        match_on: 'text' or 'tokens' matching mode
        initial_length: Length of input tokens
    """
    
    def __init__(
        self, 
        stops: List[str], 
        tokenizer, 
        match_on: str = 'text', 
        initial_length: Optional[int] = None
    ):
        """Initialize stopping criteria.
        
        Args:
            stops: List of stop sequences
            tokenizer: HuggingFace tokenizer
            match_on: 'text' or 'tokens' - how to match stops
            initial_length: Number of input tokens (for text matching)
        """
    
    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor
    ) -> bool:
        """Check if generation should stop.
        
        Args:
            input_ids: Generated token IDs so far
            scores: Model output scores (unused)
        
        Returns:
            bool: True if should stop, False otherwise
        """
```

#### Function: `remove_split_layer()`

```python
def remove_split_layer(device_map_in: Dict[str, int]) -> Dict[str, int]:
    """Modify device map to prevent layer splitting across GPUs.
    
    Ensures that individual model layers are not spread across multiple
    devices, which can cause performance issues.
    
    Args:
        device_map_in: Original device map
    
    Returns:
        dict: Modified device map with split layers consolidated
    
    Raises:
        ValueError: If more than one layer is split
    
    Side Effects:
        Logs information about split layers found
    """
```

#### Class: `HuggingfaceModel`

```python
class HuggingfaceModel(BaseModel):
    """HuggingFace transformer model wrapper.
    
    Supports:
        - Llama (1B, 7B, 8B, 13B, 70B)
        - Mistral (7B, 8x7B)
        - Falcon (7B, 40B)
    
    Attributes:
        model: HuggingFace model instance
        tokenizer: HuggingFace tokenizer
        model_name: Name of the model
        stop_sequences: List of stop strings
        token_limit: Maximum context length
        max_new_tokens: Maximum new tokens to generate
        cache_dir: Model cache directory
    """
    
    def __init__(
        self,
        model_name: str,
        stop_sequences: Optional[Union[str, List[str]]] = None,
        max_new_tokens: Optional[int] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize HuggingFace model.
        
        Args:
            model_name: Model identifier. Supported:
                - "Llama-3.2-1B", "Llama-2-7b", "Llama-3.1-70B"
                - "Mistral-7B-v0.1", "Mixtral-8x7B-v0.1"
                - "falcon-7b", "falcon-40b"
                Add "-8bit" suffix for 8-bit quantization
                Add "-4bit" suffix for 4-bit quantization (Mistral only)
            stop_sequences: Stop sequences for generation:
                - "default": Use STOP_SEQUENCES
                - "detailed": Use STOP_SEQUENCES_DETAILED
                - List[str]: Custom stop sequences
            max_new_tokens: Maximum tokens to generate (required)
            cache_dir: Custom cache directory (optional)
        
        Raises:
            ValueError: If max_new_tokens is None or model not supported
        
        Example:
            >>> model = HuggingfaceModel(
            ...     model_name="Llama-3.2-1B",
            ...     stop_sequences="default",
            ...     max_new_tokens=50
            ... )
        """
    
    def predict(
        self,
        input_data: str,
        temperature: float,
        return_full: bool = False
    ) -> Union[str, Tuple[str, List[float], torch.Tensor, List[int], List[str]]]:
        """Generate answer with token-level information.
        
        Args:
            input_data: Input prompt text
            temperature: Sampling temperature:
                - 0.0: Greedy decoding (deterministic)
                - > 0.0: Sampling (stochastic)
            return_full: If True, return only full answer string
        
        Returns:
            If return_full=False (default):
                Tuple of:
                - answer (str): Generated text with stop sequences removed
                - log_likelihoods (List[float]): Log P(token|context) for each token
                - embedding (torch.Tensor): Last token embedding, shape (1, hidden_dim)
                - token_ids (List[int]): Token IDs for generated tokens
                - tokens (List[str]): Token strings for generated tokens
            
            If return_full=True:
                str: Full answer including input prompt
        
        Raises:
            ValueError: If generation exceeds token limit or other errors
        
        Side Effects:
            - Clears GPU cache after generation
            - Logs warnings for truncation or unusual behavior
        
        Example:
            >>> answer, log_liks, emb, token_ids, tokens = model.predict(
            ...     "Question: What is 2+2?\nAnswer:",
            ...     temperature=0.0
            ... )
            >>> print(answer)
            '4'
            >>> print(len(tokens), tokens)
            1 ['4']
        """
    
    def get_p_true(self, input_data: str) -> float:
        """Get log-probability of generating 'A' (True).
        
        Used for p_true baseline uncertainty estimation.
        
        Args:
            input_data: Input prompt (will append ' A')
        
        Returns:
            float: Negative loss for generating 'A'
                  Higher values indicate higher probability
        
        Example:
            >>> p_true = model.get_p_true(
            ...     "Is Paris the capital of France? Answer: (A) True (B) False\nAnswer:"
            ... )
            >>> print(p_true)
            -0.123
        """
```

---

## Uncertainty Measures API

### `uncertainty_measures.rw_gnll`

#### Function: `initialize_similarity_model()`

```python
def initialize_similarity_model(
    model_name: str = 'cross-encoder/stsb-roberta-large'
) -> CrossEncoder:
    """Initialize semantic similarity model.
    
    Args:
        model_name: HuggingFace model name for cross-encoder
                   Default: 'cross-encoder/stsb-roberta-large'
                   Other options:
                   - 'cross-encoder/stsb-distilroberta-base' (faster)
                   - 'cross-encoder/stsb-roberta-base' (balanced)
    
    Returns:
        CrossEncoder: Initialized similarity model
    
    Raises:
        ImportError: If sentence-transformers not installed
    
    Side Effects:
        Logs model loading progress
    
    Example:
        >>> sim_model = initialize_similarity_model()
        Loading similarity model: cross-encoder/stsb-roberta-large
        Similarity model loaded successfully
    """
```

#### Function: `compute_similarity()`

```python
def compute_similarity(
    similarity_model: CrossEncoder,
    text1: str,
    text2: str
) -> float:
    """Compute semantic similarity between two texts.
    
    Args:
        similarity_model: Initialized cross-encoder model
        text1: First text
        text2: Second text
    
    Returns:
        float: Normalized similarity score in [0, 1]
               0 = completely dissimilar
               1 = identical
    
    Notes:
        - Uses linear normalization for STSB models (0-5 range)
        - Uses sigmoid for models with negative scores
    
    Example:
        >>> sim = compute_similarity(
        ...     model,
        ...     "The cat sat on the mat",
        ...     "The cat sat on the mat"
        ... )
        >>> print(sim)
        1.0
        
        >>> sim = compute_similarity(
        ...     model,
        ...     "The cat sat on the mat",
        ...     "Dogs are playing"
        ... )
        >>> print(sim)
        0.15
    """
```

#### Function: `remove_token_at_position()`

```python
def remove_token_at_position(
    response_text: str,
    tokenizer,
    token_position: int
) -> str:
    """Remove token at specified position from response.
    
    Used to create ablated responses for relevance computation:
    y \ {y_t} = response with token t removed
    
    Args:
        response_text: Response text as string
        tokenizer: HuggingFace tokenizer
        token_position: 0-based index of token to remove
    
    Returns:
        str: Response with specified token removed
    
    Raises:
        ValueError: If token_position is out of bounds
    
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
        >>> response = "The quick brown fox"
        >>> ablated = remove_token_at_position(response, tokenizer, 1)
        >>> print(ablated)
        'The brown fox'  # 'quick' removed
    """
```

#### Function: `compute_token_relevance_weights()`

```python
def compute_token_relevance_weights(
    prompt_x: str,
    response_y_greedy: str,
    tokenizer,
    similarity_model,
    cache: Optional[Dict[Tuple[str, int], float]] = None,
    show_progress: bool = False
) -> List[float]:
    """Compute relevance weights for each token in response.
    
    For each token t:
        R_T(y_t) = 1 - similarity(x ∪ y, x ∪ (y \ {y_t}))
    
    Where:
        - x = prompt
        - y = response
        - y \ {y_t} = response with token t removed
    
    Args:
        prompt_x: Input prompt (context + question)
        response_y_greedy: Generated response text
        tokenizer: HuggingFace tokenizer
        similarity_model: Initialized similarity model
        cache: Optional cache dict for similarity scores
               Key: (response_text, token_position)
               Value: similarity score
        show_progress: Show progress bar for token processing
    
    Returns:
        List[float]: Relevance weight for each token
                    Higher values = more relevant to prompt
                    Range: [0, 1]
    
    Side Effects:
        - Updates cache dict if provided
        - Shows tqdm progress bar if show_progress=True
    
    Example:
        >>> weights = compute_token_relevance_weights(
        ...     prompt_x="What is the capital of France?",
        ...     response_y_greedy="The capital is Paris",
        ...     tokenizer=tokenizer,
        ...     similarity_model=sim_model
        ... )
        >>> print(weights)
        [0.2, 0.3, 0.4, 0.9]  # "Paris" has highest relevance
    """
```

#### Function: `compute_rw_gnll()`

```python
def compute_rw_gnll(
    entry: Dict[str, Any],
    similarity_model,
    tokenizer,
    cache: Optional[Dict[Tuple[str, int], float]] = None,
    return_relevance_weights: bool = False
) -> Tuple[float, Optional[List[float]]]:
    """Compute Relevance-Weighted G-NLL score.
    
    Formula:
        RW-G-NLL = Σ [R_T(y_t) · (-log P(y_t))] / Σ R_T(y_t)
    
    Args:
        entry: Dictionary with:
            - 'question': str
            - 'context': str (optional)
            - 'most_likely_answer': dict with:
                - 'response': str
                - 'token_log_likelihoods': List[float]
                - 'token_ids': List[int] (recommended)
                - 'tokens': List[str] (recommended)
        similarity_model: Initialized similarity model
        tokenizer: HuggingFace tokenizer
        cache: Optional cache for similarity scores
        return_relevance_weights: Return weights along with score
    
    Returns:
        Tuple of:
            - rw_gnll_score (float): Relevance-weighted NLL
            - relevance_weights (Optional[List[float]]): Weights per token
              (None unless return_relevance_weights=True)
    
    Raises:
        KeyError: If required fields missing from entry
        ValueError: If token counts don't align
    
    Example:
        >>> entry = {
        ...     'question': 'What is the capital of France?',
        ...     'context': '',
        ...     'most_likely_answer': {
        ...         'response': 'Paris',
        ...         'token_log_likelihoods': [-0.234, -0.156],
        ...         'token_ids': [40, 80285],
        ...         'tokens': ['Par', 'is']
        ...     }
        ... }
        >>> rw_gnll, weights = compute_rw_gnll(
        ...     entry, sim_model, tokenizer,
        ...     return_relevance_weights=True
        ... )
        >>> print(f"RW-G-NLL: {rw_gnll:.3f}")
        RW-G-NLL: 0.195
    """
```

---

### `uncertainty_measures.sar`

#### Function: `compute_sar_for_entry()`

```python
def compute_sar_for_entry(
    entry: Dict[str, Any],
    responses: List[Dict[str, Any]],
    similarity_model,
    tokenizer,
    cache: Optional[Dict[Tuple[str, int], float]] = None,
    show_progress: bool = False
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Compute SAR score across multiple sampled responses.
    
    Formula:
        SAR = (1/M) * Σ_m [Σ_t R_T(y_t^m) * (-log P(y_t^m))] / [Σ_t R_T(y_t^m)]
    
    Args:
        entry: Dictionary with 'question' and optional 'context'
        responses: List of response dicts, each with:
            - 'response': str
            - 'token_log_likelihoods': List[float]
            - 'token_ids': List[int] (recommended)
            - 'tokens': List[str] (recommended)
        similarity_model: Initialized similarity model
        tokenizer: HuggingFace tokenizer
        cache: Optional cache for similarity scores
        show_progress: Show progress bar for samples
    
    Returns:
        Tuple of:
            - sar_score (float): Average SAR across samples
            - details (Dict): Dictionary with:
                - 'num_samples': int
                - 'per_sample_scores': List[float]
                - 'per_sample_relevance_sums': List[float]
                - 'mean_sar': float
                - 'std_sar': float
                - 'num_valid_samples': int
    
    Example:
        >>> entry = {'question': 'What is...', 'context': ''}
        >>> responses = [
        ...     {'response': 'Answer A', 'token_log_likelihoods': [...]},
        ...     {'response': 'Answer B', 'token_log_likelihoods': [...]},
        ...     # ... more samples
        ... ]
        >>> sar, details = compute_sar_for_entry(
        ...     entry, responses, sim_model, tokenizer
        ... )
        >>> print(f"SAR: {sar:.3f} ± {details['std_sar']:.3f}")
        SAR: 0.456 ± 0.123
    """
```

#### Function: `compute_sar()`

```python
def compute_sar(
    entry: Dict[str, Any],
    similarity_model,
    tokenizer,
    cache: Optional[Dict[Tuple[str, int], float]] = None,
    num_samples: Optional[int] = None,
    show_progress: bool = False
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Compute SAR score from entry with 'responses' field.
    
    Convenience function that extracts responses from entry['responses']
    and calls compute_sar_for_entry().
    
    Args:
        entry: Dictionary with 'question', 'context', and 'responses'
        similarity_model: Initialized similarity model
        tokenizer: HuggingFace tokenizer
        cache: Optional cache for similarity scores
        num_samples: Limit number of samples (None = use all)
        show_progress: Show progress bar
    
    Returns:
        Tuple of (sar_score, details_dict)
        See compute_sar_for_entry() for details format
    
    Example:
        >>> entry = {
        ...     'question': 'What is...',
        ...     'context': '',
        ...     'responses': [
        ...         {'response': 'A', 'token_log_likelihoods': [...]},
        ...         {'response': 'B', 'token_log_likelihoods': [...]},
        ...     ]
        ... }
        >>> sar, _ = compute_sar(entry, sim_model, tokenizer)
    """
```

---

### `uncertainty_measures.p_true`

#### Function: `get_p_true_from_pred()`

```python
def get_p_true_from_pred(
    pred: Dict[str, Any],
    model
) -> float:
    """Get p_true score for a prediction.
    
    Args:
        pred: Dictionary with 'input_data' key
        model: Model instance with get_p_true() method
    
    Returns:
        float: P(True) score
    """
```

---

### `uncertainty_measures.semantic_entropy`

#### Function: `compute_semantic_entropy()`

```python
def compute_semantic_entropy(
    responses: List[str],
    log_likelihoods: List[float],
    similarity_threshold: float = 0.5
) -> float:
    """Compute semantic entropy over responses.
    
    Groups semantically similar responses and computes entropy
    over semantic clusters rather than exact strings.
    
    Args:
        responses: List of generated responses
        log_likelihoods: Log-likelihood for each response
        similarity_threshold: Threshold for semantic clustering
    
    Returns:
        float: Semantic entropy value
    """
```

---

## Data Loading API

### `data.data_utils`

#### Function: `load_ds()`

```python
def load_ds(
    dataset_name: str,
    seed: int,
    add_options: Optional[bool] = None
) -> Tuple[Union[List, datasets.Dataset], Union[List, datasets.Dataset]]:
    """Load dataset and normalize format.
    
    Supported datasets:
        - 'squad': SQuAD v2.0 (with unanswerable questions)
        - 'trivia_qa': TriviaQA in SQuAD format
        - 'svamp': Math word problems
        - 'nq': Natural Questions (open-domain)
        - 'bioasq': BioASQ biomedical QA
    
    Args:
        dataset_name: Name of dataset to load
        seed: Random seed for train/test split
        add_options: Whether to include multiple-choice options (if available)
    
    Returns:
        Tuple of (train_dataset, validation_dataset)
        Each example has format:
            {
                'question': str,
                'answers': {'text': List[str]},
                'context': str (may be empty),
                'id': str
            }
    
    Raises:
        ValueError: If dataset_name not recognized
        FileNotFoundError: If local dataset file not found (bioasq)
    
    Side Effects:
        - Downloads dataset from HuggingFace hub (if not cached)
        - For bioasq: reads from local file
    
    Example:
        >>> train, val = load_ds('trivia_qa', seed=42)
        >>> print(len(train), len(val))
        87600 21900
        >>> print(val[0].keys())
        dict_keys(['question', 'answers', 'context', 'id'])
    """
```

---

## Utilities API

### `utils.utils`

#### Function: `setup_logger()`

```python
def setup_logger() -> None:
    """Configure logging with INFO level and formatted output.
    
    Side Effects:
        Sets up root logger with:
        - Level: INFO
        - Format: '%(asctime)s - %(levelname)s - %(message)s'
        - Handler: StreamHandler (console output)
    """
```

#### Function: `init_model()`

```python
def init_model(args) -> BaseModel:
    """Initialize model from arguments.
    
    Args:
        args: Namespace or dict with:
            - model_name: str
            - model_max_new_tokens: int
            - (other model-specific args)
    
    Returns:
        BaseModel: Initialized model instance
    
    Raises:
        ValueError: If model type not supported
    
    Example:
        >>> from types import SimpleNamespace
        >>> args = SimpleNamespace(
        ...     model_name="Llama-3.2-1B",
        ...     model_max_new_tokens=50
        ... )
        >>> model = init_model(args)
    """
```

#### Function: `get_metric()`

```python
def get_metric(metric_name: str) -> Callable:
    """Get evaluation metric function.
    
    Args:
        metric_name: One of:
            - 'squad': F1 and Exact Match
            - 'rouge': ROUGE-L score
            - 'llm_<model>': LLM-as-a-judge (e.g., 'llm_llama-3.1-70b')
    
    Returns:
        Callable: Function that takes (prediction, reference) and returns score
    
    Raises:
        ValueError: If metric_name not recognized
    
    Example:
        >>> metric = get_metric('squad')
        >>> score = metric("Paris", ["Paris", "paris"])
        >>> print(score['f1'])
        1.0
    """
```

#### Function: `construct_fewshot_prompt_from_indices()`

```python
def construct_fewshot_prompt_from_indices(
    dataset: List[Dict],
    indices: List[int],
    brief: str,
    arg: bool,
    make_prompt: Callable
) -> str:
    """Construct few-shot prompt from examples.
    
    Args:
        dataset: List of examples
        indices: Indices of examples to use for few-shot
        brief: Brief instruction text
        arg: Whether to include brief in each example
        make_prompt: Function to format single example
    
    Returns:
        str: Formatted few-shot prompt
    
    Example:
        >>> prompt = construct_fewshot_prompt_from_indices(
        ...     dataset=train_dataset,
        ...     indices=[0, 5, 10, 15, 20],
        ...     brief="Answer concisely.",
        ...     arg=True,
        ...     make_prompt=format_qa_example
        ... )
        >>> print(prompt[:100])
        'Answer concisely.
        
        Question: What is the capital of France?
        Answer: Paris
        
        Question: ...'
    """
```

#### Function: `get_make_prompt()`

```python
def get_make_prompt(args) -> Callable:
    """Get prompt construction function for dataset.
    
    Args:
        args: Namespace with 'dataset' and other prompt-related fields
    
    Returns:
        Callable: Function(example) -> str that formats example
    
    Example:
        >>> make_prompt = get_make_prompt(args)
        >>> formatted = make_prompt(dataset[0])
        >>> print(formatted)
        'Question: What is...?
        Answer: ...'
    """
```

#### Function: `split_dataset()`

```python
def split_dataset(
    dataset: List[Dict]
) -> Tuple[List[int], List[int]]:
    """Split dataset into answerable and unanswerable indices.
    
    Args:
        dataset: List of examples with 'answers' field
    
    Returns:
        Tuple of (answerable_indices, unanswerable_indices)
    
    Example:
        >>> answerable, unanswerable = split_dataset(squad_dataset)
        >>> print(len(answerable), len(unanswerable))
        86821 43498
    """
```

#### Constants

```python
BRIEF_PROMPTS: Dict[str, str] = {
    'short': "Answer the following question with a short answer (1-5 words):",
    'detailed': "Answer the following question with a detailed explanation:",
    'concise': "Answer concisely:",
    'none': ""
}
```

---

### `utils.eval_utils`

#### Function: `compute_rouge_scores()`

```python
def compute_rouge_scores(
    prediction: str,
    reference: Union[str, List[str]]
) -> Dict[str, float]:
    """Compute ROUGE-L scores.
    
    Args:
        prediction: Predicted answer
        reference: Reference answer(s)
    
    Returns:
        dict: {'rouge-l': score} where score is in [0, 1]
    
    Example:
        >>> scores = compute_rouge_scores(
        ...     "The capital is Paris",
        ...     "Paris"
        ... )
        >>> print(scores['rouge-l'])
        0.67
    """
```

#### Function: `compute_squad_metrics()`

```python
def compute_squad_metrics(
    prediction: str,
    references: List[str]
) -> Dict[str, float]:
    """Compute SQuAD F1 and Exact Match scores.
    
    Args:
        prediction: Predicted answer
        references: List of acceptable answers
    
    Returns:
        dict: {'f1': score, 'exact_match': score}
              Both in [0, 1] range
    
    Example:
        >>> metrics = compute_squad_metrics(
        ...     "Paris, France",
        ...     ["Paris", "paris"]
        ... )
        >>> print(metrics)
        {'f1': 0.67, 'exact_match': 0.0}
    ```
```

#### Function: `llm_judge_evaluate()`

```python
def llm_judge_evaluate(
    question: str,
    context: str,
    prediction: str,
    reference: List[str],
    model_name: str = 'llama-3.1-70b'
) -> float:
    """Use LLM to judge answer correctness.
    
    Args:
        question: Question text
        context: Context passage (may be empty)
        prediction: Model's answer
        reference: Reference answers
        model_name: Judge model to use
    
    Returns:
        float: Correctness score in [0, 1]
               1 = completely correct
               0 = completely incorrect
    
    Raises:
        APIError: If LLM API call fails
    
    Side Effects:
        Makes API call to OpenAI or HuggingFace
    
    Example:
        >>> score = llm_judge_evaluate(
        ...     question="What is the capital of France?",
        ...     context="",
        ...     prediction="The capital of France is Paris.",
        ...     reference=["Paris"],
        ...     model_name='llama-3.1-70b'
        ... )
        >>> print(score)
        1.0
    """
```

---

## Analysis API

### `analysis.phase1_baseline_metrics`

#### Function: `compute_baseline_metrics()`

```python
def compute_baseline_metrics(
    data: Dict[str, Dict],
    output_dir: str
) -> Dict[str, Any]:
    """Compute baseline metrics from generation results.
    
    Args:
        data: Pickle data (validation_generations.pkl)
        output_dir: Directory for output files
    
    Returns:
        dict: Metrics including:
            - 'accuracy': mean, std, per_example
            - 'gnll': mean, std, per_example
            - 'token_statistics': mean/std/min/max length
    
    Side Effects:
        Writes to output_dir:
        - baseline_metrics.json
        - token_statistics.json
        - baseline_metrics.csv
    """
```

---

### `analysis.phase1_5_token_nll_analysis`

#### Function: `analyze_token_nlls()`

```python
def analyze_token_nlls(
    data: Dict[str, Dict],
    sample_size: int,
    output_dir: str
) -> Dict[str, Any]:
    """Analyze token-level NLL patterns.
    
    Args:
        data: Pickle data
        sample_size: Number of examples to analyze in detail
        output_dir: Directory for outputs
    
    Returns:
        dict: Analysis results with:
            - 'examples': Detailed token-level info
            - 'statistics': Aggregate statistics
    
    Side Effects:
        Writes to output_dir:
        - nll_vs_position.png
        - nll_distribution.png
        - position_nll_heatmap.png
        - sentence_level_nll_examples.json
        - example_{01-10}_token_nlls.png
    """
```

---

### `analysis.phase2_token_importance`

#### Function: `compute_token_importance()`

```python
def compute_token_importance(
    data: Dict[str, Dict],
    model_name: str,
    similarity_model_name: str,
    sample_size: int,
    output_dir: str
) -> Dict[str, Any]:
    """Compute token relevance weights and importance scores.
    
    Args:
        data: Pickle data
        model_name: Model name for tokenizer
        similarity_model_name: Similarity model name
        sample_size: Number of examples to process
        output_dir: Directory for outputs
    
    Returns:
        dict: Results with relevance-weighted NLL scores
    
    Side Effects:
        Writes to output_dir:
        - relevance_vs_position.png
        - token_importance_examples.json
    """
```

---

### `analysis.phase5_comparative_analysis`

#### Function: `compute_auroc_comparison()`

```python
def compute_auroc_comparison(
    data: Dict[str, Dict],
    model_name: str,
    similarity_model_name: str,
    output_dir: str,
    use_rouge: bool = False,
    rouge_threshold: float = 0.3
) -> Dict[str, float]:
    """Compare AUROC of different uncertainty metrics.
    
    Args:
        data: Pickle data
        model_name: Model name for tokenizer
        similarity_model_name: Similarity model name
        output_dir: Directory for outputs
        use_rouge: Use ROUGE for correctness (instead of LLM judge)
        rouge_threshold: Threshold for ROUGE-based correctness
    
    Returns:
        dict: AUROC scores for each uncertainty metric
            - 'gnll': G-NLL AUROC
            - 'rw_gnll': RW-G-NLL AUROC
            - 'sar': SAR AUROC (if multi-sample)
            - 'semantic_entropy': SE AUROC (if multi-sample)
            - 'length': Length baseline AUROC
    
    Side Effects:
        Writes to output_dir:
        - auroc_comparison.csv
        - roc_curves.png
        - cost_performance_plot.png
    """
```

---

## Type Definitions

### Common Types

```python
# Entry format (pickle structure)
Entry = TypedDict('Entry', {
    'question': str,
    'answers': List[str],
    'context': str,
    'most_likely_answer': TypedDict('MLA', {
        'response': str,
        'token_log_likelihoods': List[float],
        'token_ids': List[int],
        'tokens': List[str],
        'embedding': np.ndarray,
        'accuracy': float
    }),
    'responses': List[TypedDict('Response', {
        'response': str,
        'token_log_likelihoods': List[float],
        'token_ids': List[int],
        'tokens': List[str],
        'embedding': np.ndarray,
        'accuracy': float
    })]
})

# Generation results (pickle file)
GenerationResults = Dict[str, Entry]

# Uncertainty cache
UncertaintyCache = Dict[Tuple[str, int], float]
```

---

## Error Codes

### Model Errors
- `ModelInitError`: Model initialization failed
- `ModelLoadError`: Model weights loading failed
- `GenerationError`: Text generation failed
- `TokenizationError`: Tokenization failed

### Data Errors
- `DatasetNotFoundError`: Dataset not found or not supported
- `DataFormatError`: Unexpected data format
- `MissingFieldError`: Required field missing from data

### Analysis Errors
- `AlignmentError`: Token count mismatch
- `MetricComputationError`: Metric computation failed
- `InsufficientDataError`: Not enough data for analysis

---

## Deprecation Notices

### Deprecated APIs (v1.x)

```python
# DEPRECATED: Old pickle format without token_ids/tokens
# Use new format with token_ids and tokens
entry = {
    'most_likely_answer': {
        'response': str,
        'token_log_likelihoods': List[float]
        # Missing: 'token_ids', 'tokens'
    }
}

# DEPRECATED: String-based stop sequence counting
# Use token-based counting instead
if answer.endswith(stop):
    answer = answer[:-len(stop)]  # WRONG
    # Use tokenizer to count actual tokens

# DEPRECATED: Re-tokenization in analysis
# Use stored tokens from generation
tokens = tokenizer.encode(response)  # WRONG
# Use: tokens = entry['most_likely_answer']['tokens']
```

---

## Version History

### v2.0 (Current)
- Added token_ids and tokens to pickle format
- Improved token alignment system
- Added RW-G-NLL and SAR implementations
- Enhanced multi-GPU support
- Added comprehensive analysis pipeline

### v1.0
- Initial release
- Basic G-NLL computation
- HuggingFace model support
- SQuAD and TriviaQA support

---

**Last Updated:** November 2025
**API Version:** 2.0


