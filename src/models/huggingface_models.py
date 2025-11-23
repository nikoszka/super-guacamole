"""Implement HuggingfaceModel models."""
import copy
import logging
import os
from collections import Counter
import torch

import accelerate

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download


from models.base_model import BaseModel
from models.base_model import STOP_SEQUENCES
from models.base_model import STOP_SEQUENCES_DETAILED


def get_hf_cache_dir():
    """Get HuggingFace model cache directory from environment variable or default.
    
    Priority order:
    1. HF_MODELS_CACHE (custom environment variable - direct model cache path)
    2. TRANSFORMERS_CACHE (standard transformers environment variable - direct model cache path)
    3. HF_HOME (standard HuggingFace environment variable - base directory, models in hub/)
    4. Default HuggingFace cache location (~/.cache/huggingface/hub)
    
    Note: HF_HOME is the base directory for all HuggingFace data. Models are stored in HF_HOME/hub/
    HF_MODELS_CACHE and TRANSFORMERS_CACHE are direct paths to the model cache.
    
    Returns:
        str: Path to cache directory, or None to use default
    """
    # Check for direct cache directory variables first
    cache_dir = os.getenv('HF_MODELS_CACHE') or os.getenv('TRANSFORMERS_CACHE')
    
    # If not set, check HF_HOME (base directory, models go in hub/)
    if not cache_dir:
        hf_home = os.getenv('HF_HOME')
        if hf_home:
            cache_dir = os.path.join(hf_home, 'hub')
    
    if cache_dir:
        logging.info(f'Using HuggingFace model cache directory: {cache_dir}')
    else:
        logging.info('Using default HuggingFace cache directory (~/.cache/huggingface/hub). '
                    'Set HF_MODELS_CACHE, TRANSFORMERS_CACHE, or HF_HOME to customize.')
    return cache_dir


def get_gpu_memory_dict():
    """Get max_memory dictionary for all available GPUs.
    
    Automatically detects GPU memory capacity and creates a max_memory dictionary
    that can be used with device_map="auto" to distribute models across all GPUs.
    
    Returns:
        dict: Dictionary mapping GPU index to available memory string (e.g., {0: '10GiB', 1: '10GiB'})
              Returns None if CUDA is not available
    """
    import torch
    if not torch.cuda.is_available():
        return None
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return None
    
    max_memory = {}
    
    for i in range(num_gpus):
        # Get GPU memory capacity in bytes
        total_memory_bytes = torch.cuda.get_device_properties(i).total_memory
        # Convert to GB
        total_memory_gb = total_memory_bytes / (1024**3)
        # Reserve ~500MB for system overhead, use rest for model
        # Use at least 1GB to avoid issues
        available_memory_gb = max(1, int(total_memory_gb - 0.5))
        max_memory[i] = f'{available_memory_gb}GiB'
    
    return max_memory


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer.
            raise ValueError(
                'More than one split layer.\n'
                f'Currently at layer {layer}.\n'
                f'In map: {device_map_in}\n'
                f'Out map: {device_map}\n')

        logging.info(f'Split layer is {layer}.')

        # Remove split for that layer.
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f'pop {name}')
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map


class HuggingfaceModel(BaseModel):
    """Hugging Face Model."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None, cache_dir=None):
        """Initialize HuggingFace model.
        
        Args:
            model_name: Name of the model to load
            stop_sequences: Sequences that stop generation
            max_new_tokens: Maximum number of new tokens to generate
            cache_dir: Custom cache directory for models (optional).
                      If None, uses environment variables HF_MODELS_CACHE, HF_HOME, or TRANSFORMERS_CACHE.
                      If still None, uses default HuggingFace cache location.
        """
        if max_new_tokens is None:
            raise
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES
        elif stop_sequences == 'detailed':
            stop_sequences = STOP_SEQUENCES_DETAILED

        # Get cache directory: parameter > environment variable > default
        if cache_dir is None:
            cache_dir = get_hf_cache_dir()
        self.cache_dir = cache_dir

        if 'llama' in model_name.lower():

            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
                eightbit = True
                fourbit = False
            elif model_name.endswith('-4bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True
                )}
                model_name = model_name[:-len('-4bit')]
                eightbit = False
                fourbit = True
            else:
                kwargs = {}
                eightbit = False
                fourbit = False

            if 'Llama-3' in model_name or 'Llama-3.1' in model_name or 'Meta-Llama-3' in model_name or 'Llama-2' in model_name:
                base = 'meta-llama'
                model_name = model_name
            else:
                base = 'huggyllama'

            self.tokenizer = AutoTokenizer.from_pretrained(
                f"{base}/{model_name}", device_map="auto",
                token_type_ids=None, cache_dir=self.cache_dir)

            # Extract model size, handling suffixes like -Instruct, -hf, etc.
            # Split model name and look for the size component (e.g., '8B', '1B', '7B', '70B')
            model_name_parts = model_name.replace('-hf', '').split('-')
            model_size = None
            
            # Look for size pattern in format like '8B', '1B', '7B', '70B', etc.
            for part in model_name_parts:
                part_lower = part.lower()
                # Check if this part is a model size (ends with 'b' and has numbers before it)
                if part_lower.endswith('b') and part_lower[:-1].isdigit():
                    model_size = part_lower
                    break
            
            if not model_size:
                # Fallback: take last part (old behavior)
                logging.warning(f"Could not detect model size from name '{model_name}', using fallback")
                model_size = model_name.split('-')[-1].lower()
            
            logging.info(f"Detected model size: {model_size}")
            llama65b = '65b' in model_name and base == 'huggyllama'
            llama70b = '70b' in model_name.lower() and base == 'meta-llama'

            # AUTOMATIC REDIRECTION TO PRE-QUANTIZED MODEL FOR 70B
            # This prevents downloading the massive 140GB checkpoint
            if llama70b and fourbit:
                logging.info("Redirecting to pre-quantized 4-bit model to save disk space...")
                base = 'unsloth'
                # unsloth names end in -bnb-4bit usually
                # e.g. unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit
                if 'Llama-3.1' in model_name:
                     model_name = 'Meta-Llama-3.1-70B-Instruct-bnb-4bit'
                elif 'Llama-3' in model_name:
                     model_name = 'Meta-Llama-3-70B-Instruct-bnb-4bit'
                
                # Clear quantization_config since unsloth models are pre-quantized
                kwargs = {}
            
            print("Initializing model: ", model_name + " and base:", base)
            if model_size in ['1b', '7b','8b', '13b'] or eightbit:
                # Use device_map="auto" which automatically distributes across all available GPUs
                # accelerate library will handle multi-GPU distribution automatically
                
                # Get max_memory for all GPUs to enable proper multi-GPU distribution
                max_memory_dict = get_gpu_memory_dict()
                if num_gpus > 1:
                    logging.info(f'Using device_map="auto" - will distribute across {num_gpus} GPU(s)')
                    if max_memory_dict:
                        logging.info(f'Max memory per GPU: {max_memory_dict}')
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    f"{base}/{model_name}", device_map="auto",
                    max_memory=max_memory_dict if max_memory_dict else None,
                    cache_dir=self.cache_dir, **kwargs,)

            elif llama70b or llama65b:
                # For 70B models, use quantization if requested (via -8bit suffix)
                if eightbit or fourbit:
                    # Load with quantization for memory efficiency
                    logging.warning('Loading 70B model with quantization. This may still require significant GPU memory.')
                    
                    # Get max_memory for all GPUs to enable proper multi-GPU distribution
                    max_memory_dict = get_gpu_memory_dict()
                    
                    # Add CPU as fallback for offloading if needed (especially for pre-quantized models)
                    if max_memory_dict and base == 'unsloth':
                        max_memory_dict['cpu'] = '100GiB'  # Allow CPU offload if GPU is full
                    
                    if max_memory_dict:
                        logging.info(f'Using max_memory per GPU for quantized model: {max_memory_dict}')
                    
                    # Prepare kwargs for from_pretrained
                    load_kwargs = {
                        'device_map': 'auto',
                        'max_memory': max_memory_dict if max_memory_dict else {0: '8GIB'},
                        'cache_dir': self.cache_dir,
                    }
                    
                    # Only add quantization_config if it exists (not for pre-quantized models)
                    if kwargs.get('quantization_config'):
                        load_kwargs['quantization_config'] = kwargs['quantization_config']
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        f"{base}/{model_name}",
                        **load_kwargs
                    )
                else:
                    # Multi-GPU loading path (automatically uses all available GPUs)
                    
                    num_gpus = torch.cuda.device_count()
                    logging.info(f'Detected {num_gpus} GPU(s) for model loading')
                    
                    path = snapshot_download(
                        repo_id=f'{base}/{model_name}',
                        allow_patterns=['*.json', '*.model', '*.safetensors'],
                        ignore_patterns=['pytorch_model.bin.index.json'],
                        cache_dir=self.cache_dir
                    )
                    config = AutoConfig.from_pretrained(f"{base}/{model_name}", cache_dir=self.cache_dir)
                    with accelerate.init_empty_weights():
                        self.model = AutoModelForCausalLM.from_config(config)
                    self.model.tie_weights()
                    max_mem = 15 * 4686198491

                    # Configure max_memory for all available GPUs
                    # Each GPU gets equal memory allocation
                    max_memory_dict = {i: max_mem for i in range(num_gpus)}
                    logging.info(f'Distributing model across {num_gpus} GPU(s) with max_memory: {max_memory_dict}')

                    device_map = accelerate.infer_auto_device_map(
                        self.model.model,
                        max_memory=max_memory_dict,
                        dtype='float16'
                    )
                    device_map = remove_split_layer(device_map)
                    full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
                    full_model_device_map["lm_head"] = 0

                    logging.info(f'Device map: {full_model_device_map}')

                    self.model = accelerate.load_checkpoint_and_dispatch(
                        self.model, path, device_map=full_model_device_map,
                        dtype='float16', skip_keys='past_key_values')
            else:
                raise ValueError

        elif 'mistral' in model_name.lower():

            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
            if model_name.endswith('-4bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_4bit=True,)}
                model_name = model_name[:-len('-4bit')]
            else:
                kwargs = {}

            model_id = f'mistralai/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False, cache_dir=self.cache_dir)

            # Get max_memory for all GPUs to enable proper multi-GPU distribution
            max_memory_dict = get_gpu_memory_dict()
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map='auto',
                max_memory=max_memory_dict if max_memory_dict else {0: '80GIB'},
                cache_dir=self.cache_dir,
                **kwargs,
            )

        elif 'falcon' in model_name:
            model_id = f'tiiuae/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False, cache_dir=self.cache_dir)

            kwargs = {'quantization_config': BitsAndBytesConfig(
                load_in_8bit=True,),
                'cache_dir': self.cache_dir}

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
                **kwargs,
            )
        else:
            raise ValueError

        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
        # Set token limit based on model capabilities
        if 'Llama-2' in model_name:
            self.token_limit = 4096
        elif 'Llama-3' in model_name or 'Llama-3.1' in model_name or 'Meta-Llama-3' in model_name:
            # Llama-3 models support 8192 tokens, but use 4096 for safety
            self.token_limit = 4096
        else:
            self.token_limit = 2048

    def predict(self, input_data, temperature, return_full=False):

        # Truncate input if it's too long (keep the end which contains the question)
        input_tokens = self.tokenizer.encode(input_data)
        max_input_tokens = self.token_limit - self.max_new_tokens - 50  # Reserve space for generation + buffer
        
        if len(input_tokens) > max_input_tokens:
            # Truncate from the beginning, keeping the end
            truncated_tokens = input_tokens[-max_input_tokens:]
            input_data = self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)
            logging.warning(
                'Input truncated from %d to %d tokens to fit within limit',
                len(input_tokens), max_input_tokens)

        # Implement prediction.
        # Determine the correct device - use the device of the first model parameter
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(input_data, return_tensors="pt").to(device)

        if 'llama' in self.model_name.lower() or 'falcon' in self.model_name or 'mistral' in self.model_name.lower():
            if 'token_type_ids' in inputs:  # Some HF models have changed.
                del inputs['token_type_ids']
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(inputs['input_ids'][0]),
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        logging.debug('temperature: %f', temperature)
        # Save n_input_token before generation (needed later for calculations)
        n_input_token = len(inputs['input_ids'][0])
        
        with torch.no_grad():
            # For greedy decoding (temperature=0), use do_sample=True with top_p=1.0
            if temperature == 0.0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    do_sample=True,
                    top_p=1.0,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=pad_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    temperature=temperature,
                    do_sample=True,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=pad_token_id,
                )
            
            # Clear input tensors from GPU memory immediately after generation
            del inputs
            torch.cuda.empty_cache()

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)

        if return_full:
            return full_answer

        # ===================================================================
        # TOKEN-BASED EXTRACTION (Reliable method)
        # ===================================================================
        # Extract only the generated tokens (ignores input tokens)
        # This is MUCH more reliable than string matching, especially with
        # few-shot prompts that contain multiple "Answer:" markers
        
        generated_token_ids = outputs.sequences[0][n_input_token:]
        # DON'T strip here - we need to detect stop sequences first!
        answer = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        logging.debug(f"✅ Token-based extraction: {len(generated_token_ids)} tokens generated")
        logging.debug(f"✅ Extracted answer (raw): {repr(answer[:100])}..." if len(answer) > 100 else f"✅ Extracted answer (raw): {repr(answer)}")
        
        # Sanity check: Compare with string-based extraction for debugging
        if full_answer.startswith(input_data):
            string_based_answer = full_answer[len(input_data):].strip()
            if answer.strip() != string_based_answer:
                logging.warning("⚠️ Token-based and string-based extraction differ!")
                logging.warning(f"🔹 Token-based: {repr(answer.strip()[:100])}")
                logging.warning(f"🔹 String-based: {repr(string_based_answer[:100])}")
                # Use token-based as it's more reliable
        else:
            logging.debug("ℹ️ Input not echoed exactly - token-based extraction used")

        # ===================================================================
        # HANDLE STOP SEQUENCES AND COMPUTE n_generated
        # ===================================================================
        # Remove stop sequences from answer if present
        # IMPORTANT: Do this BEFORE stripping, so we can detect whitespace stop sequences like '\n\n'
        sliced_answer = answer
        num_stop_tokens = 0  # Track how many tokens were in the stop sequence
        
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    # Found a stop sequence - remove it
                    sliced_answer = answer[:-len(stop)]
                    # Calculate how many tokens were in the stop sequence
                    stop_tokens_encoded = self.tokenizer.encode(stop, add_special_tokens=False)
                    num_stop_tokens = len(stop_tokens_encoded)
                    logging.debug(f"🛑 Removed stop sequence {repr(stop)} ({num_stop_tokens} tokens)")
                    break
            
            # Verify stop sequences were removed successfully
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                error_msg = 'Error: Stop words not removed successfully!'
                error_msg += f'Answer: >{answer}< '
                error_msg += f'Sliced Answer: >{sliced_answer}<'
                if 'falcon' not in self.model_name.lower():
                    raise ValueError(error_msg)
                else:
                    logging.error(error_msg)

        # NOW strip whitespace from the final answer (after stop sequence removal)
        sliced_answer = sliced_answer.strip()

        # Compute n_generated from ORIGINAL token IDs (not re-tokenization!)
        # This ensures perfect alignment between token_ids, tokens, and log_likelihoods
        n_generated = len(generated_token_ids) - num_stop_tokens
        
        # Ensure we have at least 1 generated token (for embeddings and likelihoods)
        if n_generated <= 0:
            logging.warning('n_generated <= 0 after removing stop tokens (only stop_words generated). Using 1 for embeddings.')
            n_generated = 1

        # Calculate token_stop_index for logging/debugging
        # This represents the total number of tokens (input + generated up to stop)
        token_stop_index = n_input_token + n_generated

        # Get the last hidden state (last layer) and the last token's embedding of the answer.
        # Note: We do not want this to be the stop token.

        # outputs.hidden_state is a tuple of len = n_generated_tokens.
        # The first hidden state is for the input tokens and is of shape
        #     (n_layers) x (batch_size, input_size, hidden_size).
        # (Note this includes the first generated token!)
        # The remaining hidden states are for the remaining generated tokens and is of shape
        #    (n_layers) x (batch_size, 1, hidden_size).

        # Note: The output embeddings have the shape (batch_size, generated_length, hidden_size).
        # We do not get embeddings for input_data! We thus subtract the n_tokens_in_input from
        # token_stop_index to arrive at the right output.

        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        # Ensure n_generated is valid (non-negative and within reasonable bounds)
        if n_generated < 1:
            logging.warning(
                'n_generated is %d, setting to 1. '
                'n_input_token: %d, token_stop_index: %d',
                n_generated, n_input_token, token_stop_index)
            n_generated = 1

        # Calculate the index we want to access
        target_idx = n_generated - 1
        
        # Handle empty hidden states
        if len(hidden) == 0:
            logging.error(
                'Hidden states tuple is empty! '
                'n_generated: %d, n_input_token: %d, token_stop_index: %d, '
                'full_answer: %s',
                n_generated, n_input_token, token_stop_index, full_answer)
            raise ValueError('Hidden states tuple is empty - cannot extract embedding')
        
        # Select the appropriate hidden state with robust bounds checking
        if len(hidden) == 1:
            logging.warning(
                'Taking first and only generation for hidden! '
                'n_generated: %d, n_input_token: %d, token_stop_index: %d, '
                'len(hidden): %d, target_idx: %d, '
                'last_token: %s, generation was: %s',
                n_generated, n_input_token, token_stop_index, len(hidden), target_idx,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer)
            last_input = hidden[0]
        elif target_idx < 0:
            # n_generated was 0 or negative (shouldn't happen after our check, but be safe)
            logging.warning(
                'target_idx is negative (%d), using first hidden state. '
                'n_generated: %d, len(hidden): %d',
                target_idx, n_generated, len(hidden))
            last_input = hidden[0]
        elif target_idx >= len(hidden):
            # Index is out of range - use the last available hidden state
            logging.error(
                'target_idx (%d) >= len(hidden) (%d). Using last hidden state. '
                'n_generated: %d, n_input_token: %d, token_stop_index: %d, '
                'last_token: %s, generation was: %s, slice_answer: %s',
                target_idx, len(hidden), n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer, sliced_answer)
            last_input = hidden[-1]
        else:
            # Normal case: index is valid
            last_input = hidden[target_idx]

        # Then access last layer for input
        last_layer = last_input[-1]
        # Then access last token in input - move to CPU immediately to free GPU memory
        last_token_embedding = last_layer[:, -1, :].cpu()
        
        # Clear hidden states from GPU memory after extracting what we need
        del hidden, last_input, last_layer
        torch.cuda.empty_cache()

        # Get log_likelihoods.
        # outputs.scores are the logits for the generated token.
        # outputs.scores is a tuple of len = n_generated_tokens.
        # Each entry is shape (bs, vocabulary size).
        # outputs.sequences is the sequence of all tokens: input and generated.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        # Transition_scores[0] only contains the scores for the first generated tokens.

        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning('Taking first and only generation for log likelihood!')
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning('Generation interrupted by max_token limit.')

        if len(log_likelihoods) == 0:
            raise ValueError

        # Extract the generated token IDs (important for exact alignment with log_likelihoods)
        generated_token_ids = outputs.sequences[0][n_input_token:n_input_token + n_generated].tolist()
        
        # Also get token strings for easier debugging/analysis
        generated_tokens = [self.tokenizer.decode([tid]) for tid in generated_token_ids]

        # Clear transition_scores from GPU memory before returning
        # Note: outputs.scores and outputs.sequences will be garbage collected when outputs goes out of scope
        del transition_scores
        torch.cuda.empty_cache()

        return sliced_answer, log_likelihoods, last_token_embedding, generated_token_ids, generated_tokens

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input."""

        input_data += ' A'
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']
        # The computation of the negative log likelihoods follows:
        # https://huggingface.co/docs/transformers/perplexity.

        target_ids_true = tokenized_prompt_true.clone()
        # Set all target_ids except the last one to -100.
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)

        loss_true = model_output_true.loss

        return -loss_true.item()
