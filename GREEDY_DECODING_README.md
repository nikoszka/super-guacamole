# Greedy Decoding Configuration for Answer Generation

This configuration modifies the answer generation script to use **greedy decoding** (temperature=0) and generate **detailed, sentence-based answers** suitable for LLM-as-judge evaluation.

## Key Changes Made

### 1. Greedy Decoding
- **Temperature**: Changed from 1.0 to 0.0 (greedy decoding)
- **Generations**: Changed from 15 to 1 (single most probable answer)
- **Sampling**: Properly configured for greedy decoding in HuggingFace models

### 2. New Detailed Prompt
- Added `'detailed'` prompt type: *"Provide a detailed, well-structured answer to the following question. Write your response as a complete sentence that thoroughly addresses the question."*
- Set as default prompt type for generating longer, more informative answers

### 3. Model Configuration
- Fixed HuggingFace model to properly handle temperature=0 with `do_sample=True` and `top_p=1.0`
- Maintains compatibility with existing model loading and generation pipeline

## How to Run

### Option 1: Use the Convenience Script
```bash
python run_greedy_decoding.py
```

### Option 2: Run Directly
```bash
cd src
python generate_answers.py \
    --model_name "Llama-2-7b-chat" \
    --dataset "squad" \
    --num_samples 100 \
    --temperature 0.0 \
    --num_generations 1 \
    --brief_prompt "detailed" \
    --enable_brief True \
    --brief_always True \
    --use_context True \
    --metric "llm_gpt-4" \
    --compute_uncertainties False
```

## Configuration Options

### Required Arguments
- `--model_name`: Model to use (e.g., "Llama-2-7b-chat", "Mistral-7B-Instruct-v0.1")
- `--dataset`: Dataset to evaluate on ("squad", "trivia_qa", "svamp", "nq", "bioasq")

### Key Parameters
- `--temperature 0.0`: Greedy decoding (most probable token)
- `--num_generations 1`: Single answer per question
- `--brief_prompt detailed`: Use detailed sentence prompt
- `--num_samples`: Number of questions to evaluate (start small for testing)
- `--num_few_shot`: Number of few-shot examples (default: 5)

### LLM-as-Judge Options
- `--metric "llm_gpt-4"`: Use GPT-4 as judge
- `--metric "llm_gpt-3.5"`: Use GPT-3.5 as judge
- `--metric "squad"`: Use SQuAD F1 metric

### Optional Settings
- `--use_context True`: Include context in prompts (recommended)
- `--compute_uncertainties False`: Skip uncertainty computation for faster runs
- `--get_training_set_generations False`: Skip training set generation

## Output Files

The script generates:
- `validation_generations.pkl`: Generated answers with metadata
- `experiment_details.pkl`: Configuration and prompt details
- Weights & Biases logs for monitoring

## Example Usage for Different Datasets

### SQuAD
```bash
python generate_answers.py --dataset squad --use_context True --metric llm_gpt-4
```

### TriviaQA
```bash
python generate_answers.py --dataset trivia_qa --use_context False --metric llm_gpt-4
```

### SVAMP (Math Word Problems)
```bash
python generate_answers.py --dataset svamp --use_context True --metric llm_gpt-4
```

## Notes

1. **Memory Usage**: Greedy decoding with single generation uses less memory than sampling
2. **Speed**: Much faster than multi-generation uncertainty estimation
3. **Quality**: Detailed prompts encourage longer, more informative answers
4. **Evaluation**: Perfect for LLM-as-judge evaluation due to single, deterministic answers

## Troubleshooting

- If you get CUDA OOM errors, reduce `--num_samples` or use a smaller model
- For very long answers, you may need to increase `--model_max_new_tokens`
- Make sure you have the required model access tokens for HuggingFace models
