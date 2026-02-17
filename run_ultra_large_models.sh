#!/bin/bash

################################################################################
# Ultra-Large Models Experiment Runner
# Runs ultra-large models (70B+ with 4-bit quantization) on both datasets
#
# Optimized for 4×11GB GPUs (44GB total)
# All models use 4-bit quantization to fit in available memory
################################################################################

set -e

ENTITY="${WANDB_ENTITY:-nikosteam}"
PROJECT="${WANDB_PROJECT:-super_guacamole}"
NUM_SAMPLES=400
NUM_FEW_SHOT=5
TEMPERATURE=0.0
BRIEF_PROMPT="${BRIEF_PROMPT:-short}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

DATASETS=("trivia_qa" "squad")

# Ultra-large models with 4-bit quantization
# These are in the 70B+ weight class
MODELS=(
    "Llama-3.1-70B-4bit:Llama:XLarge"
    "Qwen2.5-72B-4bit:Qwen:XLarge"
    "Mistral-Large-Instruct-2407-4bit:Mistral:XLarge"
)

echo "================================================================================"
echo "Ultra-Large Models Experiments (70B+ Weight Class)"
echo "================================================================================"
echo "Models:"
echo "  - Llama-3.1-70B-4bit (70B params, ~35GB with 4-bit)"
echo "  - Qwen2.5-72B-4bit (72B params, ~36GB with 4-bit)"
echo "  - Mistral-Large-2-4bit (123B params, ~62GB with 4-bit, may use CPU offload)"
echo ""
echo "Datasets: TriviaQA, SQuAD"
echo "Total: ${#MODELS[@]} models × ${#DATASETS[@]} datasets = $((${#MODELS[@]} * ${#DATASETS[@]})) experiments"
echo ""
echo "⚠️  NOTE: These models use 4-bit quantization and distribute across your 4 GPUs"
echo "    Mistral Large 2 is 123B and may require CPU offloading"
echo "    Monitor GPU memory with: watch -n 1 nvidia-smi"
echo "================================================================================"

for model_info in "${MODELS[@]}"; do
    MODEL=$(echo "$model_info" | cut -d: -f1)
    FAMILY=$(echo "$model_info" | cut -d: -f2)
    SIZE=$(echo "$model_info" | cut -d: -f3)
    
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "================================================================================"
        echo "Running: $MODEL on $DATASET"
        echo "Started: $(date)"
        echo "================================================================================"
        
        cd src
        python generate_answers.py \
            --model_name "$MODEL" \
            --dataset "$DATASET" \
            --num_samples "$NUM_SAMPLES" \
            --num_few_shot "$NUM_FEW_SHOT" \
            --temperature "$TEMPERATURE" \
            --num_generations 1 \
            --brief_prompt "$BRIEF_PROMPT" \
            --enable_brief \
            --brief_always \
            --no-compute_uncertainties \
            --no-compute_p_true \
            --no-get_training_set_generations \
            --use_context \
            --entity "$ENTITY" \
            --project "$PROJECT" \
            --experiment_lot "ultra_large_${FAMILY}_${SIZE}_${DATASET}_${TIMESTAMP}"
        cd ..
        
        echo "✅ Completed: $MODEL on $DATASET"
        echo "Completed: $(date)"
        
        # Extra cleanup time for large models
        echo "Cleaning up GPU memory..."
        sleep 20
    done
done

echo ""
echo "================================================================================"
echo "✅ All ultra-large model experiments completed!"
echo "================================================================================"
