#!/bin/bash

################################################################################
# Large Models Experiment Runner
# Runs all large models (7B-8B with 8-bit quantization) on both datasets
################################################################################

set -e

ENTITY="${WANDB_ENTITY:-nikosteam}"
PROJECT="${WANDB_PROJECT:-super_guacamole}"
NUM_SAMPLES=400
NUM_FEW_SHOT=0
TEMPERATURE=0.0
BRIEF_PROMPT="${BRIEF_PROMPT:-manual}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

DATASETS=("trivia_qa" "squad")
MODELS=(
    "Llama-3.1-8B:Llama:Large"
    "Qwen3-8B:Qwen:Large"
    "Ministral-3-8B-Instruct-2512:Mistral:Large"
)

echo "================================================================================"
echo "Large Models Experiments"
echo "================================================================================"
echo "Models: Llama-3.1-8B, Qwen3-8B, Ministral-3-8B-Instruct-2512 (fp16, no quantization)"
echo "Datasets: TriviaQA, SQuAD"
echo "Total: ${#MODELS[@]} models × ${#DATASETS[@]} datasets = $((${#MODELS[@]} * ${#DATASETS[@]})) experiments"
echo "================================================================================"

for model_info in "${MODELS[@]}"; do
    MODEL=$(echo "$model_info" | cut -d: -f1)
    FAMILY=$(echo "$model_info" | cut -d: -f2)
    SIZE=$(echo "$model_info" | cut -d: -f3)
    
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "Running: $MODEL on $DATASET"
        echo "Started: $(date)"
        
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
            --experiment_lot "large_models_${FAMILY}_${DATASET}_${TIMESTAMP}"
        cd ..
        
        echo "✅ Completed: $MODEL on $DATASET"
        sleep 15  # Memory cleanup (larger models need more time)
    done
done

echo ""
echo "================================================================================"
echo "✅ All large model experiments completed!"
echo "================================================================================"
