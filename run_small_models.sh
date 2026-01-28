#!/bin/bash

################################################################################
# Small Models Experiment Runner
# Runs all small models (1B-1.5B) on both datasets
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
MODELS=(
    "Llama-3.2-1B:Llama:Small"
    "Qwen2.5-1.5B:Qwen:Small"
    "Mistral-7B-v0.3-8bit:Mistral:Small"
)

echo "================================================================================"
echo "Small Models Experiments"
echo "================================================================================"
echo "Models: Llama-3.2-1B, Qwen2.5-1.5B, Mistral-7B-v0.3-8bit"
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
            --experiment_lot "small_models_${FAMILY}_${DATASET}_${TIMESTAMP}"
        cd ..
        
        echo "✅ Completed: $MODEL on $DATASET"
        sleep 10  # Memory cleanup
    done
done

echo ""
echo "================================================================================"
echo "✅ All small model experiments completed!"
echo "================================================================================"
