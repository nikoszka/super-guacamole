#!/bin/bash

################################################################################
# Multi-Model Family Comparison Experiment Runner
# 
# This script runs experiments across 3 model families (Llama, Qwen, Mistral)
# with 2 models per family (small and large) on 2 datasets (TriviaQA, SQuAD).
#
# Total: 6 models × 2 datasets = 12 experiment runs
#
# Optimized for: Single A100 40GB GPU with balanced speed/accuracy
################################################################################

set -e  # Exit on error

# Configuration
ENTITY="${WANDB_ENTITY:-nikosteam}"
PROJECT="${WANDB_PROJECT:-super_guacamole}"
NUM_SAMPLES=400
NUM_FEW_SHOT=5
TEMPERATURE=0.0
NUM_GENERATIONS=1
MAX_NEW_TOKENS=150

# Answer type: "short" for brief answers, "detailed" for long answers
BRIEF_PROMPT="${BRIEF_PROMPT:-short}"

# Datasets to test
DATASETS=("trivia_qa" "squad")

# Model families with quantization
# Format: "ModelName:FamilyLabel:SizeLabel"
MODELS=(
    "Llama-3.2-1B:Llama:Small"
    "Llama-3-8B-8bit:Llama:Large"
    "Qwen2.5-1.5B:Qwen:Small"
    "Qwen2.5-7B-8bit:Qwen:Large"
    "Mistral-7B-v0.3-8bit:Mistral:Small"
    "Mistral-7B-Instruct-v0.3-8bit:Mistral:Large"
)

# Optional: Add Mixtral for even larger model comparison
# "Mixtral-8x7B-Instruct-v0.1-4bit:Mistral:XLarge"

# Logging
LOG_DIR="experiment_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/multi_model_experiments_${TIMESTAMP}.log"

echo "================================================================================"
echo "Multi-Model Family Comparison Experiments"
echo "================================================================================"
echo "Start time: $(date)"
echo "Configuration:"
echo "  Entity: $ENTITY"
echo "  Project: $PROJECT"
echo "  Num samples: $NUM_SAMPLES"
echo "  Few-shot: $NUM_FEW_SHOT"
echo "  Temperature: $TEMPERATURE (greedy decoding)"
echo "  Brief prompt: $BRIEF_PROMPT"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo ""
echo "Models to test: ${#MODELS[@]}"
for model_info in "${MODELS[@]}"; do
    model_name=$(echo "$model_info" | cut -d: -f1)
    family=$(echo "$model_info" | cut -d: -f2)
    size=$(echo "$model_info" | cut -d: -f3)
    echo "  - $model_name [$family, $size]"
done
echo ""
echo "Datasets to test: ${#DATASETS[@]}"
for dataset in "${DATASETS[@]}"; do
    echo "  - $dataset"
done
echo ""
echo "Total experiments: $((${#MODELS[@]} * ${#DATASETS[@]}))"
echo "Estimated time: 12-24 hours (depending on GPU speed)"
echo "================================================================================"
echo ""

# Ask for confirmation
read -p "Press Enter to continue or Ctrl+C to cancel..." </dev/tty

# Counter for progress tracking
TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#DATASETS[@]}))
CURRENT_EXPERIMENT=0

# Run experiments
for model_info in "${MODELS[@]}"; do
    # Parse model info
    MODEL=$(echo "$model_info" | cut -d: -f1)
    FAMILY=$(echo "$model_info" | cut -d: -f2)
    SIZE=$(echo "$model_info" | cut -d: -f3)
    
    for DATASET in "${DATASETS[@]}"; do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
        
        echo ""
        echo "================================================================================"
        echo "Experiment $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS"
        echo "================================================================================"
        echo "Model: $MODEL [$FAMILY, $SIZE]"
        echo "Dataset: $DATASET"
        echo "Started: $(date)"
        echo "================================================================================"
        
        # Create experiment name
        EXPERIMENT_NAME="multi_model_${FAMILY}_${SIZE}_${DATASET}_${TIMESTAMP}"
        
        # Log file for this experiment
        EXPERIMENT_LOG="$LOG_DIR/${EXPERIMENT_NAME}.log"
        
        # Run generation
        cd src
        
        echo "Running generation..."
        if python generate_answers.py \
            --model_name "$MODEL" \
            --dataset "$DATASET" \
            --num_samples "$NUM_SAMPLES" \
            --num_few_shot "$NUM_FEW_SHOT" \
            --temperature "$TEMPERATURE" \
            --num_generations "$NUM_GENERATIONS" \
            --model_max_new_tokens "$MAX_NEW_TOKENS" \
            --brief_prompt "$BRIEF_PROMPT" \
            --enable_brief \
            --brief_always \
            --no-compute_uncertainties \
            --no-compute_p_true \
            --no-get_training_set_generations \
            --use_context \
            --entity "$ENTITY" \
            --project "$PROJECT" \
            --experiment_lot "$EXPERIMENT_NAME" \
            2>&1 | tee "../$EXPERIMENT_LOG"; then
            
            echo "✅ Experiment $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS completed successfully!"
            echo "   Model: $MODEL on $DATASET"
            echo "   Log: $EXPERIMENT_LOG"
            
        else
            echo "❌ Experiment $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS FAILED!"
            echo "   Model: $MODEL on $DATASET"
            echo "   Check log: $EXPERIMENT_LOG"
            echo ""
            echo "Continue with next experiment? (y/n)"
            read -p "Choice: " continue_choice </dev/tty
            if [[ "$continue_choice" != "y" ]]; then
                echo "Experiments stopped by user."
                cd ..
                exit 1
            fi
        fi
        
        cd ..
        
        # Memory cleanup - wait a bit between experiments
        echo "Cleaning up GPU memory..."
        sleep 15
        
        echo "Completed: $(date)"
        echo ""
    done
done

echo ""
echo "================================================================================"
echo "All Experiments Completed!"
echo "================================================================================"
echo "End time: $(date)"
echo "Total experiments run: $CURRENT_EXPERIMENT"
echo ""
echo "Logs directory: $LOG_DIR"
echo "Master log: $MASTER_LOG"
echo ""
echo "Next steps:"
echo "  1. Check wandb for run IDs: https://wandb.ai/$ENTITY/$PROJECT"
echo "  2. Run evaluation with LLM judge (if needed)"
echo "  3. Compute G-NLL and RW-G-NLL AUROC metrics"
echo "  4. Create comparative analysis visualizations"
echo "================================================================================"
