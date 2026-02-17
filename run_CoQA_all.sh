#!/bin/bash

################################################################################
# CoQA Dataset - All Models Experiment Runner
# Runs ALL models (small → large → ultra-large) on CoQA dataset
#
# CoQA: Conversational Question Answering Challenge
# - 127K questions from 8K conversations
# - Questions require understanding conversational context
# - Free-form answers with text spans
#
# Total: 9 models (3 small + 3 large + 3 ultra-large) on CoQA
################################################################################

set -e

ENTITY="${WANDB_ENTITY:-nikosteam}"
PROJECT="${WANDB_PROJECT:-super_guacamole}"
NUM_SAMPLES=400
NUM_FEW_SHOT=5
TEMPERATURE=0.0
BRIEF_PROMPT="${BRIEF_PROMPT:-short}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# CoQA is conversational, so we always use context
DATASET="coqa"

echo "================================================================================"
echo "CoQA Dataset - Complete Model Suite"
echo "================================================================================"
echo "Dataset: CoQA (Conversational Question Answering)"
echo "Total Models: 9 (Small: 3, Large: 3, Ultra-Large: 3)"
echo "Estimated Time: 24-30 hours for all experiments"
echo ""
echo "Model Categories:"
echo "  📦 SMALL (1-7B):       Fast, baseline performance"
echo "  📦 LARGE (7-8B):       Balanced, strong performance"
echo "  📦 ULTRA-LARGE (70B+): Slow, maximum performance"
echo "================================================================================"
echo ""

# ============================================================================
# PHASE 1: SMALL MODELS (1B-7B with quantization)
# ============================================================================
echo "================================================================================"
echo "PHASE 1/3: Small Models (1B-7B)"
echo "================================================================================"
echo "Models:"
echo "  - Llama-3.2-1B (1B params)"
echo "  - Qwen2.5-1.5B (1.5B params)"
echo "  - Mistral-7B-v0.3-8bit (7B params, 8-bit quantized)"
echo ""
echo "Expected time: ~6-8 hours for all 3 models"
echo "================================================================================"

SMALL_MODELS=(
    "Llama-3.2-1B:Llama:Small"
    "Qwen2.5-1.5B:Qwen:Small"
    "Mistral-7B-v0.3-8bit:Mistral:Small"
)

for model_info in "${SMALL_MODELS[@]}"; do
    MODEL=$(echo "$model_info" | cut -d: -f1)
    FAMILY=$(echo "$model_info" | cut -d: -f2)
    SIZE=$(echo "$model_info" | cut -d: -f3)
    
    echo ""
    echo "🔄 Running SMALL model: $MODEL on CoQA"
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
        --experiment_lot "coqa_small_${FAMILY}_${TIMESTAMP}"
    cd ..
    
    echo "✅ Completed: $MODEL on CoQA"
    echo "Completed: $(date)"
    sleep 10  # Memory cleanup
done

echo ""
echo "================================================================================"
echo "✅ PHASE 1 COMPLETE: All small models finished on CoQA"
echo "================================================================================"
sleep 5

# ============================================================================
# PHASE 2: LARGE MODELS (7-8B without quantization)
# ============================================================================
echo ""
echo "================================================================================"
echo "PHASE 2/3: Large Models (7-8B)"
echo "================================================================================"
echo "Models:"
echo "  - Llama-3.1-8B (8B params, fp16)"
echo "  - Qwen3-8B (8B params, fp16)"
echo "  - Mistral-7B-Instruct-v0.3 (7B params, fp16)"
echo ""
echo "Expected time: ~8-10 hours for all 3 models"
echo "================================================================================"

LARGE_MODELS=(
    "Llama-3.1-8B:Llama:Large"
    "Qwen3-8B:Qwen:Large"
    "Mistral-7B-Instruct-v0.3:Mistral:Large"
)

for model_info in "${LARGE_MODELS[@]}"; do
    MODEL=$(echo "$model_info" | cut -d: -f1)
    FAMILY=$(echo "$model_info" | cut -d: -f2)
    SIZE=$(echo "$model_info" | cut -d: -f3)
    
    echo ""
    echo "🔄 Running LARGE model: $MODEL on CoQA"
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
        --experiment_lot "coqa_large_${FAMILY}_${TIMESTAMP}"
    cd ..
    
    echo "✅ Completed: $MODEL on CoQA"
    echo "Completed: $(date)"
    sleep 15  # Memory cleanup (larger models need more time)
done

echo ""
echo "================================================================================"
echo "✅ PHASE 2 COMPLETE: All large models finished on CoQA"
echo "================================================================================"
sleep 5

# ============================================================================
# PHASE 3: ULTRA-LARGE MODELS (70B+ with 4-bit quantization)
# ============================================================================
echo ""
echo "================================================================================"
echo "PHASE 3/3: Ultra-Large Models (70B+)"
echo "================================================================================"
echo "Models:"
echo "  - Llama-3.1-70B-Instruct-4bit (70B params)"
echo "  - Qwen2.5-72B-4bit (72B params)"
echo "  - Mistral-Large-2-4bit (123B params, uses CPU offload)"
echo ""
echo "Expected time: ~10-12 hours for all 3 models"
echo "⚠️  NOTE: These models use 4-bit quantization and distribute across 4 GPUs"
echo "    Mistral Large 2 also uses CPU offloading (slower but highest quality)"
echo "================================================================================"

ULTRA_LARGE_MODELS=(
    "Llama-3.1-70B-Instruct-4bit:Llama:XLarge"
    "Qwen2.5-72B-4bit:Qwen:XLarge"
    "Mistral-Large-Instruct-2407-4bit:Mistral:XLarge"
)

for model_info in "${ULTRA_LARGE_MODELS[@]}"; do
    MODEL=$(echo "$model_info" | cut -d: -f1)
    FAMILY=$(echo "$model_info" | cut -d: -f2)
    SIZE=$(echo "$model_info" | cut -d: -f3)
    
    echo ""
    echo "🔄 Running ULTRA-LARGE model: $MODEL on CoQA"
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
        --experiment_lot "coqa_xlarge_${FAMILY}_${TIMESTAMP}"
    cd ..
    
    echo "✅ Completed: $MODEL on CoQA"
    echo "Completed: $(date)"
    
    # Extra cleanup time for ultra-large models
    echo "Cleaning up GPU memory..."
    sleep 20
done

echo ""
echo "================================================================================"
echo "✅ PHASE 3 COMPLETE: All ultra-large models finished on CoQA"
echo "================================================================================"

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
echo "================================================================================"
echo "🎉 ALL EXPERIMENTS COMPLETE!"
echo "================================================================================"
echo "Dataset: CoQA (Conversational Question Answering)"
echo "Total Models Run: 9"
echo ""
echo "Breakdown:"
echo "  ✅ Small Models (3):       Llama-3.2-1B, Qwen2.5-1.5B, Mistral-7B-v0.3"
echo "  ✅ Large Models (3):       Llama-3.1-8B, Qwen3-8B, Mistral-7B-Instruct-v0.3"
echo "  ✅ Ultra-Large Models (3): Llama-3.1-70B, Qwen2.5-72B, Mistral-Large-2"
echo ""
echo "Completed: $(date)"
echo "Total Runtime: Started at ${TIMESTAMP}"
echo ""
echo "Next Steps:"
echo "  1. Check WandB for results: https://wandb.ai/${ENTITY}/${PROJECT}"
echo "  2. Compute AUROC: python compute_multi_model_auroc.py"
echo "  3. Compare with TriviaQA and SQuAD results"
echo "================================================================================"
