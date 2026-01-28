#!/bin/bash

################################################################################
# Quick Baseline Validation Script
# 
# This script runs a quick validation with just 10 samples to verify:
# 1. Model loading works
# 2. Dataset loading works  
# 3. Generation pipeline works
# 4. Results are saved correctly
#
# Use this before running the full multi_model_experiments.sh script
################################################################################

set -e

# Configuration
ENTITY="${WANDB_ENTITY:-nikosteam}"
PROJECT="${WANDB_PROJECT:-super_guacamole}"
NUM_SAMPLES=10  # Quick validation with just 10 samples
NUM_FEW_SHOT=5
TEMPERATURE=0.0
BRIEF_PROMPT="short"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================================"
echo "Quick Baseline Validation"
echo "================================================================================"
echo "This will run quick tests (10 samples each) to validate the pipeline."
echo ""
echo "Models to test:"
echo "  - Llama-3.2-1B (baseline)"
echo ""
echo "Datasets to test:"
echo "  - trivia_qa"
echo "  - squad"
echo ""
echo "================================================================================"
echo ""

# Test 1: Llama-3.2-1B on TriviaQA
echo "Test 1/2: Llama-3.2-1B on TriviaQA..."
cd src
python generate_answers.py \
    --model_name "Llama-3.2-1B" \
    --dataset "trivia_qa" \
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
    --experiment_lot "baseline_validation_trivia_${TIMESTAMP}"

echo "✅ Test 1/2 completed!"
echo ""

# Test 2: Llama-3.2-1B on SQuAD
echo "Test 2/2: Llama-3.2-1B on SQuAD..."
python generate_answers.py \
    --model_name "Llama-3.2-1B" \
    --dataset "squad" \
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
    --experiment_lot "baseline_validation_squad_${TIMESTAMP}"

cd ..

echo "✅ Test 2/2 completed!"
echo ""
echo "================================================================================"
echo "✅ ALL VALIDATION TESTS PASSED!"
echo "================================================================================"
echo ""
echo "The pipeline is working correctly. You can now run the full experiments with:"
echo "  ./run_multi_model_experiments.sh"
echo ""
echo "Or run individual model families with the phase scripts."
echo "================================================================================"
