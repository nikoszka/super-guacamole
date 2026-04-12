#!/bin/bash

################################################################################
# SE/SAR Baseline Comparison Runner
#
# Two-phase overnight job for a single Titan V (12 GB):
#   Phase A – Generate multi-sample responses (temperature=0.5, 5 samples)
#   Phase B – Run Phase 5 with SAR + Semantic Entropy enabled
#
# Selected configurations (3 model families × 1 dataset each):
#   1. Llama-3.1-8B-8bit  on coqa       (Llama family, strong NLL signal)
#   2. Mistral-7B-v0.3-8bit on trivia_qa (Mistral family, strong baseline)
#   3. Qwen2.5-1.5B       on trivia_qa  (Qwen family, small enough for FP16)
#
# Estimated wall-clock: 4-8 hours total on a Titan V
################################################################################

set -e

ENTITY="${WANDB_ENTITY:-nikosteam}"
PROJECT="${WANDB_PROJECT:-super_guacamole}"
NUM_SAMPLES=400
NUM_FEW_SHOT=0
TEMPERATURE=0.5
NUM_GENERATIONS=5
BRIEF_PROMPT="${BRIEF_PROMPT:-manual}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="results/se_sar_comparison_${TIMESTAMP}"

CONFIGS=(
    "Llama-3.1-8B-8bit:coqa:Llama:Large"
    "Mistral-7B-v0.3-8bit:trivia_qa:Mistral:Large"
    "Qwen2.5-1.5B:trivia_qa:Qwen:Small"
)

mkdir -p "$LOGDIR"

echo "================================================================================"
echo "SE/SAR Baseline Comparison — Overnight Run"
echo "================================================================================"
echo "Temperature:   $TEMPERATURE"
echo "Generations:   $NUM_GENERATIONS per example (+ 1 greedy = $((NUM_GENERATIONS + 1)) total)"
echo "Samples:       $NUM_SAMPLES examples per dataset"
echo ""
echo "Configurations:"
for cfg in "${CONFIGS[@]}"; do
    MODEL=$(echo "$cfg" | cut -d: -f1)
    DATASET=$(echo "$cfg" | cut -d: -f2)
    echo "  - $MODEL on $DATASET"
done
echo ""
echo "Log directory:  $LOGDIR"
echo "Started:        $(date)"
echo "================================================================================"

# =============================================================================
# PHASE A: Generate multi-sample responses
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  PHASE A: Multi-sample generation (temperature=$TEMPERATURE, n=$NUM_GENERATIONS)                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"

for cfg in "${CONFIGS[@]}"; do
    MODEL=$(echo "$cfg" | cut -d: -f1)
    DATASET=$(echo "$cfg" | cut -d: -f2)
    FAMILY=$(echo "$cfg" | cut -d: -f3)
    SIZE=$(echo "$cfg" | cut -d: -f4)

    echo ""
    echo "────────────────────────────────────────────────────────────────────────────"
    echo "  Generating: $MODEL on $DATASET"
    echo "  Started:    $(date)"
    echo "────────────────────────────────────────────────────────────────────────────"

    cd src
    python generate_answers.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --num_few_shot "$NUM_FEW_SHOT" \
        --temperature "$TEMPERATURE" \
        --num_generations "$NUM_GENERATIONS" \
        --brief_prompt "$BRIEF_PROMPT" \
        --enable_brief \
        --brief_always \
        --no-compute_uncertainties \
        --no-compute_p_true \
        --no-get_training_set_generations \
        --use_context \
        --entity "$ENTITY" \
        --project "$PROJECT" \
        --experiment_lot "se_sar_${FAMILY}_${DATASET}_${TIMESTAMP}" \
        2>&1 | tee "../${LOGDIR}/gen_${MODEL}_${DATASET}.log"
    cd ..

    echo "  Completed generation: $MODEL on $DATASET at $(date)"

    echo "  Waiting 30s for GPU memory cleanup..."
    sleep 30
done

echo ""
echo "================================================================================"
echo "  PHASE A complete — all multi-sample generations finished at $(date)"
echo "================================================================================"

# =============================================================================
# PHASE B: Run Phase 5 with SAR + Semantic Entropy
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  PHASE B: Computing SAR + Semantic Entropy on generated responses          ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  This phase loads DeBERTa-v2-xlarge-mnli (~3.6 GB) for SE"
echo "  and cross-encoder/stsb-roberta-large (~1.4 GB) for SAR."
echo "  No LLM needed — fits easily on Titan V."
echo ""

# Discover the pickle files that were just generated.
# They land in src/boldis/uncertainty/wandb/{Size}/{dataset}/{model}/run-*/files/validation_generations.pkl
# We look for runs created in this session (matching our experiment_lot pattern).

for cfg in "${CONFIGS[@]}"; do
    MODEL=$(echo "$cfg" | cut -d: -f1)
    DATASET=$(echo "$cfg" | cut -d: -f2)
    FAMILY=$(echo "$cfg" | cut -d: -f3)
    SIZE=$(echo "$cfg" | cut -d: -f4)

    # Strip quantization suffix for directory lookup
    MODEL_CLEAN=$(echo "$MODEL" | sed 's/-8bit$//' | sed 's/-4bit$//')

    echo "────────────────────────────────────────────────────────────────────────────"
    echo "  Looking for pickle: $SIZE / $DATASET / $MODEL_CLEAN"

    # Find the most recent pickle for this model/dataset/size
    PICKLE_PATH=$(find "src/boldis/uncertainty/wandb/${SIZE}/${DATASET}/${MODEL_CLEAN}" \
        -name "validation_generations.pkl" -type f -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "$PICKLE_PATH" ]; then
        echo "  WARNING: No pickle found for $MODEL on $DATASET — skipping Phase 5"
        echo "  Searched: src/boldis/uncertainty/wandb/${SIZE}/${DATASET}/${MODEL_CLEAN}/"
        continue
    fi

    echo "  Found: $PICKLE_PATH"

    OUTPUT_DIR="${LOGDIR}/phase5_se_sar/${SIZE}/${DATASET}/${MODEL_CLEAN}"
    mkdir -p "$OUTPUT_DIR"

    echo "  Running Phase 5 with SAR + SE enabled..."
    echo "  Started: $(date)"

    python src/analysis/phase5_comparative_analysis.py \
        --pickle-path "$PICKLE_PATH" \
        --model-name "$MODEL_CLEAN" \
        --output-dir "$OUTPUT_DIR" \
        --num-samples-sar 5 \
        --num-samples-se 5 \
        2>&1 | tee "${LOGDIR}/phase5_${MODEL}_${DATASET}.log"

    echo "  Phase 5 complete: $MODEL on $DATASET at $(date)"
    echo "  Results: $OUTPUT_DIR/"
    echo ""

    sleep 10
done

echo ""
echo "================================================================================"
echo "  PHASE B complete at $(date)"
echo "================================================================================"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  ALL DONE                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Results directory: $LOGDIR"
echo "  Finished:          $(date)"
echo ""
echo "  Phase 5 outputs (AUROC CSVs, ROC curves, cost plots):"
for cfg in "${CONFIGS[@]}"; do
    MODEL=$(echo "$cfg" | cut -d: -f1)
    DATASET=$(echo "$cfg" | cut -d: -f2)
    SIZE=$(echo "$cfg" | cut -d: -f4)
    MODEL_CLEAN=$(echo "$MODEL" | sed 's/-8bit$//' | sed 's/-4bit$//')
    echo "    - ${LOGDIR}/phase5_se_sar/${SIZE}/${DATASET}/${MODEL_CLEAN}/"
done
echo ""
echo "  To compare with your existing Phase 5 results (G-NLL only):"
echo "    diff results/pipeline/{Size}/{dataset}/{model}/phase5/auroc_comparison.csv \\"
echo "         ${LOGDIR}/phase5_se_sar/{Size}/{dataset}/{model}/auroc_comparison.csv"
echo ""
