#!/bin/bash

################################################################################
# Semantic Entropy Replication (Kuhn et al., ICLR 2023)
#
# Replicates the SE experiment with paper-matching settings on our models/datasets.
# Uses our existing prompt format (manual brief) for consistency with other results.
#
# Paper settings:
#   - 10 multinomial samples per question (temperature=0.5)
#   - DeBERTa-v2-xlarge-mnli for NLI clustering
#   - Non-strict bidirectional entailment
#   - Datasets: TriviaQA, CoQA
#
# Selected models:
#   - Llama-3.2-1B  (Small representative)
#   - Llama-3.1-8B  (Large representative)
#
# Two phases:
#   Phase A – Generate 10 high-temperature responses per example
#   Phase B – Run Phase 5 with Semantic Entropy + SAR + all baselines
#
# Estimated wall-clock: ~10-18 hours total (overnight run)
################################################################################

set -e

ENTITY="${WANDB_ENTITY:-nikosteam}"
PROJECT="${WANDB_PROJECT:-super_guacamole}"
NUM_SAMPLES=400
NUM_FEW_SHOT=0
TEMPERATURE=0.5
NUM_GENERATIONS=10
BRIEF_PROMPT="${BRIEF_PROMPT:-manual}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="results/se_replication_${TIMESTAMP}"

CONFIGS=(
    "Llama-3.2-1B:trivia_qa:Llama:Small"
    "Llama-3.2-1B:coqa:Llama:Small"
    "Llama-3.1-8B:trivia_qa:Llama:Large"
    "Llama-3.1-8B:coqa:Llama:Large"
)

mkdir -p "$LOGDIR"

echo "================================================================================"
echo "Semantic Entropy Replication (Kuhn et al., ICLR 2023)"
echo "================================================================================"
echo "Temperature:    $TEMPERATURE (paper optimal from Fig 3b ablation)"
echo "Generations:    $NUM_GENERATIONS per example (+ 1 primary = $((NUM_GENERATIONS + 1)) total)"
echo "Samples:        $NUM_SAMPLES examples per dataset"
echo "Brief prompt:   $BRIEF_PROMPT (our format, for consistency)"
echo ""
echo "Configurations:"
for cfg in "${CONFIGS[@]}"; do
    MODEL=$(echo "$cfg" | cut -d: -f1)
    DATASET=$(echo "$cfg" | cut -d: -f2)
    SIZE=$(echo "$cfg" | cut -d: -f4)
    echo "  - $MODEL on $DATASET ($SIZE)"
done
echo ""
echo "Log directory:  $LOGDIR"
echo "Started:        $(date)"
echo "================================================================================"

# =============================================================================
# PHASE A: Generate multi-sample responses
# =============================================================================

echo ""
echo "========================================================================"
echo "  PHASE A: Multi-sample generation (temp=$TEMPERATURE, n=$NUM_GENERATIONS)"
echo "========================================================================"

for cfg in "${CONFIGS[@]}"; do
    MODEL=$(echo "$cfg" | cut -d: -f1)
    DATASET=$(echo "$cfg" | cut -d: -f2)
    FAMILY=$(echo "$cfg" | cut -d: -f3)
    SIZE=$(echo "$cfg" | cut -d: -f4)

    echo ""
    echo "------------------------------------------------------------------------"
    echo "  Generating: $MODEL on $DATASET ($SIZE)"
    echo "  Started:    $(date)"
    echo "------------------------------------------------------------------------"

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
        --experiment_lot "se_replication_${FAMILY}_${DATASET}_${TIMESTAMP}" \
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
# PHASE B: Run Phase 5 with Semantic Entropy + SAR + all baselines
# =============================================================================

echo ""
echo "========================================================================"
echo "  PHASE B: Computing Semantic Entropy + SAR + baselines"
echo "========================================================================"
echo ""
echo "  DeBERTa-v2-xlarge-mnli (~3.6 GB) for SE clustering"
echo "  cross-encoder/stsb-roberta-large (~1.4 GB) for SAR"
echo ""

for cfg in "${CONFIGS[@]}"; do
    MODEL=$(echo "$cfg" | cut -d: -f1)
    DATASET=$(echo "$cfg" | cut -d: -f2)
    FAMILY=$(echo "$cfg" | cut -d: -f3)
    SIZE=$(echo "$cfg" | cut -d: -f4)

    echo "------------------------------------------------------------------------"
    echo "  Looking for pickle: $SIZE / $DATASET / $MODEL"

    PICKLE_PATH=$(find "src/boldis/uncertainty/wandb/${SIZE}/${DATASET}/${MODEL}" \
        -name "validation_generations.pkl" -type f -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "$PICKLE_PATH" ]; then
        echo "  WARNING: No pickle found for $MODEL on $DATASET — skipping Phase 5"
        echo "  Searched: src/boldis/uncertainty/wandb/${SIZE}/${DATASET}/${MODEL}/"
        continue
    fi

    echo "  Found: $PICKLE_PATH"

    OUTPUT_DIR="${LOGDIR}/phase5/${SIZE}/${DATASET}/${MODEL}"
    mkdir -p "$OUTPUT_DIR"

    echo "  Running Phase 5 with SE + SAR enabled..."
    echo "  Started: $(date)"

    python src/analysis/phase5_comparative_analysis.py \
        --pickle-path "$PICKLE_PATH" \
        --model-name "$MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --num-samples-se "$NUM_GENERATIONS" \
        --num-samples-sar "$NUM_GENERATIONS" \
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
# PHASE C: Summary comparison
# =============================================================================

echo ""
echo "========================================================================"
echo "  PHASE C: Summary"
echo "========================================================================"
echo ""

echo "SE Replication Results vs Existing Pipeline (G-NLL baseline):"
echo ""
printf "%-15s %-12s %-10s %-10s %-10s %-10s\n" "Model" "Dataset" "SE AUROC" "G-NLL" "Avg NLL" "SAR"
printf "%-15s %-12s %-10s %-10s %-10s %-10s\n" "---------------" "------------" "----------" "----------" "----------" "----------"

for cfg in "${CONFIGS[@]}"; do
    MODEL=$(echo "$cfg" | cut -d: -f1)
    DATASET=$(echo "$cfg" | cut -d: -f2)
    SIZE=$(echo "$cfg" | cut -d: -f4)

    RESULTS_JSON="${LOGDIR}/phase5/${SIZE}/${DATASET}/${MODEL}/all_metrics_results.json"
    if [ -f "$RESULTS_JSON" ]; then
        SE=$(python3 -c "import json; d=json.load(open('$RESULTS_JSON')); print(f\"{d['aurocs'].get('semantic_entropy', 'N/A'):.4f}\" if isinstance(d['aurocs'].get('semantic_entropy'), (int,float)) else 'N/A')" 2>/dev/null || echo "N/A")
        GNLL=$(python3 -c "import json; d=json.load(open('$RESULTS_JSON')); print(f\"{d['aurocs'].get('g_nll', 'N/A'):.4f}\" if isinstance(d['aurocs'].get('g_nll'), (int,float)) else 'N/A')" 2>/dev/null || echo "N/A")
        AVGNLL=$(python3 -c "import json; d=json.load(open('$RESULTS_JSON')); print(f\"{d['aurocs'].get('average_neg_log_likelihood', 'N/A'):.4f}\" if isinstance(d['aurocs'].get('average_neg_log_likelihood'), (int,float)) else 'N/A')" 2>/dev/null || echo "N/A")
        SAR=$(python3 -c "import json; d=json.load(open('$RESULTS_JSON')); print(f\"{d['aurocs'].get('sar', 'N/A'):.4f}\" if isinstance(d['aurocs'].get('sar'), (int,float)) else 'N/A')" 2>/dev/null || echo "N/A")
        printf "%-15s %-12s %-10s %-10s %-10s %-10s\n" "$MODEL" "$DATASET" "$SE" "$GNLL" "$AVGNLL" "$SAR"
    else
        printf "%-15s %-12s %-10s\n" "$MODEL" "$DATASET" "(no results)"
    fi
done

echo ""
echo "================================================================================"
echo "  ALL DONE"
echo "================================================================================"
echo ""
echo "  Results directory: $LOGDIR"
echo "  Finished:          $(date)"
echo ""
echo "  Phase 5 outputs per run:"
for cfg in "${CONFIGS[@]}"; do
    MODEL=$(echo "$cfg" | cut -d: -f1)
    DATASET=$(echo "$cfg" | cut -d: -f2)
    SIZE=$(echo "$cfg" | cut -d: -f4)
    echo "    - ${LOGDIR}/phase5/${SIZE}/${DATASET}/${MODEL}/"
done
echo ""
echo "  Each contains: auroc_comparison.csv, roc_curves.png, all_metrics_results.json"
echo ""
echo "  Paper reference (Kuhn et al., ICLR 2023, OPT-30B):"
echo "    TriviaQA SE AUROC ~0.80, CoQA SE AUROC ~0.76"
echo ""
