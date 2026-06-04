#!/bin/bash
# Ablation inference: no_rationale and no_harmless variants.
# Models: llama3.2-3, mistral, gemma2 — fold 0 only (as in the paper).
# Run from the project root: bash scripts/inference_ablation.sh [checkpoint]

set -e

MODELS=("llama3.2-3" "mistral" "gemma2")
ABLATIONS=("no_rationale" "no_harmless")
CHECKPOINT="${1:-checkpoint-640}"

for ablation in "${ABLATIONS[@]}"; do
    echo "=== [ABLATION] Inference: $ablation  (checkpoint=$CHECKPOINT) ==="
    for model in "${MODELS[@]}"; do
        echo "  Model: $model"
        python ablation/inference.py \
            --model "$model" --fold 0 \
            --ablation "$ablation" \
            --checkpoint "$CHECKPOINT"
        echo "  $model done."
    done
done

echo "Ablation inference complete."
