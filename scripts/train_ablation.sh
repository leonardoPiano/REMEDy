#!/bin/bash
# Ablation training: no_rationale and no_harmless variants.
# Models: llama3.2-3, mistral, gemma2 — fold 0 only (as in the paper).
# Run from the project root: bash scripts/train_ablation.sh [--epochs N]

set -e

MODELS=("llama3.2-3" "mistral" "gemma2")
ABLATIONS=("no_rationale" "no_harmless")
FOLD=0
EPOCHS=2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs) EPOCHS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

for ablation in "${ABLATIONS[@]}"; do
    echo "=== [ABLATION] Variant: $ablation ==="
    for model in "${MODELS[@]}"; do
        echo "  Model: $model  Fold: $FOLD"
        python ablation/train.py \
            --model "$model" \
            --fold  "$FOLD"  \
            --ablation "$ablation" \
            --epochs "$EPOCHS"
        echo "  $model done."
    done
done

echo "Ablation training complete."
