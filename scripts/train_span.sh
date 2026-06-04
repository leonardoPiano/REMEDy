#!/bin/bash
# Fine-tune GLiNER span-based models (gliner, nuner) for 3 folds.
# Run from the project root: bash scripts/train_span.sh [--epochs N]

set -e

MODELS=("gliner" "nuner")
FOLDS=(0 1 2)
EPOCHS=15

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs) EPOCHS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

for model in "${MODELS[@]}"; do
    echo "=== [SPAN] Training: $model ==="
    for fold in "${FOLDS[@]}"; do
        echo "  Fold $fold"
        python span-based/train.py --model "$model" --fold "$fold" --epochs "$EPOCHS"
        echo "  Fold $fold done."
    done
done

echo "Span-based training complete."
