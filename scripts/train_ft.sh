#!/bin/bash
# Fine-tune all LLM models (llama3.2-3, llama3.1-8, mistral, gemma2) for 3 folds.
# Run from the project root: bash scripts/train_ft.sh [--epochs N]

set -e

MODELS=("llama3.2-3" "llama3.1-8" "mistral" "gemma2")
FOLDS=(0 1 2)
EPOCHS=2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs) EPOCHS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

for model in "${MODELS[@]}"; do
    echo "=== [FT] Training: $model ==="
    for fold in "${FOLDS[@]}"; do
        echo "  Fold $fold"
        python train.py --model "$model" --fold "$fold" --epochs "$EPOCHS"
        echo "  Fold $fold done."
    done
done

echo "FT training complete."
