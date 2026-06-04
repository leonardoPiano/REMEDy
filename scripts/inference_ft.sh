#!/bin/bash
# FT inference for all LLM models across all datasets.
#
# Strategy:
#   Fold 0  →  standard + HateXplain + CoSafe  (--dataset all)
#   Fold 1,2 → standard datasets only           (--dataset standard)
#   HateXplain and CoSafe are fold-independent; running them once on fold-0 is enough.
#
# Run from the project root: bash scripts/inference_ft.sh [checkpoint]

set -e

MODELS=("llama3.2-3" "llama3.1-8" "mistral" "gemma2")
CHECKPOINT="${1:-checkpoint-640}"

for model in "${MODELS[@]}"; do
    echo "=== [FT] Inference: $model  (checkpoint=$CHECKPOINT) ==="

    echo "  Fold 0 — all datasets (standard + HateXplain + CoSafe)"
    python inference.py \
        --model "$model" --fold 0 --mode FT \
        --checkpoint "$CHECKPOINT" --dataset all

    for fold in 1 2; do
        echo "  Fold $fold — standard datasets"
        python inference.py \
            --model "$model" --fold "$fold" --mode FT \
            --checkpoint "$CHECKPOINT" --dataset standard
    done
done

echo "FT inference complete."
