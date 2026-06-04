#!/bin/bash
# Zero-shot inference for all LLM models on all datasets (fold-0 only).
# Run from the project root: bash scripts/inference_zero.sh

set -e

MODELS=("llama3.2-3" "llama3.1-8" "mistral" "gemma2")

for model in "${MODELS[@]}"; do
    echo "=== [ZERO] Inference: $model ==="
    python inference.py \
        --model "$model" --fold 0 --mode ZERO --dataset all
done

echo "Zero-shot inference complete."
