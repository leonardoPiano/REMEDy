#!/bin/bash
# Span-based inference for all GLiNER models across all folds and datasets.
# Covers: standard (Remedy + OOD, 3 folds) and HateXplain.
# Run from the project root: bash scripts/inference_span.sh

set -e

echo "=== [SPAN] Inference: all models, all datasets ==="
python span-based/inference.py --model all --dataset all

echo "Span-based inference complete."
