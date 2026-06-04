#!/bin/bash
# Master inference script — runs ALL inference experiments in sequence:
#   1. FT inference          (inference_ft.sh)
#   2. Zero-shot inference   (inference_zero.sh)
#   3. Span-based inference  (inference_span.sh)
#   4. Ablation inference    (inference_ablation.sh)
#
# Run from the project root: bash scripts/inference.sh [checkpoint]
# The optional checkpoint argument (default: checkpoint-640) is forwarded
# to FT and ablation inference scripts.
# To skip a stage, comment out the corresponding line below.

set -e

CHECKPOINT="${1:-checkpoint-640}"

echo "########################################"
echo "# 1/4  FT Inference  (ckpt=$CHECKPOINT)"
echo "########################################"
bash scripts/inference_ft.sh "$CHECKPOINT"

echo ""
echo "########################################"
echo "# 2/4  Zero-Shot Inference"
echo "########################################"
bash scripts/inference_zero.sh

echo ""
echo "########################################"
echo "# 3/4  Span-Based Inference"
echo "########################################"
bash scripts/inference_span.sh

echo ""
echo "########################################"
echo "# 4/4  Ablation Inference  (ckpt=$CHECKPOINT)"
echo "########################################"
bash scripts/inference_ablation.sh "$CHECKPOINT"

echo ""
echo "All inference experiments complete."
