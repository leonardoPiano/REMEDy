#!/bin/bash
# Master training script — runs ALL training experiments in sequence:
#   1. LLM fine-tuning       (train_ft.sh)
#   2. Span-based training   (train_span.sh)
#   3. Ablation training     (train_ablation.sh)
#
# Run from the project root: bash scripts/train.sh
# To skip a stage, comment out the corresponding line below.

set -e

echo "########################################"
echo "# 1/3  LLM Fine-Tuning"
echo "########################################"
bash scripts/train_ft.sh

echo ""
echo "########################################"
echo "# 2/3  Span-Based Training (GLiNER)"
echo "########################################"
bash scripts/train_span.sh

echo ""
echo "########################################"
echo "# 3/3  Ablation Training"
echo "########################################"
bash scripts/train_ablation.sh

echo ""
echo "All training experiments complete."
