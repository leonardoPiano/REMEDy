#!/bin/bash
# Run all guardian (baseline) inference experiments across all 3 folds.
#
# Guardians:
#   llamaguard   Meta LlamaGuard-3-8B            (vLLM)
#   wildguard    AllenAI WildGuard               (HF CausalLM — excludes WildGuard benchmark)
#   shieldgemma  Google ShieldGemma-9B           (HF CausalLM)
#   duoguard     DuoGuard 0.5B / 1.5B / LLaMA   (HF Classifier, 3 variants)
#
# Output: output/fold-{0,1,2}/parsed/GUARD/{guardian_name}/
#
# Run from the project root: bash scripts/guardians.sh [cuda_device]
# Default CUDA device: 0

set -e

CUDA="${1:-0}"
FOLDS=(0 1 2)

# ---------------------------------------------------------------------------
echo "########################################"
echo "# 1/4  LlamaGuard"
echo "########################################"
for fold in "${FOLDS[@]}"; do
    echo "  Fold $fold"
    python -m guardians.llamaguard --fold "$fold" --cuda "$CUDA"
done

# ---------------------------------------------------------------------------
echo ""
echo "########################################"
echo "# 2/4  WildGuard"
echo "########################################"
for fold in "${FOLDS[@]}"; do
    echo "  Fold $fold"
    python -m guardians.wildguard --fold "$fold" --cuda "$CUDA"
done

# ---------------------------------------------------------------------------
echo ""
echo "########################################"
echo "# 3/4  ShieldGemma"
echo "########################################"
for fold in "${FOLDS[@]}"; do
    echo "  Fold $fold"
    python -m guardians.shieldgemma --fold "$fold" --cuda "$CUDA"
done

# ---------------------------------------------------------------------------
echo ""
echo "########################################"
echo "# 4/4  DuoGuard  (all 3 variants)"
echo "########################################"
for fold in "${FOLDS[@]}"; do
    echo "  Fold $fold"
    python -m guardians.duoguard --all --fold "$fold" --cuda "$CUDA"
done

echo ""
echo "Guardian inference complete."
