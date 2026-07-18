#!/usr/bin/env bash
# 5-fold cross-validation for all 3 ablation variants on LIVE
# Estimated run time: ~23 hours total
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

run_variant() {
    local variant=$1 fold=$2
    local prefix="variant${variant}_live_fold${fold}"
    echo "==========================================="
    echo "  Variant ${variant} Fold ${fold}   $(date)"
    echo "==========================================="
    python3 "train_${variant,,}.py" \
        --dataset live \
        --data-dir "$DATA_DIR" \
        --fold "$fold" \
        --num-folds 5 \
        --epochs 40 \
        --batch-size 2 \
        --cpu-workers 8 \
        --save-prefix "$prefix" \
        2>&1 | tee "${prefix}.log"
}

for variant in a b c; do
    for fold in 0 1 2 3 4; do
        run_variant "$variant" "$fold"
    done
done

echo "==========================================="
echo "  Cross-validation complete!   $(date)"
echo "==========================================="
