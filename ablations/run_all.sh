#!/usr/bin/env bash
# Run all 3 ablation variants on LIVE Fold 4 (0-indexed)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

echo "==========================================="
echo "  Variant A   $(date)"
echo "==========================================="
python3 train_a.py \
    --dataset live \
    --data-dir "$DATA_DIR" \
    --fold 4 \
    --epochs 40 \
    --batch-size 2 \
    --cpu-workers 8 \
    --save-prefix "variantA_live_fold4" \
    2>&1 | tee "variantA_live_fold4.log"

echo "==========================================="
echo "  Variant B   $(date)"
echo "==========================================="
python3 train_b.py \
    --dataset live \
    --data-dir "$DATA_DIR" \
    --fold 4 \
    --epochs 40 \
    --batch-size 2 \
    --cpu-workers 8 \
    --save-prefix "variantB_live_fold4" \
    2>&1 | tee "variantB_live_fold4.log"

echo "==========================================="
echo "  Variant C   $(date)"
echo "==========================================="
python3 train_c.py \
    --dataset live \
    --data-dir "$DATA_DIR" \
    --fold 4 \
    --epochs 40 \
    --batch-size 2 \
    --cpu-workers 8 \
    --save-prefix "variantC_live_fold4" \
    2>&1 | tee "variantC_live_fold4.log"

echo "==========================================="
echo "  Face-Only (grid pos emb + face-ID)   $(date)"
echo "==========================================="
python3 train_face_only.py \
    --dataset live \
    --data-dir "$DATA_DIR" \
    --fold 4 \
    --epochs 40 \
    --batch-size 2 \
    --cpu-workers 8 \
    --save-prefix "faceonly_live_fold4" \
    2>&1 | tee "faceonly_live_fold4.log"

echo "==========================================="
echo "  RoPE-Only (3D RoPE, no face emb)   $(date)"
echo "==========================================="
python3 train_rope_only.py \
    --dataset live \
    --data-dir "$DATA_DIR" \
    --fold 4 \
    --epochs 40 \
    --batch-size 2 \
    --cpu-workers 8 \
    --save-prefix "ropeonly_live_fold4" \
    2>&1 | tee "ropeonly_live_fold4.log"

echo "==========================================="
echo "  All done   $(date)"
echo "==========================================="
