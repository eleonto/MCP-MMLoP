#!/usr/bin/env bash
# =============================================================================
# Reproduce MCP-MMLoP on the PRPD (Custom6) dataset.
#
# Paper protocol (Sec. 4.1): stratified 8:1:1 split, results averaged over
# five random seeds. Override SEEDS / GPU / CLIP_MODEL_PATH via environment:
#
#   GPU=0 \
#   CLIP_MODEL_PATH=/path/to/ViT-B-16.pt \
#   DATA_ROOT=dataset \
#   bash run_mcp_mmlop.sh
#
# If CLIP_MODEL_PATH is unset, the loader automatically searches ./ViT-B-16.pt,
# the project root and ~/.cache/clip, and finally downloads it from OpenAI.
# =============================================================================
set -e

GPU="${GPU:-0}"
SEEDS="${SEEDS:-1 2 3 4 5}"
DATA_ROOT="${DATA_ROOT:-dataset}"
CLIP_WEIGHTS="${CLIP_MODEL_PATH:-ViT-B-16.pt}"

export CUDA_VISIBLE_DEVICES="${GPU}"

for SEED in ${SEEDS}; do
    OUT="output/mcp_mmloplop/seed${SEED}"
    echo "================================================================"
    echo " Training MCP-MMLoP | seed=${SEED} | output=${OUT}"
    echo "================================================================"

    python -u train.py \
        --seed "${SEED}" \
        --trainer MMLoP \
        --backbone "ViT-B/16" \
        --dataset-name "Custom6" \
        --dataset-config-file configs/datasets/custom6.yaml \
        --config-file configs/trainers/MMLoP/mcp_mmlop_vit_b16.yaml \
        --output-dir "${OUT}" \
        DATASET.NAME Custom6 \
        DATASET.ROOT "${DATA_ROOT}" \
        DATASET.NUM_SHOTS -1 \
        DATASET.SUBSAMPLE_CLASSES all \
        TRAINER.MMLOP.CLIP_MODEL_PATH "${CLIP_WEIGHTS}"

    echo " seed=${SEED} finished. Per-seed metrics are in ${OUT}/log.txt"
done

echo "All seeds finished. Aggregate per-seed results with:"
echo "  python Dassl.pytorch/tools/parse_test_res.py output/mcp_mmloplop --test-log"
