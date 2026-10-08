#!/bin/sh
# MMLop.sh - MMLoP 小样本比例实验
# 顺序运行 4 个不同训练集比例(0.2/0.4/0.6/0.8)的实验
# 每个实验有独立的 output-dir,互不干扰

export CUDA_VISIBLE_DEVICES=3

RATIOS="32"

for ratio in $RATIOS; do
    echo "================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [MMLoP] 开始实验: ratio=$ratio"
    echo "  DATASET.ROOT = /mnt/sdb/data/yjl/MMLoP/dataset/${ratio}"
    echo "  output-dir  = output/custom6/train1/mmlop_ratio${ratio}"
    echo "================================================"

    # 删除该比例目录下的预处理缓存(避免读到旧路径)
    rm -f "/mnt/sdb/data/yjl/MMLoP/dataset/${ratio}/custom6_preprocessed.pkl"

    python -u train.py \
        --seed 1 \
        --trainer MMLoP \
        --backbone "ViT-B/16" \
        --dataset-name "Custom6" \
        --config-file configs/trainers/MMLoP/vit_b16.yaml \
        --output-dir "output/custom6/train1/mmlop_ratio${ratio}" \
        DATASET.NAME Custom6 \
        DATASET.ROOT "/mnt/sdb/data/yjl/MMLoP/dataset/${ratio}" \
        DATASET.NUM_SHOTS -1 \
        DATASET.SUBSAMPLE_CLASSES all \
        TRAINER.MMLOP.N_CTX_TEXT 4 \
        TRAINER.MMLOP.N_CTX_VISION 4 \
        TRAINER.MMLOP.LORA_RANK 1 \
        TRAINER.MMLOP.PROMPT_DEPTH_TEXT 9 \
        TRAINER.MMLOP.PROMPT_DEPTH_VISION 9 \
        TRAINER.MMLOP.USE_SCL_LOSS True \
        TRAINER.MMLOP.SCL_LOGIT_LOSS js \
        TRAINER.MMLOP.TEXT_LOSS_WEIGHT 25.0 \
        TRAINER.MMLOP.IMAGE_LOSS_WEIGHT 10.0 \
        TRAINER.MMLOP.USE_CTX_INIT_EMBED True \
        TRAINER.MMLOP.CTX_INIT_PHRASE "a photo of a" \
        OPTIM.MAX_EPOCH 70 \
        DATALOADER.TRAIN_X.BATCH_SIZE 4 \
        TRAINER.MMLOP.CLIP_MODEL_PATH /mnt/sdb/data/yjl/MMLoP/ViT-B-16.pt

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [MMLoP] 实验 ratio=$ratio 完成"
    echo ""
done

echo "================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [MMLoP] 全部 4 个实验完成!"
echo "================================================"
