CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --seed 1 \
    --trainer CompLoRA \
    --backbone "ViT-B/16" \
    --dataset-name "Custom6" \
    --config-file configs/trainers/CompLoRA/vit_b16.yaml \
    --output-dir output/custom6/train1/complora_vit_b16 \
    DATASET.NAME Custom6 \
    DATASET.ROOT dataset \
    DATASET.NUM_SHOTS -1 \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.COMP_LORA.RANK 2 \
    TRAINER.COMP_LORA.ALPHA 1 \
    TRAINER.COMP_LORA.N_PRINCIPAL 16 \
    TRAINER.COMP_LORA.PROJECT_OUTPUT True \
    TRAINER.COMP_LORA.PREC fp32 \
    TRAINER.COMP_LORA.TEXT_TEMPLATE "a photo of a {}." \
    OPTIM.NAME adamw \
    OPTIM.LR 0.0002 \
    OPTIM.MAX_EPOCH 70 \
    DATALOADER.TRAIN_X.BATCH_SIZE 4