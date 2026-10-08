CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --seed 1 \
    --trainer CLIP_LoRA \
    --backbone "ViT-B/16" \
    --dataset-name "Custom6" \
    --config-file configs/trainers/CLIP_LoRA/vit_b16.yaml \
    --output-dir output/custom6/train1/clip_lora_vit_b16 \
    DATASET.NAME Custom6 \
    DATASET.ROOT dataset \
    DATASET.NUM_SHOTS -1 \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.CLIP_LORA.RANK 2 \
    TRAINER.CLIP_LORA.ALPHA 1 \
    TRAINER.CLIP_LORA.PREC fp32 \
    TRAINER.CLIP_LORA.TEXT_TEMPLATE "a photo of a {}." \
    OPTIM.NAME adamw \
    OPTIM.LR 0.0002 \
    OPTIM.MAX_EPOCH 70 \
    DATALOADER.TRAIN_X.BATCH_SIZE 4