CUDA_VISIBLE_DEVICES=3 python -u train.py \
    --seed 1 \
    --trainer BlockLoRA \
    --backbone "ViT-B/16" \
    --dataset-name "Custom6" \
    --config-file configs/trainers/BlockLoRA/vit_b16.yaml \
    --output-dir output/custom6/train1/blocklora_vit_b16 \
    DATASET.NAME Custom6 \
    DATASET.ROOT dataset \
    DATASET.NUM_SHOTS -1 \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.BLOCK_LORA.RANK 2 \
    TRAINER.BLOCK_LORA.ALPHA 1 \
    TRAINER.BLOCK_LORA.N_BLOCKS 2 \
    TRAINER.BLOCK_LORA.PREC fp32 \
    TRAINER.BLOCK_LORA.TEXT_TEMPLATE "a photo of a {}." \
    OPTIM.NAME adamw \
    OPTIM.LR 0.0002 \
    OPTIM.MAX_EPOCH 70 \
    DATALOADER.TRAIN_X.BATCH_SIZE 4