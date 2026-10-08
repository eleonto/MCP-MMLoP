CUDA_VISIBLE_DEVICES=3 python -u train.py \
    --seed 1 \
    --trainer MMA \
    --backbone "ViT-B/16" \
    --dataset-name "Custom6" \
    --config-file configs/trainers/MMA/vit_b16.yaml \
    --output-dir output/custom6/train1/mma_vit_b16 \
    DATASET.NAME Custom6 \
    DATASET.ROOT dataset \
    DATASET.NUM_SHOTS -1 \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.MMA.ADAPTER_DIM 32 \
    TRAINER.MMA.ADAPTER_START 5 \
    TRAINER.MMA.ADAPTER_END 12 \
    TRAINER.MMA.ADAPTER_SCALE 0.001 \
    TRAINER.MMA.PREC fp32 \
    TRAINER.MMA.TEXT_TEMPLATE "a photo of a {}." \
    OPTIM.NAME adamw \
    OPTIM.LR 0.0015 \
    OPTIM.MAX_EPOCH 70 \
    DATALOADER.TRAIN_X.BATCH_SIZE 4