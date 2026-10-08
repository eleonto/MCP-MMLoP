CUDA_VISIBLE_DEVICES=1 python -u train.py \
    --seed 1 \
    --trainer MaPLe \
    --backbone "ViT-B/16" \
    --dataset-name "Custom6" \
    --config-file configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml \
    --output-dir output/custom6/train1/maple_vit_b16 \
    DATASET.NAME Custom6 \
    DATASET.ROOT dataset \
    DATASET.NUM_SHOTS -1 \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.MAPLE.N_CTX 4 \
    TRAINER.MAPLE.PROMPT_DEPTH 9 \
    TRAINER.MAPLE.CTX_INIT "a photo of a" \
    TRAINER.MAPLE.PREC fp16 \
    OPTIM.MAX_EPOCH 100 \
    OPTIM.LR 0.0035 \
    DATALOADER.TRAIN_X.BATCH_SIZE 4