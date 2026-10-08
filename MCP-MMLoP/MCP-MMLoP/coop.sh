CUDA_VISIBLE_DEVICES=1 python -u train.py \
    --seed 1 \
    --trainer CoOp \
    --backbone "ViT-B/16" \
    --dataset-name "Custom6" \
    --config-file configs/trainers/CoOp/vit_b16.yaml \
    --output-dir output/custom6/train1/coop_vit_b16 \
    DATASET.NAME Custom6 \
    DATASET.ROOT dataset \
    DATASET.NUM_SHOTS -1 \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.COOP.N_CTX 4 \
    TRAINER.COOP.CTX_INIT "a photo of a" \
    TRAINER.COOP.PREC fp16 \
    OPTIM.MAX_EPOCH 70 \
    OPTIM.LR 0.002 \
    DATALOADER.TRAIN_X.BATCH_SIZE 4