CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --seed 1 \
    --trainer CoCoOp \
    --backbone "ViT-B/16" \
    --dataset-name "Custom6" \
    --config-file configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml \
    --output-dir output/custom6/train1/cocoop_vit_b16 \
    DATASET.NAME Custom6 \
    DATASET.ROOT dataset \
    DATASET.NUM_SHOTS -1 \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.COCOOP.N_CTX 4 \
    TRAINER.COCOOP.CTX_INIT "a photo of a" \
    TRAINER.COCOOP.PREC fp16 \
    OPTIM.MAX_EPOCH 70 \
    OPTIM.LR 0.002 \
    DATALOADER.TRAIN_X.BATCH_SIZE 4