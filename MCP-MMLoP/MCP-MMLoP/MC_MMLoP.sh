python -u train.py \
    --seed 1 \
    --trainer MMLoP \
    --backbone "ViT-B/16" \
    --dataset-name "Custom6" \
    --config-file configs/trainers/MMLoP/vit_b16.yaml \
    --output-dir output/custom6/train1/vit_b16 \
    DATASET.NAME Custom6 \
    DATASET.ROOT dataset \
    DATASET.NUM_SHOTS -1 \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.MMLOP.N_CTX_TEXT 4 \
    TRAINER.MMLOP.N_CTX_VISION 4 \
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
    TRAINER.MMLOP.CLIP_MODEL_PATH ViT-B-16.pt