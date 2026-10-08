CUDA_VISIBLE_DEVICES=1 python -u train.py \
    --seed 1 \
    --trainer PromptSRC \
    --backbone "ViT-B/16" \
    --dataset-name "Custom6" \
    --config-file configs/trainers/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx.yaml \
    --output-dir output/custom6/train1/promptsrc_vit_b16 \
    DATASET.NAME Custom6 \
    DATASET.ROOT dataset \
    DATASET.NUM_SHOTS -1 \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.PROMPTSRC.N_CTX_TEXT 4 \
    TRAINER.PROMPTSRC.N_CTX_VISION 4 \
    TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT 9 \
    TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION 9 \
    TRAINER.PROMPTSRC.CTX_INIT "a photo of a" \
    TRAINER.PROMPTSRC.PREC fp16 \
    TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT 25 \
    TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT 10 \
    TRAINER.PROMPTSRC.GPA_MEAN 15 \
    TRAINER.PROMPTSRC.GPA_STD 1 \
    OPTIM.MAX_EPOCH 70 \
    DATALOADER.TRAIN_X.BATCH_SIZE 4