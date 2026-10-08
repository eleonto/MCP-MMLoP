CUDA_VISIBLE_DEVICES=1 python -u train.py \
    --seed 1 \
    --trainer IVLP \
    --backbone "ViT-B/16" \
    --dataset-name "Custom6" \
    --config-file configs/trainers/IVLP/vit_b16_c2_ep20_batch4_4+4ctx.yaml \
    --output-dir output/custom6/train1/ivlp_vit_b16 \
    DATASET.NAME Custom6 \
    DATASET.ROOT dataset \
    DATASET.NUM_SHOTS -1 \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.IVLP.N_CTX_TEXT 4 \
    TRAINER.IVLP.N_CTX_VISION 4 \
    TRAINER.IVLP.PROMPT_DEPTH_TEXT 9 \
    TRAINER.IVLP.PROMPT_DEPTH_VISION 9 \
    TRAINER.IVLP.CTX_INIT "a photo of a" \
    TRAINER.IVLP.PREC fp16 \
    OPTIM.MAX_EPOCH 70 \
    DATALOADER.TRAIN_X.BATCH_SIZE 4