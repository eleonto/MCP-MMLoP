CUDA_VISIBLE_DEVICES=3 python -u train.py \
    --seed 1 \
    --trainer CoPrompt \
    --backbone "ViT-B/16" \
    --dataset-name "Custom6" \
    --config-file configs/trainers/CoPrompt/vit_b16.yaml \
    --output-dir output/custom6/train1/coprompt_vit_b16 \
    DATASET.NAME Custom6 \
    DATASET.ROOT dataset \
    DATASET.NUM_SHOTS -1 \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.COPROMPT.N_CTX 2 \
    TRAINER.COPROMPT.CTX_INIT "a photo of a" \
    TRAINER.COPROMPT.PROMPT_DEPTH 9 \
    TRAINER.COPROMPT.PREC fp16 \
    TRAINER.COPROMPT.CC_LAMBDA 1.0 \
    TRAINER.COPROMPT.IMG_ADAPTER_MOMENTUM 0.1 \
    TRAINER.COPROMPT.TXT_ADAPTER_MOMENTUM 0.2 \
    TRAINER.COPROMPT.STD_TEMPLATE "a photo of a {}." \
    TRAINER.COPROMPT.ALT_TEMPLATE "a picture of a {}." \
    OPTIM.LR 0.0035 \
    OPTIM.MAX_EPOCH 70 \
    DATALOADER.TRAIN_X.BATCH_SIZE 4