"""
评估脚本：加载已训练的 MMLoP 权重，在测试集上推理，
把 accuracy / macro_f1 / perclass_accuracy 写入 CSV 文件。

用法示例（与 train.py 的 --eval-only 等价，但额外输出 CSV）：
    CUDA_VISIBLE_DEVICES=2 python -u eval_to_csv.py \
        --seed 1 \
        --trainer MMLoP \
        --backbone "ViT-B/16" \
        --dataset-name "Custom6" \
        --config-file configs/trainers/MMLoP/vit_b16.yaml \
        --output-dir output/custom6/train2/mcm_lop_intra5.0_inter5.0 \
        --model-dir output/custom6/train2/mcm_lop_intra5.0_inter5.0 \
        --load-epoch 70 \
        --csv-path output/custom6/train2/mcm_lop_intra5.0_inter5.0/test_results.csv \
        DATASET.NAME Custom6 \
        DATASET.ROOT dataset \
        DATASET.NUM_SHOTS -1 \
        DATASET.SUBSAMPLE_CLASSES all \
        TRAINER.MMLOP.N_CTX_TEXT 4 \
        TRAINER.MMLOP.N_CTX_VISION 4 \
        TRAINER.MMLOP.PROMPT_DEPTH_TEXT 9 \
        TRAINER.MMLOP.PROMPT_DEPTH_VISION 9 \
        TRAINER.MMLOP.LORA_RANK 1 \
        TRAINER.MMLOP.USE_SCL_LOSS True \
        TRAINER.MMLOP.SCL_LOGIT_LOSS js \
        TRAINER.MMLOP.USE_CTX_INIT_EMBED True \
        TRAINER.MMLOP.CTX_INIT_PHRASE "a photo of a" \
        TRAINER.MMLOP.USE_MCP True \
        TRAINER.MMLOP.NUM_CENTERS 1 \
        TRAINER.MMLOP.PROTO_TEMP 0.07 \
        TRAINER.MMLOP.EMA_MOMENTUM 0.01 \
        TRAINER.MMLOP.PROTO_FUSION_WEIGHT 1.0 \
        TRAINER.MMLOP.TEXT_LOSS_WEIGHT 25.0 \
        TRAINER.MMLOP.IMAGE_LOSS_WEIGHT 10.0 \
        TRAINER.MMLOP.INTRA_LOSS_WEIGHT 5.0 \
        TRAINER.MMLOP.INTER_LOSS_WEIGHT 5.0 \
        TRAINER.MMLOP.INTER_MARGIN 0.2 \
        TRAINER.MMLOP.CLIP_MODEL_PATH /mnt/sdb/data/yjl/MMLoP/ViT-B-16.pt
"""

import argparse
import csv
import os.path as osp
from collections import OrderedDict

import torch
from tqdm import tqdm

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom dataset (importing registers "Custom6" with Dassl)
import datasets.custom6

# custom trainers
import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.maple
import trainers.independentVL
import trainers.promptsrc
import trainers.mmlop
import trainers.clip_lora
import trainers.comp_lora
import trainers.block_lora
import trainers.mma
import trainers.coprompt


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.seed:
        cfg.SEED = args.seed
    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains
    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    _bn = cfg.MODEL.BACKBONE.NAME
    if not _bn or ('/' in _bn) or ('\\' in _bn) or _bn.endswith('.yaml') or _bn.endswith('.yml'):
        print(f"[WARN] cfg.MODEL.BACKBONE.NAME 无效 ('{_bn}')，回退到 ViT-B/16")
        cfg.MODEL.BACKBONE.NAME = "ViT-B/16"
    if args.dataset_name:
        cfg.DATASET.NAME = args.dataset_name
    _dn = cfg.DATASET.NAME
    if not _dn or ('/' in _dn) or ('\\' in _dn) or _dn.endswith('.yaml') or _dn.endswith('.yml'):
        print(f"[WARN] cfg.DATASET.NAME 无效 ('{_dn}')，回退到 Custom6")
        cfg.DATASET.NAME = "Custom6"
    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    from yacs.config import CfgNode as CN

    cfg.TEST.PER_CLASS_RESULT = True

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CTX_INIT = ""
    cfg.TRAINER.COOP.PREC = "fp16"
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.MMLOP = CN()
    cfg.TRAINER.MMLOP.N_CTX_TEXT = 4
    cfg.TRAINER.MMLOP.N_CTX_VISION = 4
    cfg.TRAINER.MMLOP.PROMPT_DEPTH_TEXT = 9
    cfg.TRAINER.MMLOP.PROMPT_DEPTH_VISION = 9
    cfg.TRAINER.MMLOP.CTX_INIT = "a photo of a"
    cfg.TRAINER.MMLOP.PREC = "fp16"
    cfg.TRAINER.MMLOP.LORA_RANK = 4
    cfg.TRAINER.MMLOP.LORA_LAMBDA = 1.0
    cfg.TRAINER.MMLOP.SHARED_U = True
    cfg.TRAINER.MMLOP.LORA_INIT = "lora"
    cfg.TRAINER.MMLOP.SVD_INIT_TEXT = "a photo of a"
    cfg.TRAINER.MMLOP.LORA_BETA = 1.0
    cfg.TRAINER.MMLOP.USE_SCL_LOSS = False
    cfg.TRAINER.MMLOP.TEXT_LOSS_WEIGHT = 25.0
    cfg.TRAINER.MMLOP.IMAGE_LOSS_WEIGHT = 10.0
    cfg.TRAINER.MMLOP.USE_GPA = False
    cfg.TRAINER.MMLOP.GPA_MEAN = 15
    cfg.TRAINER.MMLOP.GPA_STD = 1
    cfg.TRAINER.MMLOP.NUM_ENSEMBLE = 1
    cfg.TRAINER.MMLOP.ENSEMBLE_LOSS_TYPE = "avg_loss"
    cfg.TRAINER.MMLOP.NORMALIZE_UV = True
    cfg.TRAINER.MMLOP.INIT_SCALE = 0.1
    cfg.TRAINER.MMLOP.INIT_STD = 0.05
    cfg.TRAINER.MMLOP.SCL_LOGIT_LOSS = "js"
    cfg.TRAINER.MMLOP.PROMPT_START_TEXT = 0
    cfg.TRAINER.MMLOP.PROMPT_START_VISION = 0
    cfg.TRAINER.MMLOP.USE_UDC = False
    cfg.TRAINER.MMLOP.USE_CTX_INIT_EMBED = False
    cfg.TRAINER.MMLOP.CTX_INIT_PHRASE = "a photo of a"
    cfg.TRAINER.MMLOP.SINGLE_TEMPLATE = False
    cfg.TRAINER.MMLOP.TEXT_TEMPLATE = "a photo of a {}."
    cfg.TRAINER.MMLOP.CLIP_MODEL_PATH = ""
    cfg.TRAINER.MMLOP.USE_MCP = False
    cfg.TRAINER.MMLOP.NUM_CENTERS = 3
    cfg.TRAINER.MMLOP.PROTO_TEMP = 0.07
    cfg.TRAINER.MMLOP.EMA_MOMENTUM = 0.01
    cfg.TRAINER.MMLOP.PROTO_FUSION_WEIGHT = 1.0
    cfg.TRAINER.MMLOP.INTRA_LOSS_WEIGHT = 0.1
    cfg.TRAINER.MMLOP.INTER_LOSS_WEIGHT = 0.1
    cfg.TRAINER.MMLOP.INTER_MARGIN = 0.2

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16
    cfg.TRAINER.COCOOP.CTX_INIT = ""
    cfg.TRAINER.COCOOP.PREC = "fp16"

    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"
    cfg.TRAINER.MAPLE.PREC = "fp16"
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

    cfg.TRAINER.PROMPTSRC = CN()
    cfg.TRAINER.PROMPTSRC.N_CTX_VISION = 4
    cfg.TRAINER.PROMPTSRC.N_CTX_TEXT = 4
    cfg.TRAINER.PROMPTSRC.CTX_INIT = "a photo of a"
    cfg.TRAINER.PROMPTSRC.PREC = "fp16"
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION = 9
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT = 9
    cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT = 25
    cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT = 10
    cfg.TRAINER.PROMPTSRC.GPA_MEAN = 15
    cfg.TRAINER.PROMPTSRC.GPA_STD = 1
    cfg.TRAINER.PROMPTSRC.USE_JS_LOSS = False

    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"
    cfg.TRAINER.IVLP.PREC = "fp16"
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9

    cfg.TRAINER.CLIP_LORA = CN()
    cfg.TRAINER.CLIP_LORA.RANK = 2
    cfg.TRAINER.CLIP_LORA.ALPHA = 1
    cfg.TRAINER.CLIP_LORA.PREC = "fp16"
    cfg.TRAINER.CLIP_LORA.TEXT_TEMPLATE = "a photo of a {}."

    cfg.TRAINER.COMP_LORA = CN()
    cfg.TRAINER.COMP_LORA.RANK = 2
    cfg.TRAINER.COMP_LORA.ALPHA = 1
    cfg.TRAINER.COMP_LORA.N_PRINCIPAL = 16
    cfg.TRAINER.COMP_LORA.PROJECT_OUTPUT = True
    cfg.TRAINER.COMP_LORA.PREC = "fp16"
    cfg.TRAINER.COMP_LORA.TEXT_TEMPLATE = "a photo of a {}."

    cfg.TRAINER.BLOCK_LORA = CN()
    cfg.TRAINER.BLOCK_LORA.RANK = 2
    cfg.TRAINER.BLOCK_LORA.ALPHA = 1
    cfg.TRAINER.BLOCK_LORA.N_BLOCKS = 2
    cfg.TRAINER.BLOCK_LORA.PREC = "fp16"
    cfg.TRAINER.BLOCK_LORA.TEXT_TEMPLATE = "a photo of a {}."

    cfg.TRAINER.MMA = CN()
    cfg.TRAINER.MMA.ADAPTER_DIM = 32
    cfg.TRAINER.MMA.ADAPTER_START = 5
    cfg.TRAINER.MMA.ADAPTER_END = 12
    cfg.TRAINER.MMA.ADAPTER_SCALE = 0.001
    cfg.TRAINER.MMA.PREC = "fp16"
    cfg.TRAINER.MMA.TEXT_TEMPLATE = "a photo of a {}."

    cfg.TRAINER.COPROMPT = CN()
    cfg.TRAINER.COPROMPT.N_CTX = 2
    cfg.TRAINER.COPROMPT.CTX_INIT = "a photo of a"
    cfg.TRAINER.COPROMPT.PROMPT_DEPTH = 9
    cfg.TRAINER.COPROMPT.PREC = "fp16"
    cfg.TRAINER.COPROMPT.CC_LAMBDA = 1.0
    cfg.TRAINER.COPROMPT.IMG_ADAPTER_MOMENTUM = 0.1
    cfg.TRAINER.COPROMPT.TXT_ADAPTER_MOMENTUM = 0.2
    cfg.TRAINER.COPROMPT.STD_TEMPLATE = "a photo of a {}."
    cfg.TRAINER.COPROMPT.ALT_TEMPLATE = "a picture of a {}."


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def run_test_with_csv(trainer, csv_path, model_dir, load_epoch):
    """复刻 SimpleTrainer.test() 但捕获全部指标并写 CSV。"""
    trainer.set_model_mode("eval")
    trainer.evaluator.reset()

    data_loader = trainer.test_loader
    split = "test"
    print(f"Evaluate on the *{split}* set")

    for batch in tqdm(data_loader):
        input_data, label = trainer.parse_batch_test(batch)
        output = trainer.model_inference(input_data)
        trainer.evaluator.process(output, label)

    # 返回完整 results dict（包含 accuracy / error_rate / macro_f1 / perclass_accuracy）
    results = trainer.evaluator.evaluate()

    # 整理要写进 CSV 的字段（缺失则填 N/A）
    fieldnames = [
        "model_dir", "load_epoch", "accuracy", "error_rate",
        "macro_f1", "perclass_accuracy",
    ]
    row = OrderedDict()
    row["model_dir"] = model_dir
    row["load_epoch"] = load_epoch
    for k in fieldnames[2:]:
        row[k] = f"{results.get(k, float('nan')):.4f}" if k in results else "N/A"

    # 追加写 CSV（如已存在则只写表头一次）
    file_exists = osp.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n[CSV] 结果已写入: {csv_path}")
    print(f"  accuracy: {row['accuracy']}")
    print(f"  macro_f1: {row['macro_f1']}")
    if row["perclass_accuracy"] != "N/A":
        print(f"  perclass_accuracy: {row['perclass_accuracy']}")

    # ===== 导出每个样本的预测 + 真实标签 + 类名，便于绘制混淆矩阵 =====
    import numpy as np
    from sklearn.metrics import confusion_matrix

    evaluator = trainer.evaluator
    y_true = list(evaluator._y_true)
    y_pred = list(evaluator._y_pred)
    lab2cname = evaluator._lab2cname

    # 标签 -> 类名 映射表（按 label 排序）
    labels_sorted = sorted(lab2cname.keys())
    label_to_name = [(lab, lab2cname[lab]) for lab in labels_sorted]

    # 1) 标签映射表 CSV
    mapping_csv = osp.join(osp.dirname(csv_path), "class_label_mapping.csv")
    with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "class_name"])
        for lab, name in label_to_name:
            w.writerow([lab, name])
    print(f"[CSV] 类名映射已写入: {mapping_csv}")

    # 2) 每个样本的 (y_true, y_pred, true_name, pred_name) CSV
    samples_csv = osp.join(osp.dirname(csv_path), "per_sample_predictions.csv")
    with open(samples_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "y_true", "y_pred", "true_name", "pred_name", "correct"])
        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            w.writerow([
                i, yt, yp,
                lab2cname.get(yt, str(yt)),
                lab2cname.get(yp, str(yp)),
                int(yt == yp),
            ])
    print(f"[CSV] 逐样本预测已写入: {samples_csv} ({len(y_true)} 条)")

    # 3) 混淆矩阵 CSV（行=真实类，列=预测类，归一化按行）
    cmat_csv = osp.join(osp.dirname(csv_path), "confusion_matrix.csv")
    cmat = confusion_matrix(y_true, y_pred, labels=labels_sorted, normalize="true")
    class_names = [lab2cname[lab] for lab in labels_sorted]
    with open(cmat_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names)
        for i, name in enumerate(class_names):
            row_vals = [f"{v*100:.2f}" for v in cmat[i]]
            w.writerow([name] + row_vals)
    print(f"[CSV] 混淆矩阵已写入: {cmat_csv}")

    # 4) 原始计数版混淆矩阵（未归一化）
    cmat_count_csv = osp.join(osp.dirname(csv_path), "confusion_matrix_count.csv")
    cmat_count = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    with open(cmat_count_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names)
        for i, name in enumerate(class_names):
            w.writerow([name] + [int(v) for v in cmat_count[i]])
    print(f"[CSV] 计数版混淆矩阵已写入: {cmat_count_csv}")

    return results


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print("** Config **")
    print(cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    run_test_with_csv(trainer, args.csv_path, args.model_dir, args.load_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--source-domains", type=str, nargs="+")
    parser.add_argument("--target-domains", type=str, nargs="+")
    parser.add_argument("--transforms", type=str, nargs="+")
    parser.add_argument("--config-file", type=str, default="")
    parser.add_argument("--dataset-config-file", type=str, default="")
    parser.add_argument("--trainer", type=str, default="")
    parser.add_argument("--backbone", type=str, default="")
    parser.add_argument("--dataset-name", type=str, default="")
    parser.add_argument("--head", type=str, default="")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory")
    parser.add_argument("--load-epoch", type=int, default=70, help="load model weights at this epoch")
    parser.add_argument("--csv-path", type=str, default="test_results.csv", help="CSV 输出路径")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    main(args)
