import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom dataset
# 仅导入本仓库实际使用的 Custom6（PRPD 六类数据集）。
# 导入即注册到 Dassl 的 DATASET_REGISTRY，之后可用 DATASET.NAME = "Custom6" 调用。
import datasets.custom6

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


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

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

    # 防御：Dassl 配置系统有时会把 NAME 覆写成配置文件路径
    _bn = cfg.MODEL.BACKBONE.NAME
    if not _bn or ('/' in _bn) or ('\\' in _bn) or _bn.endswith('.yaml') or _bn.endswith('.yml'):
        print(f"[WARN] cfg.MODEL.BACKBONE.NAME 无效 ('{_bn}')，回退到 ViT-B/16")
        cfg.MODEL.BACKBONE.NAME = "ViT-B/16"

    if args.dataset_name:
        cfg.DATASET.NAME = args.dataset_name

    # 防御：DATASET.NAME 为空或被覆写成路径时，兜底到 Custom6
    _dn = cfg.DATASET.NAME
    if not _dn or ('/' in _dn) or ('\\' in _dn) or _dn.endswith('.yaml') or _dn.endswith('.yml'):
        print(f"[WARN] cfg.DATASET.NAME 无效 ('{_dn}')，回退到 Custom6")
        cfg.DATASET.NAME = "Custom6"

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    # 默认输出每个类别的测试结果（accuracy per class）
    cfg.TEST.PER_CLASS_RESULT = True

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

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
    cfg.TRAINER.MMLOP.SINGLE_TEMPLATE = False          # use one template instead of all ImageNet templates
    cfg.TRAINER.MMLOP.TEXT_TEMPLATE = "a photo of a {}."  # template used when SINGLE_TEMPLATE=True
    cfg.TRAINER.MMLOP.CLIP_MODEL_PATH = ""             # 本地 CLIP 权重路径；为空则用 ~/.cache/clip（找不到会联网下载）
    # —— MCP：多中心原型（Multi-Center Prototype），论文 Sec. 3.4-3.7 ——
    # 论文最终配置见 configs/trainers/MMLoP/mcp_mmlop_vit_b16.yaml；
    # 此处的默认值是 MMLoP 基线默认（关闭 MCP 即退化为纯 MMLoP）。
    cfg.TRAINER.MMLOP.USE_MCP = False                  # True = MCP-MMLoP；False = MMLoP 基线
    cfg.TRAINER.MMLOP.NUM_CENTERS = 3                  # K：每类子原型数（论文 K=3）
    cfg.TRAINER.MMLOP.PROTO_TEMP = 0.07                # τ：原型路径温度（EMA 熵权重用；分类温度为 CLIP logit_scale=1/0.07）
    cfg.TRAINER.MMLOP.EMA_MOMENTUM = 0.01              # 代码语义为“更新系数 alpha”，alpha = 1-η；论文 η=0.99 对应 0.01
    cfg.TRAINER.MMLOP.PROTO_FUSION_WEIGHT = 1.0        # β：原型路径融合权重 ℓ_c = s_c + β·g_c（论文 β=0.5）
    cfg.TRAINER.MMLOP.INTRA_LOSS_WEIGHT = 0.1          # λ_intra：类内紧凑权重（论文 5.0）
    cfg.TRAINER.MMLOP.INTER_LOSS_WEIGHT = 0.1          # λ_inter：类间分离权重（论文 5.0）
    cfg.TRAINER.MMLOP.INTER_MARGIN = 0.2               # m：类间分离合页间隔（论文 0.2）

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9  # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for PromptSRC
    cfg.TRAINER.PROMPTSRC = CN()
    cfg.TRAINER.PROMPTSRC.N_CTX_VISION = 4  # number of context vectors at the vision branch
    cfg.TRAINER.PROMPTSRC.N_CTX_TEXT = 4  # number of context vectors at the language branch
    cfg.TRAINER.PROMPTSRC.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROMPTSRC.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT = 25
    cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT = 10
    cfg.TRAINER.PROMPTSRC.GPA_MEAN = 15
    cfg.TRAINER.PROMPTSRC.GPA_STD = 1
    cfg.TRAINER.PROMPTSRC.USE_JS_LOSS = False
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting(J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.CLIP_LORA = CN()
    cfg.TRAINER.CLIP_LORA.RANK = 2
    cfg.TRAINER.CLIP_LORA.ALPHA = 1
    cfg.TRAINER.CLIP_LORA.PREC = "fp16"
    cfg.TRAINER.CLIP_LORA.TEXT_TEMPLATE = "a photo of a {}."

    cfg.TRAINER.COMP_LORA = CN()
    cfg.TRAINER.COMP_LORA.RANK = 2
    cfg.TRAINER.COMP_LORA.ALPHA = 1
    cfg.TRAINER.COMP_LORA.N_PRINCIPAL = 16      # paper: exclude top 16 principal directions
    cfg.TRAINER.COMP_LORA.PROJECT_OUTPUT = True  # paper projects both input and output sides
    cfg.TRAINER.COMP_LORA.PREC = "fp16"
    cfg.TRAINER.COMP_LORA.TEXT_TEMPLATE = "a photo of a {}."

    cfg.TRAINER.BLOCK_LORA = CN()
    cfg.TRAINER.BLOCK_LORA.RANK = 2
    cfg.TRAINER.BLOCK_LORA.ALPHA = 1
    cfg.TRAINER.BLOCK_LORA.N_BLOCKS = 2         # paper default: Block-LoRA(2,2)
    cfg.TRAINER.BLOCK_LORA.PREC = "fp16"
    cfg.TRAINER.BLOCK_LORA.TEXT_TEMPLATE = "a photo of a {}."

    # Config for MMA (CVPR 2024)
    cfg.TRAINER.MMA = CN()
    cfg.TRAINER.MMA.ADAPTER_DIM = 32          # shared bottleneck dimension
    cfg.TRAINER.MMA.ADAPTER_START = 5         # first layer to adapt (0-indexed)
    cfg.TRAINER.MMA.ADAPTER_END = 12          # one past the last layer to adapt
    cfg.TRAINER.MMA.ADAPTER_SCALE = 0.001     # residual scaling factor
    cfg.TRAINER.MMA.PREC = "fp16"
    cfg.TRAINER.MMA.TEXT_TEMPLATE = "a photo of a {}."

    # Config for CoPrompt (ICLR 2024)
    cfg.TRAINER.COPROMPT = CN()
    cfg.TRAINER.COPROMPT.N_CTX = 2                        # compound prompt tokens (MaPLe style)
    cfg.TRAINER.COPROMPT.CTX_INIT = "a photo of a"
    cfg.TRAINER.COPROMPT.PROMPT_DEPTH = 9                 # compound prompt depth
    cfg.TRAINER.COPROMPT.PREC = "fp16"
    cfg.TRAINER.COPROMPT.CC_LAMBDA = 1.0                  # consistency loss weight
    cfg.TRAINER.COPROMPT.IMG_ADAPTER_MOMENTUM = 0.1       # α for image feature adapter
    cfg.TRAINER.COPROMPT.TXT_ADAPTER_MOMENTUM = 0.2       # α for text feature adapter
    cfg.TRAINER.COPROMPT.STD_TEMPLATE = "a photo of a {}."
    cfg.TRAINER.COPROMPT.ALT_TEMPLATE = "a picture of a {}."


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--dataset-name", type=str, default="", help="name of dataset (e.g. Custom6)")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
