"""
t-SNE 可视化脚本：特征提取 + t-SNE 降维 + 多子图绘图 + 轮廓系数计算
适用于 MMLoP / MC-MLoP（PyTorch + CLIP backbone）故障分类模型。

⚠️ 关键踩坑提示（两处）：
1. t-SNE 输入必须是 softmax 之前的特征（本脚本用 image_features，L2 归一化后、
   logits 计算之前），绝对不能用 softmax 后的概率向量！
   概率分布会让 t-SNE 的局部几何结构失真，类内/类间距离关系被破坏。
2. 轮廓系数 silhouette_score 必须在【原始高维特征】上计算，绝不能在 t-SNE 降维后的
   2D 数据上算！t-SNE 降维会扭曲距离关系，2D 上的轮廓系数没有数学意义。

用法示例：
    CUDA_VISIBLE_DEVICES=1 python tsne_visualization.py \
        --seed 1 \
        --trainer MMLoP \
        --backbone "ViT-B/16" \
        --dataset-name Custom6 \
        --config-file configs/trainers/MMLoP/vit_b16.yaml \
        --output-dir output/custom6/train1/mcm_lop \
        --model-dir output/custom6/train1/mcm_lop \
        --perplexity 30.0 \
        --save-prefix output/custom6/train1/mcm_lop/tsne \
        DATASET.NAME Custom6 \
        DATASET.ROOT dataset \
        DATASET.NUM_SHOTS -1 \
        DATASET.SUBSAMPLE_CLASSES all \
        TRAINER.MMLOP.USE_MCP True \
        TRAINER.MMLOP.NUM_CENTERS 3 \
        TRAINER.MMLOP.CLIP_MODEL_PATH /mnt/sdb/data/yjl/MMLoP/ViT-B-16.pt
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")  # 无界面保存图片
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# ── 全局字体设置：Times New Roman（SCI 论文常用）──
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"  # 数学符号配套

# 复用 train.py 的 setup_cfg（它内部已注册 MC-MLoP 的所有 cfg 项）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as train_module  # noqa: E402
from dassl.engine import build_trainer  # noqa: E402
from dassl.utils import set_random_seed, setup_logger, collect_env_info  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# 类别映射：标签索引 → (中文, 英文)
# 顺序必须与 datasets/custom6.py 的 FIXED_CLASSNAMES 完全一致
#   0=尖刺, 1=悬浮, 2=沿面, 3=气隙, 4=颗粒, 5=正常
# ──────────────────────────────────────────────────────────────────────
CLASS_INFO = [
    (0, "尖刺", "Tip Discharge"),
    (1, "悬浮", "Suspension Discharge"),
    (2, "沿面", "Surface Discharge"),
    (3, "气隙", "Air Gap Discharge"),
    (4, "颗粒", "Metal Discharge"),
    (5, "正常", "Normal"),
]
ENGLISH_NAMES = [c[2] for c in CLASS_INFO]


def extract_features(model, test_loader, device):
    """遍历 test_loader，批量提取全部测试样本的【softmax 前特征】与真实标签。

    ⚠️ 踩坑1：这里提取的是 image_features（L2 归一化后、logits 计算之前），
       绝对不能用 softmax 后的概率向量！概率分布会破坏 t-SNE 的局部几何结构。

    兼容多种模型（通过 prompt_learner() 返回值长度自动判别）：
      - MMLoP / MC-MLoP   (3元组): prompt_learner() + image_encoder.forward_with_lora
      - CoPrompt / MaPLe  (4元组): prompt_learner() → shared_ctx, deep_vision → image_encoder(img, ctx, deep)
      - CoOp / CoCoOp     (1元组): prompt_learner() → image_encoder(image)
      - CLIP-LoRA / Block-LoRA / Comp-LoRA (无 prompt_learner): clip_model.encode_image
    """
    # 解包 DataParallel（多卡训练时模型被包一层）
    if hasattr(model, "module"):
        model = model.module

    model.eval()
    has_pl = hasattr(model, "prompt_learner")
    # 预先探测 prompt_learner() 的返回结构，决定走哪条特征提取路径
    # 注意：CoCoOp 的 prompt_learner 需要 im_features 参数，不能无参调用，
    # 但 image_features 提取本身不需要 prompt_learner 输出，所以单独处理
    if has_pl:
        # 检查 prompt_learner.forward 是否需要参数（CoCoOp 需要 im_features）
        # 用 __code__.co_varnames 而非 inspect.signature，避免 PyTorch 包装干扰
        fwd_args = model.prompt_learner.forward.__code__.co_varnames
        fwd_nargs = model.prompt_learner.forward.__code__.co_argcount
        needs_input = fwd_nargs > 1  # 除了 self 外还有参数
        if needs_input:
            # CoCoOp：image_encoder(image) 不依赖 prompt
            mode = "coop_cocoop"
            print(f"[特征提取] CoCoOp 路径 (forward 需 {fwd_nargs-1} 个输入, 跳过 prompt_learner)")
        else:
            with torch.no_grad():
                pl_out = model.prompt_learner()
            pl_len = len(pl_out) if isinstance(pl_out, (tuple, list)) else 1
            if pl_len == 3:
                mode = "mmlop"
                print("[特征提取] MMLoP/MC-MLoP 路径 (3元组 → forward_with_lora)")
            elif pl_len == 4:
                mode = "coprompt_maple"
                print("[特征提取] CoPrompt/MaPLe 路径 (4元组 → image_encoder(img, ctx, deep))")
            else:
                mode = "coop_cocoop"
                print("[特征提取] CoOp 路径 (image_encoder(image))")
    else:
        mode = "lora"
        print("[特征提取] CLIP-LoRA / Block-LoRA / Comp-LoRA 路径 (clip_model.encode_image)")

    feats_list, labels_list = [], []
    n_batches = len(test_loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Dassl 的 batch 是 dict：{"img": ..., "label": ...}
            image = batch["img"].to(device)
            label = batch["label"].to(device)

            # —— 拿 image_features（softmax 前）——
            if mode == "mmlop":
                # MMLoP / MC-MLoP：复现 CustomCLIP.forward 前半段
                prompts, vision_lora_deltas, text_lora_deltas = model.prompt_learner()
                if hasattr(model.image_encoder, "forward_with_lora"):
                    image_features = model.image_encoder.forward_with_lora(
                        image.type(model.dtype), vision_lora_deltas
                    )
                else:
                    image_features = model.image_encoder(image.type(model.dtype))
            elif mode == "coprompt_maple":
                # CoPrompt / MaPLe：prompt_learner 返回 4 元组
                # (prompts, shared_ctx, deep_text, deep_vision)
                prompts, shared_ctx, deep_text, deep_vision = model.prompt_learner()
                image_features = model.image_encoder(
                    image.type(model.dtype), shared_ctx, deep_vision
                )
            elif mode == "coop_cocoop":
                # CoOp / CoCoOp：image_encoder 只需 image，不需要 prompt_learner 输出
                image_features = model.image_encoder(image.type(model.dtype))
            else:
                # CLIP-LoRA / Block-LoRA / Comp-LoRA / 标准 CLIP / zero-shot
                # zero-shot 时 model 本身就是 clip_model（有 encode_image）
                if hasattr(model, "encode_image"):
                    image_features = model.encode_image(image.to(dtype=model.dtype))
                else:
                    image_features = model.clip_model.encode_image(image.to(dtype=model.dtype))

            # 与各 forward 里一致：L2 归一化（这是 logits 计算前的特征）
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

            feats_list.append(image_features.float().cpu())
            labels_list.append(label.cpu())

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == n_batches:
                print(f"  已处理 {batch_idx + 1}/{n_batches} 批")

    features = torch.cat(feats_list, dim=0).numpy().astype(np.float32)
    labels = torch.cat(labels_list, dim=0).numpy().astype(np.int64)
    return features, labels


def run_tsne(features, perplexity=30.0, seed=1):
    """t-SNE 降维到 2D。

    Args:
        features: (N, d) 原始高维特征（softmax 前）
        perplexity: t-SNE 困惑度，典型值 5~50；样本数多时取大些。
        seed: 随机种子，保证可复现。

    Returns:
        embedding: (N, 2) 降维后的二维坐标
    """
    n_samples = features.shape[0]
    # perplexity 必须 < n_samples，自动夹紧
    perplexity = min(perplexity, max(5.0, n_samples - 1))
    print(
        f"[t-SNE] 降维：n_samples={n_samples}, n_features={features.shape[1]}, "
        f"perplexity={perplexity}, seed={seed}"
    )
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",           # 用 PCA 初始化，比随机更稳定
        learning_rate="auto",
        metric="cosine",      # 与 L2 归一化特征配套，等价于角度距离
        n_iter=1000,
    )
    embedding = tsne.fit_transform(features)
    print(f"[t-SNE] 完成，输出 shape={embedding.shape}")
    return embedding


# ── 类别配色（红/绿/蓝/黄/紫/灰，与 CLASS_INFO 顺序一一对应）──
CLASS_COLORS = ["red", "green", "blue", "gold", "purple", "gray"]


def plot_tsne_subplots(tsne_2d, labels, save_prefix, dpi=300):
    """绘制多子图，每个子图对应一个类别。

    布局：2 行 × 3 列共 6 个子图，每个子图聚焦一个类别：
      - 该类样本：用该类专属颜色高亮显示
      - 其他类样本：浅灰背景，作为参照
    这样能直观看出每个类别在 t-SNE 空间中的聚集程度与可分性。
    """
    n_classes = len(ENGLISH_NAMES)
    colors = CLASS_COLORS

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for cls_idx, en_name in enumerate(ENGLISH_NAMES):
        ax = axes[cls_idx]
        mask_self = labels == cls_idx
        mask_other = ~mask_self

        # 其他类：浅灰背景
        ax.scatter(
            tsne_2d[mask_other, 0],
            tsne_2d[mask_other, 1],
            c="lightgray",
            s=12,
            alpha=0.35,
            label="Other classes",
            edgecolors="none",
        )
        # 当前类：高亮
        ax.scatter(
            tsne_2d[mask_self, 0],
            tsne_2d[mask_self, 1],
            c=[colors[cls_idx]],
            s=22,
            alpha=0.85,
            label=en_name,
            edgecolors="black",
            linewidths=0.3,
        )
        ax.set_title(en_name, fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=9, framealpha=0.8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("t-SNE of MC-MLoP Features (per class)", fontsize=15, y=1.00)
    plt.tight_layout()

    png_path = f"{save_prefix}.png"
    svg_path = f"{save_prefix}.svg"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"[绘图] 已保存: {png_path} (dpi={dpi})")
    print(f"[绘图] 已保存: {svg_path} (矢量图，适合 SCI 论文)")


def plot_tsne_single(tsne_2d, labels, save_prefix, dpi=300):
    """绘制单图：6 类样本用不同颜色画在同一张图上。

    相比多子图，单图能更直观地看到各类别在 t-SNE 空间中的整体位置分布、
    类间分离程度和类内聚集程度，适合论文主图展示。
    """
    n_classes = len(ENGLISH_NAMES)
    colors = CLASS_COLORS

    fig, ax = plt.subplots(figsize=(10, 8))

    for cls_idx, en_name in enumerate(ENGLISH_NAMES):
        mask = labels == cls_idx
        ax.scatter(
            tsne_2d[mask, 0],
            tsne_2d[mask, 1],
            c=[colors[cls_idx]],
            s=28,
            alpha=0.8,
            label=en_name,
            edgecolors="black",
            linewidths=0.3,
        )

    # ax.set_title("t-SNE of MC-MMLoP Features", fontsize=15)
    ax.legend(
        loc="best",
        fontsize=11,
        framealpha=0.9,
        title="Class",
        title_fontsize=12,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    png_path = f"{save_prefix}_single.png"
    svg_path = f"{save_prefix}_single.svg"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"[绘图] 已保存: {png_path} (dpi={dpi})")
    print(f"[绘图] 已保存: {svg_path} (矢量图，适合 SCI 论文)")


def compute_silhouette(features, labels):
    """计算轮廓系数。

    ⚠️ 踩坑2：轮廓系数必须在【原始高维特征】上计算，绝不能在 t-SNE 降维后的
       2D 数据上算！t-SNE 降维会扭曲距离关系，2D 上的轮廓系数没有数学意义。

    使用 cosine 距离（与 L2 归一化特征配套，衡量角度差异）。
    """
    print(
        f"[轮廓系数] 在原始高维特征上计算: "
        f"n_samples={features.shape[0]}, n_features={features.shape[1]}"
    )
    # silhouette_score 要求至少 2 个类且每类至少 2 个样本
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print("[轮廓系数] 样本类别数 < 2，无法计算，跳过")
        return float("nan")
    score = silhouette_score(features, labels, metric="cosine")
    print(f"[轮廓系数] silhouette_score (cosine) = {score:.4f}")
    return score


def main():
    parser = argparse.ArgumentParser(
        description="t-SNE 可视化：特征提取 + 降维 + 绘图 + 轮廓系数"
    )
    # —— 与 train.py 对齐的基础参数（用于 setup_cfg）——
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="训练输出目录（用于定位日志/配置）")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config-file", type=str, default="")
    parser.add_argument("--dataset-config-file", type=str, default="")
    parser.add_argument("--trainer", type=str, default="MMLoP")
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--dataset-name", type=str, default="Custom6")
    parser.add_argument("--head", type=str, default="")
    parser.add_argument("--source-domains", type=str, nargs="+")
    parser.add_argument("--target-domains", type=str, nargs="+")
    parser.add_argument("--transforms", type=str, nargs="+")
    parser.add_argument("--eval-only", action="store_true",
                        help="占位，与 train.py 对齐")
    parser.add_argument("--no-train", action="store_true",
                        help="占位，不触发训练")
    # —— 可视化专用参数 ——
    parser.add_argument("--model-dir", type=str, default="",
                        help="加载已训练模型的目录，默认等于 --output-dir")
    parser.add_argument("--load-epoch", type=int, default=None,
                        help="加载指定 epoch 的权重；不指定则用 model-best.pth.tar")
    parser.add_argument("--zero-shot", action="store_true",
                        help="zero-shot 模式：直接用原始 CLIP 权重，不加载任何 checkpoint")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE 困惑度（样本数多时取大，5~50）")
    parser.add_argument("--save-prefix", type=str, default="tsne_mc_mlop",
                        help="输出图片前缀，会生成 {prefix}.png 和 {prefix}.svg")
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
        help="修改 config 选项（与 train.py 一致的 key value 对)",
    )
    args = parser.parse_args()

    # —— 1. 构建 cfg（复用 train.py 的 setup_cfg，确保 MC-MLoP 配置项已注册）——
    cfg = train_module.setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print("** System info **\n{}\n".format(collect_env_info()))

    # —— 2. 构建 trainer（加载 CLIP、构建 dataset/test_loader、构建模型）——
    trainer = build_trainer(cfg)

    # —— 3. 加载已训练的模型权重（zero-shot 模式跳过）——
    if args.zero_shot:
        print("[加载] zero-shot 模式：不加载 checkpoint，直接用原始 CLIP 权重")
        # trainer.model 已经是 build_model 构建好的 CLIP（含预训练权重）
    else:
        model_dir = args.model_dir if args.model_dir else args.output_dir
        print(f"[加载] 从目录加载已训练模型: {model_dir}")
        trainer.load_model(model_dir, epoch=args.load_epoch)

    # —— 4. 步骤1：提取测试集特征（softmax 前）——
    print("\n========== 步骤1: 提取测试集特征（softmax 前） ==========")
    # zsclip 把 CLIP 存在 self.clip_model 而非 self.model
    vis_model = trainer.clip_model if args.zero_shot else trainer.model
    features, labels = extract_features(
        vis_model, trainer.test_loader, trainer.device
    )
    print(f"特征矩阵: shape={features.shape}, dtype={features.dtype}")
    print(f"标签数组: shape={labels.shape}, 类别分布={np.bincount(labels)}")

    # —— 5. 步骤2：t-SNE 降维 ——
    print("\n========== 步骤2: t-SNE 降维 ==========")
    tsne_2d = run_tsne(features, perplexity=args.perplexity, seed=args.seed)

    # —— 6. 步骤3：绘图（多子图 + 单图）——
    print("\n========== 步骤3: 绘图 ==========")
    # 确保 save_prefix 所在目录存在
    save_dir = os.path.dirname(args.save_prefix)
    if save_dir and not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # 多子图：每类一个子图（聚焦单类，其他类灰色背景）
    plot_tsne_subplots(tsne_2d, labels, save_prefix=args.save_prefix, dpi=300)
    # 单图：6 类同图，不同颜色，看整体位置分布
    plot_tsne_single(tsne_2d, labels, save_prefix=args.save_prefix, dpi=300)

    # —— 7. 步骤4：轮廓系数（原始高维特征）——
    print("\n========== 步骤4: 轮廓系数（原始高维特征） ==========")
    score = compute_silhouette(features, labels)

    print("\n========== 全部完成 ==========")
    print(f"轮廓系数 (原始高维特征, cosine): {score:.4f}")
    print(f"多子图: {args.save_prefix}.png / {args.save_prefix}.svg")
    print(f"单图:   {args.save_prefix}_single.png / {args.save_prefix}_single.svg")


if __name__ == "__main__":
    main()
