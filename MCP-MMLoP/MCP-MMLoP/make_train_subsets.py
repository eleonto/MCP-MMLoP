#!/usr/bin/env python3
"""
按比例切分 train 集,生成 4 个符号链接子集目录。

输入:dataset/train_full/<类>/*.png  (用户需先把原 train 重命名为 train_full)
输出:dataset/train_subset_<ratio>/<类>/*.png (符号链接,指向 train_full 中的对应图片)

每类按固定 seed=1 抽样 ratio 比例的图片,确保可复现且各类独立抽样。
符号链接使用相对路径(../../train_full/<类>/<img>),dataset 目录整体移动时不会失效。

使用方法:
    # 1) 先备份原训练集(只需第一次)
    mv dataset/train dataset/train_full

    # 2) 生成 4 个比例子集
    python make_train_subsets.py

    # 3) 切换训练集(把 train 软链接到某个子集)
    rm -f dataset/train && ln -s train_subset_0.2 dataset/train

    # 4) 还原(实验结束后)
    rm -f dataset/train && ln -s train_full dataset/train
"""
import os
import random
import sys

DATASET_DIR = "dataset"
SRC_NAME = "train_full"           # 完整训练集目录名(用户已重命名)
RATIOS = [0.2, 0.4, 0.6, 0.8]
SEED = 1                            # 与 train.py --seed 一致
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

# 与 datasets/custom6.py 的 FIXED_CLASSNAMES 保持一致
CLASS_DIRS = ["尖刺", "悬浮", "沿面", "气隙", "颗粒", "正常"]


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    out = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(IMG_EXTS):
            out.append(f)
    return out


def clear_dir(path):
    """清空目录下所有内容(不删目录本身)。"""
    if not os.path.isdir(path):
        return
    for entry in os.listdir(path):
        full = os.path.join(path, entry)
        if os.path.islink(full) or os.path.isfile(full):
            os.remove(full)
        elif os.path.isdir(full):
            # 递归删除子目录(子目录里都是符号链接)
            for sub in os.listdir(full):
                sub_full = os.path.join(full, sub)
                if os.path.islink(sub_full) or os.path.isfile(sub_full):
                    os.remove(sub_full)
            os.rmdir(full)


def make_subset(ratio, rng):
    subset_dir = os.path.join(DATASET_DIR, f"train_subset_{ratio}")
    clear_dir(subset_dir)
    os.makedirs(subset_dir, exist_ok=True)

    print(f"\n=== ratio={ratio} -> {subset_dir} ===")
    print(f"  {'class':<10} {'sampled':>8} {'total':>6}")
    total_sampled = 0
    total_full = 0
    for cls in CLASS_DIRS:
        src_cls_dir = os.path.join(DATASET_DIR, SRC_NAME, cls)
        if not os.path.isdir(src_cls_dir):
            print(f"  [ERROR] 源目录不存在: {src_cls_dir}")
            sys.exit(1)

        imgs = list_images(src_cls_dir)
        if not imgs:
            print(f"  [ERROR] {cls}/ 下没有图片")
            sys.exit(1)

        # 按比例抽样,至少 1 张
        n_take = max(1, round(len(imgs) * ratio))
        # 确保不超过总数
        n_take = min(n_take, len(imgs))
        sampled = rng.sample(imgs, n_take)

        dst_cls_dir = os.path.join(subset_dir, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        # 符号链接:从 dataset/train_subset_X/<类>/<img>
        #          指向 dataset/train_full/<类>/<img>
        # 相对路径: ../../train_full/<类>/<img>
        for img in sampled:
            link_target = os.path.join("..", "..", SRC_NAME, cls, img)
            link_name = os.path.join(dst_cls_dir, img)
            if os.path.lexists(link_name):
                os.remove(link_name)
            os.symlink(link_target, link_name)

        print(f"  {cls:<10} {n_take:>8} {len(imgs):>6}")
        total_sampled += n_take
        total_full += len(imgs)

    print(f"  {'合计':<10} {total_sampled:>8} {total_full:>6}")


def main():
    src_dir = os.path.join(DATASET_DIR, SRC_NAME)
    if not os.path.isdir(src_dir):
        print(f"[ERROR] 找不到完整训练集目录: {src_dir}")
        print(f"请先备份原训练集:  mv {DATASET_DIR}/train {DATASET_DIR}/{SRC_NAME}")
        sys.exit(1)

    print(f"数据集目录: {DATASET_DIR}")
    print(f"完整训练集: {SRC_NAME}/")
    print(f"比例: {RATIOS}")
    print(f"seed: {SEED}")

    rng = random.Random(SEED)
    for ratio in RATIOS:
        make_subset(ratio, rng)

    print("\n" + "=" * 60)
    print("完成。4 个子集已生成。")
    print("=" * 60)
    print("切换方法(把 train 软链接到某个子集):")
    for ratio in RATIOS:
        print(f"  ratio={ratio}:  rm -f {DATASET_DIR}/train && ln -s train_subset_{ratio} {DATASET_DIR}/train")
    print(f"\n还原(用回完整训练集):")
    print(f"  rm -f {DATASET_DIR}/train && ln -s {SRC_NAME} {DATASET_DIR}/train")
    print(f"\n注意:切换后请删除 custom6 的预处理缓存,确保重新扫描:")
    print(f"  rm -f {DATASET_DIR}/custom6_preprocessed.pkl")


if __name__ == "__main__":
    main()
