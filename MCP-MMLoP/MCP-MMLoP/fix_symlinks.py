#!/usr/bin/env python3
"""
修复 dataset/0.X/train/<类>/ 下失效的符号链接。

问题:符号链接目标为 ../../train_full/<类>/<img>,
     但实际路径应是 ../../../train_full/<类>/<img>(多一层 0.X/)。

修复策略:把每个符号链接替换为指向正确目标的符号链接。
     (用 cp -L 复制成真实文件也行,但符号链接更省空间。)

用法:python fix_symlinks.py
"""
import os

DATASET_DIR = "dataset"
RATIOS = ["0.2", "0.4", "0.6", "0.8"]
CLASS_DIRS = ["尖刺", "悬浮", "沿面", "气隙", "颗粒", "正常"]


def fix_one_ratio(ratio):
    """修复 dataset/<ratio>/train/<类>/ 下所有符号链接。"""
    print(f"\n=== 修复 {ratio}/train/ ===")
    total_fixed = 0
    total_broken = 0
    total_ok = 0
    for cls in CLASS_DIRS:
        cls_dir = os.path.join(DATASET_DIR, ratio, "train", cls)
        if not os.path.isdir(cls_dir):
            print(f"  [WARN] 目录不存在: {cls_dir}")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            if not os.path.islink(fpath):
                continue
            target = os.readlink(fpath)
            # 检查链接是否有效
            if os.path.exists(fpath):
                total_ok += 1
                continue
            # 失效链接,修复
            total_broken += 1
            # 期望指向 ../../../train_full/<类>/<img>
            new_target = os.path.join("..", "..", "..", "train_full", cls, fname)
            os.remove(fpath)
            os.symlink(new_target, fpath)
            # 验证
            if os.path.exists(fpath):
                total_fixed += 1
            else:
                print(f"  [FAIL] {fpath} -> {new_target} 仍然无效!")
        print(f"  {cls}: ok={total_ok}, broken={total_broken}, fixed={total_fixed}")
    print(f"  小计: ok={total_ok}, broken={total_broken}, fixed={total_fixed}")
    return total_fixed


def main():
    print(f"数据集目录: {DATASET_DIR}")
    print(f"比例: {RATIOS}")
    print(f"类别: {CLASS_DIRS}")

    grand_fixed = 0
    for ratio in RATIOS:
        grand_fixed += fix_one_ratio(ratio)

    print(f"\n{'='*60}")
    print(f"总计修复 {grand_fixed} 个失效符号链接")
    print("=" * 60)
    print("现在可以运行训练:")
    print("  sh MC_MMLoP.sh   (确保 DATASET.ROOT 指向 dataset/0.2 等)")


if __name__ == "__main__":
    main()
