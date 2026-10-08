"""
Custom6：自定义 **6 类**图像数据集（基于 ImageFolder 风格的 train/val/test 目录）。

用户指定的固定类别顺序（整数 label 由下决定，不再由 sorted/扫描顺序决定）：
    0 = 尖刺 (spike)
    1 = 悬浮 (suspension / floating)
    2 = 沿面 (surface / creeping)
    3 = 气隙 (air-gap / void)
    4 = 颗粒 (particle / bubble)
    5 = 正常 (normal)

期望的目录结构（相对于 cfg.DATASET.ROOT，例如 "dataset"）：
    dataset/
    ├── train/
    │   ├── 尖刺/    *.png / *.jpg
    │   ├── 悬浮/    ...
    │   ├── 沿面/    ...
    │   ├── 气隙/    ...
    │   ├── 颗粒/    ...
    │   └── 正常/    ...
    ├── val/         (类别目录同上；若为空则自动从 train 每类切 10%)
    └── test/        (类别目录同上)

关键特性：
    * 类别顺序完全固定：与 Linux/Windows、文件系统遍历顺序无关，复现实验稳定。
      若 train/ 或 test/ 缺少某一类，会直接抛错提示"缺哪类"，避免静默漏读。
    * 路径鲁棒：ROOT 为相对路径时，先按 os.getcwd() 解析；若不存在再按本文件所在
      项目根（train.py 同级）解析；仍不存在则打印清晰的错误信息和当前 cwd。
    * 自动补 val：val/ 目录为空时，从 train 每类随机切出 10% 放到 val（固定 seed=0）。
    * 少样本兼容：沿用 Dassl DatasetBase.generate_fewshot_dataset，
      DATASET.NUM_SHOTS > 0 时直接复用。
"""

import os
import pickle
import random
from typing import Dict, List, Tuple

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

# 支持的图片后缀
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

# ---------------------------------------------------------------------------
# 用户指定的 **固定** 6 类顺序（label = index）。
# 改类别顺序/增加类别时，只改这里 + 对应目录即可。
# ---------------------------------------------------------------------------
FIXED_CLASSNAMES: Tuple[str, ...] = (
    "尖刺",   # 0
    "悬浮",   # 1
    "沿面",   # 2
    "气隙",   # 3
    "颗粒",   # 4
    "正常",   # 5
)
FIXED_CLASS_TO_LABEL: Dict[str, int] = {c: i for i, c in enumerate(FIXED_CLASSNAMES)}


def _list_images(folder: str) -> List[str]:
    """递归或非递归列出 folder 下所有图片，返回相对(绝对)路径列表。"""
    if not os.path.isdir(folder):
        return []
    out: List[str] = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


def _resolve_root(root_cfg: str, fallback_base: str) -> str:
    """
    解析数据集根目录的绝对路径。

    规则（对相对路径生效）：
        1. 先按当前工作目录解析（最常见的"cd 到仓库后运行 train.py"）。
        2. 若 1 不存在，按 ``fallback_base``（train.py 所在目录）解析，
           确保从 IDE 不同的启动配置也能找到数据。
        3. 仍不存在时抛出带 cwd 和期望结构的 ValueError。
    """
    if os.path.isabs(root_cfg):
        candidate = os.path.abspath(os.path.expanduser(root_cfg))
    else:
        # 相对路径：先按 cwd
        candidate = os.path.abspath(os.path.expanduser(root_cfg))
        if not os.path.isdir(candidate):
            candidate = os.path.abspath(
                os.path.join(fallback_base, root_cfg)
            )

    if not os.path.isdir(candidate):
        cwd = os.getcwd()
        raise ValueError(
            "[Custom6] 找不到数据集根目录：\n"
            f"  传入 ROOT = '{root_cfg}'\n"
            f"  当前工作目录 cwd = '{cwd}'\n"
            f"  最终尝试路径 = '{candidate}'\n"
            "请确保：\n"
            "  1) 目录存在；\n"
            "  2) 该目录下有 train/ 、val/ 、test/ 三个子目录；\n"
            "  3) 每一个子目录里有以类别命名的文件夹（例如 尖刺/、正常/ ……），"
            "里面放 .png / .jpg 图片。"
        )
    return candidate


def _scan_split(split_dir: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    扫描某个 split（train/val/test）目录。
    返回：
        class_to_paths: {类名: [图片绝对路径列表]}
        classnames:     排序后的类名列表（稳定顺序，保证跨 split 一致）
    """
    class_to_paths: Dict[str, List[str]] = {}
    if not os.path.isdir(split_dir):
        return class_to_paths, []

    # 仅取 split_dir 的一级子目录作为类别
    for entry in sorted(os.listdir(split_dir)):
        full = os.path.join(split_dir, entry)
        if not os.path.isdir(full):
            continue
        imgs = _list_images(full)
        if imgs:
            class_to_paths[entry] = imgs

    classnames = sorted(class_to_paths.keys())
    return class_to_paths, classnames


def _build_items(
    class_to_paths: Dict[str, List[str]],
    class_to_label: Dict[str, int],
) -> List[Datum]:
    """把 {类名: 图片路径列表} 转成 Dassl Datum 列表，label 用映射表。"""
    items: List[Datum] = []
    for classname, paths in class_to_paths.items():
        label = class_to_label[classname]
        for p in paths:
            items.append(Datum(impath=p, label=label, classname=classname))
    return items


@DATASET_REGISTRY.register()
class Custom6(DatasetBase):
    """
    自定义 ImageFolder 风格的 6 类数据集适配器（类别顺序完全固定，见 FIXED_CLASSNAMES）。

    使用方法：
        在 train.py 中 import datasets.custom6 后，Dassl 会自动把它注册到
        DATASET_REGISTRY；然后在 yaml 或命令行设置：
            DATASET.NAME = "Custom6"
            DATASET.ROOT = "dataset"   (相对路径，推荐)
    """

    def __init__(self, cfg):
        # 鲁棒解析数据集根（带相对路径兜底）
        fallback_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root = _resolve_root(cfg.DATASET.ROOT, fallback_base)

        self.dataset_dir = root
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")
        self.test_dir = os.path.join(self.dataset_dir, "test")

        # Dassl 把 few-shot 切分结果缓存到 DATASET.ROOT 下（与原 MMLoP 保持一致）
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        self.preprocessed_pkl = os.path.join(self.dataset_dir, "custom6_preprocessed.pkl")

        # 优先读取缓存（加速二次启动）
        if os.path.exists(self.preprocessed_pkl):
            print(f"[Custom6] 读取缓存 {self.preprocessed_pkl}")
            with open(self.preprocessed_pkl, "rb") as f:
                cache = pickle.load(f)
            train = cache["train"]
            val = cache["val"]
            test = cache["test"]
        else:
            train, val, test = self._build_splits()
            with open(self.preprocessed_pkl, "wb") as f:
                pickle.dump(
                    {"train": train, "val": val, "test": test},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"[Custom6] 缓存写入 {self.preprocessed_pkl}")

        # ---------- 少样本抽样：沿用 Dassl 基类方法 ----------
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots and num_shots >= 1:
            seed = int(cfg.SEED)
            cache_path = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )
            if os.path.exists(cache_path):
                print(f"[Custom6] 读取 few-shot 缓存 {cache_path}")
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                train = data["train"]
                # val 也做 few-shot，与 MMLoP 默认做法一致
                if "val" in data:
                    val = data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                # val 上限是 min(num_shots, 4) 每类，避免 val 被抽成空集
                val_num = max(1, min(num_shots, 4)) if val else 0
                val_sampled = (
                    self.generate_fewshot_dataset(val, num_shots=val_num)
                    if val and val_num
                    else val
                )
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        {"train": train, "val": val_sampled},
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                print(f"[Custom6] few-shot 缓存写入 {cache_path}")

        # ---------- 统计并打印 ----------
        self._print_stats(train, val, test)

        # SUBSAMPLE_CLASSES 默认 "all"；为兼容 MMLoP 的 yaml，这里也处理
        subsample = cfg.get("DATASET.SUBSAMPLE_CLASSES", "all")
        if subsample not in (None, "all", "All", "ALL"):
            # 自定义数据集默认用全部 6 类；base/new 在此类小数据集一般不需要，
            # 这里打印 warning 而不是硬报错，保留可运行性。
            print(
                f"[Custom6] 警告：DATASET.SUBSAMPLE_CLASSES='{subsample}' "
                "在 Custom6 上不支持，已回退为使用全部 6 个类别。"
            )

        super().__init__(train_x=train, val=val, test=test)

    # ------------------------------------------------------------------
    def _build_splits(self) -> Tuple[List[Datum], List[Datum], List[Datum]]:
        """扫描 train/val/test 目录，并在 val 缺失时从 train 切 10%。

        类别顺序完全由 FIXED_CLASSNAMES 决定：
            - train 和 test 目录都必须包含 **全部 6 个类别**；缺哪类就直接报错，
              避免静默漏读某类导致类别不平衡 / 标签错位。
            - 遇到目录下的"额外类"（非 6 类之一）会打印 warning 并跳过，
              不抛错（方便用户先放临时调试目录）。
        """
        train_cp, _ = _scan_split(self.train_dir)
        val_cp, _ = _scan_split(self.val_dir)
        test_cp, _ = _scan_split(self.test_dir)

        all_classnames = list(FIXED_CLASSNAMES)
        class_to_label = dict(FIXED_CLASS_TO_LABEL)

        # ---------- 基本完整性检查：train / test 必须覆盖全部 6 类 ----------
        for split_name, split_cp in (("train", train_cp), ("test", test_cp)):
            missing = [c for c in all_classnames if c not in split_cp or not split_cp[c]]
            if missing:
                raise ValueError(
                    f"[Custom6] {split_name}/ 目录缺少类别或类别为空：{missing}\n"
                    f"  期望 6 类（按顺序）：{all_classnames}\n"
                    f"  扫描路径 = '{self.train_dir if split_name == 'train' else self.test_dir}'"
                )

        # 打印"警告：扫描到但被忽略的额外类别"
        all_found = set(train_cp) | set(val_cp) | set(test_cp)
        extra = sorted(c for c in all_found if c not in FIXED_CLASS_TO_LABEL)
        if extra:
            print(
                f"[Custom6] 警告：扫描到额外目录（不属于 6 类，已忽略）：{extra}；"
                f"如需使用请修改 datasets/custom6.py 中的 FIXED_CLASSNAMES。"
            )

        # 打印最终使用的固定类别映射
        print(
            f"[Custom6] 固定类别映射 (共 {len(all_classnames)} 类，由 FIXED_CLASSNAMES 定义)：\n  "
            + " ;  ".join(f"{FIXED_CLASS_TO_LABEL[c]}={c}" for c in all_classnames)
        )

        # ---------- 只保留 6 个指定类（val 允许缺类，后面还会自动补） ----------
        train_cp = {c: list(train_cp[c]) for c in all_classnames if c in train_cp}
        val_cp = {c: list(val_cp[c]) for c in all_classnames if c in val_cp}
        test_cp = {c: list(test_cp[c]) for c in all_classnames if c in test_cp}

        # ---------- val 自动补齐：从 train 每类随机抽取 10%（最少 1 张） ----------
        if not val_cp:
            print(
                "[Custom6] val/ 目录为空，将从 train/ 每类随机抽取 10% "
                "（最少 1 张）作为验证集（固定 seed=0，可复现）。"
            )
            rng = random.Random(0)
            new_val: Dict[str, List[str]] = {}
            for cls in all_classnames:
                paths = train_cp[cls]
                if not paths:
                    continue
                n_take = max(1, round(len(paths) * 0.1))
                if n_take >= len(paths):
                    # 至少给 train 留 1 张
                    n_take = len(paths) - 1
                if n_take <= 0:
                    continue
                taken = rng.sample(paths, n_take)
                taken_set = set(taken)
                new_val[cls] = taken
                train_cp[cls] = [p for p in paths if p not in taken_set]
            val_cp = new_val

        # ---------- 组装 Datum ----------
        train_items = _build_items(train_cp, class_to_label)
        val_items = _build_items(val_cp, class_to_label)
        test_items = _build_items(test_cp, class_to_label)

        if not train_items:
            raise ValueError(
                f"[Custom6] 没有任何训练图片！请检查 {self.train_dir} 下是否"
                "包含类别子目录及图片。"
            )
        if not test_items:
            raise ValueError(
                f"[Custom6] 没有任何测试图片！请检查 {self.test_dir} 下是否"
                "包含类别子目录及图片。"
            )

        return train_items, val_items, test_items

    # ------------------------------------------------------------------
    @staticmethod
    def _print_stats(train, val, test) -> None:
        def stat(items):
            if not items:
                return 0, {}
            per_cls: Dict[str, int] = {}
            for it in items:
                per_cls[it.classname] = per_cls.get(it.classname, 0) + 1
            return len(items), per_cls

        n_train, per_train = stat(train)
        n_val, per_val = stat(val)
        n_test, per_test = stat(test)

        # 统一按 FIXED_CLASSNAMES 顺序输出，和 label 对应
        print("[Custom6] 各 split 图片数量（按固定类别顺序）：")
        print(f"  {'label':>5}  {'class':<10} {'train':>6} {'val':>6} {'test':>6}")
        for c in FIXED_CLASSNAMES:
            label = FIXED_CLASS_TO_LABEL[c]
            print(
                f"  {label:>5}  {c:<10} {per_train.get(c, 0):>6} "
                f"{per_val.get(c, 0):>6} {per_test.get(c, 0):>6}"
            )
        print(
            f"  {'':>5}  {'合计':<10} {n_train:>6} {n_val:>6} {n_test:>6}"
        )
