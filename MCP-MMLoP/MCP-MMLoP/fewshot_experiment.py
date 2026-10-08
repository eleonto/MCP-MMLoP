#!/usr/bin/env python3
"""
按训练集比例的小样本实验
扫描 train_ratio (0.2, 0.4, 0.6, 0.8) 对性能的影响

前置条件:
  1) 已运行 make_train_subsets.py 生成 4 个子集目录
     (即 dataset/train_full/ 和 dataset/train_subset_0.X/ 都已就绪)
  2) dataset/train 当前可以是任意状态(脚本会自动切换软链接)

流程(每次 ratio):
  - 删除 custom6_preprocessed.pkl(强制重新扫描 train 目录)
  - 把 dataset/train 软链接切换到 train_subset_<ratio>
  - 运行 train.py,完整沿用 MC_MMLoP.sh 的超参
  - 解析日志
全部完成后,把 dataset/train 还原为指向 train_full。
"""
import os
import subprocess
import re
import time
import shutil
from datetime import datetime

RATIOS = [0.2, 0.4, 0.6, 0.8]

GPU_ID = "1"
DATASET_DIR = "dataset"
OUTPUT_BASE = "output/custom6/fewshot_ratio_exp"
COMBINED_LOG = os.path.join(OUTPUT_BASE, "fewshot_ratio_log.txt")

# MC_MMLoP.sh 的全部超参(原样沿用,仅 NUM_SHOTS 保持 -1,因为我们用物理切分实现"少样本")
BASE_CMD = [
    "python", "-u", "train.py",
    "--seed", "1",
    "--trainer", "MMLoP",
    "--backbone", "ViT-B/16",
    "--dataset-name", "Custom6",
    "--config-file", "configs/trainers/MMLoP/vit_b16.yaml",
    "DATASET.NAME", "Custom6",
    "DATASET.ROOT", "dataset",
    "DATASET.NUM_SHOTS", "-1",
    "DATASET.SUBSAMPLE_CLASSES", "all",
    "TRAINER.MMLOP.N_CTX_TEXT", "4",
    "TRAINER.MMLOP.N_CTX_VISION", "4",
    "TRAINER.MMLOP.PROMPT_DEPTH_TEXT", "9",
    "TRAINER.MMLOP.PROMPT_DEPTH_VISION", "9",
    "TRAINER.MMLOP.LORA_RANK", "1",
    "TRAINER.MMLOP.USE_SCL_LOSS", "True",
    "TRAINER.MMLOP.SCL_LOGIT_LOSS", "js",
    "TRAINER.MMLOP.USE_CTX_INIT_EMBED", "True",
    "TRAINER.MMLOP.CTX_INIT_PHRASE", "a photo of a",
    "TRAINER.MMLOP.USE_MCP", "True",
    "TRAINER.MMLOP.NUM_CENTERS", "3",
    "TRAINER.MMLOP.PROTO_TEMP", "0.07",
    "TRAINER.MMLOP.EMA_MOMENTUM", "0.01",
    "TRAINER.MMLOP.PROTO_FUSION_WEIGHT", "1.0",
    "TRAINER.MMLOP.TEXT_LOSS_WEIGHT", "25.0",
    "TRAINER.MMLOP.IMAGE_LOSS_WEIGHT", "10.0",
    "TRAINER.MMLOP.INTRA_LOSS_WEIGHT", "5.0",
    "TRAINER.MMLOP.INTER_LOSS_WEIGHT", "0.0",
    "TRAINER.MMLOP.INTER_MARGIN", "0.2",
    "OPTIM.MAX_EPOCH", "70",
    "DATALOADER.TRAIN_X.BATCH_SIZE", "4",
    "TRAINER.MMLOP.CLIP_MODEL_PATH", "/mnt/sdb/data/yjl/MMLoP/ViT-B-16.pt",
]


# ---------------------- 日志解析(与 momentum_experiment.py 一致) ----------------------
def parse_summary(output):
    m = re.search(
        r"Summary\s*\|\s*train_loss=([\d.]+)\s+train_acc=([\d.]+)%\s*\|\s*"
        r"val_loss\s*=([\d.]+)\s+val_acc\s*=([\d.]+)%\s*\|\s*"
        r"test_loss\s*=([\d.]+)\s+test_acc\s*=([\d.]+)%\s*\|\s*"
        r"best_acc\s*=([\d.]+)%\s*@epoch\s+(\d+)",
        output,
    )
    if m:
        return {
            "train_loss": float(m.group(1)), "train_acc": float(m.group(2)),
            "val_loss": float(m.group(3)), "val_acc": float(m.group(4)),
            "test_loss": float(m.group(5)), "test_acc": float(m.group(6)),
            "best_acc": float(m.group(7)), "best_epoch": int(m.group(8)),
        }
    return None


def parse_perclass(output):
    results = {}
    for m in re.finditer(
        r"class:\s*(\d+)\s*\(([^)]+)\)\s*total:\s*(\d+)\s*correct:\s*(\d+)\s*acc:\s*([\d.]+)%",
        output,
    ):
        cls_id = int(m.group(1))
        cls_name = m.group(2).strip()
        total = int(m.group(3))
        correct = int(m.group(4))
        acc = float(m.group(5))
        results[cls_id] = {"name": cls_name, "total": total, "correct": correct, "acc": acc}
    m_avg = re.search(r"average:\s*([\d.]+)%", output)
    if m_avg:
        results["average"] = float(m_avg.group(1))
    return results


def parse_overall(output):
    m = re.search(r"accuracy:\s*([\d.]+)%.*?macro_f1:\s*([\d.]+)%", output, re.DOTALL)
    if m:
        return {"accuracy": float(m.group(1)), "macro_f1": float(m.group(2))}
    return None


# ---------------------- 数据集目录切换 ----------------------
def switch_train_to(target_name):
    """
    把 dataset/train 软链接切换到 dataset/<target_name>(target_name 是相对 dataset/ 的目录名)。
    如果 dataset/train 已经是普通目录(没被替换过),会报错提示用户先备份。
    """
    train_link = os.path.join(DATASET_DIR, "train")
    target_abs = os.path.join(DATASET_DIR, target_name)

    if not os.path.isdir(target_abs):
        raise RuntimeError(f"目标子集目录不存在: {target_abs}")

    # 检查 train/ 是否是普通目录(用户还没备份)
    if os.path.isdir(train_link) and not os.path.islink(train_link):
        # 是普通目录,说明用户没按流程备份
        # 不自动删,直接报错,避免丢失原始数据
        raise RuntimeError(
            f"{train_link} 是普通目录而非软链接。\n"
            f"请先备份原始训练集(只需一次):\n"
            f"  mv {DATASET_DIR}/train {DATASET_DIR}/train_full && "
            f"python make_train_subsets.py\n"
            f"然后重新运行本脚本。"
        )

    # 删除原软链接(如果存在)
    if os.path.islink(train_link):
        os.remove(train_link)
    elif os.path.exists(train_link):
        # 不应该走到这里(前面已检查),保险起见报错
        raise RuntimeError(f"{train_link} 存在但既不是目录也不是软链接,无法处理,请手动检查")

    # 创建新软链接(相对路径)
    os.symlink(target_name, train_link)
    print(f"  [switch] {train_link} -> {target_name}")


def clear_custom6_cache():
    """删除 custom6_preprocessed.pkl,确保下次启动时重新扫描 train 目录。"""
    pkl = os.path.join(DATASET_DIR, "custom6_preprocessed.pkl")
    if os.path.exists(pkl):
        os.remove(pkl)
        print(f"  [clear] 已删除缓存 {pkl}")


def restore_train_full():
    """还原 dataset/train 指向 train_full(完整训练集)。"""
    train_link = os.path.join(DATASET_DIR, "train")
    if os.path.islink(train_link):
        os.remove(train_link)
        os.symlink("train_full", train_link)
        print(f"[restore] {train_link} -> train_full")
    clear_custom6_cache()


# ---------------------- 实验主流程 ----------------------
def run_experiment(ratio, run_idx, total):
    run_name = f"mcm_lop_ratio{ratio}"
    output_dir = os.path.join(OUTPUT_BASE, run_name)

    print(f"\n{'='*80}")
    print(f"[{run_idx}/{total}] train_ratio={ratio}")
    print(f"output_dir: {output_dir}")
    print(f"{'='*80}")

    # 切换数据集
    try:
        clear_custom6_cache()
        switch_train_to(f"train_subset_{ratio}")
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        return {
            "ratio": ratio, "run_idx": run_idx,
            "error": True, "error_msg": str(e),
            "elapsed_sec": 0,
        }

    # 组装命令
    cmd = BASE_CMD + ["--output-dir", output_dir]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID

    t0 = time.time()
    # 注意:不捕获 stdout,实时打印训练进度(用户可观察)
    # 但为了解析,我们也保存一份完整输出到变量
    result = subprocess.run(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - t0

    full_output = result.stdout
    # 实时打印最后 50 行(便于排查)
    tail_lines = full_output.splitlines()[-50:]
    print("\n--- 训练输出最后 50 行 ---")
    for line in tail_lines:
        print(line)
    print("--- end ---\n")

    summary = parse_summary(full_output)
    perclass = parse_perclass(full_output)
    overall = parse_overall(full_output)

    if result.returncode != 0 and summary is None:
        print(f"[ERROR] Run failed with returncode={result.returncode}")
        tail = full_output[-3000:]
        print(tail)
        return {
            "ratio": ratio, "run_idx": run_idx,
            "error": True, "returncode": result.returncode,
            "elapsed_sec": elapsed,
            "error_tail": tail,
        }

    return {
        "ratio": ratio, "run_idx": run_idx,
        "error": False,
        "summary": summary,
        "perclass": perclass,
        "overall": overall,
        "elapsed_sec": elapsed,
    }


# ---------------------- 日志写入 ----------------------
def write_log(results):
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    with open(COMBINED_LOG, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"MC-MLoP 训练集比例小样本实验\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"train_ratio: {RATIOS}\n")
        f.write(f"固定参数: NUM_CENTERS=3, INTRA=5.0, INTER=0.0, EMA=0.01, MAX_EPOCH=70, LORA_RANK=1\n")
        f.write(f"GPU: {GPU_ID}\n")
        f.write(f"总实验数: {len(RATIOS)}\n")
        f.write("=" * 100 + "\n\n")

        for r in results:
            f.write("-" * 80 + "\n")
            f.write(f"实验 #{r['run_idx']}: train_ratio={r['ratio']}\n")
            f.write(f"耗时: {r['elapsed_sec']/60:.1f} 分钟\n")

            if r.get("error"):
                f.write(f"[ERROR] returncode={r.get('returncode', 'N/A')}\n")
                if "error_msg" in r:
                    f.write(f"  msg: {r['error_msg']}\n")
                if "error_tail" in r:
                    f.write(f"  output tail:\n{r['error_tail']}\n")
                f.write("\n")
                continue

            s = r["summary"]
            if s:
                f.write(f"Summary:\n")
                f.write(f"  train_loss={s['train_loss']:.4f}  train_acc={s['train_acc']:.2f}%\n")
                f.write(f"  val_loss ={s['val_loss']:.4f}  val_acc ={s['val_acc']:.2f}%\n")
                f.write(f"  test_loss={s['test_loss']:.4f}  test_acc ={s['test_acc']:.2f}%\n")
                f.write(f"  best_acc ={s['best_acc']:.2f}%  @epoch {s['best_epoch']}\n")

            o = r["overall"]
            if o:
                f.write(f"Overall: accuracy={o['accuracy']:.2f}%  macro_f1={o['macro_f1']:.2f}%\n")

            pc = r["perclass"]
            if pc:
                f.write(f"Per-class:\n")
                for cls_id in sorted(k for k in pc.keys() if isinstance(k, int)):
                    c = pc[cls_id]
                    f.write(f"  class {cls_id} ({c['name']}): total={c['total']} correct={c['correct']} acc={c['acc']:.2f}%\n")
                if "average" in pc:
                    f.write(f"  average: {pc['average']:.2f}%\n")

            f.write("\n")

        f.write("=" * 100 + "\n")
        f.write("汇总表:\n")
        f.write(f"{'#':>3}  {'ratio':>6}  {'best_acc':>9}  {'best_ep':>7}  {'test_acc':>9}  {'val_acc':>8}  {'train_acc':>10}  {'macro_f1':>9}  {'elapsed(m)':>10}\n")
        f.write("-" * 95 + "\n")
        for r in results:
            if r.get("error"):
                f.write(f"{r['run_idx']:>3}  {r['ratio']:>6.2f}  {'ERROR':>9}  {'':>7}  {'':>9}  {'':>8}  {'':>10}  {'':>9}  {r['elapsed_sec']/60:>10.1f}\n")
                continue
            s = r["summary"] or {}
            o = r.get("overall") or {}
            f.write(
                f"{r['run_idx']:>3}  {r['ratio']:>6.2f}  "
                f"{s.get('best_acc', 0):>9.2f}  {s.get('best_epoch', 0):>7}  "
                f"{s.get('test_acc', 0):>9.2f}  {s.get('val_acc', 0):>8.2f}  "
                f"{s.get('train_acc', 0):>10.2f}  "
                f"{o.get('macro_f1', 0):>9.2f}  {r['elapsed_sec']/60:>10.1f}\n"
            )
        f.write("=" * 100 + "\n")

    print(f"\n[完成] 日志已保存到: {COMBINED_LOG}")


def main():
    # 前置检查
    train_full = os.path.join(DATASET_DIR, "train_full")
    if not os.path.isdir(train_full):
        print(f"[FATAL] 未找到 {train_full}/。请先执行:")
        print(f"  mv {DATASET_DIR}/train {DATASET_DIR}/train_full && python make_train_subsets.py")
        return

    for ratio in RATIOS:
        subset = os.path.join(DATASET_DIR, f"train_subset_{ratio}")
        if not os.path.isdir(subset):
            print(f"[FATAL] 未找到 {subset}/。请先运行: python make_train_subsets.py")
            return

    total = len(RATIOS)
    results = []
    try:
        for run_idx, ratio in enumerate(RATIOS, 1):
            r = run_experiment(ratio, run_idx, total)
            results.append(r)
            write_log(results)
    finally:
        # 无论实验是否全部完成,都还原 train 指向 train_full
        print("\n[finally] 还原 dataset/train -> train_full")
        restore_train_full()

    print(f"\n全部 {total} 次实验完成!")


if __name__ == "__main__":
    main()
