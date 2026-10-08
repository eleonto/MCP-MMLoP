#!/usr/bin/env python3
"""
批量超参数实验：INTRA_LOSS_WEIGHT × INTER_LOSS_WEIGHT 网格搜索
5 × 5 = 25 次实验，每次记录 Summary 和 per-class 结果
"""
import os
import subprocess
import re
import time
from datetime import datetime

INTRA_VALUES = [ 20.0]
INTER_VALUES = [25.0]

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
    "TRAINER.MMLOP.INTER_MARGIN", "0.2",
    "OPTIM.MAX_EPOCH", "70",
    "DATALOADER.TRAIN_X.BATCH_SIZE", "4",
    "TRAINER.MMLOP.CLIP_MODEL_PATH", "/mnt/sdb/data/yjl/MMLoP/ViT-B-16.pt",
]

GPU_ID = "1"
OUTPUT_BASE = "output/custom6/train6"
COMBINED_LOG = os.path.join(OUTPUT_BASE, "hyperparam_grid_search_log.txt")


def parse_summary(output):
    m = re.search(r"Summary\s*\|\s*train_loss=([\d.]+)\s+train_acc=([\d.]+)%\s*\|\s*val_loss\s*=([\d.]+)\s+val_acc\s*=([\d.]+)%\s*\|\s*test_loss\s*=([\d.]+)\s+test_acc\s*=([\d.]+)%\s*\|\s*best_acc\s*=([\d.]+)%\s*@epoch\s+(\d+)", output)
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
    for m in re.finditer(r"class:\s*(\d+)\s*\(([^)]+)\)\s*total:\s*(\d+)\s*correct:\s*(\d+)\s*acc:\s*([\d.]+)%", output):
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


def run_experiment(intra_w, inter_w, run_idx, total):
    run_name = f"mcm_lop_intra{intra_w}_inter{inter_w}"
    output_dir = os.path.join(OUTPUT_BASE, run_name)

    cmd = [
        "python", "-u", "train.py",
        "--seed", "1",
        "--trainer", "MMLoP",
        "--backbone", "ViT-B/16",
        "--dataset-name", "Custom6",
        "--config-file", "configs/trainers/MMLoP/vit_b16.yaml",
        "--output-dir", output_dir,
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
        "TRAINER.MMLOP.INTER_MARGIN", "0.2",
        "OPTIM.MAX_EPOCH", "70",
        "DATALOADER.TRAIN_X.BATCH_SIZE", "4",
        "TRAINER.MMLOP.CLIP_MODEL_PATH", "/mnt/sdb/data/yjl/MMLoP/ViT-B-16.pt",
        "TRAINER.MMLOP.INTRA_LOSS_WEIGHT", str(intra_w),
        "TRAINER.MMLOP.INTER_LOSS_WEIGHT", str(inter_w),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID

    print(f"\n{'='*80}")
    print(f"[{run_idx}/{total}] intra={intra_w}, inter={inter_w}")
    print(f"output_dir: {output_dir}")
    print(f"{'='*80}")

    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    full_output = result.stdout + result.stderr

    summary = parse_summary(full_output)
    perclass = parse_perclass(full_output)
    overall = parse_overall(full_output)

    if result.returncode != 0 and summary is None:
        print(f"[ERROR] Run failed with returncode={result.returncode}")
        tail = result.stderr if result.stderr else result.stdout
        print(tail[-2000:])
        return {
            "intra": intra_w, "inter": inter_w, "run_idx": run_idx,
            "error": True, "returncode": result.returncode,
            "elapsed_sec": elapsed,
        }

    return {
        "intra": intra_w, "inter": inter_w, "run_idx": run_idx,
        "error": False,
        "summary": summary,
        "perclass": perclass,
        "overall": overall,
        "elapsed_sec": elapsed,
    }


def write_log(results):
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    with open(COMBINED_LOG, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"MC-MLoP 超参数网格搜索实验\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"INTRA_LOSS_WEIGHT: {INTRA_VALUES}\n")
        f.write(f"INTER_LOSS_WEIGHT: {INTER_VALUES}\n")
        f.write(f"总实验数: {len(INTRA_VALUES) * len(INTER_VALUES)}\n")
        f.write("=" * 100 + "\n\n")

        for r in results:
            f.write("-" * 80 + "\n")
            f.write(f"实验 #{r['run_idx']}: INTRA={r['intra']}, INTER={r['inter']}\n")
            f.write(f"耗时: {r['elapsed_sec']/60:.1f} 分钟\n")

            if r.get("error"):
                f.write(f"[ERROR] returncode={r['returncode']}\n\n")
                continue

            s = r["summary"]
            if s:
                f.write(f"Summary:\n")
                f.write(f"  train_loss={s['train_loss']:.4f}  train_acc={s['train_acc']:.2f}%\n")
                f.write(f"  val_loss ={s['val_loss']:.4f}  val_acc ={s['val_acc']:.2f}%\n")
                f.write(f"  test_loss={s['test_loss']:.4f}  test_acc={s['test_acc']:.2f}%\n")
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
        f.write("汇总表 (best_acc / test_acc):\n")
        f.write(f"{'#':>3}  {'intra':>6}  {'inter':>6}  {'best_acc':>9}  {'best_ep':>7}  {'test_acc':>9}  {'val_acc':>8}  {'train_acc':>10}  {'elapsed(m)':>10}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            if r.get("error"):
                f.write(f"{r['run_idx']:>3}  {r['intra']:>6.1f}  {r['inter']:>6.1f}  {'ERROR':>9}  {'':>7}  {'':>9}  {'':>8}  {'':>10}  {r['elapsed_sec']/60:>10.1f}\n")
                continue
            s = r["summary"] or {}
            f.write(
                f"{r['run_idx']:>3}  {r['intra']:>6.1f}  {r['inter']:>6.1f}  "
                f"{s.get('best_acc', 0):>9.2f}  {s.get('best_epoch', 0):>7}  "
                f"{s.get('test_acc', 0):>9.2f}  {s.get('val_acc', 0):>8.2f}  "
                f"{s.get('train_acc', 0):>10.2f}  {r['elapsed_sec']/60:>10.1f}\n"
            )
        f.write("=" * 100 + "\n")

    print(f"\n[完成] 日志已保存到: {COMBINED_LOG}")


def main():
    total = len(INTRA_VALUES) * len(INTER_VALUES)
    results = []
    run_idx = 0
    for intra_w in INTRA_VALUES:
        for inter_w in INTER_VALUES:
            run_idx += 1
            r = run_experiment(intra_w, inter_w, run_idx, total)
            results.append(r)
            write_log(results)

    print(f"\n全部 {total} 次实验完成！")


if __name__ == "__main__":
    main()
