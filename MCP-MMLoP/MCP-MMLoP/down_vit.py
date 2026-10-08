"""下载 CLIP 视觉权重到本地（ViT-B/32、ViT-L/14）。"""
import os
import sys

# 优先使用项目本地的 clip/ 包（而非 pip 安装的 openai-clip）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clip import clip

BACKBONES = [ "ViT-L/14"]
CACHE_DIR = os.path.dirname(os.path.abspath(__file__))  # 下载到脚本所在目录（项目根）
os.makedirs(CACHE_DIR, exist_ok=True)

for name in BACKBONES:
    filename = name.replace("/", "-") + ".pt"  # ViT-B-32.pt / ViT-L-14.pt
    dest = os.path.join(CACHE_DIR, filename)
    if os.path.exists(dest):
        print(f"[已存在] {dest} ({os.path.getsize(dest)//1024//1024}MB)，跳过")
        continue
    url = clip._MODELS[name]
    print(f"[下载中] {name} -> {dest}")
    print(f"  URL: {url}")
    path = clip._download(url, os.path.dirname(dest))
    print(f"[完成] {path} ({os.path.getsize(path)//1024//1024}MB)")

print("\n全部完成，权重保存在:", CACHE_DIR)
"""下载 CLIP 视觉权重到本地（ViT-B/32、ViT-L/14）。"""
import os
import sys

# 优先使用项目本地的 clip/ 包（而非 pip 安装的 openai-clip）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clip import clip

BACKBONES = ["ViT-B/32", "ViT-L/14"]
CACHE_DIR = os.path.dirname(os.path.abspath(__file__))  # 下载到脚本所在目录（项目根）
os.makedirs(CACHE_DIR, exist_ok=True)

for name in BACKBONES:
    filename = name.replace("/", "-") + ".pt"  # ViT-B-32.pt / ViT-L-14.pt
    dest = os.path.join(CACHE_DIR, filename)
    if os.path.exists(dest):
        print(f"[已存在] {dest} ({os.path.getsize(dest)//1024//1024}MB)，跳过")
        continue
    url = clip._MODELS[name]
    print(f"[下载中] {name} -> {dest}")
    print(f"  URL: {url}")
    path = clip._download(url, os.path.dirname(dest))
    print(f"[完成] {path} ({os.path.getsize(path)//1024//1024}MB)")

print("\n全部完成，权重保存在:", CACHE_DIR)
