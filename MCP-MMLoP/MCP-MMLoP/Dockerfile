# ==============================================================================
#  MMLoP (Custom6) Dockerfile - 离线可构建版（以本机 fec-gpu:latest 为基础）
# ------------------------------------------------------------------------------
# 背景：构建环境无法稳定从 Docker Hub 拉 nvidia/cuda 基础镜像，
#       复用本机已有的 fec-gpu:latest（Ubuntu 22.04 + CUDA11.8 + conda
#       Python 3.10 + torch 2.1.2 + cuDNN）作为运行时，等价性完全满足
#       MMLoP（ViT-B/16 提示训练对 torch 版本要求宽松，1.12~2.1 均可运行）。
#
#  用途    ：镜像内直接  python train.py ...  或  bash scripts/mmlop/run_custom6.sh
#  输出 tar：docker save mmlop:custom6 -o mmlop-custom6.tar   (约 10~12GB)
# ==============================================================================
ARG BASE_IMAGE=fec-gpu:latest
FROM ${BASE_IMAGE} AS runtime
LABEL maintainer="MMLoP"
LABEL description="MMLoP runtime (reused fec-gpu base: Py3.10 / torch2.1.2 / CUDA 11.8) with Dassl + Custom6"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# --------- 时区/UTF-8，系统包（apt 优先阿里源） -----------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
      sed -i 's|ports.ubuntu.com|mirrors.aliyun.com|g; s|archive.ubuntu.com|mirrors.aliyun.com|g; s|security.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list || true; \
    fi; \
    apt-get update -y || true; \
    apt-get install -y --no-install-recommends \
        curl wget build-essential ca-certificates git locales tzdata \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 vim unzip zip \
    && rm -rf /var/lib/apt/lists/* || true

# fec-gpu 已提供 conda & torch，但确保在 PATH 顶部
ENV PATH=/opt/conda/bin:${PATH}
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=Asia/Shanghai
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
 && echo "Asia/Shanghai" > /etc/timezone \
 && locale-gen en_US.UTF-8 || true \
 && echo "conda=$(which conda); python=$(which python)" \
 && python --version \
 && python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# --------- PIP 切换国内阿里云镜像 + 安装 Dassl / MMLoP / clip 依赖 -----------
# 注：清华 pypi 源对部分 wheel 返回 403，改用阿里云源；不升级 pip（23.3.1 已够用）
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com

WORKDIR /workspace/MMLoP

# 先拷 requirements，利用 docker layer cache，代码改动不触发重新 pip
COPY Dassl.pytorch/requirements.txt   /tmp/dassl-requirements.txt
COPY requirements.txt                 /tmp/mmlop-requirements.txt

RUN set -eux; \
    python -m pip install --no-cache-dir -r /tmp/dassl-requirements.txt; \
    python -m pip install --no-cache-dir -r /tmp/mmlop-requirements.txt; \
    python -m pip install --no-cache-dir \
        tensorboard six yacs tabulate gdown ftfy regex tqdm scipy scikit-learn

# 拷仓库源码（dataset/output/pkl 被 .dockerignore 过滤；含 ViT-B-16.pt）
COPY . /workspace/MMLoP

# 将 CLIP ViT-B/16 权重烘焙到 clip 默认缓存目录（离线服务器无需联网下载）
RUN set -eux; \
    mkdir -p /root/.cache/clip; \
    cp /workspace/MMLoP/ViT-B-16.pt /root/.cache/clip/ViT-B-16.pt; \
    rm -f /workspace/MMLoP/ViT-B-16.pt; \
    ls -la /root/.cache/clip/

# 安装 Dassl（develop 模式，镜像内后续调代码都立即生效）
RUN cd /workspace/MMLoP/Dassl.pytorch && python setup.py develop

# 最终 import 验证：dassl / clip / train / Custom6 数据集注册
RUN set -eux; \
    cd /workspace/MMLoP; \
    python -c "import dassl; print('dassl', getattr(dassl,'__version__','ok'))"; \
    python -c "import clip; print('clip ok')"; \
    python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"; \
    python -c "import train;  print('train.py import ok')"; \
    python -c "import datasets.custom6; \
               from dassl.data.datasets import DATASET_REGISTRY; \
               assert DATASET_REGISTRY.get('Custom6') is not None, 'Custom6 not registered!'; \
               print('Custom6 registered OK')"

# --------- 默认工作目录 / 挂载点 / 提示 CMD --------------------------------
ENV MMLOP_ROOT=/workspace/MMLoP
ENV DATASET_DIR=/workspace/MMLoP/dataset
WORKDIR /workspace/MMLoP

VOLUME ["/workspace/MMLoP/dataset", "/workspace/MMLoP/output"]

CMD [ "bash", "-c", "echo 'MMLoP (Custom6) 容器已启动（基于fec-gpu复用版）。\n示例训练命令：\n  docker run --gpus all -v \\$PWD/dataset:/workspace/MMLoP/dataset -v \\$PWD/output:/workspace/MMLoP/output --rm -it mmlop:custom6 \\\n    bash scripts/mmlop/run_custom6.sh --gpu 0 --epoch 30\n\n评估：\n  bash scripts/mmlop/eval_custom6.sh --gpu 0 --model-dir output/custom6/train/vit_b16 --load-epoch 30\n'" ]
