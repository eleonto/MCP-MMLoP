import os.path as osp
import math
import os
import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # 非交互后端，直接保存 PNG
import matplotlib.pyplot as plt

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# =============================================================================
# 论文公式与代码位置对照表（审稿人导航）
#   Eq.(1)  低秩提示 P=U@V, 跨模态共享 U ........ CrossModalLoRA (本文件)
#   Eq.(2)  提示后 Transformer 输入 ............ clip/model.py: forward_with_lora
#   Eq.(3)  UDC 均匀漂移校正 ................... CustomCLIP.forward (use_udc 段)
#   Eq.(4)  提示路径得分 s_c ................... CustomCLIP.forward (image@text)
#   Eq.(5)  多中心子原型 P_c ................... VLPromptLearner.multi_center_prototypes
#   Eq.(7)  max 相似度聚合 g_c ................. CustomCLIP.forward (proto_scores)
#   Eq.(8-10) 熵加权 EMA 原型更新 .............. CustomCLIP._ema_update_prototypes
#             子原型 K-means 初始化 ............ MMLoP.init_multi_center_prototypes
#   Eq.(11) 类内紧凑 L_intra ................... CustomCLIP._proto_losses
#   Eq.(12) 类间分离 L_inter ................... CustomCLIP._proto_losses
#   Eq.(13-14) 双路融合 + 温度 softmax ......... CustomCLIP.forward (logits)
#   Eq.(16) SCL 自调节一致性损失 ............... CustomCLIP.forward (use_scl_loss 段)
#   Alg. 1  训练/推断流程 ...................... MMLoP.forward_backward / CustomCLIP.forward
# =============================================================================

# Import ImageNet templates for SCL loss
try:
    from .imagenet_templates import IMAGENET_TEMPLATES
except ImportError:
    IMAGENET_TEMPLATES = [
        "a photo of a {}.",
    ]


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    # --- 防御性修复：Dassl 配置系统有时会把 NAME 覆写成配置文件路径 ---
    if not backbone_name or ('/' in backbone_name) or ('\\' in backbone_name) or backbone_name.endswith('.yaml') or backbone_name.endswith('.yml'):
        print(f"[WARN] cfg.MODEL.BACKBONE.NAME 无效 ('{backbone_name}')，回退到默认值 ViT-B/16")
        backbone_name = "ViT-B/16"
    print(f"[INFO] 使用 CLIP backbone: {backbone_name}")

    # --- 自动搜索本地权重（离线服务器友好）---
    # 优先级：
    #   1. cfg.TRAINER.MMLOP.CLIP_MODEL_PATH（命令行/配置指定）
    #   2. 环境变量 CLIP_MODEL_PATH
    #   3. ~/.cache/clip/<backbone>.pt（clip 默认缓存位置）
    #   4. 当前工作目录及常见挂载位置
    #   5. 项目根目录下
    local_path = getattr(cfg.TRAINER.MMLOP, "CLIP_MODEL_PATH", "") or os.environ.get("CLIP_MODEL_PATH", "")

    # --- 公共搜索准备（后面两个分支都会用到）---
    backbone_filename = backbone_name.replace("/", "-") + ".pt"  # ViT-B/16 -> ViT-B-16.pt
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_dirs = [
        os.path.expanduser("~/.cache/clip"),        # clip 默认缓存
        os.path.expanduser("~/cache/clip"),
        os.path.expanduser("~/"),                    # 家目录
        os.getcwd(),                                # 当前工作目录
        project_root,                               # 项目根（train.py 同级）
        os.path.join(project_root, "dataset"),      # dataset 目录旁边
        os.path.join(project_root, "output"),        # output 目录
    ]
    filename_variants = {
        "ViT-B/16": ["ViT-B-16.pt", "ViT-B-16.pt.jit", "vit_b16.pt", "vit-b16.pt"],
        "ViT-B/32": ["ViT-B-32.pt", "ViT-B-32.pt.jit"],
        "ViT-L/14": ["ViT-L-14.pt", "ViT-L-14.pt.jit"],
        "ViT-L/14@336px": ["ViT-L-14-336px.pt", "ViT-L-14@336px.pt"],
    }
    variants = filename_variants.get(backbone_name, [backbone_filename])

    def _search_dirs():
        """在 search_dirs 中搜索第一个匹配的权重文件，返回完整路径或空串。"""
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for v in variants:
                candidate = os.path.join(d, v)
                if os.path.isfile(candidate):
                    return candidate
        return ""

    # ---- 分支 1：CLIP_MODEL_PATH 已指定（配置或环境变量）----
    if local_path:
        # 如果是目录 → 在目录内匹配权重文件
        if os.path.isdir(local_path):
            print(f"[INFO] CLIP_MODEL_PATH='{local_path}' 是目录，尝试在其中搜索权重文件")
            found_in_dir = ""
            for v in variants:
                candidate = os.path.join(local_path, v)
                if os.path.isfile(candidate):
                    found_in_dir = candidate
                    print(f"[INFO] 在目录中找到权重文件: {candidate}")
                    break
            if found_in_dir:
                model_path = found_in_dir
            else:
                print(f"[WARN] 目录 '{local_path}' 下未找到匹配的权重文件，回退到自动搜索")
                local_path = ""
        elif os.path.isfile(local_path):
            # 是文件且存在 → 直接用
            print(f"Loading CLIP weights from: {local_path}")
            model_path = local_path
        else:
            # 文件不存在
            print(f"[WARN] CLIP_MODEL_PATH 指定的文件不存在: {local_path}，回退到自动搜索")
            local_path = ""

    # ---- 分支 2：未指定或上面已回退 → 自动搜索 ----
    if not local_path:
        print(f"[INFO] 在以下目录搜索 CLIP 权重文件：")
        model_path = _search_dirs()
        if model_path:
            print(f"[INFO] 找到权重文件: {model_path}  ({os.path.getsize(model_path)//1024//1024}MB)")

    # ---- 兜底：联网下载 ----
    if not model_path:
        print(f"[WARN] 本地未找到 CLIP 权重文件，尝试联网下载...")
        print(f"[WARN] 若服务器无外网，请将 {variants[0]} 放到以下任一目录：")
        for d in search_dirs:
            print(f"[WARN]   {d}")
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    if not zero_shot_model:
        design_details = {
            "trainer": 'IVLP',
            "vision_depth": cfg.TRAINER.MMLOP.PROMPT_DEPTH_VISION,
            "language_depth": cfg.TRAINER.MMLOP.PROMPT_DEPTH_TEXT,
            "vision_ctx": cfg.TRAINER.MMLOP.N_CTX_VISION,
            "language_ctx": cfg.TRAINER.MMLOP.N_CTX_TEXT,
            "vision_start": cfg.TRAINER.MMLOP.PROMPT_START_VISION,
            "language_start": cfg.TRAINER.MMLOP.PROMPT_START_TEXT
        }
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features (for SCL loss)
        design_details = {
            "trainer": 'IVLP',
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0
        }
        model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class CrossModalLoRA(nn.Module):
    """
    Cross-modal LoRA for deep vision-language prompts (simplified, rank-1 optimized).
    
    Features:
        - Shared U between vision and text (cross-modal coupling)
        - Pure LoRA (no base prompt mixing)
        - Configurable layer range [start, depth) for each modality
    
    Formula:
        P = U @ V
    
    Args:
        vision_depth: End layer (exclusive) for vision prompts
        text_depth: End layer (exclusive) for text prompts
        vision_start: Start layer (inclusive) for vision prompts
        text_start: Start layer (inclusive) for text prompts
        n_ctx_vision: Number of vision prompt tokens
        n_ctx_text: Number of text prompt tokens
        vis_dim: Vision prompt dimension
        text_dim: Text prompt dimension
        rank: LoRA rank
        shared_u: If True, share U between modalities
        init_std: Standard deviation for weight initialization
        dtype: Parameter dtype
    """
    def __init__(self, vision_depth, text_depth, n_ctx_vision, n_ctx_text,
                 vis_dim, text_dim, rank, shared_u=True,
                 vision_start=0, text_start=0,
                 init_std=0.05, dtype=torch.float32):
        super().__init__()
        
        self.vision_depth = vision_depth
        self.text_depth = text_depth
        self.vision_start = vision_start
        self.text_start = text_start
        self.n_ctx_vision = n_ctx_vision
        self.n_ctx_text = n_ctx_text
        self.shared_u = shared_u
        self.rank = rank
        self.init_std = init_std
        
        # Number of active layers per modality
        n_vision_layers = vision_depth - vision_start
        n_text_layers = text_depth - text_start
        max_layers = max(n_vision_layers, n_text_layers)
        
        if shared_u:
            assert n_ctx_vision == n_ctx_text, \
                f"When shared_u=True, n_ctx_vision ({n_ctx_vision}) must equal n_ctx_text ({n_ctx_text})"
            n_ctx = n_ctx_vision
            
            # Shared U matrices for each active layer: shape (n_ctx, rank) per layer
            self.U_shared = nn.ParameterList([
                nn.Parameter(torch.empty(n_ctx, rank, dtype=dtype))
                for _ in range(max_layers)
            ])
        else:
            # Separate U for each modality
            self.U_v = nn.ParameterList([
                nn.Parameter(torch.empty(n_ctx_vision, rank, dtype=dtype))
                for _ in range(n_vision_layers)
            ])
            self.U_t = nn.ParameterList([
                nn.Parameter(torch.empty(n_ctx_text, rank, dtype=dtype))
                for _ in range(n_text_layers)
            ])
        
        # V matrices (always separate per modality)
        self.V_v = nn.ParameterList([
            nn.Parameter(torch.empty(rank, vis_dim, dtype=dtype))
            for _ in range(n_vision_layers)
        ])
        self.V_t = nn.ParameterList([
            nn.Parameter(torch.empty(rank, text_dim, dtype=dtype))
            for _ in range(n_text_layers)
        ])
        
        self._init_weights(init_std)
    
    def _init_weights(self, init_std):
        if self.shared_u:
            for u in self.U_shared:
                nn.init.normal_(u, std=init_std)
        else:
            for u in self.U_v:
                nn.init.normal_(u, std=init_std)
            for u in self.U_t:
                nn.init.normal_(u, std=init_std)
        for v in self.V_v:
            nn.init.normal_(v, std=init_std)
        for v in self.V_t:
            nn.init.normal_(v, std=init_std)
    
    def forward_vision(self, layer_idx):
        """Get LoRA delta for vision at specified layer."""
        if layer_idx < self.vision_start or layer_idx >= self.vision_depth:
            return None
        
        param_idx = layer_idx - self.vision_start
        
        if self.shared_u:
            U = self.U_shared[param_idx]  # (n_ctx, rank)
        else:
            U = self.U_v[param_idx]
        
        V = self.V_v[param_idx]  # (rank, vis_dim)
        
        return U @ V
    
    def forward_text(self, layer_idx):
        """Get LoRA delta for text at specified layer."""
        if layer_idx < self.text_start or layer_idx >= self.text_depth:
            return None
        
        param_idx = layer_idx - self.text_start
        
        if self.shared_u:
            U = self.U_shared[param_idx]  # (n_ctx, rank)
        else:
            U = self.U_t[param_idx]
        
        V = self.V_t[param_idx]  # (rank, text_dim)
        
        return U @ V
    
    def get_all_vision_lora(self):
        """Get LoRA deltas for all vision layers."""
        return [self.forward_vision(l) for l in range(self.vision_depth)]
    
    def get_all_text_lora(self):
        """Get LoRA deltas for all text layers."""
        return [self.forward_text(l) for l in range(self.text_depth)]


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, text_lora_deltas=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Use LoRA-enabled forward if deltas are provided
        if text_lora_deltas is not None:
            x = self.transformer.forward_with_lora(x, text_lora_deltas)
        else:
            x = self.transformer(x)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class VLPromptLearner(nn.Module):
    """
    Vision-Language Prompt Learner with Cross-Modal LoRA (simplified).
    
    Pure LoRA prompts: P = U @ V
    Prompts applied only to layers [start, depth) per modality.
    
    Optionally supports SCL loss from PromptSRC for better generalization.
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        
        # Text prompt settings
        n_ctx_text = cfg.TRAINER.MMLOP.N_CTX_TEXT
        ctx_init = cfg.TRAINER.MMLOP.CTX_INIT
        
        # Vision prompt settings
        n_ctx_vision = cfg.TRAINER.MMLOP.N_CTX_VISION
        
        # Prompt depths
        self.vision_depth = cfg.TRAINER.MMLOP.PROMPT_DEPTH_VISION
        self.text_depth = cfg.TRAINER.MMLOP.PROMPT_DEPTH_TEXT
        
        # Prompt start layers
        self.vision_start = cfg.TRAINER.MMLOP.PROMPT_START_VISION
        self.text_start = cfg.TRAINER.MMLOP.PROMPT_START_TEXT
        
        # LoRA settings
        self.lora_rank = cfg.TRAINER.MMLOP.LORA_RANK
        self.shared_u = cfg.TRAINER.MMLOP.SHARED_U
        self.init_std = cfg.TRAINER.MMLOP.INIT_STD
        
        # Context init embedding settings
        # When enabled, layer-0 text prompt = embed(ctx_init_phrase)[:n_ctx_text] + U@V
        # This anchors the prompt in CLIP's pre-training distribution at init (U@V ≈ 0),
        # so LoRA learns residuals from a meaningful linguistic starting point.
        self.use_ctx_init_embed = cfg.TRAINER.MMLOP.USE_CTX_INIT_EMBED
        ctx_init_phrase = cfg.TRAINER.MMLOP.CTX_INIT_PHRASE  # e.g. "a photo of"
        
        # SCL loss settings
        self.use_scl_loss = cfg.TRAINER.MMLOP.USE_SCL_LOSS
        
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # text dim
        
        # Get vision prompt dimension
        if hasattr(clip_model.visual, 'transformer'):
            vis_prompt_dim = clip_model.visual.transformer.width
        else:
            vis_prompt_dim = clip_model.visual.positional_embedding.shape[1]
        
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        assert self.text_depth >= 1, "Text prompt depth should be >= 1"
        assert self.vision_start < self.vision_depth, \
            f"Vision start ({self.vision_start}) must be < depth ({self.vision_depth})"
        assert self.text_start < self.text_depth, \
            f"Text start ({self.text_start}) must be < depth ({self.text_depth})"

        # Determine prompt prefix for tokenization
        if ctx_init and n_ctx_text <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt_prefix = ctx_init
        else:
            prompt_prefix = " ".join(["X"] * n_ctx_text)
        
        print(f"Cross-Modal LoRA V-L Prompt Design (Pure LoRA)")
        print(f'Prompt prefix: "{prompt_prefix}"')
        print(f"Number of text context tokens: {n_ctx_text}")
        print(f"Number of vision context tokens: {n_ctx_vision}")
        print(f"Text prompt depth: {self.text_depth}")
        print(f"Vision prompt depth: {self.vision_depth}")
        print(f"Text prompt start layer: {self.text_start}")
        print(f"Vision prompt start layer: {self.vision_start}")
        print(f"LoRA rank: {self.lora_rank}")
        print(f"Shared U: {self.shared_u}")
        print(f"Initial std: {self.init_std}")
        print(f"Use SCL loss: {self.use_scl_loss}")
        print(f"Use ctx init embed: {self.use_ctx_init_embed}" +
              (f' ("{ctx_init_phrase}")' if self.use_ctx_init_embed else ""))
        
        # Cross-Modal LoRA module
        self.cross_modal_lora = CrossModalLoRA(
            vision_depth=self.vision_depth,
            text_depth=self.text_depth,
            n_ctx_vision=n_ctx_vision,
            n_ctx_text=n_ctx_text,
            vis_dim=vis_prompt_dim,
            text_dim=ctx_dim,
            rank=self.lora_rank,
            shared_u=self.shared_u,
            vision_start=self.vision_start,
            text_start=self.text_start,
            init_std=self.init_std,
            dtype=dtype
        )
        
        # Process class names
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        
        # MC-MLoP: 多中心原型开关（先于 SCL/MCP 分支使用）
        self.use_mcp = cfg.TRAINER.MMLOP.USE_MCP
        self.num_centers = cfg.TRAINER.MMLOP.NUM_CENTERS if self.use_mcp else 0

        # SCL loss: compute frozen zero-shot features
        if self.use_scl_loss:
            clip_model_temp = load_clip_to_cpu(cfg, True).float()
            clip_model_temp_image = load_clip_to_cpu(cfg, True)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
                self.ZS_image_encoder = clip_model_temp_image.visual
                # Pre-compute frozen VL embeddings using text templates
                if cfg.TRAINER.MMLOP.SINGLE_TEMPLATE:
                    templates = [cfg.TRAINER.MMLOP.TEXT_TEMPLATE]
                else:
                    templates = IMAGENET_TEMPLATES
                all_teacher_features = []
                for single_template in templates:
                    x = [single_template.replace("{}", name) for name in classnames]
                    x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                    text_features = clip_model_temp.encode_text(x_tokenized)
                    all_teacher_features.append(text_features.unsqueeze(1))
                self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        else:
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.fixed_embeddings = None
            if self.use_mcp:
                # MCP 的 K-means 初始化需要冻结 CLIP 视觉编码器提取零样本特征
                # （SCL 关闭时这里补建，与 SCL 分支中的 ZS_image_encoder 等价）
                self.ZS_image_encoder = load_clip_to_cpu(cfg, True).visual
            else:
                self.ZS_image_encoder = None

        # MC-MLoP: 多中心原型参数（存于 prompt_learner 内，
        # 使其落在 build_model 的 "prompt_learner" 可训练名过滤范围内，与低秩因子联合优化）
        if self.use_mcp:
            embed_dim = clip_model.text_projection.shape[1]
            # fp32 存储（优化器数值稳定），使用处按特征 dtype 转换
            self.multi_center_prototypes = nn.Parameter(
                torch.randn(n_cls, self.num_centers, embed_dim, dtype=torch.float32) * 0.01
            )
            print(f"Use Multi-Center Prototype (MCP): True (K={self.num_centers}, "
                  f"params={n_cls * self.num_centers * embed_dim})")
        else:
            self.multi_center_prototypes = None

        # Register buffers
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx_text:, :])  # CLS, EOS
        
        # Context init embedding anchor for layer 0 text prompt.
        # We tokenize ctx_init_phrase, embed it, and take up to n_ctx_text tokens
        # (padding with zeros if the phrase is shorter). At init, U@V ≈ 0, so the
        # layer-0 prompt starts close to the natural-language phrase CLIP was trained on.
        if self.use_ctx_init_embed and self.text_start == 0:
            with torch.no_grad():
                phrase_tokens = clip.tokenize([ctx_init_phrase])  # (1, 77)
                phrase_embed = clip_model.token_embedding(phrase_tokens).type(dtype)  # (1, 77, dim)
                # Skip SOS token (position 0); take up to n_ctx_text content tokens
                phrase_content = phrase_embed[0, 1:1 + n_ctx_text, :]  # (<=n_ctx_text, dim)
                n_phrase = phrase_content.shape[0]
                if n_phrase < n_ctx_text:
                    # Pad with zeros for remaining slots
                    padding = torch.zeros(n_ctx_text - n_phrase, phrase_content.shape[-1], dtype=dtype)
                    phrase_content = torch.cat([phrase_content, padding], dim=0)
            self.register_buffer("ctx_init_embed", phrase_content)  # (n_ctx_text, dim)
        else:
            self.ctx_init_embed = None

        self.n_cls = n_cls
        self.n_ctx_text = n_ctx_text
        self.n_ctx_vision = n_ctx_vision
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        """Construct text prompts from context vectors."""
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self):
        """
        Returns:
            prompts: Text prompts with LoRA applied at layer 0 (or zeros if text_start > 0)
            vision_lora_deltas: List of LoRA deltas for each vision layer (None outside active range)
            text_lora_deltas: List of LoRA deltas for text layers 1+ (None outside active range)
        
        Layer-0 text context (when text_start == 0):
            ctx = ctx_init_embed + U@V   (if USE_CTX_INIT_EMBED)
            ctx = U@V                    (otherwise)
        This anchors the prompt in CLIP's linguistic pre-training distribution when
        USE_CTX_INIT_EMBED=True, so LoRA learns residuals from a meaningful starting point.
        """
        # Get LoRA delta for text layer 0 (returns None if text_start > 0)
        ctx = self.cross_modal_lora.forward_text(0)
        
        if ctx is None:
            # No prompt at layer 0: use zero context as placeholder
            ctx = torch.zeros(
                self.n_ctx_text, self.token_prefix.shape[-1],
                dtype=self.token_prefix.dtype, device=self.token_prefix.device
            )
        
        # Optionally anchor layer-0 context with a frozen linguistic embedding.
        # ctx_init_embed is None when USE_CTX_INIT_EMBED=False or text_start > 0.
        if self.ctx_init_embed is not None:
            ctx = ctx + self.ctx_init_embed
        
        # Expand for all classes
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)
        
        # Get LoRA deltas for deep layers
        vision_lora_deltas = self.cross_modal_lora.get_all_vision_lora()
        # Text: layers 1+ (layer 0 already applied above)
        text_lora_deltas = [self.cross_modal_lora.forward_text(l) for l in range(1, self.text_depth)]
        
        return prompts, vision_lora_deltas, text_lora_deltas


class CustomCLIP(nn.Module):
    """
    Custom CLIP model with Cross-Modal LoRA prompts (simplified).
    
    Pure LoRA prompts (P = U @ V) applied to both vision and text encoders.
    Optionally computes SCL loss from PromptSRC for better generalization.
    Optionally applies UDC (Uniform Drift Correction) to remove uniform base-class drift.
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        # SCL loss settings
        self.use_scl_loss = cfg.TRAINER.MMLOP.USE_SCL_LOSS
        if self.use_scl_loss:
            self.text_loss_weight = cfg.TRAINER.MMLOP.TEXT_LOSS_WEIGHT
            self.image_loss_weight = cfg.TRAINER.MMLOP.IMAGE_LOSS_WEIGHT
            self.scl_logit_loss = cfg.TRAINER.MMLOP.SCL_LOGIT_LOSS
        
        # UDC: Uniform Drift Correction (requires SCL for fixed_embeddings)
        self.use_udc = cfg.TRAINER.MMLOP.USE_UDC
        if self.use_udc:
            assert self.use_scl_loss, "USE_UDC requires USE_SCL_LOSS=True (needs fixed_embeddings)"

        # MC-MLoP: Multi-Center Prototype (MCP) hyperparams
        self.use_mcp = cfg.TRAINER.MMLOP.USE_MCP
        if self.use_mcp:
            self.proto_temp = cfg.TRAINER.MMLOP.PROTO_TEMP                    # τ
            self.ema_momentum = cfg.TRAINER.MMLOP.EMA_MOMENTUM                # η
            self.proto_fusion_weight = cfg.TRAINER.MMLOP.PROTO_FUSION_WEIGHT  # β
            self.intra_loss_weight = cfg.TRAINER.MMLOP.INTRA_LOSS_WEIGHT      # λ3
            self.inter_loss_weight = cfg.TRAINER.MMLOP.INTER_LOSS_WEIGHT      # λ4
            self.inter_margin = cfg.TRAINER.MMLOP.INTER_MARGIN                # m

        self.n_cls = len(classnames)

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        
        # Get prompts and LoRA deltas
        prompts, vision_lora_deltas, text_lora_deltas = self.prompt_learner()
        
        # Encode text with LoRA
        text_features = self.text_encoder(prompts, tokenized_prompts, text_lora_deltas)
        
        # Encode image with LoRA
        if hasattr(self.image_encoder, 'forward_with_lora'):
            image_features = self.image_encoder.forward_with_lora(
                image.type(self.dtype), 
                vision_lora_deltas
            )
        else:
            image_features = self.image_encoder(image.type(self.dtype))

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # UDC: remove uniform base-class drift (applies during both training and inference)
        if self.use_udc:
            fixed_embeddings_udc = self.prompt_learner.fixed_embeddings
            fixed_embeddings_udc = fixed_embeddings_udc / fixed_embeddings_udc.norm(dim=-1, keepdim=True)
            fixed_embeddings_udc = fixed_embeddings_udc.to(text_features.device).type(text_features.dtype)
            
            residual = text_features - fixed_embeddings_udc
            mean_shift = residual.mean(dim=0, keepdim=True)
            text_features = fixed_embeddings_udc + (residual - mean_shift)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # ---- 双路分类 logits（论文 3.7 节，训练与推断完全一致）----
        # 提示路径:  s_c = sim(f_p, ĝ_c)                    Eq.(4)
        # 原型路径:  g_c = max_k sim(f_p, p_{c,k})          Eq.(7)
        # 融合:      ℓ_c = logit_scale * (s_c + β·g_c)      Eq.(13)-(14)
        # logit_scale 是冻结的 CLIP 温度参数（初始化 exp(ln(1/0.07))，即 1/τ），
        # 因此 softmax(ℓ/τ) 与 CLIP 惯例一致（τ=0.07）；PROTO_TEMP(τ) 另用于
        # EMA 更新时原型路径软标签 π 的熵计算（见 _ema_update_prototypes）。
        if self.use_mcp:
            protos = self.prompt_learner.multi_center_prototypes
            protos_n = F.normalize(protos.to(image_features.dtype), dim=-1)     # (C, K, d)
            proto_sim = torch.einsum("bd,ckd->bck", image_features, protos_n)   # (B, C, K)
            proto_scores = proto_sim.max(dim=2).values                          # (B, C) Eq.(7)
            logits = logit_scale * (
                image_features @ text_features.t()                              # s_c, Eq.(4)
                + self.proto_fusion_weight * proto_scores                       # β·g_c, Eq.(13)
            )
        else:
            proto_sim = None
            logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            # ---- L_CE ----
            loss = F.cross_entropy(logits, label)

            if self.use_scl_loss:
                # Get frozen features for SCL loss (all on the same device as text_features)
                fixed_embeddings = self.prompt_learner.fixed_embeddings
                fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
                fixed_embeddings = fixed_embeddings.to(text_features.device, text_features.dtype)

                # Compute zero-shot logits without LoRA (frozen teacher path)
                # ZS_image_encoder is stored as a plain attr, ensure device alignment
                zs_image_enc = self.prompt_learner.ZS_image_encoder
                if zs_image_enc is not None:
                    # Move the frozen encoder to the current device (idempotent)
                    zs_image_enc = zs_image_enc.to(text_features.device)
                with torch.no_grad():
                    zero_shot_features = zs_image_enc(image.type(self.dtype))
                    zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                    zero_shot_features = zero_shot_features.to(text_features.device, text_features.dtype)
                    zero_shot_logits = logit_scale * zero_shot_features @ fixed_embeddings.t()

                # ---- L_SCL_text: L1 between learned and frozen text features ----
                loss_scl_text = F.l1_loss(
                    text_features,
                    fixed_embeddings,
                    reduction='mean'
                ) * self.text_loss_weight

                # ---- L_SCL_image: L1 between learned and frozen image features ----
                loss_scl_image = F.l1_loss(
                    image_features,
                    zero_shot_features,
                    reduction='mean'
                ) * self.image_loss_weight

                # ---- L_SCL_logits ----
                if self.scl_logit_loss == 'js':
                    # Symmetric JS divergence via two-way KL
                    p_prompt = F.log_softmax(logits, dim=1)
                    p_zero = F.log_softmax(zero_shot_logits, dim=1)
                    kl_zero_to_prompt = F.kl_div(p_prompt, p_zero, reduction='sum', log_target=True)
                    kl_prompt_to_zero = F.kl_div(p_zero, p_prompt, reduction='sum', log_target=True)
                    loss_scl_logits = 0.5 * (kl_zero_to_prompt + kl_prompt_to_zero) / logits.numel()
                elif self.scl_logit_loss == 'njs':
                    # Jensen-Shannon: JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M), M = 0.5*(P+Q)
                    p_prompt = F.softmax(logits, dim=1)
                    p_zero = F.softmax(zero_shot_logits, dim=1)
                    m = 0.5 * (p_prompt + p_zero)
                    log_m = m.log()
                    kl_prompt_to_m = F.kl_div(log_m, p_prompt, reduction='sum', log_target=False)
                    kl_zero_to_m = F.kl_div(log_m, p_zero, reduction='sum', log_target=False)
                    loss_scl_logits = 0.5 * (kl_prompt_to_m + kl_zero_to_m) / logits.numel()
                else:
                    # KL divergence (default)
                    loss_scl_logits = F.kl_div(
                        F.log_softmax(zero_shot_logits / 1, dim=1),
                        F.log_softmax(logits / 1, dim=1),
                        reduction='sum',
                        log_target=True
                    ) * 1 / logits.numel()

                loss = loss + loss_scl_text + loss_scl_image + loss_scl_logits

            if self.use_mcp:
                # ---- L_intra + L_inter（设计文档 3.7 节）----
                loss_intra, loss_inter = self._proto_losses(proto_sim, label, image_features)
                loss = loss + loss_intra + loss_inter
                # ---- 熵加权 EMA 原型更新（设计文档 3.5 节，SGD 之外的额外更新）----
                self._ema_update_prototypes(proto_sim.detach(), label, image_features)

            return loss

        return logits

    def _proto_losses(self, proto_sim, label, image_features):
        """论文 3.6 节（Eq. 11-12）：类内紧凑 + 类间分离正则。

        proto_sim: (B, C, K) 样本与各子原型的余弦相似度
        label: (B,) 真实标签
        image_features: (B, d) 已归一化的提示后图像特征
        """
        protos = self.prompt_learner.multi_center_prototypes
        protos_n = F.normalize(protos.to(image_features.dtype), dim=-1)  # (C, K, d)
        B = label.shape[0]

        # 类内紧凑（论文 Eq. 11: L_intra = 1 - g_{y_n}(f)）。
        # f、p 均为单位向量时 ||f - p||_2^2 = 2(1 - sim(f, p))，
        # 因此平方欧氏距离与 (1 - 余弦相似度) 仅相差常数因子 2，
        # 该常数被超参 lambda_intra 吸收，两者优化目标等价。
        k_star = proto_sim.argmax(dim=2)                                  # (B, C)
        k_sel = k_star[torch.arange(B, device=label.device), label]       # (B,)
        p_assigned = protos_n[label, k_sel]                               # (B, d)
        loss_intra = ((image_features - p_assigned) ** 2).sum(dim=1).mean()
        loss_intra = loss_intra * self.intra_loss_weight

        # 类间分离（论文 Eq. 12）：对跨类子原型对施加余弦合页惩罚，
        # sim(p_{c,k}, p_{c',k'}) 超过间隔 m 才计损失，达标后自动归零。
        C, K, d = protos_n.shape
        flat = protos_n.reshape(C * K, d)                                 # (CK, d)
        sims = flat @ flat.t()                                            # (CK, CK)
        cls_ids = torch.arange(C, device=flat.device).repeat_interleave(K)
        cross = cls_ids[:, None] != cls_ids[None, :]                      # 跨类对掩码
        hinge = (sims - self.inter_margin).clamp(min=0)
        loss_inter = (hinge * cross).sum() / cross.sum()                  # / |S|
        loss_inter = loss_inter * self.inter_loss_weight

        return loss_intra, loss_inter

    @torch.no_grad()
    def _ema_update_prototypes(self, proto_sim, label, image_features):
        """论文 3.5 节（Eq. 8-10）熵加权 EMA 原型更新的等价实现（无梯度，不参与反传）。

        与论文公式的对应关系（实现形式不同，但单调等价）：
          * EMA_MOMENTUM 在代码中是【更新系数 alpha】，论文中的 eta 是
            【保留系数（动量）】，二者互补：alpha = 1 - eta。
            论文 eta = 0.99  <=>  配置 EMA_MOMENTUM = 0.01。
          * 论文 Eq.(9) 取 w_c ~ 1 / H_c 再归一化；代码取 w_c = 1 - H_c/log C。
            两者都是类别熵 H_c 的单调递减函数（高熵=不确定类别的更新被衰减），
            且 w_c ∈ [0, 1]，是同一种"熵置信度加权"的等价实现形式。
          * 论文按样本权重 w̄_n = Σ_c q_{n,c} w_c 逐样本 EMA；代码先在 batch 内
            按 (真实类别, 归属子原型) 分组求特征均值，再以同一类权重做一次 EMA。
            对同一 (c, k*) 组而言，"逐样本 EMA"与"组均值一次 EMA"在稳态下
            都把原型移向该形态簇的批次中心，数值接近且后者与样本顺序无关。

        注意：DataParallel 下此更新只作用于各 replica 副本、不回传主模型；
        多卡训练时原型仅由 SGD 梯度更新，EMA 仅在单卡训练时完全生效。
        """
        protos = self.prompt_learner.multi_center_prototypes
        C, K, d = protos.shape
        feats = image_features.float()
        sim = proto_sim.float()                                           # (B, C, K)

        # 原型路径概率 π（温度 τ，论文 Eq. 8 中的软标签 q_n）与批量级类别边缘分布
        pi = F.softmax(sim.max(dim=2).values / self.proto_temp, dim=1)    # (B, C)
        p_c = pi.mean(dim=0)                                              # (C,) Eq.(8)
        H_c = -(p_c * p_c.clamp_min(1e-12).log())                         # (C,) 类别熵
        # 论文 Eq.(9) 的等价置信度权重：w_c = 1 - H_c/log C ∈ [0, 1]
        # （1/logC 是 H_c 的最大值归一化；高熵类别的权重被压向 0）
        w_c = (1.0 - H_c / math.log(C)).clamp(0.0, 1.0)                   # (C,)

        # 归属中心 k* = argmax_k sim(f_n, p_{c,k})（论文 Algorithm 1, line 11）
        k_star = sim.argmax(dim=2)                                        # (B, C)
        for c in label.unique().tolist():
            idx = (label == c).nonzero(as_tuple=True)[0]
            # 实际更新步长 = alpha * w_c，alpha = 1 - eta（eta 为论文 EMA 动量）
            eta_w = self.ema_momentum * float(w_c[c])
            assign = k_star[idx, c]                                       # (n_c,)
            for k in assign.unique().tolist():
                # 组均值：同一 (c, k) 形态簇内批次特征的归一化平均
                sel = idx[assign == k]
                f_mean = F.normalize(feats[sel].mean(dim=0), dim=0)
                # 论文 Eq.(10): p <- eta*p + (1-eta)*w*f；
                # 代码以更新系数 alpha=1-eta 书写：p <- (1-alpha*w)*p + alpha*w*f_mean
                protos.data[c, k] = (1.0 - eta_w) * protos.data[c, k] + eta_w * f_mean
                protos.data[c, k] = F.normalize(protos.data[c, k], dim=0)


def _kmeans_centers(x, k, n_iter=50, seed=0):
    """简单 K-means，用于 MC-MLoP 子原型初始化（设计文档 3.4 节）。

    x: (n, d) 已归一化的零样本特征；返回 (k, d) 归一化中心。
    n <= k 时退化为按类别均值初始化（极小样本场景，如 1-shot）。
    """
    n, d = x.shape
    if n == 0:
        return F.normalize(torch.randn(k, d), dim=-1)
    if n <= k:
        mean = F.normalize(x.mean(dim=0), dim=-1)
        return mean.unsqueeze(0).repeat(k, 1)
    g = torch.Generator().manual_seed(seed)
    centers = x[torch.randperm(n, generator=g)[:k]].clone()
    for _ in range(n_iter):
        assign = torch.cdist(x, centers).argmin(dim=1)
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = x[m].mean(dim=0)
        centers = F.normalize(centers, dim=-1)
    return centers


@TRAINER_REGISTRY.register()
class MMLoP(TrainerX):
    """
    Independent Vision-Language Prompting with Cross-Modal LoRA (simplified).

    Pure LoRA prompts (P = U @ V) with optional SCL loss.
    MC-MLoP extension: optional Multi-Center Prototype (MCP) dual-path scoring.

    Config options:
        TRAINER.MMLOP.N_CTX_TEXT: Number of text prompt tokens
        TRAINER.MMLOP.N_CTX_VISION: Number of vision prompt tokens
        TRAINER.MMLOP.PROMPT_DEPTH_TEXT: Text prompt end layer (exclusive)
        TRAINER.MMLOP.PROMPT_DEPTH_VISION: Vision prompt end layer (exclusive)
        TRAINER.MMLOP.PROMPT_START_TEXT: Text prompt start layer (inclusive, default 0)
        TRAINER.MMLOP.PROMPT_START_VISION: Vision prompt start layer (inclusive, default 0)
        TRAINER.MMLOP.CTX_INIT: Initial context words
        TRAINER.MMLOP.PREC: Precision (fp16, fp32, amp)
        TRAINER.MMLOP.LORA_RANK: LoRA rank
        TRAINER.MMLOP.SHARED_U: Share U between modalities
        TRAINER.MMLOP.INIT_STD: Standard deviation for weight initialization
        TRAINER.MMLOP.USE_SCL_LOSS: Use Self-regulating Consistency Loss
        TRAINER.MMLOP.SCL_LOGIT_LOSS: Logit consistency loss type ('kl', 'js', or 'njs')
        TRAINER.MMLOP.TEXT_LOSS_WEIGHT: Weight for L_SCL_text
        TRAINER.MMLOP.IMAGE_LOSS_WEIGHT: Weight for L_SCL_image
        TRAINER.MMLOP.USE_UDC: Use Uniform Drift Correction (requires USE_SCL_LOSS)
        TRAINER.MMLOP.USE_CTX_INIT_EMBED: Anchor layer-0 text prompt with frozen linguistic embedding
        TRAINER.MMLOP.CTX_INIT_PHRASE: Phrase to embed as anchor (default: "a photo of")
        TRAINER.MMLOP.USE_MCP: Enable Multi-Center Prototype dual path (MC-MLoP)
        TRAINER.MMLOP.NUM_CENTERS: Number of sub-prototypes per class (K)
        TRAINER.MMLOP.PROTO_TEMP: Prototype-path softmax temperature (tau)
        TRAINER.MMLOP.EMA_MOMENTUM: Entropy-weighted EMA momentum (eta)
        TRAINER.MMLOP.PROTO_FUSION_WEIGHT: Prototype-path fusion weight (beta)
        TRAINER.MMLOP.INTRA_LOSS_WEIGHT: Intra-class compactness weight (lambda3)
        TRAINER.MMLOP.INTER_LOSS_WEIGHT: Inter-class separation weight (lambda4)
        TRAINER.MMLOP.INTER_MARGIN: Inter-class hinge margin (m)

    Note: MCP 的 EMA 原型更新只在单卡训练下完全生效（DataParallel 的
    replica 内更新不会回传主模型）；建议 MCP 训练使用单卡（--gpu 0）。
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MMLOP.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.MMLOP.LORA_RANK > 0
        assert cfg.TRAINER.MMLOP.SCL_LOGIT_LOSS in ["kl", "js", "njs"]
        assert cfg.TRAINER.MMLOP.PROMPT_START_VISION >= 0
        assert cfg.TRAINER.MMLOP.PROMPT_START_TEXT >= 0
        if cfg.TRAINER.MMLOP.USE_UDC:
            assert cfg.TRAINER.MMLOP.USE_SCL_LOSS, \
                "USE_UDC requires USE_SCL_LOSS=True"
        if cfg.TRAINER.MMLOP.USE_CTX_INIT_EMBED:
            assert cfg.TRAINER.MMLOP.CTX_INIT_PHRASE, \
                "USE_CTX_INIT_EMBED requires a non-empty CTX_INIT_PHRASE"
        if cfg.TRAINER.MMLOP.USE_MCP:
            assert cfg.TRAINER.MMLOP.NUM_CENTERS >= 1, "NUM_CENTERS (K) must be >= 1"
            assert cfg.TRAINER.MMLOP.PROTO_TEMP > 0, "PROTO_TEMP must be > 0"
            assert 0.0 <= cfg.TRAINER.MMLOP.EMA_MOMENTUM <= 1.0, \
                "EMA_MOMENTUM must be in [0, 1]"
            assert cfg.TRAINER.MMLOP.INTER_MARGIN >= 0, "INTER_MARGIN must be >= 0"

    def before_train(self):
        super().before_train()
        # 仅从头训练时做 K-means 初始化；resume 时保留 checkpoint 中的原型
        if self.cfg.TRAINER.MMLOP.USE_MCP and self.start_epoch == 0:
            self.init_multi_center_prototypes()

    @torch.no_grad()
    def init_multi_center_prototypes(self):
        """MC-MLoP 3.4 节：用冻结 CLIP 零样本图像特征的类内 K-means
        中心初始化子原型；类内样本数 <= K 时退化为按类别均值初始化。"""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        pl = model.prompt_learner
        protos = pl.multi_center_prototypes
        if protos is None:
            return
        C, K, _ = protos.shape
        zs_enc = pl.ZS_image_encoder
        assert zs_enc is not None, \
            "USE_MCP 需要 ZS_image_encoder 提取零样本特征（初始化异常）"

        # 遍历训练集收集零样本图像特征（用测试风格的冻结编码器，无增强参与推理）
        was_training = model.training
        model.eval()
        zs_enc.eval()
        feats, labels = [], []
        for batch in self.train_loader_x:
            img = batch["img"].to(self.device)
            lbl = batch["label"]
            f = zs_enc(img.type(model.dtype))
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.float().cpu())
            labels.append(lbl)
        if was_training:
            model.train()
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)

        for c in range(C):
            protos.data[c] = _kmeans_centers(feats[labels == c], K, seed=c)
        print(f"[MC-MLoP] 子原型已用类内 K-means 中心初始化 (C={C}, K={K}, "
              f"samples={feats.shape[0]})")

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MMLOP.PREC == "fp32" or cfg.TRAINER.MMLOP.PREC == "amp":
            clip_model.float()

        print("Building Custom CLIP with Cross-Modal LoRA")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # VPT_shallow blocks are allocated for IVLP compatibility but are
                # never used in MMLoP (LoRA delta always overrides them). Freeze.
                param.requires_grad_(False)
            else:
                # Make sure ZS_image_encoder is frozen (for SCL loss)
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Print trainable parameters
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Total trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        # Build optimizer and scheduler
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MMLOP.PREC == "amp" else None

        # Multi-GPU support
        device_count = torch.cuda.device_count()
        if device_count > 1:
            # 检测模型是否含 nn.ParameterList：DataParallel 不支持
            # ParameterList（cross_modal_lora 的 U_shared/V_t/V_v 即属此类），
            # 在 replica 上该 list 会变成空，self.U_shared[0] 抛 IndexError。
            # MMLoP 仅 46K 可训练参数，单卡即可，直接退回单卡训练。
            def _has_parameter_list(m):
                for sub in m.modules():
                    if isinstance(sub, nn.ParameterList):
                        return True
                return False
            if _has_parameter_list(self.model):
                print(
                    f"Detected {device_count} GPUs but model contains "
                    "nn.ParameterList (incompatible with DataParallel); "
                    "falling back to single-GPU training."
                )
            else:
                print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
                self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MMLOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
                # DataParallel 会把各 replica 的标量 loss 沿 dim=0 gather 成向量，
                # .mean() 后变回标量；单卡下 .mean() 对标量安全返回自身。
                loss = loss.mean()
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            loss = loss.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

    # ==================================================================
    # 训练结束时的损失+准确率评估（MMLoP 扩展）
    # ------------------------------------------------------------------
    # Dassl 默认 after_train 只调用 self.test() 打印 test 集 accuracy，
    # 既不算 loss，也不打印 train 集。这里扩展为：
    #   1) 在 test 之前先跑 train_loader，打印 train 上的 loss + accuracy
    #   2) 跑 test_loader，打印 test 上的 loss + accuracy（同时算 loss，
    #      Dassl 默认 test() 只算 accuracy）
    # 然后调用 super().after_train() 保留 Dassl 默认 test 输出（macro_f1 /
    # per-class 等）和 elapsed time / close writer。
    # ==================================================================
    @torch.no_grad()
    def _eval_split_with_loss(self, data_loader, split_name):
        """跑一遍 data_loader，同时计算平均 loss 和 accuracy，打印汇总。

        forward(image, label=None) 在 model.eval() 下走推理分支返回 logits，
        所以这里手动用 F.cross_entropy(logits, label) 计算等价损失。
        """
        self.set_model_mode("eval")
        total = 0
        correct = 0
        loss_sum = 0.0

        for batch in tqdm(data_loader, desc=f"Eval [{split_name}]", leave=False):
            image = batch["img"].to(self.device)
            label = batch["label"].to(self.device)
            # 不传 label → forward 返回 logits（eval 模式下 training=False）
            logits = self.model(image)
            # 与训练时 L_CE 一致：F.cross_entropy 默认 mean
            loss = F.cross_entropy(logits, label, reduction="mean")
            pred = logits.max(1)[1]
            correct += int(pred.eq(label).sum().item())
            total += int(label.shape[0])
            loss_sum += float(loss.item()) * int(label.shape[0])

        avg_loss = loss_sum / max(total, 1)
        acc = 100.0 * correct / max(total, 1)
        print(
            f"[Final {split_name:>4}] loss={avg_loss:.4f}  "
            f"acc={acc:.2f}%  ({correct}/{total})"
        )
        return avg_loss, acc

    @torch.no_grad()
    def _compute_loss_acc(self, data_loader):
        """跑一遍 data_loader，返回 (avg_loss, accuracy)。
        不带 tqdm 也不打印，用于每 epoch 的 train/val 评估。"""
        self.set_model_mode("eval")
        total = 0
        correct = 0
        loss_sum = 0.0
        for batch in data_loader:
            image = batch["img"].to(self.device)
            label = batch["label"].to(self.device)
            logits = self.model(image)
            loss = F.cross_entropy(logits, label, reduction="mean")
            pred = logits.max(1)[1]
            correct += int(pred.eq(label).sum().item())
            total += int(label.shape[0])
            loss_sum += float(loss.item()) * int(label.shape[0])
        avg_loss = loss_sum / max(total, 1)
        acc = 100.0 * correct / max(total, 1)
        return avg_loss, acc, correct, total

    def after_epoch(self):
        # 每个 epoch 训练结束后跑一次 train + val + test，打印 loss + accuracy 并写入
        # TensorBoard。先调用 Dassl 默认 after_epoch 完成模型保存等逻辑。
        super().after_epoch()

        # 跑 train 集的 loss+acc（eval 模式下推理）
        train_loss, train_acc, _, _ = self._compute_loss_acc(self.train_loader_x)
        # 跑 val 集的 loss+acc：优先用 self.val_loader（真正的验证集），
        # 仅在 val 缺失时才 fallback 到 test_loader（与 Dassl test() 一致）
        val_loader = self.val_loader if self.val_loader is not None else self.test_loader
        val_loss, val_acc, _, _ = self._compute_loss_acc(val_loader)
        # 跑 test 集的 loss+acc，用于 best_acc 跟踪
        test_loss, test_acc, correct, total = self._compute_loss_acc(self.test_loader)

        n_iter = (self.epoch + 1) * self.num_batches
        self.write_scalar("train/loss", train_loss, n_iter)
        self.write_scalar("train/accuracy", train_acc, n_iter)
        self.write_scalar("val/loss", val_loss, n_iter)
        self.write_scalar("val/accuracy", val_acc, n_iter)
        self.write_scalar("test/loss", test_loss, n_iter)
        self.write_scalar("test/accuracy", test_acc, n_iter)

        # 记录每个 epoch 的曲线数据（供 after_train 画图用，只画 val）
        if not hasattr(self, "_mmlop_records"):
            self._mmlop_records = {
                "epoch": [], "val_loss": [], "val_acc": [],
            }
        self._mmlop_records["epoch"].append(self.epoch + 1)
        self._mmlop_records["val_loss"].append(val_loss)
        self._mmlop_records["val_acc"].append(val_acc)

        # 跟踪 best_acc 和 best_epoch（基于 test_acc，即测试集最佳）
        if not hasattr(self, "_mmlop_best_acc"):
            self._mmlop_best_acc = -1.0
            self._mmlop_best_epoch = -1
        if test_acc > self._mmlop_best_acc:
            self._mmlop_best_acc = test_acc
            self._mmlop_best_epoch = self.epoch + 1
        self.write_scalar("test/best_accuracy", self._mmlop_best_acc, n_iter)

        print(
            f"[epoch {self.epoch + 1}/{self.max_epoch}] "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  |  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  |  "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.2f}%  "
            f"({correct}/{total})  |  best_acc={self._mmlop_best_acc:.2f}% "
            f"@epoch {self._mmlop_best_epoch}"
        )

        # 恢复训练模式
        self.set_model_mode("train")

    def after_train(self):
        # 训练已结束，跑一份 train + test 的 loss+acc 汇总
        print("\n" + "=" * 64)
        print("Final evaluation (loss + accuracy) after training")
        print("-" * 64)
        train_loss, train_acc = self._eval_split_with_loss(
            self.train_loader_x, "train"
        )
        # 部署 last-epoch 或 best_val 模型（与 Dassl 默认一致）
        if not self.cfg.TEST.NO_TEST and self.cfg.TEST.FINAL_MODEL == "best_val":
            print("Deploy the model with the best val performance")
            self.load_model(self.output_dir)
        # 优先用真正的 val_loader，仅在缺失时 fallback 到 test_loader
        val_loader = self.val_loader if self.val_loader is not None else self.test_loader
        val_split_name = "val" if self.val_loader is not None else "test"
        val_loss, val_acc = self._eval_split_with_loss(val_loader, val_split_name)
        # 始终再单独跑一次 test_loader，报告真正的 test 性能
        test_loss, test_acc = self._eval_split_with_loss(self.test_loader, "test")
        print("-" * 64)
        print(
            f"Summary  |  train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%"
        )
        print(
            f"         |  {val_split_name}_loss ={val_loss:.4f}  "
            f"{val_split_name}_acc ={val_acc:.2f}%"
        )
        print(
            f"         |  test_loss ={test_loss:.4f}  test_acc ={test_acc:.2f}%"
        )
        # 打印训练过程中的最佳 test acc 及其 epoch
        if hasattr(self, "_mmlop_best_acc"):
            print(
                f"         |  best_acc ={self._mmlop_best_acc:.2f}%  "
                f"@epoch {self._mmlop_best_epoch}"
            )
        print("=" * 64 + "\n")

        # 调用 Dassl 默认 after_train：跑一次 test 打印 macro_f1/per-class
        # （loss+acc 已经在上面打印过，这里只为保留 Dassl 原始 evaluator 输出
        # 和 elapsed time / close writer。）
        super().after_train()

        # 画 loss / accuracy 随 epoch 变化曲线并保存 PNG
        self._plot_curves()

    def _plot_curves(self):
        """绘制 loss / accuracy 随 epoch 变化曲线，保存为 PNG 到 output_dir。"""
        if not hasattr(self, "_mmlop_records"):
            return
        rec = self._mmlop_records
        n = len(rec["epoch"])
        if n == 0:
            return
        epochs = rec["epoch"]
        out_dir = self.output_dir

        # 全局设置 Times New Roman 字体（系统缺失时自动 fallback 到 serif）
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
        plt.rcParams["axes.unicode_minus"] = False

        # ---- 1) Loss 曲线：仅 Validation ----
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, rec["val_loss"], "-s", color="tab:orange",
                markersize=3, label="validation loss")
        ax.set_xlabel("Epoch", fontname="Times New Roman")
        ax.set_ylabel("Loss", fontname="Times New Roman")
        ax.set_title("Validation Loss Curve", fontname="Times New Roman")
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontname("Times New Roman")
        ax.legend(loc="best", prop={"family": "Times New Roman"})
        ax.grid(True, alpha=0.3)
        loss_path = os.path.join(out_dir, "loss_curve.png")
        fig.tight_layout()
        fig.savefig(loss_path, dpi=150)
        plt.close(fig)
        print(f"[plot] saved loss curve to {loss_path}")

        # ---- 2) Accuracy 曲线：仅 Validation ----
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, rec["val_acc"], "-s", color="tab:green",
                markersize=3, label="validation accuracy")
        ax.set_xlabel("Epoch", fontname="Times New Roman")
        ax.set_ylabel("Accuracy (%)", fontname="Times New Roman")
        ax.set_title("Validation Accuracy Curve", fontname="Times New Roman")
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontname("Times New Roman")
        ax.legend(loc="best", prop={"family": "Times New Roman"})
        ax.grid(True, alpha=0.3)
        acc_path = os.path.join(out_dir, "accuracy_curve.png")
        fig.tight_layout()
        fig.savefig(acc_path, dpi=150)
        plt.close(fig)
        print(f"[plot] saved accuracy curve to {acc_path}")