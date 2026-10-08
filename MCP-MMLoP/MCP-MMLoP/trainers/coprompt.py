"""
CoPrompt: Consistency-regularized Prompt Tuning (ICLR 2024).

Reference:
    "CoPrompt: Prompt-based Consistency Learning for Vision-Language Models"
    arXiv:2306.01195.

Method:
    MaPLe compound prompts (N_CTX=2, PROMPT_DEPTH=9) + post-hoc feature adapters
    applied to encoder outputs + consistency regularisation from a frozen CLIP teacher.

    Feature adapter: Linear(d → d//4) → ReLU → Linear(d//4 → d) → ReLU
    Adapted features: (1 - m) * z + m * adapter(z)
        image_adapter momentum m=0.1, text_adapter momentum m=0.2

    Loss: L = L_ce + λ * L_cc
        L_cc = 2
               - cos(frozen_text(std_prompts), text_adapter(frozen_text(alt_prompts)))
               - cos(frozen_image(image), img_adapter(frozen_image(image)))

    NOTE: The paper uses GPT-4 generated class descriptions as alt_prompts.
    This implementation uses "a picture of a {}." as the alternative template
    (same accuracy-vs-simplicity tradeoff, identical parameter count and FLOPS).
    Replace ALT_TEMPLATE in the config to plug in GPT descriptions.
"""

import copy
import math
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


# ── CLIP loading ─────────────────────────────────────────────────────────────

def load_clip_to_cpu(cfg, use_maple_design=True):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    if use_maple_design:
        # MaPLe blocks for compound prompts
        n_ctx = cfg.TRAINER.COPROMPT.N_CTX
        design_details = {
            "trainer": "MaPLe",
            "vision_depth": 0, "language_depth": 0,
            "vision_ctx": 0, "language_ctx": 0,
            "maple_length": n_ctx,
        }
    else:
        # Plain CoOp blocks for frozen teacher (no prompt injection needed)
        design_details = {
            "trainer": "CoOp",
            "vision_depth": 0, "language_depth": 0,
            "vision_ctx": 0, "language_ctx": 0,
        }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


# ── Compound prompt learner (MaPLe-style) ────────────────────────────────────

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiModalPromptLearner(nn.Module):
    """Compound prompt learner identical to MaPLe, reading from COPROMPT config."""

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls   = len(classnames)
        n_ctx   = cfg.TRAINER.COPROMPT.N_CTX
        ctx_init = cfg.TRAINER.COPROMPT.CTX_INIT
        dtype   = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]   # text width (512 for ViT-B/16)
        vis_dim = 768                                    # visual width for ViT-B/16

        assert cfg.TRAINER.COPROMPT.PROMPT_DEPTH >= 1

        self.compound_prompts_depth = cfg.TRAINER.COPROMPT.PROMPT_DEPTH

        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt_tok = clip.tokenize(ctx_init)
            with torch.no_grad():
                emb = clip_model.token_embedding(prompt_tok).type(dtype)
            ctx_vectors = emb[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f"[CoPrompt] Initial context: \"{prompt_prefix}\"")
        print(f"[CoPrompt] Compound prompt depth: {self.compound_prompts_depth}")

        self.proj = nn.Linear(ctx_dim, vis_dim).to(dtype)
        self.ctx = nn.Parameter(ctx_vectors)

        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(n_ctx, ctx_dim))
            for _ in range(self.compound_prompts_depth - 1)
        ])
        for p in self.compound_prompts_text:
            nn.init.normal_(p, std=0.02)

        single_layer = nn.Linear(ctx_dim, vis_dim)
        self.compound_prompt_projections = _get_clones(single_layer,
                                                        self.compound_prompts_depth - 1)

        classnames = [c.replace("_", " ") for c in classnames]
        name_lens = [len(_tokenizer.encode(n)) for n in classnames]
        prompts = [prompt_prefix + " " + n + "." for n in classnames]
        tokenized = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            emb = clip_model.token_embedding(tokenized).type(dtype)

        self.register_buffer("token_prefix", emb[:, :1, :])       # SOS
        self.register_buffer("token_suffix", emb[:, 1 + n_ctx:, :])  # CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

        visual_deep = [
            layer(self.compound_prompts_text[i])
            for i, layer in enumerate(self.compound_prompt_projections)
        ]
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer         = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final            = clip_model.ln_final
        self.text_projection     = clip_model.text_projection
        self.dtype               = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        outputs = self.transformer([x, compound_prompts_deeper_text, 0])
        x = outputs[0].permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        x = x @ self.text_projection
        return x


# ── Feature adapter ──────────────────────────────────────────────────────────

class FeatureAdapter(nn.Module):
    """Bottleneck residual adapter with momentum blending."""

    def __init__(self, d: int, momentum: float):
        super().__init__()
        r = max(d // 4, 1)
        self.net = nn.Sequential(
            nn.Linear(d, r), nn.ReLU(inplace=True),
            nn.Linear(r, d), nn.ReLU(inplace=True),
        )
        self.m = momentum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.m) * x + self.m * self.net(x)


# ── Full CoPrompt model ───────────────────────────────────────────────────────

class CoPromptModel(nn.Module):
    """
    Forward pass returns either:
        training: (logits, L_ce, L_cc)   — caller sums the losses
        eval    : logits                  — plain similarity scores
    """

    def __init__(self, cfg, classnames, clip_model, clip_teacher):
        super().__init__()
        self.prompt_learner   = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder    = clip_model.visual
        self.text_encoder     = TextEncoder(clip_model)
        self.logit_scale      = clip_model.logit_scale
        self.dtype            = clip_model.dtype

        # Frozen teacher (no gradient)
        self.teacher = clip_teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Feature adapters (on normalized encoder outputs)
        vis_dim = clip_model.visual.output_dim
        txt_dim = clip_model.ln_final.weight.shape[0]
        self.image_adapter = FeatureAdapter(vis_dim, cfg.TRAINER.COPROMPT.IMG_ADAPTER_MOMENTUM)
        self.text_adapter  = FeatureAdapter(txt_dim, cfg.TRAINER.COPROMPT.TXT_ADAPTER_MOMENTUM)

        self.lmbda = cfg.TRAINER.COPROMPT.CC_LAMBDA

        # Pre-compute frozen teacher text features for two templates
        std_template = cfg.TRAINER.COPROMPT.STD_TEMPLATE
        alt_template = cfg.TRAINER.COPROMPT.ALT_TEMPLATE
        self._std_texts = [std_template.format(c.replace("_", " ")) for c in classnames]
        self._alt_texts = [alt_template.format(c.replace("_", " ")) for c in classnames]
        self._frozen_std_feats = None
        self._frozen_alt_feats = None
        self._cached_text_feats = None   # cache for eval inference

    def _teacher_text(self, texts, device):
        tokens = torch.cat([clip.tokenize(t) for t in texts]).to(device)
        with torch.no_grad():
            feats = self.teacher.encode_text(tokens)
        return F.normalize(feats.float(), dim=-1)

    def _teacher_image(self, image):
        with torch.no_grad():
            feats = self.teacher.encode_image(image.to(self.dtype))
        return F.normalize(feats.float(), dim=-1)

    def forward(self, image: torch.Tensor, label=None):
        device = image.device
        logit_scale = self.logit_scale.exp()
        tokenized_prompts = self.tokenized_prompts

        # ── student features (with compound prompts) ─────────────────────────
        prompts, shared_ctx, deep_text, deep_vision = self.prompt_learner()

        if self.training:
            text_feats = self.text_encoder(prompts, tokenized_prompts, deep_text)
            self._cached_text_feats = None
        else:
            if self._cached_text_feats is None:
                with torch.no_grad():
                    text_feats = self.text_encoder(prompts, tokenized_prompts, deep_text)
                    text_feats = F.normalize(text_feats.float(), dim=-1)
                    self._cached_text_feats = F.normalize(
                        self.text_adapter(text_feats), dim=-1
                    )
            image_feats = self.image_encoder(image.type(self.dtype), shared_ctx, deep_vision)
            image_feats = F.normalize(image_feats.float(), dim=-1)
            image_adapted = F.normalize(self.image_adapter(image_feats), dim=-1)
            return logit_scale * image_adapted @ self._cached_text_feats.t()

        text_feats  = F.normalize(text_feats.float(), dim=-1)
        image_feats = self.image_encoder(image.type(self.dtype), shared_ctx, deep_vision)
        image_feats = F.normalize(image_feats.float(), dim=-1)

        # Apply adapters
        text_adapted  = F.normalize(self.text_adapter(text_feats), dim=-1)
        image_adapted = F.normalize(self.image_adapter(image_feats), dim=-1)

        logits = logit_scale * image_adapted @ text_adapted.t()

        # ── classification loss ───────────────────────────────────────────────
        L_ce = F.cross_entropy(logits, label)

        # ── consistency loss ──────────────────────────────────────────────────
        # Text branch: frozen CLIP std features vs. text_adapter(frozen CLIP alt features)
        if self._frozen_std_feats is None:
            self._frozen_std_feats = self._teacher_text(self._std_texts, device)
            self._frozen_alt_feats = self._teacher_text(self._alt_texts, device)
        frozen_std = self._frozen_std_feats.to(device)
        frozen_alt = self._frozen_alt_feats.to(device)
        alt_adapted = F.normalize(self.text_adapter(frozen_alt), dim=-1)
        L_cc_text = 1.0 - (frozen_std * alt_adapted).sum(dim=-1).mean()

        # Image branch: frozen CLIP image features vs. img_adapter(frozen CLIP features)
        frozen_img = self._teacher_image(image)
        img_adapted_frozen = F.normalize(self.image_adapter(frozen_img), dim=-1)
        L_cc_img = 1.0 - (frozen_img * img_adapted_frozen).sum(dim=-1).mean()

        L_cc = L_cc_text + L_cc_img

        return logits, L_ce, L_cc


# ── Trainer ───────────────────────────────────────────────────────────────────

@TRAINER_REGISTRY.register()
class CoPrompt(TrainerX):
    """
    CoPrompt (ICLR 2024): compound prompts + feature adapters + consistency loss.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COPROMPT.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.COPROMPT.PROMPT_DEPTH >= 1

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg, use_maple_design=True)

        # Frozen teacher uses plain CoOp-style blocks (no MaPLe prompt injection)
        clip_teacher = load_clip_to_cpu(cfg, use_maple_design=False)

        prec = cfg.TRAINER.COPROMPT.PREC
        if prec in ["fp32", "amp"]:
            clip_model.float()
            clip_teacher.float()

        print("Building CoPrompt model")
        self.model = CoPromptModel(cfg, classnames, clip_model, clip_teacher)

        # Freeze backbone, keep prompts + adapters trainable
        print("Freezing backbone; training compound prompts and feature adapters")
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name or "image_adapter" in name or "text_adapter" in name:
                param.requires_grad_(True)
            elif "VPT" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        enabled = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print(f"Trainable: {enabled}")
        total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable params: {total:,}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CoPrompt", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if prec == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use DataParallel")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COPROMPT.PREC
        lmbda = self.cfg.TRAINER.COPROMPT.CC_LAMBDA

        if prec == "amp":
            with autocast():
                _, L_ce, L_cc = self.model(image, label)
                loss = L_ce + lmbda * L_cc
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            _, L_ce, L_cc = self.model(image, label)
            loss = L_ce + lmbda * L_cc
            self.model_backward_and_update(loss)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return {"loss": loss.item(), "L_ce": L_ce.item(), "L_cc": L_cc.item()}

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = f"model.pth.tar-{epoch}"

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            for key in ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]:
                state_dict.pop(key, None)

            print(f'Loading weights to {name} from "{model_path}" (epoch={epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
