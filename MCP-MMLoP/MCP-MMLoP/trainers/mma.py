"""
MMA: Multi-Modal Adapter for Vision-Language Models (CVPR 2024).

Reference:
    "Multi-Modal Adapter for Vision-Language Models" (CVPR 2024).

Method:
    Lightweight adapters are inserted after each transformer block in layers
    [ADAPTER_START, ADAPTER_END) of both vision and text encoders.

    Adapter(x) = x + scale * W_up( shared_proj( ReLU( W_down(x) ) ) )

    where W_down: d→r and W_up: r→d are modal-specific linear projections, and
    shared_proj: r→r is the CROSS-MODAL bottleneck shared between vision and text
    adapters at the same depth.  This sharing forces the two modalities to align
    in the same low-dimensional subspace.

    Init: W_down ~ Kaiming, W_up = 0 (so delta = 0 at start), shared_proj = I.

Architecture (ViT-B/16):
    Vision blocks: width=768, 12 layers; adapters at layers ADAPTER_START..ADAPTER_END-1
    Text  blocks:  width=512, 12 layers; adapters at layers ADAPTER_START..ADAPTER_END-1
    adapter_dim=32, scale=0.001

Only adapter weights and shared_proj parameters are trained; everything else is frozen.
"""

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


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {
        "trainer": "CoOp", "vision_depth": 0, "language_depth": 0,
        "vision_ctx": 0, "language_ctx": 0,
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class MMAAdapter(nn.Module):
    """Single-stream MMA adapter with cross-modal shared bottleneck."""

    def __init__(self, d_in: int, adapter_dim: int, shared_proj: nn.Linear,
                 scale: float = 0.001):
        super().__init__()
        self.down = nn.Linear(d_in, adapter_dim, bias=False)
        self.up = nn.Linear(adapter_dim, d_in, bias=False)
        self.shared = shared_proj  # same object for visual and text at this depth
        self.scale = scale

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (L, N, d_in) — CLIP LND layout
        h = F.relu(self.down(x))
        h = self.shared(h)
        return x + self.scale * self.up(h)


class MMABlock(nn.Module):
    """Wraps a ResidualAttentionBlock and appends an MMAAdapter."""

    def __init__(self, block: nn.Module, adapter: MMAAdapter):
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.block(x))


def inject_mma_adapters(clip_model: nn.Module, adapter_start: int, adapter_end: int,
                        adapter_dim: int, scale: float) -> nn.ModuleList:
    """
    Replace resblocks[adapter_start:adapter_end] in both visual and text
    transformers with MMABlock wrappers. Returns the list of shared projections
    (one per adapted layer) so they can be registered as model parameters.
    """
    shared_projs = nn.ModuleList()

    vis_blocks = clip_model.visual.transformer.resblocks
    txt_blocks = clip_model.transformer.resblocks

    # Infer widths from the first adapted block's MHA embed_dim
    vis_dim = vis_blocks[adapter_start].attn.embed_dim
    txt_dim = txt_blocks[adapter_start].attn.embed_dim

    for i in range(adapter_start, adapter_end):
        shared_proj = nn.Linear(adapter_dim, adapter_dim, bias=False)
        with torch.no_grad():
            # Identity init: shared_proj starts as pass-through
            nn.init.eye_(shared_proj.weight)

        vis_adapter = MMAAdapter(vis_dim, adapter_dim, shared_proj, scale)
        txt_adapter = MMAAdapter(txt_dim, adapter_dim, shared_proj, scale)

        vis_blocks._modules[str(i)] = MMABlock(vis_blocks._modules[str(i)], vis_adapter)
        txt_blocks._modules[str(i)] = MMABlock(txt_blocks._modules[str(i)], txt_adapter)

        shared_projs.append(shared_proj)

    return shared_projs


def set_mma_trainable(model: nn.Module):
    """Freeze everything; unfreeze only MMAAdapter parameters (incl. shared_proj)."""
    for p in model.parameters():
        p.requires_grad_(False)

    for mod in model.modules():
        if isinstance(mod, MMAAdapter):
            mod.down.weight.requires_grad_(True)
            mod.up.weight.requires_grad_(True)
            mod.shared.weight.requires_grad_(True)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MMA] Trainable params: {total:,}")


class MMAModel(nn.Module):
    """CLIP + MMA adapters; uses a fixed text template for classification."""

    def __init__(self, cfg, classnames, clip_model, shared_projs):
        super().__init__()
        self.clip_model = clip_model
        self.shared_projs = shared_projs  # register as submodule so params are visible
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        template = cfg.TRAINER.MMA.TEXT_TEMPLATE
        self.prompts = [template.format(c.replace("_", " ")) for c in classnames]
        self._cached_text_features = None

    def _encode_text(self, device):
        tokenized = torch.cat([clip.tokenize(p) for p in self.prompts]).to(device)
        feats = self.clip_model.encode_text(tokenized)
        return F.normalize(feats.float(), dim=-1).to(self.dtype)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        device = image.device

        if self.training:
            text_features = self._encode_text(device)
            self._cached_text_features = None
        else:
            if self._cached_text_features is None:
                self._cached_text_features = self._encode_text(device)
            text_features = self._cached_text_features

        image_features = self.clip_model.encode_image(image.to(self.dtype))
        image_features = F.normalize(image_features.float(), dim=-1).to(self.dtype)

        return self.logit_scale.exp() * image_features @ text_features.t()


@TRAINER_REGISTRY.register()
class MMA(TrainerX):
    """
    MMA: Multi-Modal Adapter (CVPR 2024).

    Inserts lightweight adapters with a cross-modal shared projection at
    layers ADAPTER_START..ADAPTER_END-1 of both vision and text encoders.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MMA.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.MMA.ADAPTER_DIM > 0
        assert 0 <= cfg.TRAINER.MMA.ADAPTER_START < cfg.TRAINER.MMA.ADAPTER_END

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        prec = cfg.TRAINER.MMA.PREC
        if prec in ["fp32", "amp"]:
            clip_model.float()

        adapter_dim   = cfg.TRAINER.MMA.ADAPTER_DIM
        adapter_start = cfg.TRAINER.MMA.ADAPTER_START
        adapter_end   = cfg.TRAINER.MMA.ADAPTER_END
        adapter_scale = cfg.TRAINER.MMA.ADAPTER_SCALE

        print(f"Injecting MMA adapters at layers [{adapter_start}, {adapter_end}) "
              f"(dim={adapter_dim}, scale={adapter_scale})")

        shared_projs = inject_mma_adapters(
            clip_model, adapter_start, adapter_end, adapter_dim, adapter_scale
        )

        self.model = MMAModel(cfg, classnames, clip_model, shared_projs)
        set_mma_trainable(self.model)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("mma", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if prec == "amp" else None

        if torch.cuda.device_count() > 1:
            print("Multiple GPUs detected, using DataParallel")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.MMA.PREC

        if prec == "amp":
            with autocast():
                loss = F.cross_entropy(self.model(image), label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss = F.cross_entropy(self.model(image), label)
            self.model_backward_and_update(loss)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return {"loss": loss.item()}

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
            print(f'Loading weights to {name} from "{model_path}" (epoch={epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
