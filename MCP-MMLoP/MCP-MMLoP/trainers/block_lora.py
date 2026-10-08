"""
Block-LoRA: Block Matrix based Low-Rank Adaptation for CLIP.

Baseline implementation for comparison with MMLoP.

Reference:
    "One Head Eight Arms: Block Matrix based Low Rank Adaptation for
    CLIP-based Few-Shot Learning", Zhou et al., arXiv:2501.16720.

Method:
    LoRA on Q, K, V projections with a shared down-projection A_s across
    n_blocks equal input chunks, and n_blocks independent up-projections B_i.

    Input x is split into n chunks: x = [x_0, ..., x_{n-1}], each of size d/n.
    Shared A_s ∈ R^(r × d/n), independent B_i ∈ R^(d × r).

    delta_W[:, i*chunk:(i+1)*chunk] = B_i @ A_s   for i in range(n_blocks)

    This reduces the A-side parameters by factor n vs standard LoRA while
    maintaining the same number of B-side parameters.
    delta = 0 at init (all B_i initialised to zeros).

    Default config: r=2, n_blocks=2 ("Block-LoRA(2,2)" from the paper).
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
    design_details = {"trainer": "IVLP", "vision_depth": 0, "language_depth": 0,
                      "vision_ctx": 0, "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class BlockLoRAMHA(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention that applies Block-LoRA to
    the Q, K, V projections.

    For each projection weight W (shape d×d):
        W_eff[:, i*chunk:(i+1)*chunk] = W[:, i*chunk:(i+1)*chunk] + B_i @ A_s * scaling

    where A_s (r × chunk) is shared across all n_blocks chunks and B_i (d × r)
    are independent per-chunk up-projections, all initialised to zeros.
    """

    def __init__(self, mha: nn.MultiheadAttention, r: int, alpha: float,
                 n_blocks: int):
        super().__init__()
        d = mha.embed_dim
        assert d % n_blocks == 0, \
            f"embed_dim ({d}) must be divisible by n_blocks ({n_blocks})"
        chunk = d // n_blocks
        dtype = mha.in_proj_weight.dtype

        self.embed_dim = d
        self.num_heads = mha.num_heads
        self.dropout = mha.dropout   # float
        self.n_blocks = n_blocks
        self.chunk = chunk
        self.scaling = alpha / r

        # Freeze and store original weights as buffers
        self.register_buffer("in_proj_weight", mha.in_proj_weight.detach())
        if mha.in_proj_bias is not None:
            self.register_buffer("in_proj_bias", mha.in_proj_bias.detach())
        else:
            self.in_proj_bias = None

        # Output projection stays frozen
        self.out_proj = mha.out_proj
        for p in self.out_proj.parameters():
            p.requires_grad_(False)

        # LoRA parameters for Q, K, V
        # A_s: shared down-projection (r, chunk)  — kaiming init
        # B_i: n_blocks independent up-projections (d, r) — zero init → delta=0
        for name in ["q", "k", "v"]:
            A = nn.Parameter(torch.empty(r, chunk, dtype=dtype))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            setattr(self, f"lora_A_{name}", A)

            B_list = nn.ParameterList([
                nn.Parameter(torch.zeros(d, r, dtype=dtype))
                for _ in range(n_blocks)
            ])
            setattr(self, f"lora_B_{name}", B_list)

    def _effective_weight(self, idx: int, name: str) -> torch.Tensor:
        """Return W_frozen + block-LoRA delta for one projection."""
        d = self.embed_dim
        W = self.in_proj_weight[idx * d : (idx + 1) * d]   # (d, d)
        A_s = getattr(self, f"lora_A_{name}")               # (r, chunk)
        B_list = getattr(self, f"lora_B_{name}")            # n_blocks × (d, r)

        # delta[:, i*chunk:(i+1)*chunk] = B_i @ A_s
        # torch.cat stacks block columns → (d, n_blocks*chunk) = (d, d)
        delta = torch.cat([B @ A_s for B in B_list], dim=1) # (d, d)

        return W + delta * self.scaling

    def forward(self, query, key, value, need_weights=False, attn_mask=None):
        W_fused = torch.cat([
            self._effective_weight(0, "q"),
            self._effective_weight(1, "k"),
            self._effective_weight(2, "v"),
        ], dim=0)

        return F.multi_head_attention_forward(
            query, key, value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=W_fused,
            in_proj_bias=self.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.dropout if self.training else 0.0,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )


def replace_mha_with_block_lora(module: nn.Module, r: int, alpha: float,
                                  n_blocks: int):
    """Recursively replace every nn.MultiheadAttention with BlockLoRAMHA."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, BlockLoRAMHA(
                child, r=r, alpha=alpha, n_blocks=n_blocks
            ))
        else:
            replace_mha_with_block_lora(child, r=r, alpha=alpha, n_blocks=n_blocks)


def set_lora_trainable(model: nn.Module):
    """Freeze everything, then unfreeze only BlockLoRAMHA LoRA parameters."""
    for p in model.parameters():
        p.requires_grad_(False)

    enabled = []
    for mod_name, mod in model.named_modules():
        if isinstance(mod, BlockLoRAMHA):
            for proj in ["q", "k", "v"]:
                A = getattr(mod, f"lora_A_{proj}")
                A.requires_grad_(True)
                enabled.append(f"{mod_name}.lora_A_{proj}")

                B_list = getattr(mod, f"lora_B_{proj}")
                for i, B in enumerate(B_list):
                    B.requires_grad_(True)
                    enabled.append(f"{mod_name}.lora_B_{proj}[{i}]")

    print("Trainable parameters (Block-LoRA, Q/K/V only):")
    for n in enabled:
        print("  ", n)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable: {total:,}")


class BlockLoRAModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        template = cfg.TRAINER.BLOCK_LORA.TEXT_TEMPLATE
        self.prompts = [template.format(name.replace("_", " ")) for name in classnames]
        self._cached_text_features = None

    def _encode_text(self, device):
        tokenized = torch.cat([clip.tokenize(p) for p in self.prompts]).to(device)
        feats = self.clip_model.encode_text(tokenized)
        return F.normalize(feats, dim=-1)

    def forward(self, image):
        device = image.device

        if self.training:
            text_features = self._encode_text(device)
            self._cached_text_features = None
        else:
            if self._cached_text_features is None:
                self._cached_text_features = self._encode_text(device)
            text_features = self._cached_text_features

        image_features = self.clip_model.encode_image(image.to(dtype=self.dtype))
        image_features = F.normalize(image_features, dim=-1)

        return self.logit_scale.exp() * image_features @ text_features.t()


@TRAINER_REGISTRY.register()
class BlockLoRA(TrainerX):
    """
    Block-LoRA trainer: shared down-projection A_s across n_blocks input chunks
    with independent up-projections B_i, applied to Q/K/V in both CLIP encoders.
    Default config matches "Block-LoRA(2,2)" from the paper (r=2, n_blocks=2).
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.BLOCK_LORA.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.BLOCK_LORA.RANK > 0
        assert cfg.TRAINER.BLOCK_LORA.N_BLOCKS > 0

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        prec = cfg.TRAINER.BLOCK_LORA.PREC
        if prec in ["fp32", "amp"]:
            clip_model.float()

        r = cfg.TRAINER.BLOCK_LORA.RANK
        alpha = cfg.TRAINER.BLOCK_LORA.ALPHA
        n_blocks = cfg.TRAINER.BLOCK_LORA.N_BLOCKS

        print(f"Injecting Block-LoRA into Q/K/V "
              f"(rank={r}, alpha={alpha}, n_blocks={n_blocks})")

        replace_mha_with_block_lora(clip_model, r=r, alpha=alpha, n_blocks=n_blocks)

        self.model = BlockLoRAModel(cfg, classnames, clip_model)
        set_lora_trainable(self.model)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("block_lora", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if prec == "amp" else None

        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected, using DataParallel")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.BLOCK_LORA.PREC

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
        x = batch["img"].to(self.device)
        y = batch["label"].to(self.device)
        return x, y

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
