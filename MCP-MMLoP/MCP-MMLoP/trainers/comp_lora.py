"""
Comp-LoRA: Complementary Subspace Low-Rank Adaptation for CLIP.

Baseline implementation for comparison with MMLoP.

Reference:
    "Complementary Subspace Low-Rank Adaptation of Vision-Language Models
    for Few-Shot Classification", Wang et al., arXiv:2501.15040.

Method:
    LoRA on Q, K, V projections, but constrained to the complementary subspace
    of each frozen weight matrix W = U S V^T:

        delta_W = (I - U_p U_p^T) @ B @ A @ (I - V_p^T V_p)

    where V_p (right) and U_p (left) are the top-p principal singular vectors.
    This prevents the update from overwriting CLIP's principal learned directions.

    delta = 0 at init (B initialised to zeros).
    Applied to Q, K, V inside each MultiheadAttention block in both encoders,
    following the paper's "linear layers of multi-head attention modules."
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


class CompLoRAMHA(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention that applies Comp-LoRA to
    the Q, K, V projections.

    For each projection weight W (shape d×d):
        W_eff = W + (I - U_p U_p^T) @ B @ A @ (I - V_p^T V_p) * scaling

    where V_p = top-n_principal right singular vectors of W (projects input
    to complementary subspace) and U_p = top-n_principal left singular vectors
    (projects output to complementary subspace).
    """

    def __init__(self, mha: nn.MultiheadAttention, r: int, alpha: float,
                 n_principal: int, project_output: bool = True):
        super().__init__()
        d = mha.embed_dim
        dtype = mha.in_proj_weight.dtype

        self.embed_dim = d
        self.num_heads = mha.num_heads
        self.dropout = mha.dropout   # float, not nn.Dropout
        self.project_output = project_output
        self.scaling = alpha / r

        # Freeze and store original weights as buffers (moved with .to(device))
        self.register_buffer("in_proj_weight", mha.in_proj_weight.detach())
        if mha.in_proj_bias is not None:
            self.register_buffer("in_proj_bias", mha.in_proj_bias.detach())
        else:
            self.in_proj_bias = None

        # Output projection stays frozen as a submodule
        self.out_proj = mha.out_proj
        for p in self.out_proj.parameters():
            p.requires_grad_(False)

        # SVD for each of Q, K, V — precompute (d,d) projection matrices as buffers
        # P_in = I - V_p^T V_p  (projects input to complement of top-p right singular space)
        # P_out = I - U_p U_p^T (projects output to complement of top-p left singular space)
        # Using full (d,d) buffers means each trainable param appears exactly once in forward.
        # no_grad: mha.in_proj_weight has requires_grad=True at init; without this, P_in/P_out
        # carry a grad_fn chain through SVD. After the first backward() frees those saved
        # intermediates, the second forward reuses P_in and backward fails with "graph freed".
        with torch.no_grad():
            W_qkv = mha.in_proj_weight.float()
            eye = torch.eye(d, dtype=torch.float32)
            for idx, name in enumerate(["q", "k", "v"]):
                W = W_qkv[idx * d : (idx + 1) * d]        # (d, d)
                U, _, Vh = torch.linalg.svd(W, full_matrices=False)
                Vp = Vh[:n_principal]                       # (n_p, d)
                P_in = eye - Vp.t() @ Vp                   # (d, d)
                self.register_buffer(f"P_in_{name}", P_in.to(dtype))
                if project_output:
                    Up = U[:, :n_principal]                 # (d, n_p)
                    P_out = eye - Up @ Up.t()              # (d, d)
                    self.register_buffer(f"P_out_{name}", P_out.to(dtype))

        # LoRA parameters — B = 0 at init so delta_W = 0
        for name in ["q", "k", "v"]:
            A = nn.Parameter(torch.empty(r, d, dtype=dtype))
            B = nn.Parameter(torch.zeros(d, r, dtype=dtype))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            setattr(self, f"lora_A_{name}", A)
            setattr(self, f"lora_B_{name}", B)

    def _effective_weight(self, idx: int, name: str) -> torch.Tensor:
        """Return W_frozen + complementary-space LoRA delta for one projection."""
        d = self.embed_dim
        W = self.in_proj_weight[idx * d : (idx + 1) * d]   # (d, d)
        A = getattr(self, f"lora_A_{name}")                  # (r, d)
        B = getattr(self, f"lora_B_{name}")                  # (d, r)
        P_in = getattr(self, f"P_in_{name}")                 # (d, d) frozen buffer

        # Each trainable tensor (A, B) appears exactly once — no diamond autograd graph.
        A_eff = A.float() @ P_in.float()                    # (r, d)
        delta = B.float() @ A_eff                           # (d, d)

        if self.project_output:
            P_out = getattr(self, f"P_out_{name}")          # (d, d) frozen buffer
            delta = P_out.float() @ delta

        return (W.float() + delta * self.scaling).to(W.dtype)

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


def replace_mha_with_comp_lora(module: nn.Module, r: int, alpha: float,
                                n_principal: int, project_output: bool = True):
    """Recursively replace every nn.MultiheadAttention with CompLoRAMHA."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, CompLoRAMHA(
                child, r=r, alpha=alpha,
                n_principal=n_principal,
                project_output=project_output,
            ))
        else:
            replace_mha_with_comp_lora(child, r=r, alpha=alpha,
                                        n_principal=n_principal,
                                        project_output=project_output)


def set_lora_trainable(model: nn.Module):
    """Freeze everything, then unfreeze only CompLoRAMHA LoRA parameters."""
    for p in model.parameters():
        p.requires_grad_(False)

    enabled = []
    for mod_name, mod in model.named_modules():
        if isinstance(mod, CompLoRAMHA):
            for proj in ["q", "k", "v"]:
                for ab in ["A", "B"]:
                    p = getattr(mod, f"lora_{ab}_{proj}")
                    p.requires_grad_(True)
                    enabled.append(f"{mod_name}.lora_{ab}_{proj}")

    print("Trainable parameters (Comp-LoRA, Q/K/V only):")
    for n in enabled:
        print("  ", n)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable: {total:,}")


class CompLoRAModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        template = cfg.TRAINER.COMP_LORA.TEXT_TEMPLATE
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
class CompLoRA(TrainerX):
    """
    Comp-LoRA: LoRA on Q/K/V attention weights, constrained to the
    complementary subspace of each pretrained weight matrix.
    Preserves CLIP's principal directions while allowing low-rank adaptation
    in the orthogonal complement.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COMP_LORA.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.COMP_LORA.RANK > 0
        assert cfg.TRAINER.COMP_LORA.N_PRINCIPAL > 0

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        prec = cfg.TRAINER.COMP_LORA.PREC
        if prec in ["fp32", "amp"]:
            clip_model.float()

        r = cfg.TRAINER.COMP_LORA.RANK
        alpha = cfg.TRAINER.COMP_LORA.ALPHA
        n_principal = cfg.TRAINER.COMP_LORA.N_PRINCIPAL
        project_output = cfg.TRAINER.COMP_LORA.PROJECT_OUTPUT

        print(f"Injecting Comp-LoRA into Q/K/V "
              f"(rank={r}, alpha={alpha}, n_principal={n_principal}, "
              f"project_output={project_output})")

        replace_mha_with_comp_lora(clip_model, r=r, alpha=alpha,
                                    n_principal=n_principal,
                                    project_output=project_output)

        self.model = CompLoRAModel(cfg, classnames, clip_model)
        set_lora_trainable(self.model)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("comp_lora", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if prec == "amp" else None

        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected, using DataParallel")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COMP_LORA.PREC

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
