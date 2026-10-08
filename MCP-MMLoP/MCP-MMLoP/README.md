# MCP-MMLoP: Multi-Center Prototype-Enhanced Multimodal Low-Rank Prompting for Phase-Resolved Partial Discharge Recognition

> **Anonymous code release for double-blind peer review.** Author-identifying
> information has been removed; the bibliographic citation will be added after
> the review period.

This repository contains the official PyTorch implementation of **MCP-MMLoP**, a
training-free (backbone-frozen) parameter-efficient adaptation of CLIP for
classifying six partial discharge (PD) patterns from **phase-resolved partial
discharge (PRPD)** scatter images. MCP-MMLoP adds only **20.8K trainable
parameters** to the frozen ViT-B/16 backbone and reaches **98.9% overall
accuracy / 99.0% macro-F1** on the held-out test set.

---

## 1. Method overview

MCP-MMLoP combines three ideas (all backbone weights stay frozen):

1. **Multimodal low-rank prompting (MMLoP).** Deep prompts are low-rank
   factors `P = U V^T` with a *shared* basis `U` across the vision and text
   branches, injected into every transformer layer of both branches.
2. **Uniform drift correction (UDC) + self-consistency loss (SCL).** UDC
   removes the class-common drift from the prompt-aligned text embeddings;
   SCL aligns the embedding/logit distributions induced by two prompt sets.
3. **Multi-center prototypes (MCP).** Each class is represented by `K`
   normalized sub-prototypes, initialized by K-means and updated online with an
   **entropy-weighted exponential moving average**. A class score is the
   maximum cosine similarity to its sub-prototypes and is fused with the
   prompt-path score; intra-class compactness and inter-class separation
   regularizers are applied to the prototypes.

Predictions use the same dual-path rule at train and test time:

```
s_c  = cos(f_p, ĝ_c)                              # prompt path      (Eq. 4)
g_c  = max_k cos(f_p, p_{c,k})                    # prototype path   (Eq. 7)
ℓ_c  = (1/τ) · (s_c + β · g_c)                    # fused logit      (Eq. 13-14)
y*   = argmax_c ℓ_c
```

See **Section 10** for a paper-equation → code-location map.

---

## 2. Key results

**Table 1: Full-supervision comparison** (paper Table 1, mean over 5 seeds, %).

| Method | Params | Tip | Susp. | Surf. | AirGap | Metal | Normal | OA | macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| Zero-shot CLIP | 0 | 89.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 20.1 | 5.7 |
| CoOp | 2.5 K | 97.4 | 88.9 | 84.6 | 80.0 | 79.5 | 96.3 | 87.9 | 88.4 |
| CoCoOp | 35.4 K | 100.0 | 83.3 | 80.8 | 92.0 | 89.7 | 96.3 | 91.4 | 90.6 |
| MaPLe | 3.56 M | 94.9 | 100.0 | 100.0 | 96.0 | 92.3 | 100.0 | 96.6 | 97.2 |
| PromptSRC | 46.08 K | 100.0 | 100.0 | 96.2 | 92.0 | 89.7 | 100.0 | 96.0 | 96.2 |
| MMA | 0.58 M | 94.9 | 100.0 | 100.0 | 96.0 | 100.0 | 100.0 | 98.3 | 98.6 |
| CoPrompt | 3.82 M | 97.4 | 100.0 | 100.0 | 92.0 | 94.9 | 100.0 | 97.1 | 97.5 |
| CLIP-LoRA | 0.18 M | 94.9 | 100.0 | 96.2 | 92.0 | 94.9 | 100.0 | 96.0 | 96.5 |
| Block-LoRA | 0.23 M | 94.9 | 100.0 | 96.2 | 96.0 | 97.4 | 100.0 | 97.1 | 97.5 |
| Comp-LoRA | 0.18 M | 94.9 | 100.0 | 96.2 | 100.0 | 97.4 | 100.0 | 97.7 | 98.1 |
| MMLoP | 11.56 K | 100.0 | 100.0 | 96.2 | 100.0 | 92.3 | 100.0 | 97.7 | 98.1 |
| **MCP-MMLoP (Ours)** | **20.8 K** | **100.0** | **100.0** | **96.2** | **100.0** | **97.4** | **100.0** | **98.9** | **99.0** |

MCP-MMLoP improves OA by **+1.2 pp** and macro-F1 by **+0.9 pp** over the MMLoP
baseline, while adding only **9.2 K** parameters (0.0092 M). The improvement over
the second-best baseline (MMA, 98.3% OA) is statistically significant (p < 0.05,
paired t-test). The largest per-class gain is on **Metal Discharge**
(92.3% → 97.4%), a class morphologically similar to Air Gap Discharge.

**Table 2: Ablation study** (paper Table 2, mean over 5 seeds, %).

| Variant | Tip | Susp. | Surf. | AirGap | Metal | Normal | OA | macro-F1 |
|---|---|---|---|---|---|---|---|---|
| Baseline (MMLoP, no prototypes) | 100.0 | 100.0 | 96.2 | 100.0 | 92.3 | 100.0 | 97.7 | 98.1 |
| w/ L_intra only | 100.0 | 100.0 | 92.3 | 96.0 | 87.2 | 100.0 | 95.4 | 95.5 |
| w/ L_inter only | 100.0 | 100.0 | 92.3 | 92.0 | 89.7 | 100.0 | 95.4 | 95.5 |
| **Full MCP-MMLoP** | **100.0** | **100.0** | **96.2** | **100.0** | **97.4** | **100.0** | **98.9** | **99.0** |

Adding either regularizer alone *without* the prototype path reduces accuracy,
confirming that the geometric constraints require prototype-path guidance to be
effective. The full system (prototype path + both regularizers) is
synergistic.

**Table 5: EMA momentum sensitivity** (paper Table 5, mean over 5 seeds, %).

| η | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 0.95 | **0.99** |
|---|---|---|---|---|---|---|---|
| OA | 96.0 | 94.8 | 95.4 | 96.6 | 96.6 | 97.7 | **98.9** |
| macro-F1 | 96.5 | 95.1 | 95.9 | 96.8 | 97.0 | 98.1 | **99.0** |

---

## 3. Repository structure

```
MCP-MMLoP/
├── train.py                          # Main entry point: registers cfg + launches Dassl trainer
├── run_mcp_mmlop.sh                  # One-click paper-protocol run (5 seeds, ViT-B/16)
├── eval_to_csv.py                    # Test a checkpoint; dumps accuracy/macro-F1/confusion matrix CSVs
├── tsne_visualization.py             # t-SNE of pre-softmax features + silhouette score (Fig. 4)
├── fewshot_experiment.py             # Train-ratio sweep via physical subsets (Fig. 3 utility)
├── make_train_subsets.py             # Build 20/40/60/80%-symlink subsets of the train set
├── hyperparam_grid_search.py         # λ_intra × λ_inter grid sweep (Table 4 utility)
├── momentum_experiment.py            # EMA momentum sweep (Table 4 utility)
│
├── trainers/
│   ├── mmlop.py                      # >>> CORE: CrossModalLoRA, VLPromptLearner,
│   │                                 #     CustomCLIP, MMLoP trainer, MCP, EMA, losses
│   ├── maple.py, promptsrc.py, coop.py, cocoop.py, independentVL.py, zsclip.py
│   └── clip_lora.py, comp_lora.py, block_lora.py, mma.py, coprompt.py   # baselines
│
├── datasets/
│   └── custom6.py                    # Custom6 dataset loader (PRPD, 6 classes, 8:1:1 split)
│
├── configs/
│   ├── datasets/custom6.yaml         # Dataset-level defaults
│   └── trainers/MMLoP/
│       ├── mcp_mmlop_vit_b16.yaml    # >>> Paper configuration (Table 4) — USE THIS
│       └── vit_b16.yaml              # Legacy development configuration (SGD/bs4/20ep)
│
├── clip/                             # Local OpenAI-CLIP package, extended with
│                                     # forward_with_lora() for low-rank prompt injection
├── Dassl.pytorch/                    # Dassl framework (installed in editable mode)
└── requirements.txt
```

---

## 4. Environment setup

```bash
# Python 3.8–3.11, CUDA-enabled PyTorch
conda create -n mcp_mmlop python=3.10 -y
conda activate mcp_mmlop

# 1) PyTorch (match your CUDA version; example for CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 2) Other dependencies
pip install -r requirements.txt

# 3) Install the bundled Dassl framework in editable mode
cd Dassl.pytorch/
python setup.py develop
cd ..
```

**CLIP ViT-B/16 weights.** Training needs the OpenAI checkpoint `ViT-B-16.pt`.
Pass it explicitly with `TRAINER.MMLOP.CLIP_MODEL_PATH` (see
`run_mcp_mmlop.sh`). If the path is omitted, the loader searches, in order,
the file name in the current directory, the project root, and
`~/.cache/clip`, and finally downloads it automatically from OpenAI when the
network allows.

**Hardware.** A single NVIDIA RTX 3090 (24 GB) is sufficient for the paper
configuration (batch size 32, ViT-B/16, AMP). Single-GPU training is
recommended: the prototype EMA update modifies the live module directly, which
is exact on one GPU but does not cross replicas under `DataParallel` (the
gradient path still works normally in either case).

---

## 5. Dataset preparation (Custom6 / PRPD)

Arrange the PRPD images as an `8:1:1` stratified split with one folder per
class. The six class folder names (Chinese) are **fixed by label index** in
`datasets/custom6.py` and must not be reordered:

```
dataset/
├── train/
│   ├── 尖刺/  ├── 悬浮/  ├── 沿面/  ├── 气隙/  ├── 颗粒/  └── 正常/
├── val/
│   └── ... (same six folders)
└── test/
    └── ... (same six folders)
```

| Label | Folder name | Paper class name            | Physical cause (paper Table 3)                     | Train / Val / Test |
|------:|-------------|-----------------------------|----------------------------------------------------|-------------------:|
| 0 | 尖刺 | Tip discharge        | Local field concentration at metal protrusion               | 307 / 38 / 39 |
| 1 | 悬浮 | Suspension discharge | Charge accumulation on floating metal parts                | 134 / 16 / 18 |
| 2 | 沿面 | Surface discharge    | Creeping discharge along oil-pressboard interface           | 196 / 24 / 26 |
| 3 | 气隙 | Air gap discharge    | Gas-void breakdown inside solid insulation                  | 200 / 25 / 25 |
| 4 | 颗粒 | Metal discharge      | PD induced by free moving metal particles                   | 307 / 38 / 39 |
| 5 | 正常 | Normal               | Background noise without insulation defect                  | 209 / 26 / 27 |
|   | **Total** |                     |                                                              | **1353 / 167 / 174** |

Notes:

- Supported extensions: `.jpg .jpeg .png .bmp .tif .tiff .webp`.
- To reproduce the paper's numbers, use the provided stratified split. If
  `val/` is empty, the loader can automatically carve 10% of `train/` as a
  validation subset (seeded); delete `dataset/custom6_preprocessed.pkl` to
  force a rescan whenever the split folders change.
- The folder names listed above are also the strings tokenized as CLIP text
  class names in the prompt `"a photo of a {class}."`. The English equivalents
  in the table are the names used throughout the paper and the result/t-SNE
  plots (`tsne_visualization.py` contains the same mapping).
- The raw PRPD dataset is not redistributed with this code because of
  laboratory data policy; the loader accepts any split organized as above.

---

## 6. Quick start (reproduces the main result)

```bash
# Default: five seeds {1..5}, GPU 0, auto-search for ViT-B-16.pt
bash run_mcp_mmlop.sh

# Or with explicit settings
GPU=0 \
CLIP_MODEL_PATH=/path/to/ViT-B-16.pt \
DATA_ROOT=dataset \
SEEDS="1 2 3 4 5" \
bash run_mcp_mmlop.sh
```

Equivalent single-run command:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --seed 1 \
    --trainer MMLoP \
    --backbone "ViT-B/16" \
    --dataset-name "Custom6" \
    --dataset-config-file configs/datasets/custom6.yaml \
    --config-file configs/trainers/MMLoP/mcp_mmlop_vit_b16.yaml \
    --output-dir output/mcp_mmloplop/seed1 \
    DATASET.NAME Custom6 DATASET.ROOT dataset \
    DATASET.NUM_SHOTS -1 DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.MMLOP.CLIP_MODEL_PATH /path/to/ViT-B-16.pt
```

Each run saves `log.txt`, training curves (`train_curve_*.png`), and
checkpoints under its output directory. Aggregate over seeds with the bundled
Dassl tool:

```bash
python Dassl.pytorch/tools/parse_test_res.py output/mcp_mmloplop --test-log
```

### Paper hyperparameters (all already set in `mcp_mmlop_vit_b16.yaml`)

| Symbol | Meaning | Value | Config key |
|---|---|---|---|
| — | Optimizer / schedule | AdamW, cosine, weight decay 5e-4 | `OPTIM.NAME/LR_SCHEDULER/WEIGHT_DECAY` |
| — | Learning rate / epochs / batch | 1e-3 / 100 / 32 | `OPTIM.LR`, `OPTIM.MAX_EPOCH`, `DATALOADER.TRAIN_X.BATCH_SIZE` |
| `T` | Prompt tokens per branch | 4 | `TRAINER.MMLOP.N_CTX_TEXT/VISION` |
| `L` | Prompted layers (shallow + deep) | 9 of 12 | `TRAINER.MMLOP.PROMPT_DEPTH_TEXT/VISION` |
| `r` | LoRA rank | 1 | `TRAINER.MMLOP.LORA_RANK` |
| `K` | Sub-prototypes per class | 3 | `TRAINER.MMLOP.NUM_CENTERS` |
| `β` | Prototype fusion weight (Eq. 13) | 0.5 | `TRAINER.MMLOP.PROTO_FUSION_WEIGHT` |
| `τ` | Temperature | 0.07 | CLIP `logit_scale` for classification; `TRAINER.MMLOP.PROTO_TEMP` for EMA entropies |
| `η` | EMA momentum (Eq. 10) | 0.99 | `TRAINER.MMLOP.EMA_MOMENTUM = 0.01` (update coefficient α = 1 − η) |
| `λ1 / λ2` | SCL text / image weights | 25 / 10 | `TEXT_LOSS_WEIGHT / IMAGE_LOSS_WEIGHT` |
| `λ_intra / λ_inter` | Prototype regularizers | 5.0 / 5.0 | `INTRA_LOSS_WEIGHT / INTER_LOSS_WEIGHT` |
| `m` | Inter-class hinge margin | 0.2 | `INTER_MARGIN` |

**Augmentation (paper Sec. 4.2).** Images are resized to 224×224 and
normalized with CLIP statistics. Horizontal/vertical flipping is **disabled**
because the phase-angle axis of PRPD patterns is physically directional;
training applies mild brightness jitter, Gaussian noise (σ = 0.01), and
rotation by at most 5°. This pipeline is encoded in the `INPUT.TRANSFORMS`
field of `mcp_mmlop_vit_b16.yaml` (`random_rotation` is an additive option in
the bundled `Dassl.pytorch/dassl/data/transforms/transforms.py`).

---

## 7. Evaluation and analysis utilities

**Checkpoint evaluation with CSV export** (accuracy, macro-F1, per-class
accuracy, per-sample predictions, normalized and count confusion matrices):

```bash
python eval_to_csv.py \
    --seed 1 --trainer MMLoP --backbone "ViT-B/16" \
    --dataset-name "Custom6" \
    --config-file configs/trainers/MMLoP/mcp_mmlop_vit_b16.yaml \
    --model-dir output/mcp_mmloplop/seed1 --load-epoch 100 \
    --csv-path output/mcp_mmloplop/seed1/test_results.csv \
    DATASET.NAME Custom6 DATASET.ROOT dataset DATASET.NUM_SHOTS -1 \
    DATASET.SUBSAMPLE_CLASSES all TRAINER.MMLOP.USE_MCP True \
    TRAINER.MMLOP.NUM_CENTERS 3 TRAINER.MMLOP.CLIP_MODEL_PATH /path/to/ViT-B-16.pt
```

**t-SNE visualization** of the pre-softmax L2-normalized features (Fig. 4),
together with a silhouette score computed in the original high-dimensional
space:

```bash
python tsne_visualization.py --model-dir output/mcp_mmloplop/seed1 \
    --config-file configs/trainers/MMLoP/mcp_mmlop_vit_b16.yaml \
    --save-prefix output/mcp_mmloplop/seed1/tsne \
    DATASET.ROOT dataset TRAINER.MMLOP.USE_MCP True TRAINER.MMLOP.NUM_CENTERS 3
```

**Few-shot experiments (Fig. 3).** K-shot variants (4/8/16/32) follow the
standard prompt-learning protocol:

```bash
python train.py ... --config-file configs/trainers/MMLoP/mcp_mmlop_vit_b16.yaml \
    DATASET.NUM_SHOTS 4
```

The train-ratio variant (20/40/60/80% of the training set) uses physical
symlink subsets: first run `python make_train_subsets.py`, then
`python fewshot_experiment.py`.

**Ablation sweeps.** `hyperparam_grid_search.py` (λ_intra × λ_inter) and
`momentum_experiment.py` (η) reproduce the hyperparameter analyses of Table 4.
These utility scripts were written for the authors' Linux servers: edit the
`GPU_ID` and `TRAINER.MMLOP.CLIP_MODEL_PATH` constants at the top before
running them elsewhere.

---

## 8. Baselines and ablations

- **Zero-shot CLIP**: `--trainer ZeroshotCLIP`.
- **Prompt-learning baselines**: `CoOp`, `CoCoOp`, `MaPLe`, `PromptSRC`
  (`independentVL.py`), with configs under `configs/trainers/`.
- **Fine-tuning / adapter baselines**: `CLIP_LoRA`, `Comp_LoRA` (compound
  LoRA), `Block_LoRA`, `MMA`, `CoPrompt`.
- **MMLoP vs MCP-MMLoP (Table 2)**: MCP-MMLoP is `--trainer MMLoP` with
  `TRAINER.MMLOP.USE_MCP True`; the plain MMLoP row is the identical command
  with `USE_MCP False`.
- **Component ablations (Table 2, w/o EMA / w/o Intra / w/o Inter)**: set
  `TRAINER.MMLOP.EMA_MOMENTUM 0` (EMA off) or
  `TRAINER.MMLOP.INTRA_LOSS_WEIGHT 0` / `INTER_LOSS_WEIGHT 0`.

---

## 9. Implementation details and deviations

For full transparency, the following differences between the paper's
narrative and this released implementation should be noted:

1. **EMA parameterization.** The config value `EMA_MOMENTUM` is the per-step
   *update coefficient* α; the paper's η is the *retention* momentum, i.e.
   α = 1 − η. Paper η = 0.99 therefore corresponds to `EMA_MOMENTUM = 0.01`.
   The entropy confidence weight is implemented as `w_c = 1 − H_c/log C`
   (a monotonic, [0,1]-bounded analogue of the paper's normalized `1/H_c`),
   and the EMA is applied per morphology-cluster mean inside each mini-batch
   rather than per sample; see the detailed comment block above
   `CustomCLIP._ema_update_prototypes` in `trainers/mmlop.py`.
2. **Intra-class loss.** Implemented with the squared Euclidean distance on
   unit-normalized features, `‖f − p‖² = 2(1 − cos(f,p))`, which is the
   paper's `1 − g_{y_n}` up to a constant factor absorbed into λ_intra.
3. **Optimizer parameter groups.** The paper text mentions two learning rates
   (1e-3 / 1e-4) for prompt and prototype parameters. This release uses one
   AdamW optimizer with a single learning rate (1e-3) over all trainable
   parameters; the separate prototype learning rate is not implemented.
4. **Early stopping.** Training runs for a fixed `MAX_EPOCH` (100 in the paper
   config). The early-stopping rule (patience 10) mentioned in the paper is not
   implemented in this release; the checkpoint at the final epoch is evaluated.
5. **Two config files.** `configs/trainers/MMLoP/mcp_mmlop_vit_b16.yaml` is
   the paper-faithful configuration. The older `vit_b16.yaml` and the top-level
   `*.sh` / sweep scripts retain development-era settings (SGD, batch size 4,
   70 epochs, hard-coded server paths such as `/mnt/sdb/...`) and are kept only
   for completeness; please use `run_mcp_mmlop.sh` for reproduction.
6. **Single-GPU runs.** The low-rank factors (`U_shared`, `U_v/U_t`,
   `V_v/V_t`) are stored as `nn.ParameterList` modules, which
   `nn.DataParallel` cannot gather (the trainer raises an explicit error in
   that case); please use one GPU per run, as in the paper. The prototype bank
   itself is a single trainable `nn.Parameter` of shape `(C, K, d)` and is
   saved in checkpoints normally. Note also that the in-place EMA update is
   only exact on a single device (the SGD gradient path is unaffected
   regardless).

---

## 10. Paper equation → code map

| Paper element | Implementation |
|---|---|
| Eq. 1, low-rank prompts `P = U V^T`, shared `U` | `CrossModalLoRA` in `trainers/mmlop.py` |
| Prompted token stream (Sec. 3.2) | `clip/model.py`: `VisionTransformer.forward_with_lora`, `Transformer.forward_with_lora` |
| Eq. 3, uniform drift correction (UDC) | `CustomCLIP.forward`, `use_udc` block |
| Eq. 4, prompt-path score `s_c` | `CustomCLIP.forward`, `image_features @ text_features.T` |
| Eq. 5–6, multi-center prototypes + K-means init | `VLPromptLearner.multi_center_prototypes`, `MMLoP.init_multi_center_prototypes`, `_kmeans_centers` |
| Eq. 7, max-similarity aggregation `g_c` | `CustomCLIP.forward`, `proto_sim.max(dim=2)` |
| Eq. 8–10, entropy-weighted EMA | `CustomCLIP._ema_update_prototypes` |
| Eq. 11, intra-class compactness | `CustomCLIP._proto_losses` (`loss_intra`) |
| Eq. 12, inter-class separation | `CustomCLIP._proto_losses` (`loss_inter`) |
| Eq. 13–14, dual-path fusion + temperature | `CustomCLIP.forward`, fused `logits` |
| Eq. 16, SCL loss (L1 + sym. KL) | `CustomCLIP.forward`, `use_scl_loss` block |
| Algorithm 1 | `CustomCLIP.forward` + `MMLoP.forward_backward` |
| Sec. 4.1, dataset / split | `datasets/custom6.py` |

---

## 11. Acknowledgements

This code builds on publicly available prompt-learning frameworks and code
releases: CoOp/CoCoOp, MaPLe, PromptSRC, the Dassl.pytorch toolbox, and
OpenAI CLIP. We thank their authors for releasing their work.

## License

Research code, provided for peer review under the terms of the MIT license
(`Dassl.pytorch` retains its original license).
