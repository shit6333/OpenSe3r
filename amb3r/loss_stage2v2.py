"""
Stage-2 V2 Loss Functions
=========================
Combines geometry losses (re-predicted by Stage-2) with semantic /
instance / memory-consistency losses from Stage-2 V1.

Hierarchy:
    Stage2LossV2.forward(predictions, batch)
        ├─ compute_depth_loss        — from loss_semantic.py  (with grad loss + outlier filter)
        ├─ compute_point_loss        — from loss_semantic.py  (with grad loss + outlier filter)
        ├─ compute_camera_loss       — from loss_semantic.py
        ├─ compute_sem_align_loss    — cosine align to LSeg/CLIP
        ├─ compute_sem_memory_loss   — pull refined_sem toward memory
        ├─ compute_ins_contrastive_loss — SupCon / hinge
        └─ compute_ins_memory_loss   — pull refined_ins toward memory

'predictions' is the dict returned by AMB3RStage2V2.forward().
'batch'       is the chunk slice from ScannetppSequence (same format as Stage-1).

All geometry losses follow the same calling convention as MultitaskLoss in
loss_semantic.py:  kwargs (gradient_loss_fn, valid_range, gamma, alpha, conf_gt)
are forwarded so outlier filtering and gradient losses are active.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from amb3r.loss_semantic import (
    compute_depth_loss,
    compute_point_loss,
    compute_camera_loss,
)
from amb3r.tools.contrastive_loss import ContrastiveLoss

# Re-import semantic / instance / memory loss helpers from Stage-2 V1
sys.path.insert(0, os.path.dirname(__file__))
from loss_stage2 import (
    compute_sem_align_loss,
    compute_sem_memory_loss,
    compute_ins_contrastive_loss,
    compute_ins_memory_loss,
)


class Stage2LossV2(nn.Module):
    """
    Combined Stage-2 V2 loss, following the same calling convention as
    MultitaskLoss (loss_semantic.py).

    Geometry config dicts (passed as **kwargs to the underlying functions):
        camera : weight + loss_type
        depth  : weight + gradient_loss_fn + valid_range + gamma + alpha
        point  : weight + gradient_loss_fn + valid_range + gamma + alpha

    Semantic / instance / memory weights:
        w_sem_align    : cosine align to LSeg GT          (default 0.5)
        w_sem_memory   : semantic memory consistency      (default 1.0)
        w_ins_contrast : instance contrastive             (default 1.0)
        w_ins_memory   : instance memory consistency      (default 1.0)
    """

    def __init__(
        self,
        # ── geometry (mirrors MultitaskLoss defaults) ──────────────────────
        camera: dict = None,
        depth:  dict = None,
        point:  dict = None,
        # ── semantic / instance / memory ──────────────────────────────────
        w_sem_align:    float = 0.5,
        w_sem_memory:   float = 1.0,
        w_ins_contrast: float = 1.0,
        w_ins_memory:   float = 1.0,
    ):
        super().__init__()

        # Use same defaults as MultitaskLoss
        self.camera = camera or {"weight": 0.5,  "loss_type": "l1"}
        self.depth  = depth  or {"weight": 1.0,  "gradient_loss_fn": "grad",
                                  "valid_range": 0.98, "gamma": 1.0, "alpha": 0.2}
        self.point  = point  or {"weight": 1.0,  "gradient_loss_fn": "normal",
                                  "valid_range": 0.98, "gamma": 1.0, "alpha": 0.2}

        self.w_sem_align    = w_sem_align
        self.w_sem_memory   = w_sem_memory
        self.w_ins_contrast = w_ins_contrast
        self.w_ins_memory   = w_ins_memory

        self.contrastive_fn = ContrastiveLoss(
            inter_mode='hinge', inter_margin=0.2, normalize_feats=True
        )

    def forward(self, predictions: dict, batch: dict) -> dict:
        all_losses = {}
        objective  = predictions['semantic_feat'].new_tensor(0.0)

        # ── 1. Camera pose loss ───────────────────────────────────────────
        if ("pose_enc_list" in predictions and
                self.camera.get("weight", 0) > 0):
            cam_cfg = {k: v for k, v in self.camera.items() if k != "weight"}
            d = compute_camera_loss(predictions, batch, **cam_cfg)
            all_losses.update(d)
            objective = objective + d["loss_camera"] * self.camera["weight"]

        # ── 2. Depth loss ─────────────────────────────────────────────────
        if ("depth" in predictions and "depthmap" in batch and
                self.depth.get("weight", 0) > 0):
            depth_cfg = {k: v for k, v in self.depth.items() if k != "weight"}
            d = compute_depth_loss(predictions, batch, **depth_cfg)
            all_losses.update(d)
            depth_loss = (d["loss_conf_depth"]
                          + d["loss_reg_depth"]
                          + d["loss_grad_depth"])
            objective = objective + depth_loss * self.depth["weight"]

        # ── 3. Point loss ─────────────────────────────────────────────────
        if ("world_points" in predictions and "pts3d" in batch and
                self.point.get("weight", 0) > 0):
            point_cfg = {k: v for k, v in self.point.items() if k != "weight"}
            d = compute_point_loss(predictions, batch, **point_cfg)
            all_losses.update(d)
            point_loss = (d["loss_conf_point"]
                          + d["loss_reg_point"]
                          + d["loss_grad_point"])
            objective = objective + point_loss * self.point["weight"]

        # ── 4. Semantic alignment to LSeg GT ─────────────────────────────
        refined_sem = predictions['semantic_feat']      # (B, T, C_sem, H, W)
        lseg_gt     = predictions['_clip_feat_gt']      # (B, T, C_clip, H, W)
        inst_mask   = batch.get('instance_mask')        # (B, T, H, W) long | None

        if inst_mask is not None and self.w_sem_align > 0:
            d = compute_sem_align_loss(
                refined_sem, lseg_gt, inst_mask, weight=self.w_sem_align
            )
            all_losses.update(d)
            objective = objective + d['loss_sem_align']

        # ── 5. Semantic memory consistency ────────────────────────────────
        mem_sem  = predictions.get('mem_sem_for_loss')
        mem_mask = predictions['mem_mask']              # (B, T, 1, Hp, Wp)

        if mem_sem is not None and mem_mask.sum() > 0 and self.w_sem_memory > 0:
            B, T, C_sem, H, W = refined_sem.shape
            Hp, Wp = mem_mask.shape[-2:]
            sem_p = F.adaptive_avg_pool2d(
                refined_sem.flatten(0, 1).float(), (Hp, Wp)
            ).view(B, T, C_sem, Hp, Wp).to(refined_sem.dtype)
            d = compute_sem_memory_loss(
                sem_p, mem_sem, mem_mask, weight=self.w_sem_memory
            )
            all_losses.update(d)
            objective = objective + d['loss_sem_memory']

        # ── 6. Instance contrastive ───────────────────────────────────────
        refined_ins = predictions['instance_feat']      # (B, T, C_ins, H, W)

        if inst_mask is not None and self.w_ins_contrast > 0:
            d = compute_ins_contrastive_loss(
                refined_ins, inst_mask, self.contrastive_fn, weight=self.w_ins_contrast
            )
            all_losses.update(d)
            objective = objective + d['loss_ins_contrast']

        # ── 7. Instance memory consistency ────────────────────────────────
        mem_ins = predictions.get('mem_ins_for_loss')

        if mem_ins is not None and mem_mask.sum() > 0 and self.w_ins_memory > 0:
            B, T, C_ins, H, W = refined_ins.shape
            Hp, Wp = mem_mask.shape[-2:]
            ins_p = F.adaptive_avg_pool2d(
                refined_ins.flatten(0, 1).float(), (Hp, Wp)
            ).view(B, T, C_ins, Hp, Wp).to(refined_ins.dtype)
            d = compute_ins_memory_loss(
                ins_p, mem_ins, mem_mask, weight=self.w_ins_memory
            )
            all_losses.update(d)
            objective = objective + d['loss_ins_memory']

        all_losses['objective'] = objective
        return all_losses
