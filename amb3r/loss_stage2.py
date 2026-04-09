"""
Stage-2 Loss Functions
======================
All functions return a dict  {loss_name: tensor}  so they can be aggregated
and logged uniformly.

Hierarchy:
    Stage2Loss.forward(predictions, batch, mem_mask)
        ├─ compute_sem_align_loss       — cosine align to LSeg/CLIP  (from Stage-1)
        ├─ compute_sem_memory_loss      — MSE pull toward memory feature
        ├─ compute_ins_contrastive_loss — SupCon / hinge  (re-uses Stage-1 logic)
        └─ compute_ins_memory_loss      — MSE pull toward memory feature

'predictions' is the dict returned by AMB3RStage2.forward().
'batch'       is views_all from the dataloader (same format as Stage-1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from amb3r.tools.contrastive_loss import ContrastiveLoss


# ---------------------------------------------------------------------------
# Semantic alignment  (inherited from Stage-1, operates on refined_sem)
# ---------------------------------------------------------------------------

def compute_sem_align_loss(
    refined_sem: torch.Tensor,   # (B, T, C_sem, Hp, Wp)
    lseg_gt: torch.Tensor,       # (B, T, C_clip, H, W)
    inst_mask: torch.Tensor,     # (B, T, H, W)  long
    weight: float = 1.0,
) -> dict:
    """
    Per-instance cosine alignment between refined semantic features and LSeg GT.
    Mirrors compute_semantic_loss() in loss_semantic.py.
    """
    B, T, C_sem, Hp, Wp = refined_sem.shape

    clip_gt_r = F.interpolate(
        lseg_gt.flatten(0, 1).float(),
        size=(Hp, Wp), mode='bilinear', align_corners=False,
    ).view(B, T, C_sem, Hp, Wp)

    mask_r = F.interpolate(
        inst_mask.flatten(0, 1).float().unsqueeze(1),
        size=(Hp, Wp), mode='nearest',
    ).long().squeeze(1).view(B, T, Hp, Wp)

    pred = F.normalize(refined_sem.float(), dim=2)
    gt   = F.normalize(clip_gt_r.float(), dim=2)

    total_loss  = refined_sem.new_tensor(0.0)
    total_count = 0

    for b in range(B):
        for t in range(T):
            pred_bt = pred[b, t].permute(1, 2, 0).reshape(-1, C_sem)
            gt_bt   = gt[b, t].permute(1, 2, 0).reshape(-1, C_sem)
            mask_bt = mask_r[b, t].reshape(-1)

            for iid in torch.unique(mask_bt).tolist():
                if iid <= 0:
                    continue
                idx = (mask_bt == iid).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() < 2:
                    continue
                gt_proto = F.normalize(gt_bt[idx].mean(0), dim=0).detach()
                cos_loss = 1.0 - (pred_bt[idx] * gt_proto).sum(dim=1)
                total_loss  = total_loss + cos_loss.sum()
                total_count += idx.numel()

    loss = total_loss / max(total_count, 1)
    return {
        'loss_sem_align':     weight * loss,
        'loss_sem_align_det': loss.detach(),
    }


# ---------------------------------------------------------------------------
# Semantic memory consistency
# ---------------------------------------------------------------------------

def compute_sem_memory_loss(
    refined_sem: torch.Tensor,   # (B, T, C_sem, Hp, Wp)
    mem_sem: torch.Tensor,       # (B, T, C_sem, Hp, Wp)  — stop-grad already
    mem_mask: torch.Tensor,      # (B, T, 1,     Hp, Wp)  in [0, 1]
    weight: float = 1.0,
) -> dict:
    """
    Pull refined_sem toward the memory feature at voxels that have been observed.
    Only contributes loss where mem_mask > 0.
    """
    pred = F.normalize(refined_sem.float(), dim=2)
    tgt  = F.normalize(mem_sem.float(), dim=2).detach()

    # cosine distance (in [0, 2]), weighted by memory confidence
    cos_sim = (pred * tgt).sum(dim=2, keepdim=True)   # (B, T, 1, Hp, Wp)
    cos_dist = (1.0 - cos_sim) * mem_mask              # zero where unseen

    denom = mem_mask.sum().clamp(min=1.0)
    loss  = cos_dist.sum() / denom

    return {
        'loss_sem_memory':     weight * loss,
        'loss_sem_memory_det': loss.detach(),
    }


# ---------------------------------------------------------------------------
# Instance contrastive  (re-uses existing ContrastiveLoss)
# ---------------------------------------------------------------------------

def compute_ins_contrastive_loss(
    refined_ins: torch.Tensor,   # (B, T, C_ins, Hp, Wp)
    inst_mask: torch.Tensor,     # (B, T, H, W)  long
    contrastive_fn: ContrastiveLoss,
    weight: float = 1.0,
) -> dict:
    """
    Same logic as compute_instance_loss() in loss_semantic.py but operates on
    refined instance features.
    """
    B, T, C_ins, Hp, Wp = refined_ins.shape

    mask_r = F.interpolate(
        inst_mask.flatten(0, 1).float().unsqueeze(1),
        size=(Hp, Wp), mode='nearest',
    ).long().squeeze(1).view(B, T, Hp, Wp)

    intra = contrastive_fn.scene_intra_loss(refined_ins, mask_r)
    inter = contrastive_fn.scene_inter_loss(refined_ins, mask_r)
    loss  = intra + inter

    return {
        'loss_ins_contrast':     weight * loss,
        'loss_ins_intra_det':    intra.detach(),
        'loss_ins_inter_det':    inter.detach(),
    }


# ---------------------------------------------------------------------------
# Instance memory consistency
# ---------------------------------------------------------------------------

def compute_ins_memory_loss(
    refined_ins: torch.Tensor,   # (B, T, C_ins, Hp, Wp)
    mem_ins: torch.Tensor,       # (B, T, C_ins, Hp, Wp)
    mem_mask: torch.Tensor,      # (B, T, 1,     Hp, Wp)
    weight: float = 1.0,
) -> dict:
    """
    Pull refined_ins toward memory feature where the voxel has been observed.
    Uses cosine distance (works for L2-normalised features).
    """
    pred = F.normalize(refined_ins.float(), dim=2)
    tgt  = F.normalize(mem_ins.float(), dim=2).detach()

    cos_dist = (1.0 - (pred * tgt).sum(dim=2, keepdim=True)) * mem_mask

    denom = mem_mask.sum().clamp(min=1.0)
    loss  = cos_dist.sum() / denom

    return {
        'loss_ins_memory':     weight * loss,
        'loss_ins_memory_det': loss.detach(),
    }


# ---------------------------------------------------------------------------
# Stage2Loss module
# ---------------------------------------------------------------------------

class Stage2Loss(nn.Module):
    """
    Aggregated Stage-2 loss.

    Loss weights (all configurable):
        w_sem_align   : semantic alignment to LSeg GT     (default 0.5)
        w_sem_memory  : semantic memory consistency       (default 1.0)
        w_ins_contrast: instance contrastive              (default 1.0)
        w_ins_memory  : instance memory consistency       (default 1.0)

    forward() input:
        predictions  — output dict of AMB3RStage2.forward()
        batch        — views_all dict (contains 'instance_mask', etc.)
    """

    def __init__(
        self,
        w_sem_align:    float = 0.5,
        w_sem_memory:   float = 1.0,
        w_ins_contrast: float = 1.0,
        w_ins_memory:   float = 1.0,
    ):
        super().__init__()
        self.w_sem_align    = w_sem_align
        self.w_sem_memory   = w_sem_memory
        self.w_ins_contrast = w_ins_contrast
        self.w_ins_memory   = w_ins_memory

        self.contrastive_fn = ContrastiveLoss(
            inter_mode='hinge', inter_margin=0.2, normalize_feats=True
        )

    def forward(self, predictions: dict, batch: dict) -> dict:
        refined_sem = predictions['semantic_feat']      # (B, T, C_sem, Hp, Wp)
        refined_ins = predictions['instance_feat']      # (B, T, C_ins, Hp, Wp)
        lseg_gt     = predictions['_clip_feat_gt']      # (B, T, C_clip, H, W)
        mem_mask    = predictions['mem_mask']            # (B, T, 1, Hp, Wp)
        # mem_sem / mem_ins from the voxel query stored during forward
        # (re-accessed via lseg_patch + stage1 keys for memory fields)
        mem_sem     = predictions.get('mem_sem_for_loss')   # may be None 1st chunk
        mem_ins     = predictions.get('mem_ins_for_loss')

        inst_mask   = batch['instance_mask']            # (B, T, H, W) long

        all_losses = {}
        objective  = refined_sem.new_tensor(0.0)

        # 1. Semantic alignment
        d = compute_sem_align_loss(
            refined_sem, lseg_gt, inst_mask, weight=self.w_sem_align
        )
        all_losses.update(d)
        objective = objective + d['loss_sem_align']

        # 2. Semantic memory consistency (skip on first chunk — no memory)
        if mem_sem is not None and mem_mask.sum() > 0:
            d = compute_sem_memory_loss(
                refined_sem, mem_sem, mem_mask, weight=self.w_sem_memory
            )
            all_losses.update(d)
            objective = objective + d['loss_sem_memory']

        # 3. Instance contrastive
        d = compute_ins_contrastive_loss(
            refined_ins, inst_mask, self.contrastive_fn, weight=self.w_ins_contrast
        )
        all_losses.update(d)
        objective = objective + d['loss_ins_contrast']

        # 4. Instance memory consistency (skip on first chunk)
        if mem_ins is not None and mem_mask.sum() > 0:
            d = compute_ins_memory_loss(
                refined_ins, mem_ins, mem_mask, weight=self.w_ins_memory
            )
            all_losses.update(d)
            objective = objective + d['loss_ins_memory']

        all_losses['objective'] = objective
        return all_losses
