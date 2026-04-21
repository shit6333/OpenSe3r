"""
loss_stage2v4.py — Loss functions for Stage-2 V4.

Contains:
    compute_cross_view_consistency_loss   Cross-chunk sem/ins consistency
    Stage2V4Loss                          Combined loss module

Reuses from amb3r.loss_stage2:
    compute_sem_align_loss                Per-instance cosine alignment
    compute_ins_contrastive_loss          Hinge-based contrastive loss

Design:
    The loss has two categories:
    A. Per-chunk losses (computed within each chunk independently):
        - sem_align:       sem_feat ↔ LSeg/CLIP ground-truth (cosine)
        - ins_contrastive: ins_feat with instance_mask labels (hinge)
        - mem_consistency: M_readout ↔ previously stored M_queried (optional)

    B. Cross-chunk losses (enforce temporal / multi-view consistency):
        - cross_sem:  current sem_feat_patch ↔ first-visit anchor (cosine)
        - cross_ins:  current ins_feat_patch ↔ first-visit anchor (cosine)
        Instance receives higher default weight because cross-view instance
        consistency is the hardest property to learn.

    Gradient flow for cross-view loss:
        loss → sem/ins_feat_patch → frozen DPT → X_fuse → MemoryFusion
        → gamma, W_q/k/v → X_mem → DifferentiableVoxelMap → BPTT chain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from amb3r.loss_stage2 import (
    compute_sem_align_loss,
    compute_ins_contrastive_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-view consistency loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_cross_view_consistency_loss(
    sem_curr:   torch.Tensor,    # [N, sem_dim] WITH grad_fn
    ins_curr:   torch.Tensor,    # [N, ins_dim] WITH grad_fn
    sem_stored: torch.Tensor,    # [N, sem_dim] DETACHED anchor
    ins_stored: torch.Tensor,    # [N, ins_dim] DETACHED anchor
    mask:       torch.Tensor,    # [N, 1] float — 1.0 = valid target
    w_sem: float = 0.5,
    w_ins: float = 2.0,
) -> dict:
    """
    Cross-chunk / cross-view feature consistency loss.

    For each patch whose 3-D voxel was visited in a previous chunk (mask=1),
    penalise disagreement between current features and the detached first-visit
    anchor stored in CrossChunkFeatureBuffer.

    Both sem and ins use cosine distance (1 − cosine_similarity).
    Instance gets higher default weight because cross-view instance consistency
    is the primary target — the per-chunk losses already handle within-chunk
    supervision.

    Returns dict with:
        loss_cross_sem, loss_cross_sem_det
        loss_cross_ins, loss_cross_ins_det
        cross_n_matched, cross_objective
    """
    matched = mask.squeeze(1).bool()
    result: dict = {}

    # ── Semantic ──────────────────────────────────────────────────────────────
    if matched.any() and w_sem > 0:
        sc = F.normalize(sem_curr[matched].float(), dim=-1)
        st = F.normalize(sem_stored[matched].float(), dim=-1)
        sem_loss = (1.0 - (sc * st).sum(dim=-1)).mean()
        result['loss_cross_sem']     = (w_sem * sem_loss).to(sem_curr.dtype)
        result['loss_cross_sem_det'] = sem_loss.detach()
    else:
        result['loss_cross_sem']     = sem_curr.new_tensor(0.0)
        result['loss_cross_sem_det'] = sem_curr.new_tensor(0.0)

    # ── Instance (higher weight) ──────────────────────────────────────────────
    if matched.any() and w_ins > 0:
        ic = F.normalize(ins_curr[matched].float(), dim=-1)
        it = F.normalize(ins_stored[matched].float(), dim=-1)
        ins_loss = (1.0 - (ic * it).sum(dim=-1)).mean()
        result['loss_cross_ins']     = (w_ins * ins_loss).to(ins_curr.dtype)
        result['loss_cross_ins_det'] = ins_loss.detach()
    else:
        result['loss_cross_ins']     = ins_curr.new_tensor(0.0)
        result['loss_cross_ins_det'] = ins_curr.new_tensor(0.0)

    result['cross_n_matched'] = matched.sum().item()
    result['cross_objective'] = result['loss_cross_sem'] + result['loss_cross_ins']
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage2V4Loss
# ─────────────────────────────────────────────────────────────────────────────

class Stage2V4Loss(nn.Module):
    """
    Combined loss for Stage-2 V4 training.

    Per-chunk losses:
        1. sem_align       — sem_feat ↔ LSeg/CLIP GT (cosine alignment)
        2. ins_contrastive — ins_feat supervised by instance_mask (hinge)
        3. mem_consistency — M_readout ↔ M_queried (optional, stabiliser)

    Note: cross-view loss is computed separately in the training loop
    (needs CrossChunkFeatureBuffer access), not inside this module.
    """

    def __init__(
        self,
        w_sem_align:    float = 0.5,
        w_ins_contrast: float = 1.0,
        w_mem_consist:  float = 0.1,
    ):
        super().__init__()
        self.w_sem_align    = w_sem_align
        self.w_ins_contrast = w_ins_contrast
        self.w_mem_consist  = w_mem_consist

        from amb3r.tools.contrastive_loss import ContrastiveLoss
        self.contrastive_fn = ContrastiveLoss(
            inter_mode='hinge', inter_margin=0.2, normalize_feats=True,
        )

    def forward(self, predictions: dict, batch: dict) -> dict:
        sem_feat  = predictions['semantic_feat']
        ins_feat  = predictions['instance_feat']
        lseg_gt   = predictions['_clip_feat_gt']
        mem_mask  = predictions['mem_mask']
        M_readout = predictions['M_readout']
        M_queried = predictions.get('M_queried')
        inst_mask = batch.get('instance_mask')

        all_losses = {}
        objective  = sem_feat.new_tensor(0.0)

        # ── 1. Semantic alignment ─────────────────────────────────────────
        if inst_mask is not None and self.w_sem_align > 0:
            d = compute_sem_align_loss(
                sem_feat, lseg_gt, inst_mask, weight=self.w_sem_align)
            all_losses.update(d)
            objective = objective + d['loss_sem_align']

        # ── 2. Instance contrastive ───────────────────────────────────────
        if inst_mask is not None and self.w_ins_contrast > 0:
            d = compute_ins_contrastive_loss(
                ins_feat, inst_mask, self.contrastive_fn,
                weight=self.w_ins_contrast)
            all_losses.update(d)
            objective = objective + d['loss_ins_contrast']

        # ── 3. Memory consistency (optional) ──────────────────────────────
        if (M_queried is not None and self.w_mem_consist > 0
                and mem_mask.sum() > 0):
            B, T, _, Hp, Wp = mem_mask.shape
            N = Hp * Wp
            mask_flat = mem_mask.reshape(B * T, N, 1)

            cur = F.normalize(M_readout.float(), dim=-1)
            tgt = F.normalize(M_queried.detach().float(), dim=-1)
            cos_dist = 1.0 - (cur * tgt).sum(-1, keepdim=True)
            mem_loss = (cos_dist * mask_flat).sum() / mask_flat.sum().clamp(1)
            mem_loss = mem_loss.to(sem_feat.dtype)

            all_losses['loss_mem_consist']     = self.w_mem_consist * mem_loss
            all_losses['loss_mem_consist_det'] = mem_loss.detach()
            objective = objective + all_losses['loss_mem_consist']

        all_losses['objective'] = objective
        return all_losses


# ─────────────────────────────────────────────────────────────────────────────
# Instance-ID cross-chunk consistency loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_instance_id_cross_chunk_loss(
    sem_curr:           torch.Tensor,    # [N, sem_dim] WITH grad_fn
    ins_curr:           torch.Tensor,    # [N, ins_dim] WITH grad_fn
    instance_mask_flat: torch.Tensor,    # [N] long — per-patch instance IDs
    feat_buffer,                         # CrossChunkFeatureBuffer (query_by_id)
    w_sem: float = 0.5,
    w_ins: float = 2.0,
) -> dict:
    """
    Cross-chunk consistency loss using globally-consistent integer instance IDs.

    For each patch whose instance ID was seen in a previous chunk (stored in
    CrossChunkFeatureBuffer via update_by_id), penalise cosine distance between
    current features and the detached first-visit prototype.

    Complementary to compute_cross_view_consistency_loss (spatial voxel matching):
    - Voxel-based: requires 3D overlap between chunks.
    - ID-based:    uses dataset instance IDs, works even without spatial overlap.

    Both sem and ins are supervised (cosine distance).

    Returns dict with:
        loss_instid_cross_sem, loss_instid_cross_sem_det
        loss_instid_cross_ins, loss_instid_cross_ins_det
        instid_n_matched, instid_objective
    """
    sem_stored, ins_stored, matched = feat_buffer.query_by_id(
        instance_mask_flat, sem_curr.device)

    zero = sem_curr.new_tensor(0.0)
    result: dict = {}

    # ── Semantic ──────────────────────────────────────────────────────────────
    if matched.any() and w_sem > 0:
        sc = F.normalize(sem_curr[matched].float(), dim=-1)
        st = F.normalize(sem_stored[matched].float(), dim=-1)
        sem_loss = (1.0 - (sc * st).sum(dim=-1)).mean()
        result['loss_instid_cross_sem']     = (w_sem * sem_loss).to(sem_curr.dtype)
        result['loss_instid_cross_sem_det'] = sem_loss.detach()
    else:
        result['loss_instid_cross_sem']     = zero
        result['loss_instid_cross_sem_det'] = zero

    # ── Instance ──────────────────────────────────────────────────────────────
    if matched.any() and w_ins > 0:
        ic = F.normalize(ins_curr[matched].float(), dim=-1)
        it = F.normalize(ins_stored[matched].float(), dim=-1)
        ins_loss = (1.0 - (ic * it).sum(dim=-1)).mean()
        result['loss_instid_cross_ins']     = (w_ins * ins_loss).to(ins_curr.dtype)
        result['loss_instid_cross_ins_det'] = ins_loss.detach()
    else:
        result['loss_instid_cross_ins']     = zero
        result['loss_instid_cross_ins_det'] = zero

    result['instid_n_matched'] = int(matched.sum().item())
    result['instid_objective'] = result['loss_instid_cross_sem'] + result['loss_instid_cross_ins']
    return result
