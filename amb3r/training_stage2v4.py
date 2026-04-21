"""
train_stage2v4.py — Stage-2 V4 training with periodic evaluation + PLY export.

Frozen  : Stage-1 backbone (VGGT + LSeg + DPT heads)
Trainable: MemoryFusionModule, MemoryTokenizer, LearnableNullToken

Training strategy: Chunked Unrolled BPTT
    - Process sequence in chunks (e.g. 4 frames each)
    - Accumulate ALL chunk losses (including cross-view consistency)
    - Single backward() after full sequence
    - DifferentiableVoxelMap preserves grad_fn across chunks (no detach)

Evaluation: every --eval_freq epochs
    - Full-sequence forward (no grad)
    - PLY export: geometry, semantic PCA, semantic text-match, instance PCA
    - Per-batch metric logging

Usage:
    torchrun --nproc_per_node=1 .claude/sandbox/train_stage2v4.py \\
        --stage1_ckpt outputs/exp_stage1_wo_lora/checkpoint-best.pth \\
        --seq_len 16 --chunk_size 4 --stride 2 \\
        --batch_size 1 --accum_iter 4 --epochs 30 --lr 3e-4 \\
        --eval_freq 5 --use_checkpoint
"""

import os
import sys
import math
import time
import argparse
import datetime
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

# ── project paths ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PROJ)
sys.path.append(os.path.join(_PROJ, 'thirdparty'))

import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler

from amb3r.model_stage1_wo_lora import AMB3RStage1FullFT
from amb3r.model_stage2v4 import AMB3RStage2V4
from amb3r.loss_stage2v4 import (
    Stage2V4Loss,
    compute_cross_view_consistency_loss,
    compute_instance_id_cross_chunk_loss,
    compute_instance_id_cross_chunk_push_loss,
)
from amb3r.memory_stage2v4 import CrossChunkFeatureBuffer

from amb3r.datasets import *
from amb3r.datasets.scannetpp_sequence import ScannetppSequence

# Coordinate normalization utilities (model output is non-metric local frame)
from vggt.train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from moge.moge.train.losses import scale_invariant_alignment
from vggt.train_utils.general import check_and_fix_inf_nan


PATCH_SIZE = 14  # VGGT patch size — matches amb3r.model_stage2v4.PATCH_SIZE


def _pixel_broadcast_update(
    voxel_store,
    M_content,      # [B*T*N_patch, D]
    W_conf,         # [B*T*N_patch, 1] or None
    pts_pixel,      # [B, T, H, W, 3]  (global frame)
    mem_mode,
    downsample=1,
):
    """
    Broadcast patch-level memory content to pixel-level xyz and update the
    voxel store. Fixes the "one patch stores to one voxel" sparsity problem:
    a 14x14 patch now writes into every voxel its pixels fall in, and
    scatter_mean inside DetachedVoxelStore.update() aggregates duplicates
    in the same voxel.

    Only changes the UPDATE path; the model still outputs patch-level
    tokens and query() remains patch-level (hits are now dense because
    the store is dense).

    Args:
        voxel_store: DifferentiableVoxelMap (mode 0) or DetachedVoxelStore (1-3)
        M_content  : patch-level features [B*T*N_patch, D]
        W_conf     : optional per-patch confidence [B*T*N_patch, 1]
        pts_pixel  : pixel-level xyz [B, T, H, W, 3] (global frame — same as
                     feat_buffer uses)
        mem_mode   : 0=BPTT, 1-3=detached
        downsample : keep every k-th pixel along H and W to reduce compute
                     (default 1 = full 14x14 per patch)
    """
    # Local import to avoid circular-ish overhead at module load
    from amb3r.model_stage2v4 import AMB3RStage2V4

    assert pts_pixel.ndim == 5 and pts_pixel.shape[-1] == 3, (
        f'pts_pixel must be [B,T,H,W,3], got {pts_pixel.shape}')
    B, T, H, W, _ = pts_pixel.shape
    Hp, Wp = H // PATCH_SIZE, W // PATCH_SIZE
    N_patch = Hp * Wp
    D = M_content.shape[-1]
    BT = B * T
    assert M_content.shape[0] == BT * N_patch, (
        f'M_content {tuple(M_content.shape)} vs B*T*N_patch={BT*N_patch}')

    # [B*T, D, Hp, Wp] → nearest-upsample to [B*T, D, H, W]
    M_img = M_content.reshape(BT, Hp, Wp, D).permute(0, 3, 1, 2).contiguous()
    M_pix = F.interpolate(M_img, size=(H, W), mode='nearest')

    pts_bt = pts_pixel.reshape(BT, H, W, 3)
    if downsample > 1:
        M_pix  = M_pix[:, :, ::downsample, ::downsample]
        pts_bt = pts_bt[:, ::downsample, ::downsample, :]

    M_flat   = M_pix.permute(0, 2, 3, 1).reshape(-1, D).contiguous()
    pts_flat = pts_bt.reshape(-1, 3).contiguous()

    if mem_mode == 0:
        assert W_conf is not None, 'mem_mode=0 requires W_conf'
        W_img  = W_conf.reshape(BT, Hp, Wp, 1).permute(0, 3, 1, 2).contiguous()
        W_pix  = F.interpolate(W_img, size=(H, W), mode='nearest')
        if downsample > 1:
            W_pix = W_pix[:, :, ::downsample, ::downsample]
        W_flat = W_pix.permute(0, 2, 3, 1).reshape(-1, 1).contiguous()
        AMB3RStage2V4.update_voxel_store(voxel_store, M_flat, W_flat, pts_flat, mem_mode=0)
    else:
        AMB3RStage2V4.update_voxel_store(voxel_store, M_flat, None, pts_flat, mem_mode=mem_mode)


def normalize_gt(views_all):
    """
    Normalize GT extrinsics / pts3d / depthmap in-place.

    Transforms dataset-global coordinates into a canonical local frame:
    centered at the first camera, optionally unit-scaled.
    Same as training_stage1_wo_lora.py.
    """
    (views_all['extrinsics'], _,
     views_all['pts3d'],
     views_all['depthmap'],
     views_all['scale']) = normalize_camera_extrinsics_and_points_batch(
        extrinsics=views_all['extrinsics'],
        cam_points=None,
        world_points=views_all['pts3d'],
        depths=views_all['depthmap'],
        scale_by_points=True,
        point_masks=views_all['valid_mask'],
        pred_points=None,
    )


def align_pred_to_gt(pred, views_all):
    """
    Scale-invariant alignment of predicted geometry to normalized GT.
    Returns scale_depth (used to rescale pose_enc translations).
    """
    bs, t, h, w, _ = views_all['pts3d'].shape

    pred['world_points'], _ = scale_invariant_alignment(
        pred['world_points'].view(bs, t * h, w, 3),
        views_all['pts3d'].view(bs, t * h, w, 3),
        views_all['valid_mask'].view(bs, t * h, w),
        trunc=None, detach=False,
    )
    pred['world_points'] = pred['world_points'].view(bs, t, h, w, 3)

    pred['depth'], scale_depth = scale_invariant_alignment(
        pred['depth'].repeat(1, 1, 1, 1, 3),
        views_all['depthmap'][..., None].repeat(1, 1, 1, 1, 3),
        views_all['valid_mask'],
        trunc=None, detach=False,
    )
    pred['depth'] = pred['depth'].view(bs, t, h, w, 3)[..., :1]
    return scale_depth


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args_parser():
    p = argparse.ArgumentParser('AMB3R Stage-2 V4 training', add_help=False)

    # ── model ─────────────────────────────────────────────────────────────
    p.add_argument('--stage1_ckpt', required=True)
    p.add_argument('--stage2_ckpt', default=None,
                   help='Resume Stage-2 V4 from this checkpoint')
    p.add_argument('--auto_resume', action='store_true', default=True,
                   help='Auto-resume from output_dir/checkpoint-last.pth if it exists')
    p.add_argument('--sem_dim',    type=int, default=512)
    p.add_argument('--ins_dim',    type=int, default=16)
    p.add_argument('--clip_dim',   type=int, default=512)
    p.add_argument('--mem_dim',    type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--num_heads',  type=int, default=4)
    p.add_argument('--ema_alpha',  type=float, default=0.5)
    p.add_argument('--use_checkpoint', action='store_true')

    # ── memory mode ──────────────────────────────────────────────────────
    p.add_argument('--mem_mode', type=int, default=2, choices=[0, 1, 2, 3],
                   help='Memory mode: 0=tokenizer(BPTT), 1=sem+ins, 2=full_vggt, 3=AE_bottleneck')
    p.add_argument('--stage1_5_ckpt', default=None, type=str,
                   help='Stage 1.5 checkpoint (required for mem_mode=3)')
    p.add_argument('--ae_bottleneck_dim', type=int, default=256,
                   help='Bottleneck dim of Stage 1.5 AE (for mem_mode=3)')
    p.add_argument('--pixel_broadcast_mem', action='store_true',
                   help='Broadcast patch-level memory to pixel-level xyz before '
                        'voxel update (reduces memory sparsity).')
    p.add_argument('--pixel_broadcast_downsample', type=int, default=2,
                   help='Keep every k-th pixel along H,W during broadcast update '
                        '(1=full 14x14 per patch, 2=7x7, etc.). Only used when '
                        '--pixel_broadcast_mem is set.')
    p.add_argument('--finetune_head', action='store_true',
                   help='Unfreeze sem/ins DPT heads (trained with lower LR).')
    p.add_argument('--head_lr_scale', type=float, default=0.1,
                   help='LR multiplier for DPT head params (relative to --lr).')

    # ── sequence / chunk ──────────────────────────────────────────────────
    p.add_argument('--seq_len',    type=int, default=16)
    p.add_argument('--seq_len_eval',    type=int, default=16)
    p.add_argument('--chunk_size', type=int, default=4)
    p.add_argument('--stride',     type=int, default=2)

    # ── dataset ───────────────────────────────────────────────────────────
    p.add_argument('--data_root',   default='/mnt/HDD4/ricky/data/scannetpp_arrow')
    p.add_argument('--resolution',  default='(518, 336)')
    p.add_argument('--num_workers', type=int, default=2)

    # ── voxel memory ──────────────────────────────────────────────────────
    p.add_argument('--mem_voxel_size', type=float, default=0.05)
    p.add_argument('--feat_voxel_size', type=float, default=0.01)

    # ── optimiser ─────────────────────────────────────────────────────────
    p.add_argument('--lr',            type=float, default=3e-4)
    p.add_argument('--weight_decay',  type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int,   default=1)
    p.add_argument('--min_lr',        type=float, default=1e-6)

    # ── training ──────────────────────────────────────────────────────────
    p.add_argument('--epochs',     type=int, default=30)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--accum_iter', type=int, default=4)
    p.add_argument('--amp', choices=[False, 'bf16', 'fp16'], default='bf16',
                   type=lambda x: False if x == 'False' else x)

    # ── loss weights ──────────────────────────────────────────────────────
    p.add_argument('--w_sem_align',    type=float, default=0.5)
    p.add_argument('--w_ins_contrast', type=float, default=1.0)
    p.add_argument('--w_mem_consist',  type=float, default=0.1)
    p.add_argument('--w_cross_sem',    type=float, default=0.5)
    p.add_argument('--w_cross_ins',    type=float, default=2.0)
    p.add_argument('--w_instid_cross_sem', type=float, default=0.5,
                   help='Weight for instance-ID cross-chunk semantic consistency loss')
    p.add_argument('--w_instid_cross_ins', type=float, default=2.0,
                   help='Weight for instance-ID cross-chunk instance consistency loss')
    p.add_argument('--w_instid_push_ins',   type=float, default=2.0,
                   help='Weight for cross-chunk inter-class push loss (instance)')
    p.add_argument('--w_instid_push_sem',   type=float, default=0.5,
                   help='Weight for cross-chunk inter-class push loss (semantic)')
    p.add_argument('--instid_push_margin',  type=float, default=0.5,
                   help='Hinge margin for push loss (cosine similarity threshold)')
    p.add_argument('--instid_push_max_neg', type=int,   default=64,
                   help='Max stored instance IDs sampled as negatives per step')
    p.add_argument('--use_geo_cross_consistency', action='store_true', default=False)
    p.add_argument('--use_instid_cross_consistency', action='store_true', default=True)
    p.add_argument('--cross_loss_n_samples', type=int, default=4096, 
                   help='Number of pixels to sample for cross-chunk consistency losses (0 = use all pixels)')

    # ── evaluation ────────────────────────────────────────────────────────
    p.add_argument('--eval_freq',   type=int, default=1,
                   help='Evaluate + PLY export every N epochs (0 = disable)')
    p.add_argument('--eval_ply_n',  type=int, default=1,
                   help='Export PLY for first N eval batches')

    # ── misc ──────────────────────────────────────────────────────────────
    p.add_argument('--output_dir',  default='./outputs/exp_stage2v4')
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--print_freq',  type=int, default=20)
    p.add_argument('--save_freq',   type=int, default=1)
    p.add_argument('--world_size',  default=1, type=int)
    p.add_argument('--local_rank',  default=-1, type=int)
    p.add_argument('--dist_url',    default='env://')

    return p


def get_dtype(args):
    if args.amp == 'bf16': return torch.bfloat16
    if args.amp == 'fp16': return torch.float16
    return torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# Sequence training (Chunked Unrolled BPTT)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_sequence(
    model: AMB3RStage2V4,
    criterion: Stage2V4Loss,
    views_all: dict,
    device: torch.device,
    args,
    dtype: torch.dtype,
) -> dict:
    """
    Full BPTT over one training sequence.

    Accumulates losses from ALL chunks and returns a single mean_loss tensor.
    The caller is responsible for calling mean_loss.backward().

    Returns {'mean_loss': Tensor, 'log_dict': dict} or {} if all chunks fail.
    """
    assert args.seq_len % args.chunk_size == 0
    n_chunks = args.seq_len // args.chunk_size

    voxel_map   = model.make_voxel_map()
    feat_buffer = model.make_feat_buffer(sem_dim=args.sem_dim, ins_dim=args.ins_dim)

    # ── Coordinate frame handling ─────────────────────────────────────────
    # 1. Capture RAW GT points in dataset global frame BEFORE normalization.
    #    These are used for DifferentiableVoxelMap queries because they are
    #    consistent across all chunks (same global coordinate system).
    # 2. normalize_gt() then transforms views_all in-place to a canonical
    #    local frame (centered at first camera, unit-scaled) for supervision.
    #    Sem/ins losses don't depend on coordinates, but having consistent
    #    normalized GT is required for geometry monitoring.
    gt_pts_global = views_all.get('pts3d')
    if gt_pts_global is not None:
        gt_pts_global = gt_pts_global.clone()  # [B, seq_len, H, W, 3] global frame

    normalize_gt(views_all)  # in-place: pts3d, extrinsics, depthmap → local frame

    gt_pts_all = gt_pts_global  # memory queries use global-frame coordinates

    total_loss   = torch.tensor(0.0, device=device)
    log_dict     = defaultdict(float)
    valid_chunks = 0

    for c in range(n_chunks):
        s = c * args.chunk_size
        e = s + args.chunk_size

        chunk = {
            k: (v[:, s:e].contiguous()
                if isinstance(v, torch.Tensor) and v.ndim >= 2 else v)
            for k, v in views_all.items()
        }
        gt_pts_chunk = gt_pts_all[:, s:e] if gt_pts_all is not None else None

        # ── Forward ──────────────────────────────────────────────────────
        with torch.autocast('cuda', dtype=dtype):
            predictions, M_content, W_conf, pts_flat = model.forward_chunk(
                chunk, voxel_map, pts_for_query=gt_pts_chunk)

        # ── Per-chunk loss ───────────────────────────────────────────────
        with torch.autocast('cuda', dtype=dtype):
            losses = criterion(predictions, chunk)

        chunk_loss = losses['objective']

        # ── Cross-view consistency loss (from chunk 1 onwards) ───────────
        # Buffer stores pixel-wise anchors; sample N pixels for loss
        cross_losses: dict = {}
        if args.use_geo_cross_consistency:
            if c > 0 and (args.w_cross_sem > 0 or args.w_cross_ins > 0):
                pts_pixel_q = gt_pts_chunk.reshape(-1, 3)  # [B*T*H*W, 3]
                sem_curr_flat = predictions['semantic_feat'].permute(0,1,3,4,2).reshape(-1, predictions['semantic_feat'].shape[2])
                ins_curr_flat = predictions['instance_feat'].permute(0,1,3,4,2).reshape(-1, predictions['instance_feat'].shape[2])
                N_px = pts_pixel_q.shape[0]
                if args.cross_loss_n_samples > 0 and N_px > args.cross_loss_n_samples:
                    idx = torch.randperm(N_px, device=device)[:args.cross_loss_n_samples]
                    pts_pixel_q = pts_pixel_q[idx]
                    sem_curr_flat = sem_curr_flat[idx]
                    ins_curr_flat = ins_curr_flat[idx]
                with torch.autocast('cuda', dtype=dtype):
                    sem_stored, ins_stored, cv_mask = feat_buffer.query(pts_pixel_q)
                    cross_losses = compute_cross_view_consistency_loss(
                        sem_curr=sem_curr_flat,
                        ins_curr=ins_curr_flat,
                        sem_stored=sem_stored,
                        ins_stored=ins_stored,
                        mask=cv_mask,
                        w_sem=args.w_cross_sem,
                        w_ins=args.w_cross_ins,
                    )
                chunk_loss = chunk_loss + cross_losses['cross_objective']

        # ── Instance-ID cross-chunk loss (from chunk 1 onwards) ──────────
        # Buffer stores pixel-wise anchors; sample N pixels for loss
        instid_losses: dict = {}
        if args.use_instid_cross_consistency:
            if c > 0 and (args.w_instid_cross_ins > 0 or args.w_instid_cross_sem > 0):
                inst_mask_raw = chunk.get('instance_mask')  # [B, T, H, W] int64
                if inst_mask_raw is not None:
                    inst_flat_px = inst_mask_raw.reshape(-1)  # [B*T*H*W]
                    sem_flat_px = predictions['semantic_feat'].permute(0,1,3,4,2).reshape(-1, predictions['semantic_feat'].shape[2])
                    ins_flat_px = predictions['instance_feat'].permute(0,1,3,4,2).reshape(-1, predictions['instance_feat'].shape[2])
                    N_px = inst_flat_px.shape[0]
                    if args.cross_loss_n_samples > 0 and N_px > args.cross_loss_n_samples:
                        idx = torch.randperm(N_px, device=device)[:args.cross_loss_n_samples]
                        inst_flat_px = inst_flat_px[idx]
                        sem_flat_px = sem_flat_px[idx]
                        ins_flat_px = ins_flat_px[idx]
                    with torch.autocast('cuda', dtype=dtype):
                        instid_losses = compute_instance_id_cross_chunk_loss(
                            sem_curr=sem_flat_px,
                            ins_curr=ins_flat_px,
                            instance_mask_flat=inst_flat_px,
                            feat_buffer=feat_buffer,
                            w_sem=args.w_instid_cross_sem,
                            w_ins=args.w_instid_cross_ins,
                        )
                    chunk_loss = chunk_loss + instid_losses['instid_objective']

        # ── Instance-ID push loss (cross-chunk inter-class repulsion) ─────────
        push_losses: dict = {}
        if args.use_instid_cross_consistency:
            if c > 0 and (args.w_instid_push_ins > 0 or args.w_instid_push_sem > 0):
                inst_mask_raw = chunk.get('instance_mask')
                if inst_mask_raw is not None:
                    inst_flat_push = inst_mask_raw.reshape(-1)
                    sem_flat_push  = predictions['semantic_feat'].permute(0,1,3,4,2).reshape(-1, predictions['semantic_feat'].shape[2])
                    ins_flat_push  = predictions['instance_feat'].permute(0,1,3,4,2).reshape(-1, predictions['instance_feat'].shape[2])
                    if args.cross_loss_n_samples > 0:
                        N_px = inst_flat_push.shape[0]
                        if N_px > args.cross_loss_n_samples:
                            idx = torch.randperm(N_px, device=device)[:args.cross_loss_n_samples]
                            inst_flat_push = inst_flat_push[idx]
                            sem_flat_push  = sem_flat_push[idx]
                            ins_flat_push  = ins_flat_push[idx]
                    with torch.autocast('cuda', dtype=dtype):
                        push_losses = compute_instance_id_cross_chunk_push_loss(
                            sem_curr=sem_flat_push,
                            ins_curr=ins_flat_push,
                            instance_mask_flat=inst_flat_push,
                            feat_buffer=feat_buffer,
                            margin=args.instid_push_margin,
                            w_sem=args.w_instid_push_sem,
                            w_ins=args.w_instid_push_ins,
                            max_neg=args.instid_push_max_neg,
                        )
                    chunk_loss = chunk_loss + push_losses['instid_push_objective']

        if not torch.isfinite(chunk_loss):
            print(f'[Stage2V4] Non-finite loss at chunk {c}, skipping.')
            if args.mem_mode == 1:
                AMB3RStage2V4.update_voxel_store(
                    voxel_map, M_content.detach(), None,
                    pts_flat.detach(), mem_mode=1)
            elif args.pixel_broadcast_mem:
                _pixel_broadcast_update(
                    voxel_map,
                    M_content.detach(),
                    W_conf.detach() if W_conf is not None else None,
                    gt_pts_chunk,
                    mem_mode=args.mem_mode,
                    downsample=args.pixel_broadcast_downsample)
            else:
                AMB3RStage2V4.update_voxel_store(
                    voxel_map,
                    M_content.detach(),
                    W_conf.detach() if W_conf is not None else None,
                    pts_flat.detach(),
                    mem_mode=args.mem_mode)
            # Still update feat_buffer with pixel-wise features
            pts_px = gt_pts_chunk.reshape(-1, 3)
            sem_px = predictions['semantic_feat'].detach().permute(0,1,3,4,2)
            ins_px = predictions['instance_feat'].detach().permute(0,1,3,4,2)
            sem_px = sem_px.reshape(-1, sem_px.shape[-1])
            ins_px = ins_px.reshape(-1, ins_px.shape[-1])
            sem_conf_px = predictions['semantic_conf'].detach().permute(0,1,3,4,2).reshape(-1)
            ins_conf_px = predictions['instance_conf'].detach().permute(0,1,3,4,2).reshape(-1)
            feat_buffer.update(pts_px, sem_px, ins_px,
                               sem_conf=sem_conf_px, ins_conf=ins_conf_px)
            inst_skip = chunk.get('instance_mask')
            if inst_skip is not None:
                feat_buffer.update_by_id(inst_skip.reshape(-1), ins_px, sem_px,
                                         ins_conf_flat=ins_conf_px,
                                         sem_conf_flat=sem_conf_px)
            continue

        total_loss   = total_loss + chunk_loss
        valid_chunks += 1

        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                log_dict[k] += v.item()
        for k, v in cross_losses.items():
            if isinstance(v, torch.Tensor) and k.endswith('_det'):
                log_dict[k] += v.item()
        if 'cross_n_matched' in cross_losses:
            log_dict['cross_n_matched'] += cross_losses['cross_n_matched']
        for k, v in instid_losses.items():
            if isinstance(v, torch.Tensor) and k.endswith('_det'):
                log_dict[k] += v.item()
        if 'instid_n_matched' in instid_losses:
            log_dict['instid_n_matched'] += instid_losses['instid_n_matched']
        for k, v in push_losses.items():
            if isinstance(v, torch.Tensor) and k.endswith('_det'):
                log_dict[k] += v.item()
        if 'instid_push_n_pairs' in push_losses:
            log_dict['instid_push_n_pairs'] += push_losses['instid_push_n_pairs']

        # ── Memory updates ───────────────────────────────────────────────
        if args.mem_mode == 1:
            # Mode 1: M_content & pts_flat are already pixel-level
            AMB3RStage2V4.update_voxel_store(
                voxel_map, M_content, None, pts_flat, mem_mode=1)
        elif args.pixel_broadcast_mem:
            _pixel_broadcast_update(
                voxel_map, M_content, W_conf, gt_pts_chunk,
                mem_mode=args.mem_mode,
                downsample=args.pixel_broadcast_downsample)
        else:
            AMB3RStage2V4.update_voxel_store(
                voxel_map, M_content, W_conf, pts_flat, mem_mode=args.mem_mode)

        # Pixel-wise features & coords for feat_buffer storage
        # (voxelize inside buffer preserves fine-grained info for small objects)
        pts_pixel      = gt_pts_chunk.reshape(-1, 3)                              # [B*T*H*W, 3] global
        sem_pixel      = predictions['semantic_feat'].detach()                    # [B,T,C,H,W]
        ins_pixel      = predictions['instance_feat'].detach()                    # [B,T,C,H,W]
        sem_pixel      = sem_pixel.permute(0,1,3,4,2).reshape(-1, sem_pixel.shape[2])   # [B*T*H*W, sem_dim]
        ins_pixel      = ins_pixel.permute(0,1,3,4,2).reshape(-1, ins_pixel.shape[2])   # [B*T*H*W, ins_dim]
        sem_conf_pixel = predictions['semantic_conf'].detach().permute(0,1,3,4,2).reshape(-1)  # [B*T*H*W]
        ins_conf_pixel = predictions['instance_conf'].detach().permute(0,1,3,4,2).reshape(-1)  # [B*T*H*W]
        feat_buffer.update(pts_pixel, sem_pixel, ins_pixel,
                           sem_conf=sem_conf_pixel, ins_conf=ins_conf_pixel)

        # Instance-ID store: pixel-wise instance_mask + features
        inst_mask_raw_full = chunk.get('instance_mask')  # [B, T, H, W]
        if inst_mask_raw_full is not None:
            feat_buffer.update_by_id(
                inst_mask_raw_full.reshape(-1),  # [B*T*H*W] int64
                ins_pixel, sem_pixel,
                ins_conf_flat=ins_conf_pixel,
                sem_conf_flat=sem_conf_pixel)

    if valid_chunks == 0:
        voxel_map.clear()
        # feat_buffer cleared after PLY export below
        feat_buffer.clear()
        return {}

    mean_loss = total_loss / valid_chunks
    log_out   = {k: v / valid_chunks for k, v in log_dict.items()}
    return {'mean_loss': mean_loss, 'log_dict': log_out}


# ─────────────────────────────────────────────────────────────────────────────
# train_one_epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, criterion, data_loader, optimizer, device, epoch,
    loss_scaler, args, log_writer=None,
):
    model.train()
    model_without_ddp = model.module if hasattr(model, 'module') else model

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    dtype      = get_dtype(args)
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        # LR schedule
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args)

        views, views_all = batch
        for key in views_all:
            if isinstance(views_all[key], torch.Tensor):
                views_all[key] = views_all[key].to(device, non_blocking=True)

        result = train_one_sequence(
            model_without_ddp, criterion, views_all, device, args, dtype)

        if not result:
            continue

        mean_loss = result['mean_loss']
        if not torch.isfinite(mean_loss):
            print(f'[Stage2V4] Non-finite sequence loss, skipping.')
            continue

        loss_scaled = mean_loss / accum_iter
        loss_scaler(
            loss_scaled, optimizer,
            parameters=model_without_ddp.trainable_params,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        metric_logger.update(loss=mean_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        if result.get('log_dict'):
            for k, v in result['log_dict'].items():
                if isinstance(v, (int, float)):
                    metric_logger.update(**{k: v})

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            step = epoch * len(data_loader) + data_iter_step
            log_writer.add_scalar('train/loss', mean_loss.item(), step)
            log_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ─────────────────────────────────────────────────────────────────────────────
# eval_stage1_epoch — one-shot stage-1 baseline eval (no memory, runs once)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_stage1_epoch(model, criterion, data_loader, device, args):
    """
    Evaluate the stage-1 backbone directly (no memory fusion).

    Runs once before any stage-2 weights are loaded so the result reflects
    pure stage-1 quality.  Saves PLYs to <output_dir>/eval_stage1/.

    Returns aggregated metric dict.
    """
    import trimesh
    import clip
    from amb3r.tools.semantic_vis_utils import (
        build_text_embeddings,
        export_back_semantic_pca_ply,
        export_back_semantic_textmatch_ply,
        save_semantic_color_legend,
        get_scannet_label_and_color_map
    )
    from amb3r.vis_instance_hdbscan import export_instance_hdbscan_ply

    model.eval()
    model_without_ddp = model.module if hasattr(model, 'module') else model
    s1 = model_without_ddp.stage1

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Eval Stage-1 Baseline'
    dtype  = get_dtype(args)

    save_path = os.path.join(args.output_dir, 'eval_stage1')
    os.makedirs(save_path, exist_ok=True)

    # ── Build text embeddings ─────────────────────────────────────────────
    labels, _, default_color_table = get_scannet_label_and_color_map("scannet20")
    color_table = default_color_table[:len(labels)]

    if misc.is_main_process():
        save_semantic_color_legend(
            labels=labels, color_table=color_table,
            save_file=os.path.join(save_path, "scannet20_legend.png"),
            title="ScanNet20 Semantic Color Legend", ncols=2,
        )

    text_feat = build_text_embeddings(
        clip_model=model_without_ddp.stage1.lseg.clip_pretrained,
        tokenizer=clip.tokenize,
        labels=labels,
        device=device,
        template="a photo of a {}",
    )

    # ── Evaluation loop ───────────────────────────────────────────────────
    n_chunks = args.seq_len_eval // args.chunk_size

    for i, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        views, views_all = batch
        for key in views_all:
            if isinstance(views_all[key], torch.Tensor):
                views_all[key] = views_all[key].to(device, non_blocking=True)

        normalize_gt(views_all)  # in-place → local frame

        all_preds: list = []
        total_loss = torch.tensor(0.0, device=device)
        valid_chunks = 0
        losses = {}

        for c in range(n_chunks):
            s = c * args.chunk_size
            e = s + args.chunk_size

            chunk = {
                k: (v[:, s:e].contiguous()
                    if isinstance(v, torch.Tensor) and v.ndim >= 2 else v)
                for k, v in views_all.items()
            }

            # ── Stage-1 forward (no memory) ──────────────────────────────
            with torch.autocast('cuda', dtype=dtype):
                images, patch_tokens = s1.front_end.encode_patch_tokens(chunk)
                B, T = images.shape[:2]
                H, W = images.shape[-2:]
                lseg_feat = s1._extract_lseg(images)         # (B,T,512,H,W)
                lseg_flat = lseg_feat.flatten(0, 1)          # (B*T,512,H,W)
                patch_tokens = s1.clip_patch_fusion(patch_tokens, lseg_flat, H, W)
                decoded = s1.front_end.decode_patch_tokens(patch_tokens, images)
                preds = s1.front_end.decode_heads(
                    images, decoded, has_backend=True, semantic_feats=lseg_feat)
                preds['_clip_feat_gt'] = lseg_feat

            # Scale-align predicted geometry to normalized GT
            with torch.autocast('cuda', dtype=dtype, enabled=False):
                chunk_gt = {
                    'pts3d':      views_all['pts3d'][:, s:e],
                    'depthmap':   views_all['depthmap'][:, s:e],
                    'valid_mask': views_all['valid_mask'][:, s:e],
                }
                align_pred_to_gt(preds, chunk_gt)

            # Loss
            with torch.autocast('cuda', dtype=dtype):
                losses = criterion(preds, chunk)
            if torch.isfinite(losses['objective']):
                total_loss += losses['objective']
                valid_chunks += 1

            all_preds.append(preds)

        # ── Metrics ──────────────────────────────────────────────────────
        if valid_chunks > 0:
            avg_loss = (total_loss / valid_chunks).item()
            metric_logger.update(loss=avg_loss)
            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and k != 'objective':
                    metric_logger.update(**{k: v.item()})

        # ── PLY export (first N batches, main process only) ──────────────
        if i < args.eval_ply_n and misc.is_main_process() and all_preds:
            dataset_name = views.get('dataset', ['seq'])[0] if isinstance(views, dict) else 'seq'
            if isinstance(dataset_name, (list, tuple)):
                dataset_name = dataset_name[0]

            wp_list, sem_list, ins_list, color_list, clip_list = [], [], [], [], []
            for ci, pred in enumerate(all_preds):
                cs = ci * args.chunk_size
                ce = cs + args.chunk_size
                wp = views_all['pts3d'][:, cs:ce].reshape(-1, 3)
                wp_list.append(wp)
                sem = pred['semantic_feat'].permute(0,1,3,4,2).reshape(-1, pred['semantic_feat'].shape[2])
                sem_list.append(sem)
                ins = pred['instance_feat'].permute(0,1,3,4,2).reshape(-1, pred['instance_feat'].shape[2])
                ins_list.append(ins)
                col = pred['images'].permute(0,1,3,4,2).reshape(-1, 3).clamp(0, 1)
                color_list.append(col)
                clip_gt = pred['_clip_feat_gt'].permute(0,1,3,4,2).reshape(-1, pred['_clip_feat_gt'].shape[2])
                clip_list.append(clip_gt)

            pts_all  = torch.cat(wp_list, dim=0)
            sem_all  = torch.cat(sem_list, dim=0)
            ins_all  = torch.cat(ins_list, dim=0)
            col_all  = torch.cat(color_list, dim=0)
            clip_all = torch.cat(clip_list, dim=0)

            pts_np = pts_all.detach().cpu().numpy()
            col_np = (col_all.detach().cpu().numpy() * 255).astype(np.uint8)
            pc = trimesh.points.PointCloud(pts_np, colors=col_np)
            pc.export(os.path.join(save_path, f'{dataset_name}_geo_{i}.ply'))

            export_back_semantic_pca_ply(
                points_xyz=pts_all, semantic_feat=sem_all,
                save_file=os.path.join(save_path, f'{dataset_name}_sem_pca_{i}.ply'),
            )
            export_back_semantic_textmatch_ply(
                points_xyz=pts_all, semantic_feat=sem_all,
                text_feat=text_feat, color_table=color_table,
                save_file=os.path.join(save_path, f'{dataset_name}_sem_text_{i}.ply'),
            )
            export_back_semantic_textmatch_ply(
                points_xyz=pts_all, semantic_feat=clip_all,
                text_feat=text_feat, color_table=color_table,
                save_file=os.path.join(save_path, f'{dataset_name}_sem_text_gt_{i}.ply'),
            )
            export_back_semantic_pca_ply(
                points_xyz=pts_all, semantic_feat=ins_all,
                save_file=os.path.join(save_path, f'{dataset_name}_ins_pca_{i}.ply'),
            )
            export_instance_hdbscan_ply(
                points_xyz=pts_all, instance_feat=ins_all,
                save_file=os.path.join(save_path, f'{dataset_name}_ins_hdbscan_{i}.ply'),
            )

    # ── Aggregate ─────────────────────────────────────────────────────────
    metric_logger.synchronize_between_processes()
    print("Stage-1 eval stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {
        f'{k}_{tag}': getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }
    return results



# ─────────────────────────────────────────────────────────────────────────────
# eval_one_epoch — evaluation with PLY visualization export
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_one_epoch(
    model, criterion, data_loader, device, epoch, args,
    log_writer=None, prefix='eval',
):
    """
    Evaluate Stage-2 V4 and export PLY visualizations.

    For each sequence:
        1. Process all chunks sequentially (filling voxel_map)
        2. Accumulate world_points + sem/ins features from all chunks
        3. Export PLY: geometry, semantic PCA, semantic text-match, instance PCA

    First `args.eval_ply_n` sequences get PLY exports.
    """
    import trimesh
    import clip
    from amb3r.tools.semantic_vis_utils import (
        build_text_embeddings,
        export_back_semantic_pca_ply,
        export_back_semantic_textmatch_ply,
        save_semantic_color_legend,
        get_scannet_label_and_color_map
    )
    # from amb3r.tools.scannet200_constants import get_scannet_label_and_color_map
    from amb3r.vis_instance_hdbscan import export_instance_hdbscan_ply

    model.eval()
    model_without_ddp = model.module if hasattr(model, 'module') else model

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = f'Eval Epoch: [{epoch}]'
    dtype  = get_dtype(args)

    save_path = os.path.join(args.output_dir, f'eval_{epoch}')
    os.makedirs(save_path, exist_ok=True)

    # ── Build text embeddings for semantic color matching ─────────────────
    labels, _, default_color_table = get_scannet_label_and_color_map("scannet20")
    color_table = default_color_table[:len(labels)]

    if misc.is_main_process():
        save_semantic_color_legend(
            labels=labels, color_table=color_table,
            save_file=os.path.join(save_path, "scannet20_legend.png"),
            title="ScanNet20 Semantic Color Legend", ncols=2,
        )

    text_feat = build_text_embeddings(
        clip_model=model_without_ddp.stage1.lseg.clip_pretrained,
        tokenizer=clip.tokenize,
        labels=labels,
        device=device,
        template="a photo of a {}",
    )

    # ── Evaluation loop ───────────────────────────────────────────────────
    # n_chunks = args.seq_len // args.chunk_size
    n_chunks = args.seq_len_eval // args.chunk_size

    for i, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        views, views_all = batch
        for key in views_all:
            if isinstance(views_all[key], torch.Tensor):
                views_all[key] = views_all[key].to(device, non_blocking=True)

        # ── Coordinate handling ───────────────────────────────────────────
        # Capture raw GT in global frame for memory queries, then normalize
        gt_pts_global = views_all.get('pts3d')
        if gt_pts_global is not None:
            gt_pts_global = gt_pts_global.clone()

        normalize_gt(views_all)  # in-place → local frame

        # ── Full-sequence forward (chunk by chunk, no grad) ──────────────
        voxel_map = model_without_ddp.make_voxel_map()
        feat_buffer = model_without_ddp.make_feat_buffer(
            sem_dim=args.sem_dim, ins_dim=args.ins_dim)

        all_preds: list = []
        total_loss = torch.tensor(0.0, device=device)
        valid_chunks = 0

        for c in range(n_chunks):
            s = c * args.chunk_size
            e = s + args.chunk_size

            # Chunk from NORMALIZED views_all (for loss computation)
            chunk = {
                k: (v[:, s:e].contiguous()
                    if isinstance(v, torch.Tensor) and v.ndim >= 2 else v)
                for k, v in views_all.items()
            }
            # Memory queries use GLOBAL-frame GT (consistent across chunks)
            gt_pts_chunk = gt_pts_global[:, s:e] if gt_pts_global is not None else None

            with torch.autocast('cuda', dtype=dtype):
                preds, M_r, W_c, pts_flat = model_without_ddp.forward_chunk(
                    chunk, voxel_map, pts_for_query=gt_pts_chunk)

            # Scale-align predicted geometry to normalized GT for metrics/PLY
            with torch.autocast('cuda', dtype=dtype, enabled=False):
                chunk_gt = {
                    'pts3d':      views_all['pts3d'][:, s:e],
                    'depthmap':   views_all['depthmap'][:, s:e],
                    'valid_mask': views_all['valid_mask'][:, s:e],
                }
                align_pred_to_gt(preds, chunk_gt)

            # Loss (on normalized/aligned data)
            with torch.autocast('cuda', dtype=dtype):
                losses = criterion(preds, chunk)
            if torch.isfinite(losses['objective']):
                total_loss += losses['objective']
                valid_chunks += 1

            all_preds.append(preds)

            # Update voxel map (detached — no BPTT needed in eval)
            _eval_mem_mode = model_without_ddp.mem_mode if hasattr(model_without_ddp, 'mem_mode') else 0
            if _eval_mem_mode == 1:
                AMB3RStage2V4.update_voxel_store(
                    voxel_map, M_r.detach(), None,
                    pts_flat.detach(), mem_mode=1)
            elif args.pixel_broadcast_mem:
                _pixel_broadcast_update(
                    voxel_map, M_r.detach(),
                    W_c.detach() if W_c is not None else None,
                    gt_pts_chunk,
                    mem_mode=_eval_mem_mode,
                    downsample=args.pixel_broadcast_downsample)
            else:
                AMB3RStage2V4.update_voxel_store(
                    voxel_map, M_r.detach(),
                    W_c.detach() if W_c is not None else None,
                    pts_flat.detach(),
                    mem_mode=_eval_mem_mode)

            # Update feat_buffer with pixel-wise features (for debug vis)
            gt_pts_px   = gt_pts_chunk.reshape(-1, 3)
            sem_px      = preds['semantic_feat'].detach().permute(0,1,3,4,2)
            ins_px      = preds['instance_feat'].detach().permute(0,1,3,4,2)
            sem_px      = sem_px.reshape(-1, sem_px.shape[-1])
            ins_px      = ins_px.reshape(-1, ins_px.shape[-1])
            sem_conf_px = preds['semantic_conf'].detach().permute(0,1,3,4,2).reshape(-1)
            ins_conf_px = preds['instance_conf'].detach().permute(0,1,3,4,2).reshape(-1)
            feat_buffer.update(gt_pts_px, sem_px, ins_px,
                               sem_conf=sem_conf_px, ins_conf=ins_conf_px)
            inst_eval = chunk.get('instance_mask')
            if inst_eval is not None:
                feat_buffer.update_by_id(inst_eval.reshape(-1), ins_px, sem_px,
                                         ins_conf_flat=ins_conf_px,
                                         sem_conf_flat=sem_conf_px)

        voxel_map.clear()

        # ── Metrics ──────────────────────────────────────────────────────
        if valid_chunks > 0:
            avg_loss = (total_loss / valid_chunks).item()
            metric_logger.update(loss=avg_loss)
            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and k != 'objective':
                    metric_logger.update(**{k: v.item()})

        # ── PLY export (first N batches, main process only) ──────────────
        if i < args.eval_ply_n and misc.is_main_process() and all_preds:
            dataset_name = views.get('dataset', ['seq'])[0] if isinstance(views, dict) else f'seq'
            if isinstance(dataset_name, (list, tuple)):
                dataset_name = dataset_name[0]

            # Accumulate points + features across all chunks
            # NOTE: use normalized GT pts (consistent frame) for geometry,
            # NOT pred['world_points'] (each chunk has its own VGGT local origin,
            # and scale_invariant_alignment cannot fix the translation offset).
            wp_list, sem_list, ins_list, color_list, clip_list = [], [], [], [], []
            for ci, pred in enumerate(all_preds):
                B, T = pred['images'].shape[:2]
                H, W = pred['images'].shape[-2:]
                cs = ci * args.chunk_size
                ce = cs + args.chunk_size
                wp = views_all['pts3d'][:, cs:ce].reshape(-1, 3)  # normalized local
                wp_list.append(wp)
                sem = pred['semantic_feat'].permute(0,1,3,4,2).reshape(-1, pred['semantic_feat'].shape[2])
                sem_list.append(sem)
                ins = pred['instance_feat'].permute(0,1,3,4,2).reshape(-1, pred['instance_feat'].shape[2])
                ins_list.append(ins)
                col = pred['images'].permute(0,1,3,4,2).reshape(-1, 3).clamp(0, 1)  # already [0,1] from encode_patch_tokens
                color_list.append(col)
                clip_gt = pred['_clip_feat_gt'].permute(0,1,3,4,2).reshape(-1, pred['_clip_feat_gt'].shape[2])
                clip_list.append(clip_gt)

            pts_all  = torch.cat(wp_list, dim=0)
            sem_all  = torch.cat(sem_list, dim=0)
            ins_all  = torch.cat(ins_list, dim=0)
            col_all  = torch.cat(color_list, dim=0)
            clip_all = torch.cat(clip_list, dim=0)

            # Geometry (RGB coloured)
            pts_np = pts_all.detach().cpu().numpy()
            col_np = (col_all.detach().cpu().numpy() * 255).astype(np.uint8)
            pc = trimesh.points.PointCloud(pts_np, colors=col_np)
            pc.export(os.path.join(save_path, f'{dataset_name}_geo_{i}.ply'))

            # Semantic PCA
            export_back_semantic_pca_ply(
                points_xyz=pts_all,
                semantic_feat=sem_all,
                save_file=os.path.join(save_path, f'{dataset_name}_sem_pca_{i}.ply'),
            )

            # Semantic text-match
            export_back_semantic_textmatch_ply(
                points_xyz=pts_all,
                semantic_feat=sem_all,
                text_feat=text_feat,
                color_table=color_table,
                save_file=os.path.join(save_path, f'{dataset_name}_sem_text_{i}.ply'),
            )

            # Semantic text-match (GT CLIP features)
            export_back_semantic_textmatch_ply(
                points_xyz=pts_all,
                semantic_feat=clip_all,
                text_feat=text_feat,
                color_table=color_table,
                save_file=os.path.join(save_path, f'{dataset_name}_sem_text_gt_{i}.ply'),
            )

            # Instance PCA
            export_back_semantic_pca_ply(
                points_xyz=pts_all,
                semantic_feat=ins_all,
                save_file=os.path.join(save_path, f'{dataset_name}_ins_pca_{i}.ply'),
            )
            
            # Instance clustering with HDBSCAN and export colored by cluster ID
            export_instance_hdbscan_ply(
                points_xyz=pts_all, instance_feat=ins_all,
                save_file=os.path.join(save_path, f'{dataset_name}_ins_hdbscan_{i}.ply'),
            )

            # ── Debug: feat_buffer content PCA vis ───────────────────────
            # Exports the stored voxel-level anchors so you can see
            # what the buffer actually contains.
            if hasattr(feat_buffer, '_voxel_keys') and feat_buffer._voxel_keys.shape[0] > 0:
                buf_keys = feat_buffer._voxel_keys  # [V] int64
                buf_sem  = feat_buffer._voxel_sem   # [V, sem_dim]
                buf_ins  = feat_buffer._voxel_ins   # [V, ins_dim]
                # Decode voxel keys back to xyz
                # _encode_keys offsets by P//2: key = (ix+H)*P^2 + (iy+H)*P + (iz+H)
                P = feat_buffer._P
                H = P // 2
                PP = P * P
                vx = buf_keys // PP
                vy = (buf_keys % PP) // P
                vz = buf_keys % P
                ix = vx - H
                iy = vy - H
                iz = vz - H
                buf_pts = torch.stack([ix, iy, iz], dim=-1).float() * feat_buffer.voxel_size
                export_back_semantic_pca_ply(
                    points_xyz=buf_pts,
                    semantic_feat=buf_sem,
                    save_file=os.path.join(save_path, f'{dataset_name}_buf_sem_pca_{i}.ply'),
                )
                export_back_semantic_pca_ply(
                    points_xyz=buf_pts,
                    semantic_feat=buf_ins,
                    save_file=os.path.join(save_path, f'{dataset_name}_buf_ins_pca_{i}.ply'),
                )
                print(f'[Eval] feat_buffer: {feat_buffer._voxel_keys.shape[0]} voxels, '
                      f'{feat_buffer.n_instance_ids} instance IDs')

        feat_buffer.clear()

    # ── Aggregate ─────────────────────────────────────────────────────────
    metric_logger.synchronize_between_processes()
    print("Eval stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {
        f'{k}_{tag}': getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(f'{prefix}/{name}', val, epoch)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── DDP ───────────────────────────────────────────────────────────────
    misc.init_distributed_mode(args)
    device = torch.device(f'cuda:{args.gpu}' if hasattr(args, 'gpu') else 'cuda')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Dataset ───────────────────────────────────────────────────────────
    res = eval(args.resolution)
    dataset_train = ScannetppSequence(
        ROOT=args.data_root, split='train',
        resolution=[res], num_frames=args.seq_len,
        stride=args.stride, seed=args.seed,
    )
    dataset_eval = ScannetppSequence(
        ROOT=args.data_root, split='val',
        resolution=[res], num_frames=args.seq_len_eval,
        stride=args.stride, seed=args.seed,
    )

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, shuffle=True)
        sampler_eval = torch.utils.data.DistributedSampler(
            dataset_eval, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    loader_eval = torch.utils.data.DataLoader(
        dataset_eval, sampler=sampler_eval,
        batch_size=1, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    stage1 = AMB3RStage1FullFT(
        metric_scale=True, clip_dim=args.clip_dim,
        sem_dim=args.sem_dim, ins_dim=args.ins_dim,
    )
    # Load AE encoder for mem_mode=3
    ae_encoder = None
    if args.mem_mode == 3:
        assert args.stage1_5_ckpt is not None, 'mem_mode=3 requires --stage1_5_ckpt'
        from amb3r.model_stage1_5 import VGGTAutoencoder
        ae_full = VGGTAutoencoder(
            input_dim=4096,  # concat(layer4, layer23)
            bottleneck_dim=args.ae_bottleneck_dim,
        )
        ae_ckpt = torch.load(args.stage1_5_ckpt, map_location='cpu', weights_only=False)
        ae_state = ae_ckpt.get('model', ae_ckpt)
        # Extract autoencoder weights (prefix: autoencoder.)
        ae_weights = {}
        for k, v in ae_state.items():
            if k.startswith('autoencoder.'):
                ae_weights[k[len('autoencoder.'):]] = v
        ae_full.load_state_dict(ae_weights, strict=True)
        ae_encoder = ae_full.encoder  # only need the encoder
        print(f'[Stage2V4] Loaded AE encoder from {args.stage1_5_ckpt}')

    model = AMB3RStage2V4(
        stage1_model=stage1,
        mem_mode=args.mem_mode,
        mem_dim=args.mem_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        mem_voxel_size=args.mem_voxel_size,
        feat_voxel_size=args.feat_voxel_size,
        ema_alpha=args.ema_alpha,
        use_checkpoint=args.use_checkpoint,
        ae_encoder=ae_encoder,
        ae_bottleneck_dim=args.ae_bottleneck_dim,
        finetune_head=args.finetune_head,
    )

    # Load Stage-1 weights
    model.load_stage1_weights(args.stage1_ckpt)
    model.prepare('bf16' if args.amp == 'bf16' else 'fp32')
    model.to(device)

    # ── Criterion (built early — needed before stage-2 weights are loaded) ────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    criterion = Stage2V4Loss(
        w_sem_align=args.w_sem_align,
        w_ins_contrast=args.w_ins_contrast,
        w_mem_consist=args.w_mem_consist,
    ).to(device)

    # ── Stage-1 baseline eval (once, before stage-2 weights overwrite heads) ──
    # Gate: use a sentinel file so a crashed partial run doesn't block retry
    _s1_done = output_dir / 'eval_stage1' / 'done'
    if args.eval_freq > 0 and not _s1_done.exists():
        print('[Stage2V4] Running stage-1 baseline eval (one-shot)...')
        stage1_eval_stats = eval_stage1_epoch(
            model, criterion, loader_eval, device, args)
        if misc.is_main_process():
            with open(output_dir / 'eval_stage1.json', 'w') as f:
                json.dump(stage1_eval_stats, f, indent=2)
            _s1_done.touch()  # mark as complete

    # Optional Stage-2 resume
    start_epoch = 0
    resume_path = args.stage2_ckpt
    if resume_path is None and args.auto_resume:
        auto_path = Path(args.output_dir) / 'checkpoint-last.pth'
        if auto_path.exists():
            resume_path = str(auto_path)
            print(f'[Stage2V4] Auto-resuming from {resume_path}')
    if resume_path:
        ckpt = torch.load(resume_path, map_location='cpu')
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f'[Stage2V4] Loaded model from {resume_path}')

    n_train = sum(p.numel() for p in model.trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'[Stage2V4] Trainable: {n_train/1e6:.2f}M / {n_total/1e6:.1f}M total')
    print(f'[Stage2V4] mem_mode={args.mem_mode}, effective_mem_dim={model.effective_mem_dim}')
    print(f'[Stage2V4] Seq={args.seq_len}, Chunk={args.chunk_size}, '
          f'n_chunks={args.seq_len // args.chunk_size}')
    print(f'[Stage2V4] Eval Seq={args.seq_len_eval}, Chunk={args.chunk_size}, '
          f'n_chunks={args.seq_len_eval // args.chunk_size}')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=args.finetune_head)
    model_without_ddp = model.module if hasattr(model, 'module') else model

    # ── Optimiser ─────────────────────────────────────────────────────────
    if args.finetune_head:
        param_groups = [
            {'params': model_without_ddp.non_head_params, 'lr': args.lr},
            {'params': model_without_ddp.head_params,
             'lr': args.lr * args.head_lr_scale, 'name': 'head'},
        ]
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=args.weight_decay)
        n_head = sum(p.numel() for p in model_without_ddp.head_params)
        print(f'[Stage2V4] Finetuning heads: {n_head/1e6:.2f}M params '
              f'at lr={args.lr * args.head_lr_scale:.1e}')
    else:
        optimizer = torch.optim.AdamW(
            model_without_ddp.trainable_params,
            lr=args.lr, weight_decay=args.weight_decay,
        )
    loss_scaler = NativeScaler()

    # Resume optimizer + epoch from checkpoint
    if resume_path:
        ckpt = torch.load(resume_path, map_location='cpu')
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f'[Stage2V4] Resumed optimizer state')
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1  # resume from NEXT epoch
            print(f'[Stage2V4] Resuming from epoch {start_epoch}')
        if 'scaler' in ckpt:
            loss_scaler.load_state_dict(ckpt['scaler'])
        del ckpt

    # ── Output dir / logging ──────────────────────────────────────────────
    # output_dir and criterion already created above (before stage-2 ckpt load)

    log_writer = None
    if misc.is_main_process():
        log_writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
        with open(output_dir / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f'[Stage2V4] Output dir: {output_dir}')

    # ── Training loop ─────────────────────────────────────────────────────
    print(f'Start Stage-2 V4 training for {args.epochs} epochs '
          f'(starting from epoch {start_epoch}).')
    start_time = time.time()

    # Run eval before training on resume (to verify checkpoint quality)
    if start_epoch > 0 and args.eval_freq > 0:
        print(f'[Stage2V4] Running eval at resume (epoch {start_epoch - 1})...')
        eval_stats = eval_one_epoch(
            model, criterion, loader_eval, device,
            start_epoch - 1, args, log_writer=log_writer,
        )
        if misc.is_main_process():
            with open(output_dir / f'eval_resume_{start_epoch - 1}.json', 'w') as f:
                json.dump(eval_stats, f, indent=2)

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, loader_train, optimizer,
            device, epoch, loss_scaler, args, log_writer=log_writer,
        )

        # ── Checkpoint ────────────────────────────────────────────────────
        if misc.is_main_process() and (epoch + 1) % args.save_freq == 0:
            s2_state = {
                k: v for k, v in model_without_ddp.state_dict().items()
                if any(k.startswith(p) for p in
                       ('null_token.',
                        'fusion_sem_early.', 'fusion_sem_late.',
                        'fusion_ins_early.', 'fusion_ins_late.',
                        'memory_tokenizer.',
                        # Legacy names for backward compat
                        'memory_fusion_early.', 'memory_fusion_late.',
                        'memory_fusion.'))
            }
            ckpt_data = {
                'epoch': epoch,
                'model': s2_state,
                'optimizer': optimizer.state_dict(),
                'scaler': loss_scaler.state_dict(),
                'args': vars(args),
            }
            torch.save(ckpt_data, str(output_dir / 'checkpoint-last.pth'))
            if (epoch + 1) % 5 == 0:
                torch.save(ckpt_data, str(output_dir / f'checkpoint-{epoch:04d}.pth'))

        # ── Periodic evaluation ───────────────────────────────────────────
        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
            eval_stats = eval_one_epoch(
                model, criterion, loader_eval, device, epoch, args,
                log_writer=log_writer,
            )
            if misc.is_main_process():
                with open(output_dir / f'eval_{epoch}.json', 'w') as f:
                    json.dump(eval_stats, f, indent=2)

        # Log
        if misc.is_main_process():
            log_stats = {
                'epoch': epoch,
                **{f'train_{k}': v for k, v in train_stats.items()},
            }
            with open(output_dir / 'log.txt', 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_str  = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time: {total_str}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
