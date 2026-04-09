"""
train_stage2v2.py
=================
Stage-2 V2 recurrent training:
  frozen Stage-1  +  trainable PTv3 BackEnd with memory conditioning
  → re-predicts geometry + semantic + instance.

Key differences from train_stage2.py (V1):
  - Model  : AMB3RStage2V2   (PTv3 backend instead of DPT memory heads)
  - Loss   : Stage2LossV2    (geometry + semantic + instance + memory)
  - Ckpt   : saves backend.* weights (not semantic_memory_head.*)
  - Eval   : test_one_epoch() + PLY export after every epoch (same as Stage-1)

Usage:
    torchrun --nproc_per_node=1 .claude/sandbox/train_stage2v2.py \\
        --stage1_ckpt outputs/exp_stage1_wo_lora/checkpoint-best.pth \\
        --seq_len 24 --chunk_size 6 --stride 2 \\
        --batch_size 1 --accum_iter 4 --epochs 30 --lr 1e-4
"""

import os
import sys
import math
import time
import argparse
import datetime
import json
import numpy as np
import trimesh
from pathlib import Path
from shutil import copyfile
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# ── project paths ─────────────────────────────────────────────────────────
_SANDBOX = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SANDBOX)
sys.path.insert(0, _SANDBOX)
sys.path.insert(0, _ROOT)
sys.path.append(os.path.join(_ROOT, 'thirdparty'))

import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler

from amb3r.model_stage1_wo_lora import AMB3RStage1FullFT
from amb3r.model_stage2v2 import AMB3RStage2V2
from amb3r.loss_stage2v2 import Stage2LossV2
from amb3r.loss_semantic import get_depth_loss, get_point_loss
from slam_semantic.semantic_voxel_map_v2 import VoxelFeatureMapV2

from amb3r.datasets import build_dataset  # register all dataset classes
from amb3r.datasets import *
from amb3r.datasets.scannetpp_sequence import ScannetppSequence
from amb3r.tools.semantic_vis_utils import (
    export_back_semantic_pca_ply,
    export_back_semantic_textmatch_ply,
    get_scannet_label_and_color_map,
    build_text_embeddings,
    save_semantic_color_legend,
)
from amb3r.vis_instance_hdbscan import export_instance_hdbscan_ply

from vggt.train_utils.general import check_and_fix_inf_nan
from vggt.train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from moge.moge.train.losses import scale_invariant_alignment
from lang_seg.modules.models.lseg_net import clip as lseg_clip

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args_parser():
    p = argparse.ArgumentParser('AMB3R Stage-2 V2 training', add_help=False)

    # ── model ────────────────────────────────────────────────────────────
    p.add_argument('--stage1_ckpt', required=True,
                   help='Path to the Stage-1 checkpoint (checkpoint-best.pth)')
    p.add_argument('--stage2_ckpt', default=None,
                   help='Optional: resume Stage-2 V2 from this checkpoint')
    p.add_argument('--sem_dim',  type=int, default=512)
    p.add_argument('--ins_dim',  type=int, default=16)
    p.add_argument('--clip_dim', type=int, default=512)
    p.add_argument('--interp_v2', action='store_true',
                   help='Voxel coordinate alignment (must match Stage-1 ckpt)')
    p.add_argument('--voxel_res', type=float, default=0.01,
                   help='Voxel resolution for PTv3 backend')

    # ── sequence / chunk ─────────────────────────────────────────────────
    p.add_argument('--seq_len',    type=int, default=24)
    p.add_argument('--chunk_size', type=int, default=6)
    p.add_argument('--stride',     type=int, default=2)

    # ── datasets ─────────────────────────────────────────────────────────
    p.add_argument('--data_root',
                   default='/mnt/HDD4/ricky/data/scannetpp_arrow')
    p.add_argument('--resolution',      default='(518, 336)')
    p.add_argument('--test_resolution', default='(518, 336)')
    p.add_argument('--num_frames_test', type=int, default=8,
                   help='Number of frames per sample in the eval dataset')
    p.add_argument('--test_dataset', default=None,
                   help='Optional override for eval dataset string '
                        '(default: auto Scannetpp val at test_resolution)')
    p.add_argument('--num_workers',      type=int, default=2)
    p.add_argument('--num_workers_test', type=int, default=0)
    p.add_argument('--batch_size_test',  type=int, default=1)

    # ── voxel memory ─────────────────────────────────────────────────────
    p.add_argument('--voxel_size', type=float, default=0.05)
    p.add_argument('--conf_scale', type=float, default=1.0)

    # ── optimiser ────────────────────────────────────────────────────────
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--weight_decay',  type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int,   default=1)
    p.add_argument('--min_lr',        type=float, default=1e-6)

    # ── training ─────────────────────────────────────────────────────────
    p.add_argument('--epochs',     type=int, default=30)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--accum_iter', type=int, default=4)
    p.add_argument('--amp',        choices=[False, 'bf16', 'fp16'], default='bf16')

    # ── geometry loss weights ─────────────────────────────────────────────
    p.add_argument('--w_geo_depth',   type=float, default=1.0)
    p.add_argument('--w_geo_pts',     type=float, default=1.0)
    p.add_argument('--w_geo_camera',  type=float, default=0.5)

    # ── semantic / instance / memory loss weights ─────────────────────────
    p.add_argument('--w_sem_align',    type=float, default=0.5)
    p.add_argument('--w_sem_memory',   type=float, default=1.0)
    p.add_argument('--w_ins_contrast', type=float, default=1.0)
    p.add_argument('--w_ins_memory',   type=float, default=1.0)

    # ── logging / saving ─────────────────────────────────────────────────
    p.add_argument('--output_dir',  default='./outputs/exp_stage2v2')
    p.add_argument('--eval_freq',   type=int, default=1)
    p.add_argument('--save_freq',   type=int, default=1)
    p.add_argument('--print_freq',  type=int, default=20)

    # ── DDP ──────────────────────────────────────────────────────────────
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--world_size',  default=1, type=int)
    p.add_argument('--local_rank',  default=-1, type=int)
    p.add_argument('--dist_url',    default='env://')

    return p


def get_dtype(args):
    if args.amp == 'bf16': return torch.bfloat16
    if args.amp == 'fp16': return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Voxel memory helpers
# ---------------------------------------------------------------------------

PATCH_SIZE = 14


def make_voxel_maps(voxel_size, sem_dim, ins_dim, conf_scale):
    sem_map = VoxelFeatureMapV2(voxel_size=voxel_size, feat_dim=sem_dim,
                                conf_scale=conf_scale)
    ins_map = VoxelFeatureMapV2(voxel_size=voxel_size, feat_dim=ins_dim,
                                conf_scale=conf_scale)
    return sem_map, ins_map


@torch.no_grad()
def update_voxel_maps(sem_map, ins_map, gt_pts, refined_sem, refined_ins, sem_conf):
    """Downsample to patch res and insert into voxel maps (stop-grad)."""
    B, T, C_sem, H, W = refined_sem.shape
    Hp, Wp = H // PATCH_SIZE, W // PATCH_SIZE
    Ht, Wt = H // 7, W // 7

    pts = gt_pts.flatten(0, 1).float().permute(0, 3, 1, 2)
    pts = F.interpolate(pts, size=(Ht, Wt), mode='bilinear', align_corners=False)
    pts = pts.permute(0, 2, 3, 1).reshape(-1, 3).cpu()

    sem_p  = F.interpolate(refined_sem.flatten(0, 1), size=(Ht, Wt), mode='bilinear', align_corners=False)
    ins_p  = F.interpolate(refined_ins.flatten(0, 1), size=(Ht, Wt), mode='bilinear', align_corners=False)
    conf_p = F.interpolate(sem_conf.flatten(0, 1),    size=(Ht, Wt), mode='bilinear', align_corners=False)

    sem_map.update(pts,
                   sem_p.permute(0, 2, 3, 1).reshape(-1, C_sem).detach().float().cpu(),
                   conf_p.squeeze(1).reshape(-1).detach().float().cpu())
    ins_map.update(pts,
                   ins_p.permute(0, 2, 3, 1).reshape(-1, refined_ins.shape[2]).detach().float().cpu(),
                   conf_p.squeeze(1).reshape(-1).detach().float().cpu())


# ---------------------------------------------------------------------------
# GT normalisation + scale-align (same as Stage-1)
# ---------------------------------------------------------------------------

def normalize_gt(views_all):
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

    if 'pose_enc_list' in pred:
        for stage_idx in range(len(pred['pose_enc_list'])):
            pred['pose_enc_list'][stage_idx][..., :3] *= scale_depth[..., None]

    return scale_depth


def instance_ids_to_colors(inst_ids: np.ndarray) -> np.ndarray:
    import colorsys
    golden = 0.6180339887
    colors = np.full((len(inst_ids), 3), 128, dtype=np.uint8)
    for iid in np.unique(inst_ids):
        if iid == 0:
            continue
        hue     = (int(iid) * golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors[inst_ids == iid] = [int(r * 255), int(g * 255), int(b * 255)]
    return colors


# ---------------------------------------------------------------------------
# Eval epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_one_epoch(model, criterion, data_loader, device, epoch, args,
                   log_writer=None, prefix='test'):
    model.eval()
    model_without_ddp = model.module if hasattr(model, 'module') else model

    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = f'Test Epoch: [{epoch}]'
    dtype  = get_dtype(args)

    save_path = os.path.join(args.output_dir, f'eval_{epoch}')
    os.makedirs(save_path, exist_ok=True)

    # ── Text embeddings for semantic vis ─────────────────────────────────
    labels, _, default_color_table = get_scannet_label_and_color_map('scannet20')
    if len(labels) <= len(default_color_table):
        color_table = default_color_table[:len(labels)]
    else:
        reps = int(np.ceil(len(labels) / len(default_color_table)))
        color_table = np.tile(default_color_table, (reps, 1))[:len(labels)]
    save_semantic_color_legend(
        labels=labels, color_table=color_table,
        save_file=os.path.join(save_path, 'scannet20_legend.png'),
        title='ScanNet20 Semantic Color Legend', ncols=2,
    )
    text_feat = build_text_embeddings(
        clip_model=model_without_ddp.stage1.lseg.clip_pretrained,
        tokenizer=lseg_clip.tokenize,
        labels=labels,
        device=device,
        template='a photo of a {}',
    )

    for i, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        views, views_all = batch
        for key in views_all:
            if isinstance(views_all[key], torch.Tensor):
                views_all[key] = views_all[key].to(device)

        with torch.autocast('cuda', dtype=dtype):
            # Eval runs without temporal memory (first-chunk behaviour)
            predictions = model_without_ddp(views_all)

        normalize_gt(views_all)
        align_pred_to_gt(predictions, views_all)

        losses = criterion(predictions, views_all)

        # Geometry metrics
        loss_depth = check_and_fix_inf_nan(
            get_depth_loss(predictions['depth'],
                           views_all['depthmap'][..., None],
                           views_all['valid_mask']), 'eval_loss_depth')
        loss_pts = check_and_fix_inf_nan(
            get_point_loss(predictions['world_points'],
                           views_all['pts3d'],
                           views_all['valid_mask']), 'eval_loss_pts')

        metric_logger.update(loss=float(losses['objective'].item()))
        metric_logger.update(loss_depth=float(loss_depth.item()))
        metric_logger.update(loss_pts=float(loss_pts.item()))
        metric_logger.update(
            loss_refine=float(loss_pts.item() + loss_depth.item())
        )
        if 'loss_camera' in losses:
            metric_logger.update(loss_camera=float(losses['loss_camera'].item()))
            metric_logger.update(
                loss_refine=float(loss_pts.item() + loss_depth.item()
                                  + losses['loss_camera'].item())
            )
        for key in ('loss_sem_align', 'loss_sem_memory',
                    'loss_ins_contrast', 'loss_ins_memory'):
            if key in losses:
                metric_logger.update(**{key: float(losses[key].item())})

        # ── PLY export (first 5 batches, main process only) ───────────────
        if i < 5 and misc.is_main_process():
            dataset_name = views[0]['dataset'][0] if views else f'sample{i}'

            # RGB image colors for point cloud colouring
            color_np = (
                (views_all['images'].permute(0, 1, 3, 4, 2).reshape(-1, 3) + 1.0) / 2.0
            ).clamp(0, 1).cpu().numpy()

            # GT point cloud (RGB)
            pcd_gt = views_all['pts3d'].detach().cpu().numpy().reshape(-1, 3)
            trimesh.points.PointCloud(
                pcd_gt, colors=(color_np * 255).astype(np.uint8)
            ).export(os.path.join(save_path, f'{dataset_name}_gt_{i}.ply'))

            # GT instance colours
            inst_ids = views_all['instance_mask'].cpu().numpy().reshape(-1)
            inst_colors = instance_ids_to_colors(inst_ids)
            trimesh.points.PointCloud(
                pcd_gt, colors=inst_colors
            ).export(os.path.join(save_path, f'{dataset_name}_inst_gt_{i}.ply'))

            # Predicted point cloud (RGB)
            pcd_pred = predictions['world_points'].detach().cpu().numpy().reshape(-1, 3)
            trimesh.points.PointCloud(
                pcd_pred, colors=(color_np * 255).astype(np.uint8)
            ).export(os.path.join(save_path, f'{dataset_name}_pred_{i}.ply'))

            pts_flat = predictions['world_points'].reshape(-1, 3)

            # Stage-2 semantic PCA
            if 'semantic_feat' in predictions:
                sem_feat = (predictions['semantic_feat']
                            .permute(0, 1, 3, 4, 2)
                            .reshape(-1, predictions['semantic_feat'].shape[2]))
                export_back_semantic_pca_ply(
                    points_xyz=pts_flat, semantic_feat=sem_feat,
                    save_file=os.path.join(save_path, f'{dataset_name}_sem_pca_{i}.ply'),
                )

                # Stage-2 semantic text-match
                sem_exp = (predictions.get('semantic_feat_expanded',
                                           predictions['semantic_feat'])
                           .permute(0, 1, 3, 4, 2)
                           .reshape(-1, predictions['semantic_feat'].shape[2]))
                export_back_semantic_textmatch_ply(
                    points_xyz=pts_flat, semantic_feat=sem_exp,
                    text_feat=text_feat, color_table=color_table,
                    save_file=os.path.join(save_path, f'{dataset_name}_sem_text_{i}.ply'),
                )

                # Stage-1 semantic text-match (baseline comparison)
                if 'sem_feat_s1' in predictions:
                    sem_s1 = (predictions['sem_feat_s1']
                              .permute(0, 1, 3, 4, 2)
                              .reshape(-1, predictions['sem_feat_s1'].shape[2]))
                    export_back_semantic_textmatch_ply(
                        points_xyz=pts_flat, semantic_feat=sem_s1,
                        text_feat=text_feat, color_table=color_table,
                        save_file=os.path.join(save_path, f'{dataset_name}_sem_text_s1_{i}.ply'),
                    )

                # GT CLIP semantic text-match
                gt_sem = (predictions['_clip_feat_gt']
                          .permute(0, 1, 3, 4, 2)
                          .reshape(-1, predictions['_clip_feat_gt'].shape[2]))
                export_back_semantic_textmatch_ply(
                    points_xyz=pts_flat, semantic_feat=gt_sem,
                    text_feat=text_feat, color_table=color_table,
                    save_file=os.path.join(save_path, f'{dataset_name}_sem_text_gt_{i}.ply'),
                )

            # Stage-2 instance PCA + HDBSCAN
            if 'instance_feat' in predictions:
                ins_feat = (predictions['instance_feat']
                            .permute(0, 1, 3, 4, 2)
                            .reshape(-1, predictions['instance_feat'].shape[2]))
                export_back_semantic_pca_ply(
                    points_xyz=pts_flat, semantic_feat=ins_feat,
                    save_file=os.path.join(save_path, f'{dataset_name}_ins_pca_{i}.ply'),
                )
                export_instance_hdbscan_ply(
                    points_xyz=pts_flat, instance_feat=ins_feat,
                    save_file=os.path.join(save_path, f'{dataset_name}_ins_hdbscan_{i}.ply'),
                    verbose=True,
                )

                # Stage-1 instance PCA (baseline comparison)
                if 'ins_feat_s1' in predictions:
                    ins_s1 = (predictions['ins_feat_s1']
                              .permute(0, 1, 3, 4, 2)
                              .reshape(-1, predictions['ins_feat_s1'].shape[2]))
                    export_back_semantic_pca_ply(
                        points_xyz=pts_flat, semantic_feat=ins_s1,
                        save_file=os.path.join(save_path, f'{dataset_name}_ins_pca_s1_{i}.ply'),
                    )

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {
        f'{k}_{tag}': getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }
    results.setdefault('loss_refine_avg', float('inf'))

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(f'{prefix}_{name}', val, 1000 * epoch)

    return results


# ---------------------------------------------------------------------------
# One training step: process one full sequence chunk-by-chunk
# ---------------------------------------------------------------------------

def train_one_sequence(model, criterion, views_all, device, args, dtype, 
                       optimizer, loss_scaler, save_vis=False, save_dir=None, step_idx=None):
    n_chunks   = args.seq_len // args.chunk_size
    chunk_size = args.chunk_size

    sem_map, ins_map = make_voxel_maps(
        args.voxel_size, args.sem_dim, args.ins_dim, args.conf_scale
    )

    # Keep a pristine copy of global-frame pts for voxel memory.
    # normalize_gt() will reassign chunk['pts3d'] to a canonical scale/orientation,
    # so we must save the global pts BEFORE any normalization.
    gt_pts_all_global = views_all['pts3d'].clone()   # (B, T_total, H, W, 3) global

    # total_loss   = torch.tensor(0.0, device=device)
    total_loss_val = 0.0
    log_dict     = defaultdict(float)
    valid_chunks = 0

    all_pts, all_sem, all_ins = [], [], []

    for c in range(n_chunks):
        s = c * chunk_size
        e = s + chunk_size

        chunk = {
            k: v[:, s:e].contiguous()
            if (isinstance(v, torch.Tensor) and v.ndim >= 2) else v
            for k, v in views_all.items()
        }

        # Global-frame pts for this chunk (for memory query + update).
        # Must be saved before normalize_gt() which replaces chunk['pts3d'].
        gt_pts_global_chunk = gt_pts_all_global[:, s:e]   # (B, T_chunk, H, W, 3)

        sem_map_arg = sem_map if c > 0 else None
        ins_map_arg = ins_map if c > 0 else None
        # print(f'[Stage2V2] Processing chunk {c+1}/{n_chunks} with frames {s}-{e-1}...')
        with torch.autocast('cuda', dtype=dtype):
            predictions = model(
                chunk,
                sem_voxel_map=sem_map_arg,
                ins_voxel_map=ins_map_arg,
                pts_for_query=gt_pts_global_chunk,   # global coords for memory query
            )

        # ── Normalise GT to canonical frame, then scale-align predictions ──
        # The model predicts world_points/depth in its own normalised frame.
        # compute_point_loss / compute_depth_loss compare pred vs batch['pts3d']
        # directly, so both must be in the same (normalised) frame first.
        normalize_gt(chunk)             # modifies chunk['pts3d'] / depthmap / extrinsics
        align_pred_to_gt(predictions, chunk)   # scale-aligns pred world_points + depth

        if save_vis:
            # all_pts.append(predictions['world_points'].detach().reshape(-1, 3))
            all_pts.append(gt_pts_global_chunk.detach().reshape(-1, 3))
            all_sem.append(predictions['semantic_feat'].detach().permute(0, 1, 3, 4, 2).reshape(-1, args.sem_dim))
            all_ins.append(predictions['instance_feat'].detach().permute(0, 1, 3, 4, 2).reshape(-1, args.ins_dim))

        with torch.autocast('cuda', dtype=dtype):
            losses = criterion(predictions, chunk)

        chunk_loss = losses['objective']
        if not torch.isfinite(chunk_loss):
            print(f'[Stage2V2] non-finite loss at chunk {c}, skipping')
            continue

        # total_loss  = total_loss + chunk_loss
        loss_scaled = chunk_loss / (n_chunks * args.accum_iter)
        loss_scaler(
            loss_scaled,
            optimizer,
            clip_grad=1.0,
            parameters=model.trainable_params,
            update_grad=False  # 只累積梯度，先不更新權重
        )
        
        total_loss_val += chunk_loss.item()
        valid_chunks += 1
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                log_dict[k] += v.item()

        sem_conf = predictions.get(
            'semantic_conf',
            torch.ones(
                predictions['semantic_feat'].shape[:2] + (1,) +
                predictions['semantic_feat'].shape[-2:],
                device=device,
            ),
        )
        # Update voxel memory using GLOBAL-frame pts (not the normalised ones).
        update_voxel_maps(
            sem_map, ins_map,
            gt_pts=gt_pts_global_chunk,
            refined_sem=predictions['semantic_feat'],
            refined_ins=predictions['instance_feat'],
            sem_conf=sem_conf,
        )
        
        del predictions, losses, chunk, chunk_loss, loss_scaled

    # Vis sequence results and voxel memory state after processing all chunks
    if save_vis and save_dir is not None:
        dataset_name = views_all.get('dataset', [f'seq_step{step_idx}'])[0]
        if isinstance(dataset_name, torch.Tensor): dataset_name = f'seq_step{step_idx}'
        elif isinstance(dataset_name, list): dataset_name = str(dataset_name[0])

        # 1. 儲存整條 Sequence 的預測結果 (Concatenate all chunks)
        if len(all_pts) > 0:
            final_pts = torch.cat(all_pts, dim=0).float().cpu()
            final_sem = torch.cat(all_sem, dim=0).float().cpu()
            final_ins = torch.cat(all_ins, dim=0).float().cpu()

            export_back_semantic_pca_ply(points_xyz=final_pts, semantic_feat=final_sem, save_file=os.path.join(save_dir, f'{dataset_name}_train_seq_sem_pca.ply'))            
            # export_back_semantic_textmatch_ply(
            #     points_xyz=final_pts, semantic_feat=final_sem,
            #     text_feat=text_feat, color_table=color_table,
            #     save_file=os.path.join(save_dir, f'{dataset_name}_train_seq_sem_text.ply'),
            # )
            export_back_semantic_pca_ply(points_xyz=final_pts, semantic_feat=final_ins, save_file=os.path.join(save_dir, f'{dataset_name}_train_seq_ins_pca.ply'))
            export_instance_hdbscan_ply(
                points_xyz=final_pts, instance_feat=final_ins,
                save_file=os.path.join(save_dir, f'{dataset_name}_train_seq_ins_hdbscan.ply'),
                verbose=True,
            )

        # 2. 儲存 Voxel Memory Map 的狀態
        if sem_map._num_voxels > 0:
            try:
                # 呼叫內建 API，它會回傳 (centers, feat_avg)
                # centers 已經是轉換好的世界座標，feat_avg 已經是除以 conf 的平均特徵
                mem_pts, mem_sem_feat = sem_map.get_all()
                _, mem_ins_feat = ins_map.get_all()
                
                # 直接轉 numpy 即可，不用再乘 args.voxel_size
                mem_pts = mem_pts.float().cpu()
                mem_sem_feat = mem_sem_feat.float().cpu()
                mem_ins_feat = mem_ins_feat.float().cpu()
                
                export_back_semantic_pca_ply(points_xyz=mem_pts, semantic_feat=mem_sem_feat, save_file=os.path.join(save_dir, f'{dataset_name}_train_mem_sem_pca.ply'))
                export_back_semantic_pca_ply(points_xyz=mem_pts, semantic_feat=mem_ins_feat, save_file=os.path.join(save_dir, f'{dataset_name}_train_mem_ins_pca.ply'))
            except Exception as e:
                print(f"[Debug] Memory 匯出再次失敗: {e}")
                print(f"[Debug] Voxel map 內部可用的變數有: {dir(sem_map)}")

    if valid_chunks == 0:
        return {}
    # mean_loss = total_loss / valid_chunks
    mean_loss_val = total_loss_val / valid_chunks
    log_dict  = {k: v / valid_chunks for k, v in log_dict.items()}
    return mean_loss_val, log_dict


# ---------------------------------------------------------------------------
# Train epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                    loss_scaler, args, log_writer=None):
    model.train()
    model_without_ddp = model.module if hasattr(model, 'module') else model

    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    dtype  = get_dtype(args)
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    for step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + step / len(data_loader)

        if step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        views, views_all = batch
        for k in views_all:
            if isinstance(views_all[k], torch.Tensor):
                views_all[k] = views_all[k].to(device)

        save_vis = ((step % 400 == 0) or (step==0)) and misc.is_main_process()
        save_dir = os.path.join(args.output_dir, f'train_vis_ep{epoch}') if save_vis else None
        if save_vis:
            os.makedirs(save_dir, exist_ok=True)

        result = train_one_sequence(model_without_ddp, criterion, views_all, device, 
                                    args, dtype, optimizer, loss_scaler,
                                    save_vis=save_vis, save_dir=save_dir, step_idx=step)
        if not result:
            continue

        mean_loss, log_dict = result

        # loss_scaled = mean_loss / accum_iter
        # loss_scaler(
        #     loss_scaled,
        #     optimizer,
        #     clip_grad=1.0,
        #     parameters=model_without_ddp.trainable_params,
        #     update_grad=(step + 1) % accum_iter == 0,
        # )
        # if (step + 1) % accum_iter == 0:
        #     optimizer.zero_grad()

        if (step + 1) % accum_iter == 0:
            # 隨便傳一個 dummy loss 進去，只是為了觸發 scaler 的 unscale_ 和 optimizer.step()
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss_scaler(
                dummy_loss,
                optimizer,
                clip_grad=1.0,
                parameters=model_without_ddp.trainable_params,
                update_grad=True
            )
            optimizer.zero_grad()

        # metric_logger.update(loss=mean_loss.item())
        metric_logger.update(loss=mean_loss)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        for k, v in log_dict.items():
            if not k.endswith('_det') and k != 'objective':
                metric_logger.update(**{k: v})

        if log_writer is not None and (step + 1) % accum_iter == 0:
            epoch_1000x = int(epoch_f * 1000)
            # log_writer.add_scalar('train/loss', mean_loss.item(), epoch_1000x)
            log_writer.add_scalar('train/loss', mean_loss, epoch_1000x)
            for k, v in log_dict.items():
                if not k.endswith('_det') and k != 'objective':
                    log_writer.add_scalar(f'train/{k}', v, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    misc.init_distributed_mode(args)
    device = torch.device('cuda')

    torch.manual_seed(args.seed + misc.get_rank())
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────
    res      = eval(args.resolution)
    test_res = eval(args.test_resolution)

    train_dataset = ScannetppSequence(
        split='train', ROOT=args.data_root,
        resolution=[res], num_frames=args.seq_len,
        num_seq=1, stride=args.stride,
    )

    if args.test_dataset:
        test_dataset_str = args.test_dataset
    else:
        test_dataset_str = (
            f"Scannetpp_Arrow(split='val', "
            f"ROOT='{args.data_root}', "
            f"resolution=[{test_res}], "
            f"num_seq=1, num_frames={args.num_frames_test})"
        )

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)

    loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )

    loader_test = build_dataset(test_dataset_str, args.batch_size_test,
                                args.num_workers_test, test=True)

    # ── Model ─────────────────────────────────────────────────────────────
    stage1 = AMB3RStage1FullFT(
        metric_scale=True,
        sem_dim=args.sem_dim,
        ins_dim=args.ins_dim,
        clip_dim=args.clip_dim,
    )
    model = AMB3RStage2V2(
        stage1_model=stage1,
        sem_dim=args.sem_dim,
        ins_dim=args.ins_dim,
        voxel_resolutions=[args.voxel_res],
        interp_v2=args.interp_v2,
    ).to(device)

    model.load_stage1_weights(args.stage1_ckpt, data_type=args.amp or 'bf16')
    if args.stage2_ckpt:
        model.load_stage2_weights(args.stage2_ckpt, strict=False)

    n_params = sum(p.numel() for p in model.backend.parameters())
    print(f'[Stage2V2] Trainable backend parameters: {n_params / 1e6:.1f} M')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
    model_without_ddp = model.module if hasattr(model, 'module') else model

    # ── Optimiser (backend params only) ───────────────────────────────────
    optimizer = torch.optim.AdamW(
        model_without_ddp.trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_scaler = NativeScaler()

    # ── Criterion ─────────────────────────────────────────────────────────
    criterion = Stage2LossV2(
        camera={"weight": args.w_geo_camera, "loss_type": "l1"},
        depth={
            "weight":           args.w_geo_depth,
            "gradient_loss_fn": "grad",
            "valid_range":      0.98,
            "gamma": 1.0, "alpha": 0.2,
        },
        point={
            "weight":           args.w_geo_pts,
            "gradient_loss_fn": "normal",
            "valid_range":      0.98,
            "gamma": 1.0, "alpha": 0.2,
        },
        w_sem_align=args.w_sem_align,
        w_sem_memory=args.w_sem_memory,
        w_ins_contrast=args.w_ins_contrast,
        w_ins_memory=args.w_ins_memory,
    )

    # ── Logging ───────────────────────────────────────────────────────────
    log_writer = None
    if misc.is_main_process():
        log_writer = SummaryWriter(log_dir=str(output_dir))
        with open(output_dir / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

    # Copy source files for reproducibility
    if misc.is_main_process():
        rec_dir = output_dir / 'recording'
        rec_dir.mkdir(parents=True, exist_ok=True)
        for fname in os.listdir(_SANDBOX):
            if fname.endswith('.py'):
                copyfile(os.path.join(_SANDBOX, fname), str(rec_dir / fname))


    # ── Resume ────────────────────────────────────────────────────────────
    
    best_so_far = float('inf')
    start_epoch = 0
    last_ckpt = output_dir / 'checkpoint-last.pth'
    if last_ckpt.exists() and args.stage2_ckpt is None:
        state = torch.load(str(last_ckpt), map_location='cpu')
        s2_state = {k: v for k, v in state['model'].items() if k.startswith('backend.')}
        model_without_ddp.load_state_dict(s2_state, strict=False)
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        start_epoch = state.get('epoch', 0) + 1
        best_so_far = state.get('best_so_far', float('inf'))
        print(f'[Stage2V2] Resumed from epoch {start_epoch - 1}, best={best_so_far:.4f}')

        # ===== 新增：斷點續傳後，先對載入的 weight 進行一次 Eval =====
        if args.eval_freq > 0:
            print(f"[Stage2V2] Evaluating resumed checkpoint from epoch {start_epoch - 1}...")
            test_stats = test_one_epoch(
                model, criterion, loader_test, device, start_epoch - 1, args,
                log_writer=log_writer, prefix='test',
            )
            val_loss = test_stats.get('loss_refine_avg', float('inf'))
            print(f'[Stage2V2] Resumed Eval val loss: {val_loss:.4f}')
            
            # 如果這個剛 load 回來的 weight 表現更好，就更新 best
            if val_loss < best_so_far:
                best_so_far = val_loss
                if misc.is_main_process():
                    state['best_so_far'] = best_so_far
                    torch.save(state, output_dir / 'checkpoint-best.pth')
                    print(f'[Stage2V2] Updated best checkpoint upon resume!')

    # ── Training loop ─────────────────────────────────────────────────────
    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        torch.cuda.empty_cache()
        train_stats_log = {}
        test_stats_log  = {}

        # 1. ── Train ──────────────────────────────────────────────────────
        train_stats = train_one_epoch(
            model, criterion, loader_train, optimizer,
            device, epoch, loss_scaler, args, log_writer=log_writer,
        )
        train_stats_log = train_stats

        # 2. ── 先 Save Checkpoints (防止 Eval 爛掉白 Train) ────────────────
        if misc.is_main_process() and (epoch + 1) % args.save_freq == 0:
            state = {
                'epoch': epoch,
                'best_so_far': best_so_far,
                'model': {
                    k: v for k, v in model_without_ddp.state_dict().items()
                    if k.startswith('backend.')
                },
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
            }
            torch.save(state, output_dir / 'checkpoint-last.pth')
            if (epoch + 1) % 5 == 0:
                torch.save(state, output_dir / f'checkpoint-{epoch:04d}.pth')
            print(f'[Stage2V2] Saved checkpoint-last.pth at epoch {epoch}')

        # 3. ── Eval + PLY export ─────────────────────────────────────────
        new_best = False
        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
            test_stats = test_one_epoch(
                model, criterion, loader_test, device, epoch, args,
                log_writer=log_writer, prefix='test',
            )
            test_stats_log = test_stats

            val_loss = test_stats.get('loss_refine_avg', float('inf'))
            if val_loss < best_so_far:
                best_so_far = val_loss
                new_best    = True
                print(f'[Stage2V2] New best val loss: {best_so_far:.4f} at epoch {epoch}')

        # 4. ── 如果 Eval 表現最好，補存 Best Checkpoint ───────────────────
        if misc.is_main_process() and new_best:
            best_state = {
                'epoch': epoch,
                'best_so_far': best_so_far,
                'model': {
                    k: v for k, v in model_without_ddp.state_dict().items()
                    if k.startswith('backend.')
                },
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
            }
            torch.save(best_state, output_dir / 'checkpoint-best.pth')

        # 5. ── Log to file ───────────────────────────────────────────────
        if misc.is_main_process():
            log_stats = {
                'epoch': epoch,
                **{f'train_{k}': v for k, v in train_stats_log.items()},
                **{f'test_{k}':  v for k, v in test_stats_log.items()},
            }
            with open(output_dir / 'log.txt', 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

        if log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    print(f'Training time {str(datetime.timedelta(seconds=int(total_time)))}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
