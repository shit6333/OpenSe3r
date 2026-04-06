"""
Stage-1 training script for AMB3RStage1.

What is trained:
    clip_patch_fusion   cross-attention CLIP->patch injection
    lora_*              LoRA adapters in VGGT frame/global blocks
    semantic_head       PatchConditionedDPTHead
    instance_head       PatchConditionedDPTHead
    sem_expander        FeatureExpander
    instance_token      learnable instance embedding

Loss (geometry + semantic + instance):
    loss_camera             camera pose (rotation, translation, focal)
    loss_depth              depth estimation
    loss_point              3D point reconstruction
    loss_semantic           per-instance CLIP prototype pull
    loss_semantic_consistency  cross-view consistency
    loss_instance           scene intra/inter contrastive
    loss_instance_consistency  cross-view consistency

Usage:
    torchrun --nproc_per_node=1 training_stage1.py \
        --batch_size 1 --accum_iter 4 --epochs 30 --lr 0.0001 \
        --lora_r 4 --lora_last_n 8 \
        --output_dir ./outputs/exp_stage1
"""

import os
import sys
import math
import time
import json
import torch
import trimesh
import argparse
import datetime
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from typing import Sized
from pathlib import Path
from shutil import copyfile
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

# ── project root on path ────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'thirdparty'))

from amb3r.model_stage1 import AMB3RStage1
from amb3r.loss_semantic import MultitaskLoss, get_depth_loss, get_point_loss
from amb3r.datasets import build_dataset
from amb3r.tools.semantic_vis_utils import (
    export_back_semantic_pca_ply,
    export_back_semantic_textmatch_ply,
    get_scannet_label_and_color_map,
    build_text_embeddings,
    save_semantic_color_legend,
)
from amb3r.vis_instance_hdbscan import export_instance_hdbscan_ply

import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from vggt.train_utils.general import check_and_fix_inf_nan
from vggt.train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from moge.moge.train.losses import scale_invariant_alignment
from lang_seg.modules.models.lseg_net import clip

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ── GPU-aware dataset / resolution selection ────────────────────────────────

def get_resolution_by_gpu():
    gpu_name = torch.cuda.get_device_name(0).lower()
    if '4090' in gpu_name:
        num_frames      = list(range(2, 3))
        num_frames_test = 10
        res_str         = "[(518, 392), (518, 336), (518, 294), (518, 266), (518, 210), (518, 154)]"
        test_res_str    = "(518, 392)"
        batch_test      = 1
    else:
        num_frames      = list(range(3, 5))
        num_frames_test = 8
        res_str         = "[(518, 336), (518, 294), (518, 266), (518, 210), (518, 154)]"
        test_res_str    = "(518, 336)"
        batch_test      = 1

    trainset = (
        f"2000 @ Scannetpp(split='train', "
        f"ROOT='/mnt/HDD4/ricky/data/data/InsScene-15K/processed_scannetpp_v2/data', "
        f"resolution={res_str}, num_seq=1, num_frames={num_frames})"
    )
    testset = (
        f"Scannetpp(split='val', "
        f"ROOT='/mnt/HDD4/ricky/data/data/InsScene-15K/processed_scannetpp_v2/data', "
        f"resolution={test_res_str}, num_seq=1, num_frames={num_frames_test})"
    )
    return res_str, num_frames, trainset, testset, batch_test


# ── argument parser ──────────────────────────────────────────────────────────

def get_args_parser():
    _, _, trainset, testset, batch_size_test = get_resolution_by_gpu()

    p = argparse.ArgumentParser('AMB3R Stage-1 training', add_help=False)

    # Model
    p.add_argument('--lora_r',        type=int,   default=4)
    p.add_argument('--lora_alpha',    type=int,   default=16)
    p.add_argument('--lora_dropout',  type=float, default=0.05)
    p.add_argument('--lora_last_n',   type=int,   default=8,
                   help='Apply LoRA to the last N frame/global blocks')
    p.add_argument('--lora_on_mlp',   action='store_true', default=False)
    p.add_argument('--sem_dim',       type=int,   default=512)
    p.add_argument('--ins_dim',       type=int,   default=16)
    p.add_argument('--clip_dim',      type=int,   default=512)

    # Checkpoints
    p.add_argument('--vggt_ckpt',  default='./checkpoints/VGGT.pt')
    p.add_argument('--lseg_ckpt',
                   default='./checkpoints/demo_e200.ckpt')
    p.add_argument('--pretrained', default=None,
                   help='Resume from a Stage-1 checkpoint')

    # Data
    p.add_argument('--train_dataset', default=trainset, type=str)
    p.add_argument('--test_dataset',  default=testset,  type=str)

    # Loss weights — geometry
    p.add_argument('--camera_weight', type=float, default=5.0)
    p.add_argument('--depth_weight',  type=float, default=1.0)
    p.add_argument('--point_weight',  type=float, default=1.0)
    # Loss weights — semantic / instance
    p.add_argument('--semantic_weight',      type=float, default=1.0)
    p.add_argument('--semantic_cons_weight', type=float, default=0.2)
    p.add_argument('--instance_weight',      type=float, default=1.0)
    p.add_argument('--instance_cons_weight', type=float, default=0.2)

    # Training
    p.add_argument('--seed',          default=0,   type=int)
    p.add_argument('--batch_size',    default=1,   type=int)
    p.add_argument('--batch_size_test', default=batch_size_test, type=int)
    p.add_argument('--accum_iter',    default=4,   type=int)
    p.add_argument('--epochs',        default=30,  type=int)
    p.add_argument('--lr',            type=float,  default=1e-4)
    p.add_argument('--blr',           type=float,  default=1.5e-4)
    p.add_argument('--min_lr',        type=float,  default=1e-6)
    p.add_argument('--weight_decay',  type=float,  default=0.05)
    p.add_argument('--warmup_epochs', type=int,    default=1)
    p.add_argument('--amp', choices=[False, 'bf16', 'fp16'], default='bf16')

    # Logging / saving
    p.add_argument('--output_dir',  default='./outputs/exp_stage1', type=str)
    p.add_argument('--eval_freq',   type=int,  default=1)
    p.add_argument('--save_freq',   default=1, type=int)
    p.add_argument('--keep_freq',   default=2, type=int)
    p.add_argument('--print_freq',  default=20, type=int)

    # Distributed
    p.add_argument('--num_workers',      default=1,  type=int)
    p.add_argument('--num_workers_test', default=0,  type=int)
    p.add_argument('--world_size',       default=1,  type=int)
    p.add_argument('--local_rank',       default=-1, type=int)
    p.add_argument('--dist_url',         default='env://')

    return p


# ── trainable parameter selection ───────────────────────────────────────────

TRAINABLE_KEYWORDS = [
    'clip_patch_fusion',
    'lora_',
    'semantic_head',
    'instance_head',
    'sem_expander',
    'instance_token',
]


def freeze_non_stage1_params(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = any(kw in name for kw in TRAINABLE_KEYWORDS)


# ── dtype helper ─────────────────────────────────────────────────────────────

def get_dtype(args):
    if args.amp:
        return torch.bfloat16 if args.amp == 'bf16' else torch.float16
    return torch.float32


# ── GT normalisation + scale-align single prediction ────────────────────────

def normalize_gt(views_all):
    """Normalise GT extrinsics / pts3d / depthmap in-place (same as training_semantic.py)."""
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
    """Scale-invariant alignment of a single prediction dict. Returns scale_depth."""
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

    for stage_idx in range(len(pred['pose_enc_list'])):
        pred['pose_enc_list'][stage_idx][..., :3] *= scale_depth[..., None]

    return scale_depth



def instance_ids_to_colors(inst_ids: np.ndarray) -> np.ndarray:
    """
    Map integer instance IDs -> RGB uint8 colors.
    ID == 0 (background / ignore) -> gray [128, 128, 128].
    Each ID > 0 gets a visually distinct color via golden-angle hue cycling.
    """
    import colorsys
    golden  = 0.6180339887
    colors  = np.full((len(inst_ids), 3), 128, dtype=np.uint8)
    for iid in np.unique(inst_ids):
        if iid == 0:
            continue
        hue     = (int(iid) * golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors[inst_ids == iid] = [int(r * 255), int(g * 255), int(b * 255)]
    return colors


# ── test epoch ───────────────────────────────────────────────────────────────

@torch.no_grad()
def test_one_epoch(model, criterion, data_loader, device, epoch, args,
                   log_writer=None, prefix='test'):
    model.eval()
    model_without_ddp = model.module if hasattr(model, 'module') else model

    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = f'Test Epoch: [{epoch}]'

    save_path = os.path.join(args.output_dir, f'eval_{epoch}')
    os.makedirs(save_path, exist_ok=True)

    # Text embeddings for semantic vis
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
        clip_model=model_without_ddp.lseg.clip_pretrained,
        tokenizer=clip.tokenize,
        labels=labels,
        device=device,
        template='a photo of a {}',
    )

    dtype = get_dtype(args)

    for i, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        views, views_all = batch
        for key in views_all:
            views_all[key] = views_all[key].to(device)

        with torch.autocast('cuda', dtype=dtype):
            pred_all = model.forward(views_all)

        # Normalize GT + scale-align prediction
        normalize_gt(views_all)
        pred = pred_all[0]
        align_pred_to_gt(pred, views_all)

        # Loss
        loss_dict = criterion(pred, views_all)

        # Geometry monitoring
        loss_depth = check_and_fix_inf_nan(
            get_depth_loss(pred['depth'], views_all['depthmap'][..., None],
                           views_all['valid_mask']), 'loss_depth')
        loss_pts   = check_and_fix_inf_nan(
            get_point_loss(pred['world_points'], views_all['pts3d'],
                           views_all['valid_mask']), 'loss_pts')

        metric_logger.update(loss=float(loss_dict['objective'].item()))
        metric_logger.update(loss_depth=float(loss_depth.item()))
        metric_logger.update(loss_pts=float(loss_pts.item()))
        if 'loss_camera' in loss_dict:
            metric_logger.update(loss_camera=float(loss_dict['loss_camera'].item()))
        metric_logger.update(
            loss_refine=float(loss_pts.item() + loss_depth.item()
                              + loss_dict.get('loss_camera', torch.tensor(0.)).item())
        )
        for key in ('loss_semantic', 'loss_semantic_consistency',
                    'loss_instance', 'loss_instance_consistency'):
            if key in loss_dict:
                metric_logger.update(**{key: float(loss_dict[key].item())})

        # Visualise first 5 batches (main process only)
        if i < 5 and misc.is_main_process():
            dataset_name = views[0]['dataset'][0]
            color_np = ((views_all['images'].permute(0, 1, 3, 4, 2).reshape(-1, 3)
                         + 1.0) / 2.0).cpu().numpy()

            # GT point cloud (RGB)
            pcd_gt_np = views_all['pts3d'].detach().cpu().numpy().reshape(-1, 3)
            trimesh.points.PointCloud(
                pcd_gt_np, colors=(color_np * 255).astype(np.uint8)
            ).export(os.path.join(save_path, f'{dataset_name}_gt_{i}.ply'))

            # Instance GT PLY (each instance ID -> unique color, shared across all views)
            inst_ids_np = views_all['instance_mask'].cpu().numpy().reshape(-1)
            inst_colors = instance_ids_to_colors(inst_ids_np)
            trimesh.points.PointCloud(
                pcd_gt_np, colors=inst_colors
            ).export(os.path.join(save_path, f'{dataset_name}_inst_gt_{i}.ply'))

            # Pred point cloud
            pcd_pred_np = pred['world_points'].detach().cpu().numpy().reshape(-1, 3)
            trimesh.points.PointCloud(
                pcd_pred_np, colors=(color_np * 255).astype(np.uint8)
            ).export(os.path.join(save_path, f'{dataset_name}_pred_{i}.ply'))

            if 'semantic_feat' in pred:
                pts_flat = pred['world_points'].reshape(-1, 3)

                # Semantic PCA
                sem_feat = (pred['semantic_feat']
                            .permute(0, 1, 3, 4, 2)
                            .reshape(-1, pred['semantic_feat'].shape[2]))
                export_back_semantic_pca_ply(
                    points_xyz=pts_flat, semantic_feat=sem_feat,
                    save_file=os.path.join(save_path, f'{dataset_name}_sem_pca_{i}.ply'),
                )

                # Semantic text-match (pred)
                sem_exp = (pred['semantic_feat_expanded']
                           .permute(0, 1, 3, 4, 2)
                           .reshape(-1, pred['semantic_feat_expanded'].shape[2]))
                export_back_semantic_textmatch_ply(
                    points_xyz=pts_flat, semantic_feat=sem_exp,
                    text_feat=text_feat, color_table=color_table,
                    save_file=os.path.join(save_path, f'{dataset_name}_sem_text_{i}.ply'),
                )

                # Semantic text-match (GT CLIP)
                gt_sem = (pred['_clip_feat_gt']
                          .permute(0, 1, 3, 4, 2)
                          .reshape(-1, pred['_clip_feat_gt'].shape[2]))
                export_back_semantic_textmatch_ply(
                    points_xyz=pts_flat, semantic_feat=gt_sem,
                    text_feat=text_feat, color_table=color_table,
                    save_file=os.path.join(save_path, f'{dataset_name}_sem_text_gt_{i}.ply'),
                )

            if 'instance_feat' in pred:
                pts_flat = pred['world_points'].reshape(-1, 3)
                ins_feat = (pred['instance_feat']
                            .permute(0, 1, 3, 4, 2)
                            .reshape(-1, pred['instance_feat'].shape[2]))
                export_back_semantic_pca_ply(
                    points_xyz=pts_flat, semantic_feat=ins_feat,
                    save_file=os.path.join(save_path, f'{dataset_name}_ins_pca_{i}.ply'),
                )
                export_instance_hdbscan_ply(
                    points_xyz=pts_flat, instance_feat=ins_feat,
                    save_file=os.path.join(save_path, f'{dataset_name}_ins_hdbscan_{i}.ply'),
                )

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {
        f'{k}_{tag}': getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }
    results['loss_refine_avg']   = results.get('loss_refine_avg',   float('inf'))
    results['loss_relative_avg'] = results.get('loss_refine_avg',   float('inf'))

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(f'{prefix}_{name}', val, 1000 * epoch)

    return results


# ── train epoch ──────────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, data_loader, optimizer, device,
                    epoch, loss_scaler, args, log_writer=None):
    assert torch.backends.cuda.matmul.allow_tf32

    model.train(True)
    model_without_ddp = model.module if hasattr(model, 'module') else model

    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header     = f'Epoch: [{epoch}]'
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir:', log_writer.log_dir)

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()
    dtype = get_dtype(args)

    for data_iter_step, batch in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, header)):

        epoch_f = epoch + data_iter_step / len(data_loader)

        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        views, views_all = batch
        for key in views_all:
            views_all[key] = views_all[key].to(device)

        # ---- Forward -------------------------------------------------------
        with torch.autocast('cuda', dtype=dtype):
            pred_all = model.forward(views_all)

        # ---- Normalize GT --------------------------------------------------
        normalize_gt(views_all)

        # ---- Scale-align single Stage-1 prediction -------------------------
        pred = pred_all[0]
        align_pred_to_gt(pred, views_all)

        # ---- Loss ----------------------------------------------------------
        loss_dict  = criterion(pred, views_all)
        loss       = loss_dict['objective']
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f'Loss is {loss_value}, stopping training', force=True)
            sys.exit(1)

        loss = loss / accum_iter
        update_grad = (data_iter_step + 1) % accum_iter == 0

        if loss_scaler is None:
            loss.backward()
            if update_grad:
                optimizer.step()
                optimizer.zero_grad()
        else:
            loss_scaler(loss, optimizer,
                        parameters=model.parameters(),
                        update_grad=update_grad)

        if update_grad:
            optimizer.zero_grad()

        # ---- Geometry monitoring (no grad) ---------------------------------
        with torch.no_grad():
            loss_depth = check_and_fix_inf_nan(
                get_depth_loss(pred['depth'], views_all['depthmap'][..., None],
                               views_all['valid_mask']), 'loss_depth')
            loss_pts   = check_and_fix_inf_nan(
                get_point_loss(pred['world_points'], views_all['pts3d'],
                               views_all['valid_mask']), 'loss_pts')

        # ---- Metric logging ------------------------------------------------
        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_depth=float(loss_depth.item()))
        metric_logger.update(loss_pts=float(loss_pts.item()))
        if 'loss_camera' in loss_dict:
            metric_logger.update(loss_camera=float(loss_dict['loss_camera'].item()))
            metric_logger.update(
                loss_refine=float(loss_pts.item() + loss_depth.item()
                                  + loss_dict['loss_camera'].item())
            )
        for key in ('loss_semantic', 'loss_semantic_consistency',
                    'loss_instance', 'loss_instance_consistency',
                    'loss_instance_intra', 'loss_instance_inter'):
            if key in loss_dict:
                metric_logger.update(**{key: float(loss_dict[key].item())})

        if update_grad and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_reduce = misc.all_reduce_mean(loss_value)
            epoch_1000x = int(epoch_f * 1000)
            if log_writer is not None:
                log_writer.add_scalar('train_loss',       loss_reduce,       epoch_1000x)
                log_writer.add_scalar('train_lr',         lr,                epoch_1000x)
                log_writer.add_scalar('train_loss_depth', loss_depth.item(), epoch_1000x)
                log_writer.add_scalar('train_loss_pts',   loss_pts.item(),   epoch_1000x)
                if 'loss_camera' in loss_dict:
                    log_writer.add_scalar('train_loss_camera',
                                          loss_dict['loss_camera'].item(), epoch_1000x)
                for key in ('loss_semantic', 'loss_semantic_consistency',
                            'loss_instance', 'loss_instance_consistency'):
                    if key in loss_dict:
                        log_writer.add_scalar(f'train_{key}',
                                              loss_dict[key].item(), epoch_1000x)

        del loss, pred, batch, pred_all, loss_dict, views, views_all
        torch.cuda.empty_cache()

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ── main ─────────────────────────────────────────────────────────────────────

def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    print('output_dir:', args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    last_ckpt  = os.path.join(args.output_dir, 'checkpoint-last.pth')
    args.resume = last_ckpt if os.path.isfile(last_ckpt) else None

    print('job dir:', os.path.dirname(os.path.realpath(__file__)))
    print(str(args).replace(', ', ',\n'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ---- Datasets ----------------------------------------------------------
    print('Building train dataset:', args.train_dataset)
    data_loader_train = build_dataset(
        args.train_dataset, args.batch_size, args.num_workers, test=False
    )
    print('Building test dataset(s):', args.test_dataset)
    data_loader_test = {
        ds.split('(')[0]: build_dataset(ds, args.batch_size_test,
                                        args.num_workers_test, test=True)
        for ds in args.test_dataset.split('+')
    }

    # ---- Model -------------------------------------------------------------
    model = AMB3RStage1(
        metric_scale=True,
        clip_dim=args.clip_dim,
        sem_dim=args.sem_dim,
        ins_dim=args.ins_dim,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_last_n=args.lora_last_n,
        lora_on_mlp=args.lora_on_mlp,
        vggt_ckpt=args.vggt_ckpt,
        lseg_ckpt=args.lseg_ckpt,
    )
    model.to(device)

    # ---- Loss (geometry + semantic + instance) -----------------------------
    _cam_cfg   = {"weight": args.camera_weight, "loss_type": "l1"}
    _depth_cfg = {"weight": args.depth_weight,  "gradient_loss_fn": "grad",
                  "valid_range": 0.98, "gamma": 1.0, "alpha": 0.2}
    _pt_cfg    = {"weight": args.point_weight,  "gradient_loss_fn": "normal",
                  "valid_range": 0.98, "gamma": 1.0, "alpha": 0.2}
    _sem_cfg   = {"weight": args.semantic_weight,
                  "consistency_weight": args.semantic_cons_weight}
    _ins_cfg   = {"weight": args.instance_weight,
                  "consistency_weight": args.instance_cons_weight}

    train_criterion = MultitaskLoss(
        camera=_cam_cfg, depth=_depth_cfg, point=_pt_cfg,
        semantic=_sem_cfg, instance=_ins_cfg,
    ).to(device)
    test_criterion = MultitaskLoss(
        camera=_cam_cfg, depth=_depth_cfg, point=_pt_cfg,
        semantic=_sem_cfg, instance=_ins_cfg,
    ).to(device)

    model_without_ddp = model

    if args.pretrained and not args.resume:
        print('Loading Stage-1 pretrained:', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(ckpt.get('model', ckpt), strict=False)

    if args.amp == 'fp16':
        model.front_end.model.aggregator.to(torch.float16)

    # ---- Freeze non-Stage-1 params -----------------------------------------
    freeze_non_stage1_params(model)

    trainable = [n for n, p in model_without_ddp.named_parameters() if p.requires_grad]
    print(f'\nTrainable parameters ({len(trainable)}):')
    for n in trainable:
        print(' ', n)

    # ---- Optimizer ---------------------------------------------------------
    eff_bs = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_bs / 256
    print(f'base lr: {args.lr * 256 / eff_bs:.2e}')
    print(f'actual lr: {args.lr:.2e}')
    print(f'accum_iter: {args.accum_iter}  eff_batch_size: {eff_bs}')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True,
            static_graph=False,
        )
        model_without_ddp = model.module

    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer    = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler  = NativeScaler()

    def save_model(epoch, fname, best_so_far):
        misc.save_model(
            args=args, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler,
            epoch=epoch, fname=fname, best_so_far=best_so_far,
        )

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            log_stats = dict(epoch=epoch,
                             **{f'train_{k}': v for k, v in train_stats.items()})
            for test_name, stats in test_stats.items():
                log_stats.update({f'{test_name}_{k}': v for k, v in stats.items()})
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    best_so_far, _ = misc.load_model(
        args=args, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler,
    )
    if best_so_far is None:
        best_so_far = float('inf')

    log_writer = (
        SummaryWriter(log_dir=args.output_dir)
        if global_rank == 0 and args.output_dir else None
    )

    # Copy source files for reproducibility
    os.makedirs(os.path.join(args.output_dir, 'recording', 'amb3r'), exist_ok=True)
    for src_dir in ['./', 'amb3r']:
        dst_dir = os.path.join(args.output_dir, 'recording', src_dir)
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            if f.endswith('.py'):
                copyfile(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    train_stats = test_stats = {}

    for epoch in range(args.start_epoch, args.epochs + 1):
        torch.cuda.empty_cache()

        if epoch > args.start_epoch:
            if args.save_freq and (epoch % args.save_freq == 0 or epoch == args.epochs):
                save_model(epoch - 1, 'last', best_so_far)

        # ---- Evaluation ------------------------------------------------
        new_best = False
        if epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
            test_stats = {}
            test_loss     = 0.0
            test_relative = 0.0
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(
                    model, test_criterion, testset, device, epoch,
                    args=args, log_writer=log_writer, prefix=test_name,
                )
                test_stats[test_name] = stats
                test_loss     += stats.get('loss_refine_avg',   0.0)
                test_relative += stats.get('loss_relative_avg', 0.0)

            test_loss     /= max(len(data_loader_test), 1)
            test_relative /= max(len(data_loader_test), 1)

            if test_relative < best_so_far:
                best_so_far = test_relative
                new_best    = True
                print(f'New best loss: {best_so_far:.4f} at epoch {epoch}')

            if new_best:
                save_model(epoch, 'best', best_so_far)

        # ---- Training --------------------------------------------------
        if epoch < args.epochs:
            train_stats = train_one_epoch(
                model, train_criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler, args,
                log_writer=log_writer,
            )

        write_log_stats(epoch, train_stats, test_stats)

    total_time = time.time() - start_time
    print('Training time:', str(datetime.timedelta(seconds=int(total_time))))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if not hasattr(args, 'start_epoch'):
        args.start_epoch = 0
    train(args)
