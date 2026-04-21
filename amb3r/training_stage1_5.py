"""
Stage-1.5 training script.

Trains a VGGTAutoencoder that compresses concat(atl[4], atl[23]) → bottleneck
→ reconstruct → frozen DPT heads → supervised by CLIP GT (sem) + instance masks.

Only the autoencoder weights are updated.  Stage-1 backbone and DPT heads are
frozen.  The trained encoder is later used in Stage 2 (mem_mode=3) as a memory
feature extractor.

Usage:
    torchrun --nproc_per_node=1 train_stage1_5.py \
        --stage1_ckpt outputs/exp_stage1_wo_lora_long/checkpoint-last.pth \
        --data_root /path/to/data_arrow \
        --output_dir outputs/exp_stage1_5 \
        --bottleneck_dim 256 \
        --epochs 30 --lr 0.0003 --batch_size 1 --accum_iter 4
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

from amb3r.model_stage1_wo_lora import AMB3RStage1FullFT
from amb3r.model_stage1_5 import Stage1_5
from amb3r.loss_semantic import (
    MultitaskLoss, get_depth_loss, get_point_loss,
    compute_semantic_loss, compute_instance_loss,
)
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
from amb3r.tools.contrastive_loss import ContrastiveLoss

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ── argument parser ──────────────────────────────────────────────────────────

def get_args_parser():
    p = argparse.ArgumentParser('AMB3R Stage-1.5 training', add_help=False)

    # Model
    p.add_argument('--stage1_ckpt', required=True,
                   help='Path to trained Stage-1 checkpoint')
    p.add_argument('--bottleneck_dim', type=int, default=256)
    p.add_argument('--ae_hidden_dim', type=int, default=1024)
    p.add_argument('--sem_dim', type=int, default=512)
    p.add_argument('--ins_dim', type=int, default=16)
    p.add_argument('--clip_dim', type=int, default=512)

    # Data
    p.add_argument('--train_dataset', type=str, default=None)
    p.add_argument('--test_dataset', type=str, default=None)
    p.add_argument('--data_root', required=True, type=str)
    p.add_argument('--resolution', type=str, default='(518, 336)')
    p.add_argument('--num_frames', type=int, default=10,
                   help='Number of frames per sample (test)')
    p.add_argument('--num_frames_train', type=str, default='[2, 3]',
                   help='List of frame counts for training')

    # Loss weights
    p.add_argument('--w_sem', type=float, default=1.0)
    p.add_argument('--w_ins', type=float, default=1.0)
    p.add_argument('--w_recon', type=float, default=0.1,
                   help='Reconstruction L2 loss weight (optional regulariser)')

    # Training
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--batch_size', default=1, type=int)
    p.add_argument('--accum_iter', default=4, type=int)
    p.add_argument('--epochs', default=30, type=int)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int, default=1)
    p.add_argument('--amp', choices=[False, 'bf16', 'fp16'], default='bf16')

    # Logging / saving
    p.add_argument('--output_dir', default='./outputs/exp_stage1_5', type=str)
    p.add_argument('--eval_freq', type=int, default=1)
    p.add_argument('--save_freq', default=1, type=int)
    p.add_argument('--print_freq', default=20, type=int)
    p.add_argument('--eval_ply_n', type=int, default=5)
    p.add_argument('--auto_resume', action='store_true', default=True)

    # Distributed
    p.add_argument('--num_workers', default=1, type=int)
    p.add_argument('--world_size', default=1, type=int)
    p.add_argument('--local_rank', default=-1, type=int)
    p.add_argument('--dist_url', default='env://')

    return p


# ── helpers ──────────────────────────────────────────────────────────────────

def get_dtype(args):
    if args.amp:
        return torch.bfloat16 if args.amp == 'bf16' else torch.float16
    return torch.float32


def normalize_gt(views_all):
    """Normalise GT extrinsics/pts3d/depthmap in-place."""
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
    """Scale-invariant alignment of prediction dict."""
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

    for stage_idx in range(len(pred.get('pose_enc_list', []))):
        pred['pose_enc_list'][stage_idx][..., :3] *= scale_depth[..., None]

    return scale_depth


def instance_ids_to_colors(inst_ids: np.ndarray) -> np.ndarray:
    import colorsys
    colors = np.full((len(inst_ids), 3), 128, dtype=np.uint8)
    golden = (1 + 5 ** 0.5) / 2 - 1
    for iid in np.unique(inst_ids):
        if iid == 0:
            continue
        hue = (int(iid) * golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors[inst_ids == iid] = [int(r * 255), int(g * 255), int(b * 255)]
    return colors


def build_datasets(args):
    """Build train and test data loaders using Scannetpp Arrow format."""
    from amb3r.datasets import build_dataset

    trainset = args.train_dataset
    testset = args.test_dataset

    res_str = args.resolution
    num_frames_train = eval(args.num_frames_train)
    if args.train_dataset == None:
        trainset = (
            f"2000 @ ScannetppArrow(split='train', "
            f"ROOT='{args.data_root}', "
            f"resolution=[{res_str}], num_seq=1, num_frames={num_frames_train})"
        )
    if args.test_dataset == None:
        testset = (
            f"ScannetppArrow(split='val', "
            f"ROOT='{args.data_root}', "
            f"resolution={res_str}, num_seq=1, num_frames={args.num_frames})"
        )

    data_loader_train = build_dataset(
        trainset, args.batch_size, args.num_workers, test=False)
    data_loader_test = {
        'Scannetpp': build_dataset(testset, 1, 0, test=True)
    }
    return data_loader_train, data_loader_test


# ── Stage 1.5 loss ──────────────────────────────────────────────────────────

def compute_stage1_5_loss(pred, views_all, args, contrastive_loss_fn):
    """
    Compute semantic + instance + (optional) reconstruction losses.

    Semantic: cosine distance between decoded sem_feat and CLIP GT prototype.
    Instance: intra + inter contrastive on decoded ins_feat.
    Recon: L2 between original concat(early,late) and AE output (optional).
    """
    loss_dict = {}
    total_loss = torch.tensor(0.0, device=pred['semantic_feat'].device)

    # ── Semantic alignment loss (reuse MultitaskLoss logic) ──────────────
    if args.w_sem > 0:
        sem_dict = compute_semantic_loss(pred, views_all, weight=args.w_sem)
        total_loss = total_loss + sem_dict['loss_semantic']
        loss_dict['loss_semantic'] = sem_dict['loss_semantic'].detach()

    # ── Instance contrastive loss ────────────────────────────────────────
    if args.w_ins > 0:
        ins_dict = compute_instance_loss(
            pred, views_all, weight=args.w_ins,
            contrastive_loss_fn=contrastive_loss_fn,
        )
        total_loss = total_loss + ins_dict['loss_instance']
        loss_dict['loss_instance'] = ins_dict['loss_instance'].detach()
        loss_dict['loss_instance_intra'] = ins_dict['loss_instance_intra']
        loss_dict['loss_instance_inter'] = ins_dict['loss_instance_inter']

    loss_dict['objective'] = total_loss
    return loss_dict


# ── train one epoch ──────────────────────────────────────────────────────────

def train_one_epoch(model, data_loader, optimizer, device, epoch,
                    loss_scaler, args, contrastive_loss_fn, log_writer=None):
    assert torch.backends.cuda.matmul.allow_tf32

    model.train(True)
    model_without_ddp = model.module if hasattr(model, 'module') else model

    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    accum_iter = args.accum_iter

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
            pred = model.forward(views_all)

        # ---- Normalize GT --------------------------------------------------
        normalize_gt(views_all)
        align_pred_to_gt(pred, views_all)

        # ---- Loss ----------------------------------------------------------
        loss_dict = compute_stage1_5_loss(pred, views_all, args, contrastive_loss_fn)
        loss = loss_dict['objective']
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
                        update_grad=update_grad, clip_grad=1.0)

        if update_grad:
            optimizer.zero_grad()

        # ---- Geometry monitoring (no grad) ---------------------------------
        with torch.no_grad():
            loss_depth = check_and_fix_inf_nan(
                get_depth_loss(pred['depth'], views_all['depthmap'][..., None],
                               views_all['valid_mask']), 'loss_depth')
            loss_pts = check_and_fix_inf_nan(
                get_point_loss(pred['world_points'], views_all['pts3d'],
                               views_all['valid_mask']), 'loss_pts')

        # ---- Metric logging ------------------------------------------------
        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_depth=float(loss_depth.item()))
        metric_logger.update(loss_pts=float(loss_pts.item()))
        for key in ('loss_semantic', 'loss_instance',
                    'loss_instance_intra', 'loss_instance_inter'):
            if key in loss_dict:
                metric_logger.update(**{key: float(loss_dict[key].item())})

        if update_grad and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_reduce = misc.all_reduce_mean(loss_value)
            epoch_1000x = int(epoch_f * 1000)
            if log_writer is not None:
                log_writer.add_scalar('train_loss', loss_reduce, epoch_1000x)
                log_writer.add_scalar('train_lr', lr, epoch_1000x)
                for key in ('loss_semantic', 'loss_instance'):
                    if key in loss_dict:
                        log_writer.add_scalar(f'train_{key}',
                                              loss_dict[key].item(), epoch_1000x)

        del loss, pred, batch, loss_dict, views, views_all
        torch.cuda.empty_cache()

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ── test one epoch ───────────────────────────────────────────────────────────

@torch.no_grad()
def test_one_epoch(model, data_loader, device, epoch, args,
                   contrastive_loss_fn, log_writer=None, prefix='test'):
    model.eval()
    model_without_ddp = model.module if hasattr(model, 'module') else model

    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = f'Test Epoch: [{epoch}]'

    save_path = os.path.join(args.output_dir, f'eval_{epoch}')
    os.makedirs(save_path, exist_ok=True)

    # Text embeddings for semantic vis
    labels, _, default_color_table = get_scannet_label_and_color_map('scannet20')
    color_table = default_color_table[:len(labels)] if len(labels) <= len(default_color_table) \
        else np.tile(default_color_table,
                     (int(np.ceil(len(labels) / len(default_color_table))), 1))[:len(labels)]
    save_semantic_color_legend(
        labels=labels, color_table=color_table,
        save_file=os.path.join(save_path, 'scannet20_legend.png'),
        title='ScanNet20 Semantic Color Legend', ncols=2,
    )
    text_feat = build_text_embeddings(
        clip_model=model_without_ddp.stage1.lseg.clip_pretrained,
        tokenizer=clip.tokenize,
        labels=labels, device=device, template='a photo of a {}',
    )

    dtype = get_dtype(args)

    for i, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        views, views_all = batch
        for key in views_all:
            views_all[key] = views_all[key].to(device)

        with torch.autocast('cuda', dtype=dtype):
            pred = model.forward(views_all)

        normalize_gt(views_all)
        align_pred_to_gt(pred, views_all)

        loss_dict = compute_stage1_5_loss(pred, views_all, args, contrastive_loss_fn)

        # Geometry monitoring
        loss_depth = check_and_fix_inf_nan(
            get_depth_loss(pred['depth'], views_all['depthmap'][..., None],
                           views_all['valid_mask']), 'loss_depth')
        loss_pts = check_and_fix_inf_nan(
            get_point_loss(pred['world_points'], views_all['pts3d'],
                           views_all['valid_mask']), 'loss_pts')

        metric_logger.update(loss=float(loss_dict['objective'].item()))
        metric_logger.update(loss_depth=float(loss_depth.item()))
        metric_logger.update(loss_pts=float(loss_pts.item()))
        metric_logger.update(
            loss_refine=float(loss_pts.item() + loss_depth.item()))
        for key in ('loss_semantic', 'loss_instance',
                    'loss_instance_intra', 'loss_instance_inter'):
            if key in loss_dict:
                metric_logger.update(**{key: float(loss_dict[key].item())})

        # ── PLY visualisation ────────────────────────────────────────────
        if i < args.eval_ply_n and misc.is_main_process():
            dataset_name = views[0]['dataset'][0]
            color_np = views_all['images'].permute(0, 1, 3, 4, 2).reshape(-1, 3)
            color_np = color_np.clamp(0, 1).cpu().numpy()

            pts_np = views_all['pts3d'].detach().cpu().numpy().reshape(-1, 3)

            # GT RGB PLY
            trimesh.points.PointCloud(
                pts_np, colors=(color_np * 255).astype(np.uint8)
            ).export(os.path.join(save_path, f'{dataset_name}_gt_{i}.ply'))

            # Instance GT PLY
            inst_ids_np = views_all['instance_mask'].cpu().numpy().reshape(-1)
            inst_colors = instance_ids_to_colors(inst_ids_np)
            trimesh.points.PointCloud(
                pts_np, colors=inst_colors
            ).export(os.path.join(save_path, f'{dataset_name}_inst_gt_{i}.ply'))

            pts_flat = torch.from_numpy(pts_np)

            # ── Decoded AE semantic features ─────────────────────────────
            if 'semantic_feat' in pred:
                sem_feat = (pred['semantic_feat']
                            .permute(0, 1, 3, 4, 2)
                            .reshape(-1, pred['semantic_feat'].shape[2]))
                export_back_semantic_pca_ply(
                    points_xyz=pts_flat, semantic_feat=sem_feat,
                    save_file=os.path.join(save_path, f'{dataset_name}_ae_sem_pca_{i}.ply'),
                )
                sem_exp = (pred['semantic_feat_expanded']
                           .permute(0, 1, 3, 4, 2)
                           .reshape(-1, pred['semantic_feat_expanded'].shape[2]))
                export_back_semantic_textmatch_ply(
                    points_xyz=pts_flat, semantic_feat=sem_exp,
                    text_feat=text_feat, color_table=color_table,
                    save_file=os.path.join(save_path, f'{dataset_name}_ae_sem_text_{i}.ply'),
                )

            # ── Original (baseline) semantic features ────────────────────
            if 'semantic_feat_original' in pred:
                sem_orig = (pred['semantic_feat_original']
                            .permute(0, 1, 3, 4, 2)
                            .reshape(-1, pred['semantic_feat_original'].shape[2]))
                export_back_semantic_pca_ply(
                    points_xyz=pts_flat, semantic_feat=sem_orig,
                    save_file=os.path.join(save_path, f'{dataset_name}_orig_sem_pca_{i}.ply'),
                )

            # ── Decoded AE instance features ─────────────────────────────
            if 'instance_feat' in pred:
                ins_feat = (pred['instance_feat']
                            .permute(0, 1, 3, 4, 2)
                            .reshape(-1, pred['instance_feat'].shape[2]))
                export_back_semantic_pca_ply(
                    points_xyz=pts_flat, semantic_feat=ins_feat,
                    save_file=os.path.join(save_path, f'{dataset_name}_ae_ins_pca_{i}.ply'),
                )
                export_instance_hdbscan_ply(
                    points_xyz=pts_flat, instance_feat=ins_feat,
                    save_file=os.path.join(save_path, f'{dataset_name}_ae_ins_hdbscan_{i}.ply'),
                )

            # ── Bottleneck PCA (sanity check on compression quality) ─────
            if 'bottleneck' in pred:
                # bottleneck is [B*T, N_patch, bottleneck_dim], need to upsample to pixel-level for PLY
                B, T = views_all['images'].shape[:2]
                H, W = views_all['images'].shape[-2:]
                Hp, Wp = H // 14, W // 14
                bn = pred['bottleneck'].reshape(B * T, Hp, Wp, -1).permute(0, 3, 1, 2)
                bn_up = F.interpolate(bn.float(), size=(H, W), mode='bilinear', align_corners=False)
                bn_flat = bn_up.permute(0, 2, 3, 1).reshape(-1, bn_up.shape[1])
                export_back_semantic_pca_ply(
                    points_xyz=pts_flat, semantic_feat=bn_flat,
                    save_file=os.path.join(save_path, f'{dataset_name}_bottleneck_pca_{i}.ply'),
                )
                # export_instance_hdbscan_ply(
                #     points_xyz=pts_flat, instance_feat=bn_flat,
                #     save_file=os.path.join(save_path, f'{dataset_name}_bottleneck_hdbscan_{i}.ply'),
                # )

            # ── GT CLIP text-match (reference) ───────────────────────────
            if '_clip_feat_gt' in pred:
                gt_sem = (pred['_clip_feat_gt']
                          .permute(0, 1, 3, 4, 2)
                          .reshape(-1, pred['_clip_feat_gt'].shape[2]))
                export_back_semantic_textmatch_ply(
                    points_xyz=pts_flat, semantic_feat=gt_sem,
                    text_feat=text_feat, color_table=color_table,
                    save_file=os.path.join(save_path, f'{dataset_name}_sem_text_gt_{i}.ply'),
                )

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {
        f'{k}_{tag}': getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }
    results['loss_refine_avg'] = results.get('loss_refine_avg', float('inf'))
    results['loss_relative_avg'] = results.get('loss_refine_avg', float('inf'))

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(f'{prefix}_{name}', val, 1000 * epoch)

    return results


# ── main ─────────────────────────────────────────────────────────────────────

def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    print('output_dir:', args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save args
    if misc.is_main_process():
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    print('job dir:', os.path.dirname(os.path.realpath(__file__)))
    print(str(args).replace(', ', ',\n'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ---- Datasets ----------------------------------------------------------
    data_loader_train, data_loader_test = build_datasets(args)

    # ---- Model -------------------------------------------------------------
    print('Building Stage-1 model...')
    stage1 = AMB3RStage1FullFT(
        metric_scale=True,
        clip_dim=args.clip_dim,
        sem_dim=args.sem_dim,
        ins_dim=args.ins_dim,
    )
    print('Loading Stage-1 weights:', args.stage1_ckpt)
    stage1.load_weights(args.stage1_ckpt, data_type=args.amp or 'bf16', strict=False)

    model = Stage1_5(
        stage1_model=stage1,
        bottleneck_dim=args.bottleneck_dim,
        ae_hidden_dim=args.ae_hidden_dim,
    )
    model.to(device)

    if args.amp == 'fp16':
        model.stage1.front_end.model.aggregator.to(torch.float16)

    # ---- Contrastive loss fn -----------------------------------------------
    contrastive_loss_fn = ContrastiveLoss(
        inter_mode='hinge', inter_margin=0.2, normalize_feats=True,
    )

    # Print trainable params
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f'\nTrainable parameters ({len(trainable)}):')
    for n in trainable:
        print(' ', n)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable: {total_trainable:,}')

    # ---- Optimizer ---------------------------------------------------------
    model_without_ddp = model
    eff_bs = args.batch_size * args.accum_iter * misc.get_world_size()
    print(f'actual lr: {args.lr:.2e}  eff_batch_size: {eff_bs}')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True,
            static_graph=False,
        )
        model_without_ddp = model.module

    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # ---- Auto-resume -------------------------------------------------------
    start_epoch = 0
    best_so_far = float('inf')

    last_ckpt = os.path.join(args.output_dir, 'checkpoint-last.pth')
    if args.auto_resume and os.path.isfile(last_ckpt):
        print(f'Auto-resuming from {last_ckpt}')
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model_without_ddp.load_state_dict(ckpt['model'], strict=False)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt and ckpt['scaler'] is not None:
            loss_scaler.load_state_dict(ckpt['scaler'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        if 'best_so_far' in ckpt:
            best_so_far = ckpt['best_so_far']
        print(f'  Resumed: epoch={start_epoch}, best_so_far={best_so_far:.4f}')
        del ckpt

    log_writer = (
        SummaryWriter(log_dir=args.output_dir)
        if global_rank == 0 and args.output_dir else None
    )

    # ---- Copy source files -------------------------------------------------
    if misc.is_main_process():
        rec_dir = os.path.join(args.output_dir, 'recording', 'amb3r')
        os.makedirs(rec_dir, exist_ok=True)
        for src_dir in ['./', 'amb3r']:
            dst_dir = os.path.join(args.output_dir, 'recording', src_dir)
            os.makedirs(dst_dir, exist_ok=True)
            for f in os.listdir(src_dir):
                if f.endswith('.py'):
                    try:
                        copyfile(os.path.join(src_dir, f), os.path.join(dst_dir, f))
                    except Exception:
                        pass

    def save_model(epoch, fname):
        if not misc.is_main_process():
            return
        state = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': loss_scaler.state_dict(),
            'epoch': epoch,
            'best_so_far': best_so_far,
            'args': vars(args),
        }
        torch.save(state, os.path.join(args.output_dir, f'checkpoint-{fname}.pth'))

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

    # ---- Training loop -----------------------------------------------------
    print(f'Start training for {args.epochs} epochs (from epoch {start_epoch})')
    start_time = time.time()
    train_stats = test_stats = {}

    for epoch in range(start_epoch, args.epochs + 1):
        torch.cuda.empty_cache()

        if epoch > start_epoch:
            if args.save_freq and (epoch % args.save_freq == 0 or epoch == args.epochs):
                save_model(epoch - 1, 'last')

        # ---- Evaluation ------------------------------------------------
        new_best = False
        if epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(
                    model, testset, device, epoch, args,
                    contrastive_loss_fn=contrastive_loss_fn,
                    log_writer=log_writer, prefix=test_name,
                )
                test_stats[test_name] = stats

            test_loss = sum(s.get('loss_avg', 0) for s in test_stats.values()) / max(len(test_stats), 1)
            if test_loss < best_so_far:
                best_so_far = test_loss
                new_best = True
                print(f'New best loss: {best_so_far:.4f} at epoch {epoch}')

            if new_best:
                save_model(epoch, 'best')

        # ---- Training --------------------------------------------------
        if epoch < args.epochs:
            train_stats = train_one_epoch(
                model, data_loader_train, optimizer, device,
                epoch, loss_scaler, args,
                contrastive_loss_fn=contrastive_loss_fn,
                log_writer=log_writer,
            )

        write_log_stats(epoch, train_stats, test_stats)

    # Final save
    save_model(args.epochs - 1, 'last')

    total_time = time.time() - start_time
    print('Training time:', str(datetime.timedelta(seconds=int(total_time))))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if not hasattr(args, 'start_epoch'):
        args.start_epoch = 0
    train(args)
