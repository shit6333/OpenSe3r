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
  - Same   : truncated BPTT, chunk-by-chunk VoxelMemory, ScannetppSequence

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
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# ── project paths ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                     # sandbox root
sys.path.insert(0, os.path.dirname(_HERE))                   # project root
sys.path.append(os.path.join(os.path.dirname(_HERE), 'thirdparty'))

import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler

from amb3r.model_stage1_wo_lora import AMB3RStage1FullFT
from amb3r.model_stage2v2 import AMB3RStage2V2
from amb3r.loss_stage2v2 import Stage2LossV2
from slam_semantic.semantic_voxel_map_v2 import VoxelFeatureMapV2

from amb3r.datasets import *                                   # register datasets
from amb3r.datasets.scannetpp_sequence import ScannetppSequence


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

    # ── dataset ──────────────────────────────────────────────────────────
    p.add_argument('--data_root',  default='/mnt/HDD4/ricky/data/scannetpp_arrow')
    p.add_argument('--resolution', default='(518, 336)')
    p.add_argument('--num_workers',type=int, default=2)

    # ── voxel memory ─────────────────────────────────────────────────────
    p.add_argument('--voxel_size', type=float, default=0.05,
                   help='Voxel grid size for the temporal memory map')
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
    p.add_argument('--w_geo_depth',        type=float, default=1.0)
    p.add_argument('--w_geo_pts',          type=float, default=1.0)
    p.add_argument('--w_geo_camera',       type=float, default=0.5)
    p.add_argument('--geo_grad_loss_fn',   default='grad',
                   help='gradient loss fn for depth/point (grad|normal|None)')
    p.add_argument('--geo_valid_range',    type=float, default=0.98,
                   help='outlier filtering quantile for depth/point loss')

    # ── semantic / instance / memory loss weights ─────────────────────────
    p.add_argument('--w_sem_align',    type=float, default=0.5)
    p.add_argument('--w_sem_memory',   type=float, default=1.0)
    p.add_argument('--w_ins_contrast', type=float, default=1.0)
    p.add_argument('--w_ins_memory',   type=float, default=1.0)

    # ── misc ─────────────────────────────────────────────────────────────
    p.add_argument('--output_dir',  default='./outputs/exp_stage2v2')
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--print_freq',  type=int, default=20)
    p.add_argument('--save_freq',   type=int, default=1)
    # DDP
    p.add_argument('--world_size',  default=1, type=int)
    p.add_argument('--local_rank',  default=-1, type=int)
    p.add_argument('--dist_url',    default='env://')

    return p


def get_dtype(args):
    if args.amp == 'bf16': return torch.bfloat16
    if args.amp == 'fp16': return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Voxel memory helpers  (identical to train_stage2.py)
# ---------------------------------------------------------------------------

PATCH_SIZE = 14


def make_voxel_maps(voxel_size, sem_dim, ins_dim, conf_scale):
    sem_map = VoxelFeatureMapV2(voxel_size=voxel_size, feat_dim=sem_dim,
                                conf_scale=conf_scale)
    ins_map = VoxelFeatureMapV2(voxel_size=voxel_size, feat_dim=ins_dim,
                                conf_scale=conf_scale)
    return sem_map, ins_map


@torch.no_grad()
def update_voxel_maps(
    sem_map, ins_map,
    gt_pts: torch.Tensor,        # (B, T, H, W, 3)  globally-aligned GT world pts
    refined_sem: torch.Tensor,   # (B, T, C_sem, H, W)
    refined_ins: torch.Tensor,   # (B, T, C_ins, H, W)
    sem_conf: torch.Tensor,      # (B, T, 1, H, W)
):
    """Update voxel maps after each chunk (stop-grad).  Identical to V1."""
    B, T, C_sem, H, W = refined_sem.shape
    Hp, Wp = H // PATCH_SIZE, W // PATCH_SIZE

    pts = gt_pts.flatten(0, 1).float().permute(0, 3, 1, 2)      # (B*T, 3, H, W)
    pts = torch.nn.functional.interpolate(pts, size=(Hp, Wp), mode='bilinear', align_corners=False)
    pts = pts.permute(0, 2, 3, 1).reshape(-1, 3).cpu()           # (B*T*Hp*Wp, 3)

    sem_p  = torch.nn.functional.interpolate(refined_sem.flatten(0,1), size=(Hp, Wp), mode='bilinear', align_corners=False)
    ins_p  = torch.nn.functional.interpolate(refined_ins.flatten(0,1), size=(Hp, Wp), mode='bilinear', align_corners=False)
    conf_p = torch.nn.functional.interpolate(sem_conf.flatten(0,1),    size=(Hp, Wp), mode='bilinear', align_corners=False)

    sem_map.update(pts,
                   sem_p.permute(0,2,3,1).reshape(-1, C_sem).detach().float().cpu(),
                   conf_p.squeeze(1).reshape(-1).detach().float().cpu())
    ins_map.update(pts,
                   ins_p.permute(0,2,3,1).reshape(-1, refined_ins.shape[2]).detach().float().cpu(),
                   conf_p.squeeze(1).reshape(-1).detach().float().cpu())


# ---------------------------------------------------------------------------
# One training step: process one full sequence chunk-by-chunk
# ---------------------------------------------------------------------------

def train_one_sequence(model, criterion, views_all, device, args, dtype):
    """
    Truncated BPTT over one sequence.
    Returns (mean_loss_tensor, log_dict) or ({},) on failure.
    """
    n_chunks   = args.seq_len // args.chunk_size
    chunk_size = args.chunk_size

    sem_map, ins_map = make_voxel_maps(
        args.voxel_size, args.sem_dim, args.ins_dim, args.conf_scale
    )

    gt_pts_all = views_all['pts3d']   # (B, T_total, H, W, 3)

    total_loss   = torch.tensor(0.0, device=device)
    log_dict     = defaultdict(float)
    valid_chunks = 0

    for c in range(n_chunks):
        s = c * chunk_size
        e = s + chunk_size

        # slice chunk
        chunk = {
            k: v[:, s:e].contiguous()
            if (isinstance(v, torch.Tensor) and v.ndim >= 2) else v
            for k, v in views_all.items()
        }
        gt_pts_chunk = gt_pts_all[:, s:e]   # (B, chunk_size, H, W, 3)

        sem_map_arg = sem_map if c > 0 else None
        ins_map_arg = ins_map if c > 0 else None

        with torch.autocast('cuda', dtype=dtype):
            predictions = model(
                chunk,
                sem_voxel_map=sem_map_arg,
                ins_voxel_map=ins_map_arg,
                pts_for_query=gt_pts_chunk,
            )

        with torch.autocast('cuda', dtype=dtype):
            losses = criterion(predictions, chunk)

        chunk_loss = losses['objective']
        if not torch.isfinite(chunk_loss):
            print(f'[Stage2V2] non-finite loss at chunk {c}, skipping')
            continue

        total_loss  = total_loss + chunk_loss
        valid_chunks += 1
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                log_dict[k] += v.item()

        # Update voxel memory (stop-grad)
        sem_conf = predictions.get(
            'semantic_conf',
            torch.ones(
                predictions['semantic_feat'].shape[:2] + (1,) + predictions['semantic_feat'].shape[-2:],
                device=device,
            ),
        )
        update_voxel_maps(
            sem_map, ins_map,
            gt_pts=gt_pts_chunk,
            refined_sem=predictions['semantic_feat'],
            refined_ins=predictions['instance_feat'],
            sem_conf=sem_conf,
        )

    if valid_chunks == 0:
        return {}

    mean_loss = total_loss / valid_chunks
    log_dict  = {k: v / valid_chunks for k, v in log_dict.items()}
    return mean_loss, log_dict


# ---------------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                    loss_scaler, args, log_writer=None):
    model.train()
    model_without_ddp = model.module if hasattr(model, 'module') else model

    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    dtype      = get_dtype(args)
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

        result = train_one_sequence(model, criterion, views_all, device, args, dtype)
        if not result:
            continue

        mean_loss, log_dict = result

        loss_scaled = mean_loss / accum_iter
        loss_scaler(
            loss_scaled,
            optimizer,
            clip_grad=1.0,
            parameters=model_without_ddp.trainable_params,
            update_grad=(step + 1) % accum_iter == 0,
        )
        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        metric_logger.update(loss=mean_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        for k, v in log_dict.items():
            if not k.endswith('_det') and k != 'objective':
                metric_logger.update(**{k: v})

        if log_writer is not None and (step + 1) % accum_iter == 0:
            log_writer.add_scalar('train/loss', mean_loss.item(),
                                  epoch * len(data_loader) + step)

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    misc.init_distributed_mode(args)
    device = torch.device('cuda')

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    # ── Dataset ──────────────────────────────────────────────────────────
    res = eval(args.resolution)

    train_dataset = ScannetppSequence(
        split='train', ROOT=args.data_root,
        resolution=[res], num_frames=args.seq_len,
        num_seq=1, stride=args.stride,
    )
    val_dataset = ScannetppSequence(
        split='val', ROOT=args.data_root,
        resolution=[res], num_frames=args.seq_len,
        num_seq=1, stride=args.stride,
    )

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
        sampler_val   = torch.utils.data.DistributedSampler(val_dataset,   shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val   = torch.utils.data.SequentialSampler(val_dataset)

    loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )

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
            "gradient_loss_fn": args.geo_grad_loss_fn,
            "valid_range":      args.geo_valid_range,
            "gamma": 1.0, "alpha": 0.2,
        },
        point={
            "weight":           args.w_geo_pts,
            "gradient_loss_fn": "normal",
            "valid_range":      args.geo_valid_range,
            "gamma": 1.0, "alpha": 0.2,
        },
        w_sem_align=args.w_sem_align,
        w_sem_memory=args.w_sem_memory,
        w_ins_contrast=args.w_ins_contrast,
        w_ins_memory=args.w_ins_memory,
    )

    # ── Output dir / logging ──────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_writer = None
    if misc.is_main_process():
        log_writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
        with open(output_dir / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────
    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()

    for epoch in range(args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, loader_train, optimizer,
            device, epoch, loss_scaler, args, log_writer=log_writer,
        )

        # ── Checkpoint (backend weights only) ────────────────────────────
        if misc.is_main_process() and (epoch + 1) % args.save_freq == 0:
            state = {
                'epoch': epoch,
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

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if misc.is_main_process():
            with open(output_dir / 'log.txt', 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    print(f'Training time {str(datetime.timedelta(seconds=int(total_time)))}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
