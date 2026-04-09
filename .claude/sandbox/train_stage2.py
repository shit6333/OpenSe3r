"""
train_stage2.py
================
Stage-2 recurrent training: memory-conditioned semantic / instance refinement.

Key differences from train_semantic.py / training_semantic.py:
  - Dataset: ScannetppSequence (consecutive frames, num_frames = seq_len)
  - Model: AMB3RStage2 (frozen Stage-1 + trainable sem/ins memory heads)
  - Loop: chunk-by-chunk Truncated BPTT
      For each sequence:
        for c in range(n_chunks):
            - run Stage1 (no_grad)
            - query VoxelMemory  (stop-grad) — indexed by GT world pts
            - run Stage2 heads   (with grad)
            - accumulate loss
            - update memory      (detach)   — indexed by GT world pts
        backward total_loss / n_chunks

Coordinate alignment
--------------------
The voxel memory is always indexed by globally-consistent GT world points
(views_all['pts3d']), which come from absolute ScanNet++ camera poses and
are therefore consistent across all chunks in the same sequence.

This is simpler and cleaner than using predicted+aligned pts during training
(no need for coordinate_alignment()). The train/inference gap is acceptable:
at inference time the SLAM pipeline passes coordinate-aligned predicted pts
via the pts_for_query argument of AMB3RStage2.forward().

Usage:
    torchrun --nproc_per_node=1 .claude/sandbox/train_stage2.py \\
        --stage1_ckpt outputs/exp_stage1_wo_lora/checkpoint-best.pth \\
        --seq_len 24 --chunk_size 6 --stride 2 \\
        --batch_size 1 --accum_iter 4 --epochs 30 --lr 5e-4
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

_HERE = os.path.dirname(os.path.abspath(__file__))   # .claude/sandbox/
_ROOT = os.path.dirname(os.path.dirname(_HERE))               # project root (two levels up)
sys.path.insert(0, _HERE)    # sandbox root (for model_stage2, loss_stage2, etc.)
sys.path.insert(0, _ROOT)    # project root (for model_stage1_wo_lora, dpt_head, etc.)
sys.path.append(os.path.join(_ROOT, 'thirdparty'))            # thirdparty/croco

import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler

from amb3r.model_stage1_wo_lora import AMB3RStage1FullFT
from amb3r.model_stage2 import AMB3RStage2
from amb3r.loss_stage2 import Stage2Loss
from slam_semantic.semantic_voxel_map_v2 import VoxelFeatureMapV2

# Dataset is built from a string via eval() like in training_semantic.py
from amb3r.datasets import *       # registers all dataset classes
from amb3r.datasets.scannetpp_sequence import ScannetppSequence   # noqa: F401


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser('AMB3R Stage-2 training', add_help=False)

    # ── model ────────────────────────────────────────────────────────────────
    parser.add_argument('--stage1_ckpt', required=True,
                        help='Path to the Stage-1 checkpoint (checkpoint-best.pth)')
    parser.add_argument('--stage2_ckpt', default=None,
                        help='Optional: resume Stage-2 from this checkpoint')
    parser.add_argument('--sem_dim',  type=int, default=512)
    parser.add_argument('--ins_dim',  type=int, default=16)
    parser.add_argument('--clip_dim', type=int, default=512,
                        help='CLIP/LSeg feature dim (used by Stage-1 constructor only)')

    # ── sequence / chunk ─────────────────────────────────────────────────────
    parser.add_argument('--seq_len',    type=int, default=24,
                        help='Total frames per training sequence')
    parser.add_argument('--chunk_size', type=int, default=6,
                        help='Frames per chunk (Stage-1 forward window)')
    parser.add_argument('--stride',     type=int, default=2,
                        help='Temporal stride when sampling consecutive frames')

    # ── dataset ──────────────────────────────────────────────────────────────
    parser.add_argument('--data_root',
                        default='/mnt/HDD4/ricky/data/scannetpp_arrow',
                        help='Root of ScanNetPP Arrow dataset')
    parser.add_argument('--resolution', default='(518, 336)')
    parser.add_argument('--num_workers', type=int, default=2)

    # ── voxel memory ─────────────────────────────────────────────────────────
    parser.add_argument('--voxel_size',  type=float, default=0.05)
    parser.add_argument('--conf_scale',  type=float, default=1.0,
                        help='Confidence saturation threshold for memory mask')

    # ── optimiser ────────────────────────────────────────────────────────────
    parser.add_argument('--lr',           type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs',type=int,   default=1)
    parser.add_argument('--min_lr',       type=float, default=1e-6)

    # ── training ─────────────────────────────────────────────────────────────
    parser.add_argument('--epochs',     type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accum_iter', type=int, default=4)
    parser.add_argument('--amp',        choices=[False, 'bf16', 'fp16'], default='bf16')

    # ── loss weights ─────────────────────────────────────────────────────────
    parser.add_argument('--w_sem_align',    type=float, default=0.5)
    parser.add_argument('--w_sem_memory',   type=float, default=1.0)
    parser.add_argument('--w_ins_contrast', type=float, default=1.0)
    parser.add_argument('--w_ins_memory',   type=float, default=1.0)

    # ── misc ─────────────────────────────────────────────────────────────────
    parser.add_argument('--output_dir', default='./outputs/exp_stage2')
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--save_freq',  type=int, default=1)
    parser.add_argument('--eval_freq',  type=int, default=1)
    # DDP
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url',   default='env://')

    return parser


def get_dtype(args):
    if args.amp == 'bf16':  return torch.bfloat16
    if args.amp == 'fp16':  return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# voxel memory management
# ---------------------------------------------------------------------------

def make_voxel_maps(voxel_size: float, sem_dim: int, ins_dim: int,
                    conf_scale: float):
    sem_map = VoxelFeatureMapV2(voxel_size=voxel_size, feat_dim=sem_dim,
                                conf_scale=conf_scale)
    ins_map = VoxelFeatureMapV2(voxel_size=voxel_size, feat_dim=ins_dim,
                                conf_scale=conf_scale)
    return sem_map, ins_map


PATCH_SIZE = 14   # must match model_stage2.PATCH_SIZE

@torch.no_grad()
def update_voxel_maps(
    sem_map, ins_map,
    gt_pts: torch.Tensor,        # (B, T, H, W, 3) — globally-aligned GT world pts
    refined_sem: torch.Tensor,   # (B, T, C_sem, H, W) — pixel-aligned
    refined_ins: torch.Tensor,   # (B, T, C_ins, H, W) — pixel-aligned
    sem_conf: torch.Tensor,      # (B, T, 1,     H, W) — pixel-aligned
):
    """
    Update voxel maps after each chunk (stop-grad).

    All feature maps are at full image resolution (H, W).  We downsample to
    patch resolution (H//14, W//14) before voxel insertion for efficiency —
    consistent with the query resolution used in AMB3RStage2.forward().

    gt_pts must be in the same globally-consistent coordinate frame as the
    model's pts_for_query (GT world pts from the dataset during training).
    """
    import torch.nn.functional as F
    B, T, C_sem, H, W = refined_sem.shape
    Hp, Wp = H // PATCH_SIZE, W // PATCH_SIZE

    # downsample gt_pts to patch resolution
    pts = gt_pts.flatten(0, 1).float().permute(0, 3, 1, 2)    # (B*T, 3, H, W)
    pts = F.interpolate(pts, size=(Hp, Wp), mode='bilinear', align_corners=False)
    pts = pts.permute(0, 2, 3, 1).reshape(-1, 3).cpu()         # (B*T*Hp*Wp, 3)

    # downsample features to patch resolution before storing
    sem_p  = F.interpolate(refined_sem.flatten(0,1), size=(Hp, Wp), mode='bilinear', align_corners=False)
    ins_p  = F.interpolate(refined_ins.flatten(0,1), size=(Hp, Wp), mode='bilinear', align_corners=False)
    conf_p = F.interpolate(sem_conf.flatten(0,1),    size=(Hp, Wp), mode='bilinear', align_corners=False)

    sem_flat  = sem_p.permute(0, 2, 3, 1).reshape(-1, C_sem)
    ins_flat  = ins_p.permute(0, 2, 3, 1).reshape(-1, refined_ins.shape[2])
    conf_flat = conf_p.squeeze(1).reshape(-1)

    sem_map.update(pts, sem_flat.detach().float().cpu(), conf_flat.detach().float().cpu())
    ins_map.update(pts, ins_flat.detach().float().cpu(), conf_flat.detach().float().cpu())


# ---------------------------------------------------------------------------
# one training step (one full sequence, chunked)
# ---------------------------------------------------------------------------

def train_one_sequence(
    model: AMB3RStage2,
    criterion: Stage2Loss,
    views_all: dict,
    device: torch.device,
    args,
    dtype: torch.dtype,
) -> dict:
    """
    Process a single sequence (B=1, N=seq_len) chunk by chunk.

    views_all['pts3d'] has shape (B, T, H, W, 3) and contains GT world
    points in absolute ScanNet++ coordinates — consistent across all chunks.

    Returns a tuple (mean_loss, log_dict), or an empty dict on failure.
    """
    n_chunks   = args.seq_len // args.chunk_size
    chunk_size = args.chunk_size

    sem_map, ins_map = make_voxel_maps(
        args.voxel_size, args.sem_dim, args.ins_dim, args.conf_scale
    )

    # GT world pts for the full sequence — (B, seq_len, H, W, 3)
    gt_pts_all = views_all['pts3d']   # already on device

    total_loss   = torch.tensor(0.0, device=device)
    log_dict     = defaultdict(float)
    valid_chunks = 0

    for c in range(n_chunks):
        s = c * chunk_size
        e = s + chunk_size

        # ── Slice this chunk from views_all (B, T, ...) ───────────────────
        chunk = {
            k: v[:, s:e].contiguous() if (isinstance(v, torch.Tensor) and v.ndim >= 2)
               else v
            for k, v in views_all.items()
        }

        # GT pts for this chunk — used for voxel query AND update
        gt_pts_chunk = gt_pts_all[:, s:e]   # (B, chunk_size, H, W, 3)

        # ── Model forward ────────────────────────────────────────────────
        # Pass voxel maps (None on first chunk → heads handle gracefully)
        # Always pass gt_pts_chunk so Stage-2 indexes the voxel map with
        # globally-consistent coordinates.
        sem_map_arg = sem_map if c > 0 else None
        ins_map_arg = ins_map if c > 0 else None

        with torch.autocast('cuda', dtype=dtype):
            predictions = model(
                chunk,
                sem_voxel_map=sem_map_arg,
                ins_voxel_map=ins_map_arg,
                pts_for_query=gt_pts_chunk,
            )

        # ── Loss ────────────────────────────────────────────────────────
        with torch.autocast('cuda', dtype=dtype):
            losses = criterion(predictions, chunk)

        chunk_loss = losses['objective']
        if not torch.isfinite(chunk_loss):
            print(f'[Stage2] non-finite loss at chunk {c}, skipping')
            continue

        total_loss  = total_loss + chunk_loss
        valid_chunks += 1

        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                log_dict[k] += v.item()

        # ── Update voxel memory (stop-grad) ──────────────────────────────
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
    log_dict = {k: v / valid_chunks for k, v in log_dict.items()}
    return mean_loss, log_dict


# ---------------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, criterion, data_loader, optimizer, device, epoch, loss_scaler, args,
    log_writer=None,
):
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

        result = train_one_sequence(
            model, criterion, views_all, device, args, dtype
        )
        if not result:
            continue

        mean_loss, log_dict = result

        # scale for gradient accumulation
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

    return {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    misc.init_distributed_mode(args)
    device = torch.device('cuda')

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    # ── Build dataset ────────────────────────────────────────────────────────
    res     = eval(args.resolution)   # e.g. (518, 336)
    seq_len = args.seq_len

    train_dataset = ScannetppSequence(
        split='train',
        ROOT=args.data_root,
        resolution=[res],
        num_frames=seq_len,
        num_seq=1,
        stride=args.stride,
    )
    val_dataset = ScannetppSequence(
        split='val',
        ROOT=args.data_root,
        resolution=[res],
        num_frames=seq_len,
        num_seq=1,
        stride=args.stride,
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
    loader_val = torch.utils.data.DataLoader(
        val_dataset, sampler=sampler_val,
        batch_size=1, num_workers=0,
        pin_memory=False, drop_last=False,
    )

    # ── Build model ─────────────────────────────────────────────────────────
    stage1 = AMB3RStage1FullFT(
        metric_scale=True,
        sem_dim=args.sem_dim,
        ins_dim=args.ins_dim,
        clip_dim=args.clip_dim,
    )
    # AMB3RStage2 does not need clip_dim (it conditions on sem_s1, not CLIP directly)
    model = AMB3RStage2(
        stage1_model=stage1,
        sem_dim=args.sem_dim,
        ins_dim=args.ins_dim,
    ).to(device)

    model.load_stage1_weights(args.stage1_ckpt, data_type=args.amp or 'bf16')
    if args.stage2_ckpt:
        model.load_stage2_weights(args.stage2_ckpt, strict=False)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )

    model_without_ddp = model.module if hasattr(model, 'module') else model

    # ── Optimiser (only Stage-2 params) ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model_without_ddp.trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_scaler = NativeScaler()

    # ── Criterion ────────────────────────────────────────────────────────────
    criterion = Stage2Loss(
        w_sem_align=args.w_sem_align,
        w_sem_memory=args.w_sem_memory,
        w_ins_contrast=args.w_ins_contrast,
        w_ins_memory=args.w_ins_memory,
    )

    # ── Output dir / logging ─────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_writer = None
    if misc.is_main_process():
        log_writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
        with open(output_dir / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()

    for epoch in range(args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, loader_train, optimizer,
            device, epoch, loss_scaler, args, log_writer=log_writer,
        )

        # ── Save checkpoint ───────────────────────────────────────────────
        if misc.is_main_process() and (epoch + 1) % args.save_freq == 0:
            state = {
                'epoch': epoch,
                'model': {
                    k: v
                    for k, v in model_without_ddp.state_dict().items()
                    if k.startswith('semantic_memory_head.') or
                       k.startswith('instance_memory_head.')
                },
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
            }
            torch.save(state, output_dir / 'checkpoint-last.pth')
            if (epoch + 1) % 5 == 0:
                torch.save(state, output_dir / f'checkpoint-{epoch:04d}.pth')

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
        }
        if misc.is_main_process():
            with open(output_dir / 'log.txt', 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    print(f'Training time {str(datetime.timedelta(seconds=int(total_time)))}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
