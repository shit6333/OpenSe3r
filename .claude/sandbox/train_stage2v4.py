"""
train_stage2v4.py
=================
Stage-2 V4: Chunked Unrolled BPTT training.

  Frozen  : Stage-1 backbone (VGGT + LSeg + DPT heads)
  Trainable: MemoryFusionModule, MemoryTokenizer, LearnableNullToken

Key differences from train_stage2v2.py (V2 with PTv3 backend):
  - Model     : AMB3RStage2V4 (no PTv3 — lightweight cross-attention + MLP)
  - Loss      : Stage2V4Loss (sem_align + ins_contrastive + optional mem_consistency)
  - BPTT mode : FULL sequence (accumulate ALL chunk losses, single backward)
                vs. V2 which does per-chunk backward (truncated BPTT)
  - Memory    : DifferentiableVoxelMap (keeps grad_fn) vs. VoxelFeatureMapV2 (stop-grad)
  - No geometry re-prediction — Stage-1 geometry is used as-is.

Sequence lifecycle:
    ┌── for each training sequence ──────────────────────────────────────────┐
    │  voxel_map = model.make_voxel_map()    # fresh, no history             │
    │  total_loss = 0                                                         │
    │  for chunk_idx in 0..n_chunks-1:                                       │
    │      preds, M_r, W_c, pts = model.forward_chunk(frames, voxel_map)    │
    │      total_loss += criterion(preds, batch)['objective']                │
    │      AMB3RStage2V4.update_voxel_map(voxel_map, M_r, W_c, pts)         │
    │      # ↑ grad_fn KEPT — no detach within sequence                      │
    │  (total_loss / n_chunks).backward()   # single backward                │
    │  optimizer.step()                                                       │
    │  voxel_map.clear() / discard                                            │
    └────────────────────────────────────────────────────────────────────────┘

Gradient checkpointing:
    Pass --use_checkpoint to wrap MemoryFusion+Tokenizer with
    torch.utils.checkpoint, saving peak memory at the cost of recomputation.

Usage:
    torchrun --nproc_per_node=1 .claude/sandbox/train_stage2v4.py \\
        --stage1_ckpt outputs/exp_stage1_wo_lora/checkpoint-best.pth \\
        --seq_len 16 --chunk_size 4 --stride 2 \\
        --batch_size 1 --accum_iter 4 --epochs 30 --lr 3e-4 \\
        --use_checkpoint
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

# ── project paths ─────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_PROJ   = os.path.dirname(_HERE)

sys.path.insert(0, _HERE)          # sandbox root (for local sandbox imports)
sys.path.insert(0, _PROJ)          # project root
sys.path.append(os.path.join(_PROJ, 'thirdparty'))

import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler

from amb3r.model_stage1_wo_lora import AMB3RStage1FullFT
from amb3r.model_stage2v4 import AMB3RStage2V4, Stage2V4Loss

from amb3r.datasets import *                          # register all dataset classes
from amb3r.datasets.scannetpp_sequence import ScannetppSequence


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args_parser():
    p = argparse.ArgumentParser('AMB3R Stage-2 V4 training', add_help=False)

    # ── model ─────────────────────────────────────────────────────────────
    p.add_argument('--stage1_ckpt', required=True,
                   help='Path to Stage-1 checkpoint (checkpoint-best.pth)')
    p.add_argument('--stage2_ckpt', default=None,
                   help='Optional: resume Stage-2 V4 from this checkpoint')
    p.add_argument('--sem_dim',   type=int, default=512)
    p.add_argument('--ins_dim',   type=int, default=16)
    p.add_argument('--clip_dim',  type=int, default=512)
    p.add_argument('--mem_dim',   type=int, default=128,
                   help='Implicit memory token dim')
    p.add_argument('--hidden_dim', type=int, default=256,
                   help='MemoryFusionModule cross-attention hidden dim')
    p.add_argument('--num_heads',  type=int, default=4,
                   help='Attention heads in MemoryFusionModule')
    p.add_argument('--ema_alpha',  type=float, default=0.5,
                   help='EMA weight for inter-frame memory update')
    p.add_argument('--use_checkpoint', action='store_true',
                   help='Gradient checkpointing for fusion+tokenizer (saves memory)')

    # ── sequence / chunk ──────────────────────────────────────────────────
    p.add_argument('--seq_len',    type=int, default=16,
                   help='Total frames per training sequence')
    p.add_argument('--chunk_size', type=int, default=4,
                   help='Frames per BPTT chunk (seq_len must be divisible)')
    p.add_argument('--stride',     type=int, default=2)

    # ── dataset ───────────────────────────────────────────────────────────
    p.add_argument('--data_root',  default='/mnt/HDD4/ricky/data/scannetpp_arrow')
    p.add_argument('--resolution', default='(518, 336)')
    p.add_argument('--num_workers', type=int, default=2)

    # ── voxel memory ──────────────────────────────────────────────────────
    p.add_argument('--voxel_size', type=float, default=0.05,
                   help='Spatial voxel grid cell size (metres)')

    # ── optimiser ─────────────────────────────────────────────────────────
    p.add_argument('--lr',            type=float, default=3e-4)
    p.add_argument('--weight_decay',  type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int,   default=1)
    p.add_argument('--min_lr',        type=float, default=1e-6)

    # ── training ──────────────────────────────────────────────────────────
    p.add_argument('--epochs',     type=int, default=30)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--accum_iter', type=int, default=4)
    p.add_argument('--amp',
                   choices=[False, 'bf16', 'fp16'], default='bf16',
                   type=lambda x: False if x == 'False' else x)

    # ── loss weights ──────────────────────────────────────────────────────
    p.add_argument('--w_sem_align',    type=float, default=0.5)
    p.add_argument('--w_ins_contrast', type=float, default=1.0)
    p.add_argument('--w_mem_consist',  type=float, default=0.1,
                   help='Memory token consistency weight (0 to disable)')

    # ── misc ──────────────────────────────────────────────────────────────
    p.add_argument('--output_dir',  default='./outputs/exp_stage2v4')
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
    Full BPTT over one complete training sequence.

    Accumulates losses from ALL chunks and returns a single mean_loss tensor
    whose grad_fn spans the entire sequence (no detach between chunks).
    The caller is responsible for calling mean_loss.backward().

    Returns
    -------
    dict with:
        'mean_loss' : scalar tensor (None if all chunks invalid)
        'log_dict'  : per-loss averages for logging
    """
    assert args.seq_len % args.chunk_size == 0, \
        f"seq_len ({args.seq_len}) must be divisible by chunk_size ({args.chunk_size})"
    n_chunks = args.seq_len // args.chunk_size

    # Fresh voxel map for this sequence — cleared after backward
    voxel_map = model.make_voxel_map()

    gt_pts_all = views_all.get('pts3d')   # [B, seq_len, H, W, 3] or None

    total_loss   = torch.tensor(0.0, device=device)
    log_dict     = defaultdict(float)
    valid_chunks = 0

    for c in range(n_chunks):
        s = c * args.chunk_size
        e = s + args.chunk_size

        # Slice chunk (keep all non-temporal keys intact)
        chunk = {
            k: (v[:, s:e].contiguous()
                if isinstance(v, torch.Tensor) and v.ndim >= 2 else v)
            for k, v in views_all.items()
        }
        gt_pts_chunk = gt_pts_all[:, s:e] if gt_pts_all is not None else None

        # ── Forward chunk (with gradient, except frozen Stage-1) ────────
        with torch.autocast('cuda', dtype=dtype):
            predictions, M_readout, W_conf, pts_flat = model.forward_chunk(
                chunk,
                voxel_map,
                pts_for_query=gt_pts_chunk,
            )

        # ── Loss ─────────────────────────────────────────────────────────
        with torch.autocast('cuda', dtype=dtype):
            losses = criterion(predictions, chunk)

        chunk_loss = losses['objective']

        if not torch.isfinite(chunk_loss):
            print(f'[Stage2V4] Non-finite loss at chunk {c}, skipping.')
            # Still update the voxel map with detached features so later
            # chunks have valid memory, even if this chunk's grad is skipped.
            AMB3RStage2V4.update_voxel_map(
                voxel_map,
                M_readout.detach(),
                W_conf.detach(),
                pts_flat.detach(),
            )
            continue

        # Accumulate — do NOT backward here (full BPTT: backward once at end)
        total_loss   = total_loss + chunk_loss
        valid_chunks += 1
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                log_dict[k] += v.item()

        # ── VoxelMap update ───────────────────────────────────────────────
        # CRITICAL: do NOT detach M_readout or W_conf between chunks.
        # Their grad_fn creates the backward-through-time chain.
        # The EMA operation inside update() preserves this chain.
        AMB3RStage2V4.update_voxel_map(voxel_map, M_readout, W_conf, pts_flat)

    if valid_chunks == 0:
        voxel_map.clear()
        return {}

    mean_loss = total_loss / valid_chunks
    log_out   = {k: v / valid_chunks for k, v in log_dict.items()}

    # voxel_map is no longer needed after this call returns; it will be garbage-
    # collected, but the tensors inside _store are still referenced by mean_loss's
    # computation graph and will be freed after loss.backward() completes.

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

        # batch = (views, views_all) — views_all has the full seq_len frames
        views, views_all = batch
        for k in views_all:
            if isinstance(views_all[k], torch.Tensor):
                views_all[k] = views_all[k].to(device)

        # ── Full BPTT over one sequence ──────────────────────────────────
        result = train_one_sequence(
            model_without_ddp, criterion, views_all, device, args, dtype
        )
        if not result:
            # All chunks were invalid (e.g. NaN pts); skip this sequence
            optimizer.zero_grad()
            continue

        mean_loss = result['mean_loss']
        log_dict  = result['log_dict']

        # ── Backward + grad accumulation ─────────────────────────────────
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

        # ── Logging ──────────────────────────────────────────────────────
        metric_logger.update(loss=mean_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        for k, v in log_dict.items():
            if not k.endswith('_det') and k != 'objective':
                metric_logger.update(**{k: v})

        if log_writer is not None and (step + 1) % accum_iter == 0:
            log_writer.add_scalar(
                'train/loss', mean_loss.item(),
                epoch * len(data_loader) + step,
            )

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    misc.init_distributed_mode(args)
    device = torch.device('cuda')

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    # ── Dataset ───────────────────────────────────────────────────────────
    res = eval(args.resolution)

    train_dataset = ScannetppSequence(
        split='train', ROOT=args.data_root,
        resolution=[res], num_frames=args.seq_len,
        num_seq=1, stride=args.stride,
    )

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)

    loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    stage1 = AMB3RStage1FullFT(
        metric_scale=True,
        sem_dim=args.sem_dim,
        ins_dim=args.ins_dim,
        clip_dim=args.clip_dim,
    )

    model = AMB3RStage2V4(
        stage1_model=stage1,
        mem_dim=args.mem_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        voxel_size=args.voxel_size,
        ema_alpha=args.ema_alpha,
        use_checkpoint=args.use_checkpoint,
    ).to(device)

    # Load Stage-1 weights (frozen baseline)
    model.load_stage1_weights(args.stage1_ckpt, data_type=args.amp or 'bf16')

    # Optionally resume Stage-2 V4 trainable weights
    if args.stage2_ckpt:
        state = torch.load(args.stage2_ckpt, map_location='cpu')
        if 'model' in state:
            state = state['model']
        # Only load keys belonging to trainable Stage-2 modules
        s2_keys = {k: v for k, v in state.items()
                   if any(k.startswith(p) for p in
                          ('null_token.', 'memory_fusion.', 'memory_tokenizer.'))}
        missing, unexpected = model.load_state_dict(s2_keys, strict=False)
        print(f'[Stage2V4] Resumed from {args.stage2_ckpt}: '
              f'{len(s2_keys)} tensors, missing={len(missing)}, '
              f'unexpected={len(unexpected)}')

    n_trainable = sum(p.numel() for p in model.trainable_params)
    n_frozen    = sum(p.numel() for p in model.stage1.parameters())
    print(f'[Stage2V4] Trainable: {n_trainable/1e6:.2f}M | '
          f'Frozen (Stage-1): {n_frozen/1e6:.1f}M')
    print(f'[Stage2V4] Seq={args.seq_len}, Chunk={args.chunk_size}, '
          f'n_chunks={args.seq_len // args.chunk_size}, '
          f'BPTT=full-sequence')
    print(f'[Stage2V4] GradCheckpoint={args.use_checkpoint}')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
    model_without_ddp = model.module if hasattr(model, 'module') else model

    # ── Optimiser (Stage-2 trainable params only) ─────────────────────────
    optimizer = torch.optim.AdamW(
        model_without_ddp.trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_scaler = NativeScaler()

    # ── Criterion ─────────────────────────────────────────────────────────
    criterion = Stage2V4Loss(
        w_sem_align=args.w_sem_align,
        w_ins_contrast=args.w_ins_contrast,
        w_mem_consist=args.w_mem_consist,
    ).to(device)

    # ── Output dir / logging ──────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_writer = None
    if misc.is_main_process():
        log_writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
        with open(output_dir / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f'[Stage2V4] Output dir: {output_dir}')

    # ── Training loop ─────────────────────────────────────────────────────
    print(f'Start Stage-2 V4 training for {args.epochs} epochs.')
    start_time = time.time()

    for epoch in range(args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, loader_train, optimizer,
            device, epoch, loss_scaler, args, log_writer=log_writer,
        )

        # ── Checkpoint — save ONLY trainable Stage-2 weights ─────────────
        if misc.is_main_process() and (epoch + 1) % args.save_freq == 0:
            s2_state = {
                k: v for k, v in model_without_ddp.state_dict().items()
                if any(k.startswith(p) for p in
                       ('null_token.', 'memory_fusion.', 'memory_tokenizer.'))
            }
            ckpt_data = {
                'epoch':     epoch,
                'model':     s2_state,
                'optimizer': optimizer.state_dict(),
                'args':      vars(args),
            }
            torch.save(ckpt_data, output_dir / 'checkpoint-last.pth')
            if (epoch + 1) % 5 == 0:
                torch.save(ckpt_data, output_dir / f'checkpoint-{epoch:04d}.pth')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if misc.is_main_process():
            with open(output_dir / 'log.txt', 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    print(f'Training time {str(datetime.timedelta(seconds=int(total_time)))}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
