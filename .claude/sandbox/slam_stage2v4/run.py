"""
run.py  —  Entry point for Stage-2 V4 Semantic SLAM.

Config priority:  slam_config_v4.yaml  (defaults)  <  CLI args  (overrides)

Usage (basic, mem_mode=3 default):
    python slam_stage2v4/run.py \
        --stage1_ckpt   checkpoints/amb3r_semantic.pt \
        --stage1_5_ckpt checkpoints/stage1_5.pth \
        --stage2_ckpt   outputs/exp_stage2v4_mode3/checkpoint-best.pth \
        --data_path     /path/to/scene

Usage (ScanNet GT depth + GT pose, semantic/instance + voxel_store PCA export):
    python slam_stage2v4/run.py \
        --stage1_ckpt   checkpoints/amb3r_semantic.pt \
        --stage1_5_ckpt checkpoints/stage1_5.pth \
        --stage2_ckpt   outputs/exp_stage2v4_mode3/checkpoint-best.pth \
        --data_path     /path/to/scene0001_00 \
        --demo_type     scannet \
        --resolution    518 336 \
        --use_gt_depth \
        --save_semantic --save_instance --save_voxel_store
"""

import os
import sys
import time
import argparse

import torch
import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# from amb3r.model_semantic import AMB3R
from amb3r.model_stage1_wo_lora import AMB3RStage1FullFT
from amb3r.model_stage2v4 import AMB3RStage2V4
from amb3r.tools.pts_vis import get_pts_mask
from amb3r.tools.semantic_vis_utils import (
    get_scannet_label_and_color_map,
    build_text_embeddings,
    save_semantic_color_legend,
    export_semantic_voxels_ply,
    export_semantic_per_point_ply,
    export_instance_voxels_ply,
    export_instance_per_point_ply,
)
from amb3r.tools.voxel_store_vis import export_voxel_store_pca_ply

from slam_stage2v4.pipeline import AMB3RV4_VO
from slam_stage2v4.pipeline_gt import AMB3RV4_VO_GT
from slam_stage2v4.pipeline_gt_depth import AMB3RV4_VO_GT_Depth
from slam_semantic.datasets.demo import DemoDataset
# from slam_semantic.datasets.scannet_slam import ScannetDemoDataset
from slam_semantic.datasets.scannet_slam_gt_depth import ScannetDemoDatasetWithDepth
from lang_seg.modules.models.lseg_net import clip

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ── Argument parser ──────────────────────────────────────────────────────────

def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Stage-2 V4 Semantic SLAM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument('--data_path',    type=str, default='./demo')
    parser.add_argument('--demo_name',    type=str, default='demo')
    parser.add_argument('--results_path', type=str, default='./outputs/slam_v4/demo')
    parser.add_argument('--demo_type',    type=str, default='demo',
                        choices=['demo', 'scannet'])
    parser.add_argument('--resolution',   type=int, nargs=2, default=None,
                        help='Image resolution (H W), e.g. 518 336')

    # ── Checkpoints ───────────────────────────────────────────────────────
    parser.add_argument('--stage1_ckpt',    type=str,
                        default='./checkpoints/amb3r_semantic.pt')
    parser.add_argument('--stage2_ckpt',    type=str, default='')
    parser.add_argument('--stage1_5_ckpt',  type=str, default='',
                        help='Stage-1.5 AE checkpoint (required for mem_mode=3)')
    parser.add_argument('--ae_bottleneck_dim', type=int, default=256)

    # ── Config override (None = use yaml value) ───────────────────────────
    parser.add_argument('--cfg_path',     type=str,   default='')
    parser.add_argument('--device',       type=str,   default=None,
                        help='Override yaml device')
    parser.add_argument('--mem_mode',     type=int,   default=None,
                        choices=[1, 2, 3])
    parser.add_argument('--mem_voxel_size', type=float, default=None)

    # ── GT modes ──────────────────────────────────────────────────────────
    parser.add_argument('--use_gt_pose',  action='store_true', default=False)
    parser.add_argument('--use_gt_depth', action='store_true', default=False,
                        help='GT depth + GT poses. Takes priority over --use_gt_pose.')

    # ── Export ────────────────────────────────────────────────────────────
    parser.add_argument('--target_point_count', type=int,   default=6_000_000)
    parser.add_argument('--edge_mask',          type=bool,  default=True)
    parser.add_argument('--conf_threshold',     type=float, default=0.0)
    parser.add_argument('--save_res',           type=bool,  default=True)
    parser.add_argument('--save_semantic',       action='store_true', default=False)
    parser.add_argument('--save_instance',       action='store_true', default=False)
    parser.add_argument('--save_voxel_store',    action='store_true', default=False,
                        help='Save voxel_store (model memory) as PCA-coloured PLY')
    parser.add_argument('--save_semantic_npz',   action='store_true', default=False)
    parser.add_argument('--save_instance_npz',   action='store_true', default=False)
    parser.add_argument('--label_set',           type=str,  default='scannet20',
                        choices=['scannet20', 'scannet200'])

    return parser


# ── Config loading ───────────────────────────────────────────────────────────

def build_cfg(args) -> OmegaConf:
    """yaml provides defaults; any CLI arg != None overrides yaml."""
    cfg_path = args.cfg_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'slam_config_v4.yaml'
    )
    cfg = OmegaConf.load(cfg_path)

    if args.device          is not None: cfg.device          = args.device
    if args.mem_mode        is not None: cfg.mem_mode        = args.mem_mode
    if args.mem_voxel_size  is not None: cfg.mem_voxel_size  = args.mem_voxel_size

    return cfg


# ── Model loading ────────────────────────────────────────────────────────────

def load_stage2v4_model(args, cfg) -> AMB3RStage2V4:
    # Stage-1 backbone
    # stage1 = AMB3R(metric_scale=True, clip_dim=512, sem_dim=512, ins_dim=16)
    stage1 = AMB3RStage1FullFT(metric_scale=True, clip_dim=512, sem_dim=512, ins_dim=16)
    if os.path.isfile(args.stage1_ckpt):
        stage1.load_weights(args.stage1_ckpt, data_type='bf16', strict=False)
        print(f"[Stage-1] Loaded: {args.stage1_ckpt}")
    else:
        print(f"[Stage-1] WARNING: checkpoint not found at {args.stage1_ckpt}")

    # AE encoder for mem_mode=3
    ae_encoder = None
    if cfg.mem_mode == 3:
        if not args.stage1_5_ckpt:
            raise ValueError('--stage1_5_ckpt is required for mem_mode=3')
        from amb3r.model_stage1_5 import VGGTAutoencoder
        ae_full = VGGTAutoencoder(
            input_dim=4096,
            bottleneck_dim=args.ae_bottleneck_dim,
        )
        ae_ckpt    = torch.load(args.stage1_5_ckpt, map_location='cpu',
                                weights_only=False)
        ae_state   = ae_ckpt.get('model', ae_ckpt)
        ae_weights = {k[len('autoencoder.'):]: v
                      for k, v in ae_state.items()
                      if k.startswith('autoencoder.')}
        ae_full.load_state_dict(ae_weights, strict=True)
        ae_encoder = ae_full.encoder
        print(f'[AE encoder] Loaded from {args.stage1_5_ckpt} '
              f'(bottleneck_dim={args.ae_bottleneck_dim})')

    model = AMB3RStage2V4(
        stage1_model=stage1,
        mem_mode=cfg.mem_mode,
        mem_voxel_size=cfg.mem_voxel_size,
        ae_encoder=ae_encoder,
        ae_bottleneck_dim=args.ae_bottleneck_dim,
    )

    if args.stage2_ckpt and os.path.isfile(args.stage2_ckpt):
        ckpt  = torch.load(args.stage2_ckpt, map_location='cpu')
        state = ckpt.get('model', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Stage-2 V4] Loaded: {args.stage2_ckpt}")
        if missing:
            print(f"  Missing    : {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected : {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    elif args.stage2_ckpt:
        print(f"[Stage-2 V4] WARNING: checkpoint not found at {args.stage2_ckpt}")
    else:
        print("[Stage-2 V4] No Stage-2 checkpoint — using random fusion weights.")

    model.cuda().eval()
    return model


# ── Coordinate helpers ───────────────────────────────────────────────────────

def transform_points(points, T):
    orig_shape = points.shape
    pts_flat = points.reshape(-1, 3)
    ones = np.ones((pts_flat.shape[0], 1), dtype=pts_flat.dtype)
    pts_h = np.concatenate([pts_flat, ones], axis=1)
    return (T @ pts_h.T).T[:, :3].reshape(orig_shape)


def left_multiply_poses(poses, T):
    return T[None] @ poses


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = get_args_parser()
    args   = parser.parse_args()
    cfg    = build_cfg(args)

    os.makedirs(args.results_path, exist_ok=True)

    print(f"\nEffective SLAM config:")
    print(f"  mem_mode            = {cfg.mem_mode}")
    print(f"  mem_voxel_size      = {cfg.mem_voxel_size} m")
    print(f"  semantic_voxel_size = "
          f"{getattr(cfg, 'semantic_voxel_size', 0.05)} m")
    print(f"  device              = {cfg.device}")
    print(f"  map_init_window     = {cfg.map_init_window}")
    print(f"  map_every           = {cfg.map_every}\n")

    # ── Dataset ───────────────────────────────────────────────────────────
    resolution = tuple(args.resolution) if args.resolution else (518, 336)
    if args.demo_type == 'scannet':
        data = ScannetDemoDatasetWithDepth(ROOT=args.data_path, resolution=resolution)
    else:
        data = DemoDataset(ROOT=args.data_path, resolution=resolution)

    # ── Model ─────────────────────────────────────────────────────────────
    model = load_stage2v4_model(args, cfg)

    # ── Data ──────────────────────────────────────────────────────────────
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    _, views_all = batch

    images     = views_all['images']
    num_frames = images.shape[1]
    print(f"Processing '{args.demo_name}': {num_frames} frames, "
          f"resolution {images.shape[-2]}×{images.shape[-1]}")

    # ── Pipeline ──────────────────────────────────────────────────────────
    cfg_path = args.cfg_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'slam_config_v4.yaml'
    )
    t0 = time.time()

    if args.use_gt_depth and args.demo_type == 'scannet':
        print('Mode: GT depth + GT poses (AMB3RV4_VO_GT_Depth)')
        pipeline = AMB3RV4_VO_GT_Depth(model, cfg_path=cfg_path)
        memory = pipeline.run(
            images        = images,
            depths_gt     = views_all['depth'][0].cpu().float(),
            poses_gt      = views_all['camera_pose'][0].cpu().float(),
            intrinsics_gt = views_all['camera_intrinsics'][0].cpu().float(),
        )
        gt_mode = True
    elif args.use_gt_pose and args.demo_type == 'scannet':
        print('Mode: GT poses (AMB3RV4_VO_GT)')
        pipeline = AMB3RV4_VO_GT(model, cfg_path=cfg_path)
        memory = pipeline.run(
            images   = images,
            poses_gt = views_all['camera_pose'][0].cpu().float(),
        )
        gt_mode = True
    else:
        print('Mode: estimated poses (AMB3RV4_VO)')
        pipeline = AMB3RV4_VO(model, cfg_path=cfg_path)
        memory = pipeline.run(images)
        gt_mode = False

    fps = num_frames / (time.time() - t0)
    print(f"\nDone — {fps:.1f} fps (wall time)")

    # ── Coordinate alignment for ScanNet ──────────────────────────────────
    # FIX: GT pipelines already store pts/poses in GT-world frame — do NOT
    # apply first_gt transform (would double-transform). Only non-GT mode
    # needs this alignment.
    T_voxel = None
    if args.demo_type == 'scannet' and not gt_mode:
        first_gt = views_all['camera_pose'][0, 0].cpu().numpy().astype(np.float32)
        T_voxel  = first_gt
        memory.poses = torch.from_numpy(
            left_multiply_poses(memory.poses.cpu().numpy().astype(np.float32), first_gt)
        )
        memory.pts = torch.from_numpy(
            transform_points(memory.pts.cpu().numpy().astype(np.float32), first_gt)
        )
    elif gt_mode:
        print("[run] GT mode — memory.pts / poses already in GT world frame; "
              "skipping T_voxel transform")

    # ── Geometry PLY ──────────────────────────────────────────────────────
    pts_pred  = memory.pts
    conf_pred = memory.conf
    kf_idx    = memory.kf_idx

    if args.edge_mask:
        pts_mask, sky_mask = get_pts_mask(
            pts_pred, views_all['images'], conf_pred,
            conf_threshold=args.conf_threshold,
        )
    else:
        pts_mask = conf_pred >= args.conf_threshold
        sky_mask = torch.zeros_like(pts_mask)

    pts_kf   = pts_pred[kf_idx].cpu().numpy().reshape(-1, 3)
    color_kf = (
        images[0, kf_idx].cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, 3) + 1.0
    ) / 2.0
    mask_kf  = pts_mask[kf_idx].reshape(-1)
    pts_kf   = pts_kf[mask_kf]
    color_kf = color_kf[mask_kf]

    if len(pts_kf) > args.target_point_count:
        print(f"Downsampling: {len(pts_kf):,} → {args.target_point_count:,}")
        idx_s = np.random.choice(len(pts_kf), args.target_point_count, replace=False)
        pts_kf   = pts_kf[idx_s]
        color_kf = color_kf[idx_s]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_kf)
    pcd.colors = o3d.utility.Vector3dVector(color_kf)
    geo_ply = os.path.join(args.results_path,
                           f"scene_{args.demo_name}_geo_kf.ply")
    o3d.io.write_point_cloud(geo_ply, pcd)
    print(f"Saved geometry PLY         → {geo_ply}")

    # ── Semantic exports ──────────────────────────────────────────────────
    if args.save_semantic and memory.semantic_voxel_map is not None:
        labels, _, color_table = get_scannet_label_and_color_map(args.label_set)
        device    = next(model.parameters()).device
        text_feat = build_text_embeddings(
            clip_model=model.stage1.lseg.clip_pretrained,
            tokenizer=clip.tokenize,
            labels=labels,
            device=device,
        ).cpu()

        vox_sem_ply = os.path.join(
            args.results_path, f"scene_{args.demo_name}_semantic_voxels.ply"
        )
        export_semantic_voxels_ply(
            memory.semantic_voxel_map, text_feat, color_table, vox_sem_ply,
            transformation=T_voxel,
        )
        print(f"Saved semantic voxel PLY   → {vox_sem_ply}")

        pp_sem_ply = os.path.join(
            args.results_path, f"scene_{args.demo_name}_semantic_per_point.ply"
        )
        export_semantic_per_point_ply(
            torch.from_numpy(pts_kf), memory.semantic_voxel_map,
            text_feat, color_table, pp_sem_ply,
            transformation=T_voxel,
        )
        print(f"Saved semantic per-pt PLY  → {pp_sem_ply}")

        legend_path = os.path.join(
            args.results_path, f"scene_{args.demo_name}_semantic_legend.png"
        )
        save_semantic_color_legend(labels, color_table, legend_path)
        print(f"Saved semantic legend       → {legend_path}")

        if args.save_semantic_npz:
            centers, feats = memory.semantic_voxel_map.get_all()
            c_np = centers.numpy().astype(np.float32)
            if T_voxel is not None:
                c_np = transform_points(c_np, T_voxel)
            np.savez_compressed(
                os.path.join(args.results_path,
                             f"scene_{args.demo_name}_semantic_voxel_feats.npz"),
                voxel_centers=c_np,
                voxel_features=feats.numpy().astype(np.float32),
            )
            print(f"Saved semantic feats npz")

    # ── Instance exports ──────────────────────────────────────────────────
    if args.save_instance and memory.instance_voxel_map is not None:
        vox_ins_ply = os.path.join(
            args.results_path, f"scene_{args.demo_name}_instance_voxels.ply"
        )
        export_instance_voxels_ply(
            memory.instance_voxel_map, vox_ins_ply, transformation=T_voxel,
        )
        print(f"Saved instance voxel PLY   → {vox_ins_ply}")

        pp_ins_ply = os.path.join(
            args.results_path, f"scene_{args.demo_name}_instance_per_point.ply"
        )
        export_instance_per_point_ply(
            torch.from_numpy(pts_kf), memory.instance_voxel_map, pp_ins_ply,
            transformation=T_voxel,
        )
        print(f"Saved instance per-pt PLY  → {pp_ins_ply}")

        if args.save_instance_npz:
            centers, feats = memory.instance_voxel_map.get_all()
            c_np = centers.numpy().astype(np.float32)
            if T_voxel is not None:
                c_np = transform_points(c_np, T_voxel)
            np.savez_compressed(
                os.path.join(args.results_path,
                             f"scene_{args.demo_name}_instance_voxel_feats.npz"),
                voxel_centers=c_np,
                voxel_features=feats.numpy().astype(np.float32),
            )
            print(f"Saved instance feats npz")

    # ── voxel_store (model memory) PCA PLY ────────────────────────────────
    if args.save_voxel_store:
        vs_ply = os.path.join(
            args.results_path, f"scene_{args.demo_name}_voxel_store_pca.ply"
        )
        try:
            export_voxel_store_pca_ply(
                memory.voxel_store, vs_ply, transformation=T_voxel,
            )
        except Exception as e:
            print(f"[WARN] voxel_store PCA export failed: {e}")

    # ── Full results npz ──────────────────────────────────────────────────
    if args.save_res:
        np.savez_compressed(
            os.path.join(args.results_path,
                         f"scene_{args.demo_name}_results.npz"),
            pts=memory.pts.cpu().numpy(),
            conf=memory.conf.cpu().numpy(),
            pose=memory.poses.cpu().numpy(),
            images=views_all['images'].cpu().squeeze(0).numpy(),
            sky_mask=sky_mask,
            kf_idx=kf_idx.cpu().numpy(),
            fps=fps,
        )
        print(f"Saved results npz")


if __name__ == '__main__':
    main()
