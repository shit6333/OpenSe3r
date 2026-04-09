"""
slam_semantic/run_gt_depth.py
==============================
Entry point for GT-pose + GT-depth semantic mapping.

Differences from slam_semantic/run.py
--------------------------------------
* Uses ScannetDemoDatasetWithDepth  (loads depth/ alongside color/)
* Uses AMB3R_VO_GT_Depth            (no scale estimation, model used for features only)
* No coordinate alignment needed:   GT pts are already in metric world space
* T_voxel is applied only during export (same as run.py for scannet scenes)

Usage
-----
    python slam_semantic/run_gt_depth.py \
        --data_path  /path/to/scene_dir \
        --demo_name  scene0000_00 \
        --results_path ./outputs/slam/gt_depth \
        --ckpt_path  ./checkpoints/amb3r_semantic.pt \
        --resolution 518 336 \
        --save_semantic --save_instance \
        --label_set scannet20
"""

import sys
import os
import time
import torch
import argparse
import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

# from amb3r.model_stage1_wo_lora import AMB3RStage1FullFT as AMB3R
from amb3r.model_semantic import AMB3R
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

from slam_semantic.pipeline_gt_depth import AMB3R_VO_GT_Depth
from slam_semantic.datasets.scannet_slam_gt_depth import ScannetDemoDatasetWithDepth
from lang_seg.modules.models.lseg_net import clip

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',      type=str, required=True)
    p.add_argument('--demo_name',      type=str, default='demo')
    p.add_argument('--results_path',   type=str, default='./outputs/slam/gt_depth')
    p.add_argument('--ckpt_path',      type=str, default='./checkpoints/amb3r_semantic.pt')
    p.add_argument('--resolution',     type=int, nargs=2, default=[518, 336])
    p.add_argument('--target_point_count', type=int, default=3_000_000)
    p.add_argument('--conf_threshold', type=float, default=0.0,
                   help='Depth validity: 0 = keep all valid depth pixels')
    p.add_argument('--edge_mask',      action='store_true', default=False)
    p.add_argument('--save_res',       action='store_true', default=True)
    p.add_argument('--save_semantic',  action='store_true', default=False)
    p.add_argument('--save_instance',  action='store_true', default=False)
    p.add_argument('--save_semantic_npz', action='store_true', default=False)
    p.add_argument('--save_instance_npz', action='store_true', default=False)
    p.add_argument('--label_set',      type=str, default='scannet20',
                   choices=['scannet20', 'scannet200'])
    return p


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str) -> AMB3R:
    model = AMB3R(metric_scale=True, clip_dim=512, sem_dim=512, ins_dim=16)
    if os.path.isfile(ckpt_path):
        model.load_weights(ckpt_path, data_type='bf16', strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint not found at {ckpt_path}; using random weights.")
    model.cuda().eval()
    return model


def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """pts (N, 3), T (4, 4) → (N, 3)"""
    ones   = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_h  = np.concatenate([pts, ones], axis=1)
    return (T @ pts_h.T).T[:, :3]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args_parser().parse_args()
    os.makedirs(args.results_path, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────
    data = ScannetDemoDatasetWithDepth(
        ROOT=args.data_path,
        resolution=tuple(args.resolution),
    )
    loader     = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    _, views_all = next(iter(loader))

    images        = views_all['images']              # (1, T, 3, H, W)
    poses_gt      = views_all['camera_pose'][0]      # (T, 4, 4)
    intrinsics_gt = views_all['camera_intrinsics'][0]# (T, 3, 3)
    depths_gt     = views_all['depth'][0]            # (T, H, W) metres

    num_frames = images.shape[1]
    print(f"Scene '{args.demo_name}': {num_frames} frames, "
          f"resolution {images.shape[-1]}×{images.shape[-2]}")

    # ── Model + pipeline ─────────────────────────────────────────────────
    model    = load_model(args.ckpt_path)
    pipeline = AMB3R_VO_GT_Depth(model)

    t0     = time.time()
    memory = pipeline.run(images, depths_gt, poses_gt, intrinsics_gt)
    fps    = num_frames / (time.time() - t0)
    print(f"Done — {fps:.1f} fps")

    # ── Coordinate frame: align with GT first frame (for ScanNet eval) ──
    # GT pts are already in world space; no alignment needed.
    # We keep T_voxel = identity (or first frame pose if scene uses
    # relative coordinates, but ScanNet poses are typically absolute).
    first_c2w = poses_gt[0].cpu().numpy().astype(np.float32)
    # T_voxel   = first_c2w   # Apply to bring to scene-absolute coords if needed
    T_voxel = np.eye(4)
    # If your poses are already absolute world coords, set T_voxel = np.eye(4)

    # ── Geometry PLY ─────────────────────────────────────────────────────
    pts_pred  = memory.pts      # (T, H, W, 3) — GT world pts
    conf_pred = memory.conf     # (T, H, W)    — depth validity
    kf_idx    = memory.kf_idx

    if args.edge_mask:
        pts_mask, sky_mask = get_pts_mask(
            pts_pred, images[0], conf_pred,
            conf_threshold=args.conf_threshold,
        )
    else:
        pts_mask = conf_pred > args.conf_threshold
        sky_mask = torch.zeros_like(pts_mask)

    pts_kf   = pts_pred[kf_idx].cpu().numpy().reshape(-1, 3)
    color_kf = (
        images[0, kf_idx].cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, 3) + 1.0
    ) / 2.0
    mask_kf  = pts_mask[kf_idx].reshape(-1)#.numpy()
    pts_kf   = pts_kf[mask_kf]
    color_kf = color_kf[mask_kf]

    # Align to first-frame GT coordinate frame
    # pts_kf = transform_points(pts_kf, first_c2w)

    sampled_indices = None
    if len(pts_kf) > args.target_point_count:
        print(f"Downsampling: {len(pts_kf):,} → {args.target_point_count:,}")
        sampled_indices = np.random.choice(
            len(pts_kf), args.target_point_count, replace=False
        )
        pts_kf   = pts_kf[sampled_indices]
        color_kf = color_kf[sampled_indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_kf)
    pcd.colors = o3d.utility.Vector3dVector(color_kf)
    geo_ply = os.path.join(args.results_path, f"scene_{args.demo_name}_geo_kf.ply")
    o3d.io.write_point_cloud(geo_ply, pcd)
    print(f"Saved geometry PLY        → {geo_ply}")

    # ── Semantic exports ──────────────────────────────────────────────────
    scale_voxel = None   # Metric scale already correct

    if args.save_semantic and memory.semantic_voxel_map is not None:
        labels, _, color_table = get_scannet_label_and_color_map(args.label_set)
        device    = next(model.parameters()).device
        text_feat = build_text_embeddings(
            clip_model=model.lseg.clip_pretrained,
            tokenizer=clip.tokenize,
            labels=labels,
            device=device,
        ).cpu()

        vox_sem_ply = os.path.join(
            args.results_path, f"scene_{args.demo_name}_semantic_voxels.ply"
        )
        export_semantic_voxels_ply(
            memory.semantic_voxel_map, text_feat, color_table, vox_sem_ply,
            transformation=T_voxel, scale=scale_voxel,
        )
        print(f"Saved semantic voxel PLY  → {vox_sem_ply}")

        pts_kf_tensor = torch.from_numpy(pts_kf)
        pp_sem_ply = os.path.join(
            args.results_path, f"scene_{args.demo_name}_semantic_per_point.ply"
        )
        export_semantic_per_point_ply(
            pts_kf_tensor, memory.semantic_voxel_map,
            text_feat, color_table, pp_sem_ply,
            transformation=T_voxel, scale=scale_voxel,
        )
        print(f"Saved semantic per-pt PLY → {pp_sem_ply}")

        legend_path = os.path.join(
            args.results_path, f"scene_{args.demo_name}_semantic_legend.png"
        )
        save_semantic_color_legend(labels, color_table, legend_path)
        print(f"Saved semantic legend      → {legend_path}")

        if args.save_semantic_npz:
            sem_centers, sem_features = memory.semantic_voxel_map.get_all()
            sem_centers_np  = sem_centers.numpy().astype(np.float32)
            sem_features_np = sem_features.numpy().astype(np.float32)
            if T_voxel is not None:
                sem_centers_np = transform_points(sem_centers_np, T_voxel)
            np.savez_compressed(
                os.path.join(args.results_path,
                             f"scene_{args.demo_name}_semantic_voxel_feats.npz"),
                voxel_centers=sem_centers_np,
                voxel_features=sem_features_np,
            )
            print("Saved semantic feats npz")

    # ── Instance exports ──────────────────────────────────────────────────
    if args.save_instance and memory.instance_voxel_map is not None:
        vox_ins_ply = os.path.join(
            args.results_path, f"scene_{args.demo_name}_instance_voxels.ply"
        )
        export_instance_voxels_ply(
            memory.instance_voxel_map, vox_ins_ply,
            transformation=T_voxel, scale=scale_voxel,
        )
        print(f"Saved instance voxel PLY  → {vox_ins_ply}")

        pp_ins_ply = os.path.join(
            args.results_path, f"scene_{args.demo_name}_instance_per_point.ply"
        )
        export_instance_per_point_ply(
            torch.from_numpy(pts_kf), memory.instance_voxel_map, pp_ins_ply,
            transformation=T_voxel, scale=scale_voxel,
        )
        print(f"Saved instance per-pt PLY → {pp_ins_ply}")

        if args.save_instance_npz:
            ins_centers, ins_features = memory.instance_voxel_map.get_all()
            ins_centers_np  = ins_centers.numpy().astype(np.float32)
            ins_features_np = ins_features.numpy().astype(np.float32)
            if T_voxel is not None:
                ins_centers_np = transform_points(ins_centers_np, T_voxel)
            np.savez_compressed(
                os.path.join(args.results_path,
                             f"scene_{args.demo_name}_instance_voxel_feats.npz"),
                voxel_centers=ins_centers_np,
                voxel_features=ins_features_np,
            )
            print("Saved instance feats npz")

    # ── Full result npz ───────────────────────────────────────────────────
    if args.save_res:
        save_path = os.path.join(
            args.results_path, f"scene_{args.demo_name}_results.npz"
        )
        np.savez_compressed(
            save_path,
            pts    = memory.pts.cpu().numpy(),
            conf   = memory.conf.cpu().numpy(),
            pose   = memory.poses.cpu().numpy(),
            images = views_all['images'].cpu().squeeze(0).numpy(),
            depth  = depths_gt.cpu().numpy(),
            sky_mask = sky_mask,
            kf_idx   = memory.kf_idx.cpu().numpy(),
            fps      = fps,
        )
        print(f"Saved npz results          → {save_path}")


if __name__ == '__main__':
    main()
