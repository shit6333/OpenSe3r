import sys
import os
import time
import torch
import argparse
import numpy as np
import open3d as o3d
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from amb3r.model_semantic import AMB3R
from amb3r.tools.pts_vis import get_pts_mask
from amb3r.tools.semantic_vis_utils import (
    get_scannet_label_and_color_map,
    build_text_embeddings,
    save_semantic_color_legend,
    # ── voxel-based export (used below) ─────────
    export_semantic_voxels_ply,
    export_semantic_per_point_ply,
    export_instance_voxels_ply,
    export_instance_per_point_ply,
    # ── raw feature export (NOT used; available for offline analysis) ─
    # export_semantic_feat_ply,
    # export_instance_feat_ply,
)

from slam_semantic.pipeline import AMB3R_VO
from slam_semantic.datasets.demo import DemoDataset
from lang_seg.modules.models.lseg_net import clip

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',      type=str,   default="./demo")
    parser.add_argument('--demo_name',      type=str,   default="demo")
    parser.add_argument('--results_path',   type=str,   default="./outputs/slam/demo")
    parser.add_argument('--ckpt_path',      type=str,   default="./checkpoints/amb3r_semantic.pt")
    parser.add_argument('--target_point_count', type=int, default=6_000_000)
    parser.add_argument('--edge_mask',      type=bool,  default=True)
    parser.add_argument('--conf_threshold', type=float, default=0.0)
    parser.add_argument("--resolution", type=int, nargs=2)
    parser.add_argument('--save_res',       type=bool,  default=True)
    # ── Semantic / instance export ─────────────────────────────────────────
    parser.add_argument('--save_semantic',  action='store_true', default=False,
                        help='Export semantic colour PLY (voxel + per-point)')
    parser.add_argument('--save_instance',  action='store_true', default=False,
                        help='Export instance feature PLY (voxel + per-point, PCA coloured)')
    parser.add_argument('--label_set',      type=str,   default='scannet20',
                        choices=['scannet20', 'scannet200'],
                        help='Label set used for text-matching export')
    return parser


def load_semantic_model(ckpt_path: str) -> AMB3R:
    model = AMB3R(metric_scale=True, clip_dim=512, sem_dim=512, ins_dim=16)
    if os.path.isfile(ckpt_path):
        model.load_weights(ckpt_path, data_type='bf16', strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint not found at {ckpt_path}; using random weights.")
    model.cuda()
    model.eval()
    return model


def main():
    parser = get_args_parser()
    args   = parser.parse_args()

    data = DemoDataset(ROOT=args.data_path, resolution=tuple(args.resolution))
    # data = DemoDataset(ROOT=args.data_path, resolution=(518, 392))
    os.makedirs(args.results_path, exist_ok=True)

    # ── Model & pipeline ──────────────────────────────────────────────────
    model    = load_semantic_model(args.ckpt_path)
    pipeline = AMB3R_VO(model)

    # ── Data ──────────────────────────────────────────────────────────────
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    batch      = next(iter(dataloader))
    _, views_all  = batch
    num_frames    = views_all['images'].shape[1]
    images        = views_all['images']             # (1, T, 3, H, W) in [-1, 1]
    print(f"Processing scene '{args.demo_name}' with {num_frames} frames...")

    # ── SLAM ──────────────────────────────────────────────────────────────
    t0     = time.time()
    memory = pipeline.run(images)
    fps    = num_frames / (time.time() - t0)
    print(f"Done — {fps:.1f} fps")
    if memory.semantic_voxel_map is not None:
        print(f"  Semantic voxels: {memory.semantic_voxel_map.num_voxels:,}")
    if memory.instance_voxel_map is not None:
        print(f"  Instance voxels: {memory.instance_voxel_map.num_voxels:,}")

    # ── Geometry point cloud ──────────────────────────────────────────────
    poses_pred = memory.poses
    pts_pred   = memory.pts       # (num_frames, H, W, 3)
    conf_pred  = memory.conf
    kf_idx     = memory.kf_idx

    if args.edge_mask:
        pts_mask, sky_mask = get_pts_mask(
            pts_pred, views_all['images'], conf_pred,
            conf_threshold=args.conf_threshold,
        )
    else:
        pts_mask = conf_pred >= args.conf_threshold
        sky_mask = torch.zeros_like(pts_mask)

    # Keyframe points (flat)
    pts_kf   = pts_pred[kf_idx].cpu().numpy().reshape(-1, 3)
    color_kf = (
        images[0, kf_idx].cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, 3) + 1.0
    ) / 2.0
    mask_kf  = pts_mask[kf_idx].reshape(-1)
    pts_kf   = pts_kf[mask_kf]
    color_kf = color_kf[mask_kf]

    # Downsample if needed — keep track of sampled indices for feature alignment
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
    geo_ply = os.path.join(args.results_path,
                            f"scene_{args.demo_name}_geo_kf.ply")
    o3d.io.write_point_cloud(geo_ply, pcd)
    print(f"Saved geometry PLY        → {geo_ply}")

    # ── Semantic exports ──────────────────────────────────────────────────
    if args.save_semantic and memory.semantic_voxel_map is not None:
        labels, _, color_table = get_scannet_label_and_color_map(args.label_set)

        device    = next(model.parameters()).device
        text_feat = build_text_embeddings(
            clip_model=model.lseg.clip_pretrained,
            tokenizer=clip.tokenize,
            labels=labels,
            device=device,
        ).cpu()

        # (A) Voxel-centre PLY  — one pt per voxel, fast
        vox_sem_ply = os.path.join(
            args.results_path, f"scene_{args.demo_name}_semantic_voxels.ply"
        )
        export_semantic_voxels_ply(
            memory.semantic_voxel_map, text_feat, color_table, vox_sem_ply
        )
        print(f"Saved semantic voxel PLY  → {vox_sem_ply}")

        # (B) Per-point PLY — query voxel map for every geometry point
        pts_kf_tensor = torch.from_numpy(pts_kf)
        pp_sem_ply = os.path.join(
            args.results_path, f"scene_{args.demo_name}_semantic_per_point.ply"
        )
        export_semantic_per_point_ply(
            pts_kf_tensor, memory.semantic_voxel_map,
            text_feat, color_table, pp_sem_ply,
        )
        print(f"Saved semantic per-pt PLY → {pp_sem_ply}")

        # Legend
        legend_path = os.path.join(
            args.results_path, f"scene_{args.demo_name}_semantic_legend.png"
        )
        save_semantic_color_legend(labels, color_table, legend_path)
        print(f"Saved semantic legend      → {legend_path}")

    # ── Instance exports ──────────────────────────────────────────────────
    if args.save_instance and memory.instance_voxel_map is not None:

        # (A) Voxel-centre PLY — PCA coloured
        vox_ins_ply = os.path.join(
            args.results_path, f"scene_{args.demo_name}_instance_voxels.ply"
        )
        export_instance_voxels_ply(memory.instance_voxel_map, vox_ins_ply)
        print(f"Saved instance voxel PLY  → {vox_ins_ply}")

        # (B) Per-point PLY
        pp_ins_ply = os.path.join(
            args.results_path, f"scene_{args.demo_name}_instance_per_point.ply"
        )
        export_instance_per_point_ply(
            torch.from_numpy(pts_kf), memory.instance_voxel_map, pp_ins_ply
        )
        print(f"Saved instance per-pt PLY → {pp_ins_ply}")

    # ── Save full result npz ──────────────────────────────────────────────
    if args.save_res:
        res_save = {
            'pts':      pts_pred.cpu().numpy(),
            'conf':     conf_pred.cpu().numpy(),
            'pose':     poses_pred.cpu().numpy(),
            'images':   views_all['images'].cpu().squeeze(0).numpy(),
            'sky_mask': sky_mask,
            'kf_idx':   kf_idx.cpu().numpy(),
            'fps':      fps,
        }
        save_path = os.path.join(
            args.results_path, f"scene_{args.demo_name}_results.npz"
        )
        np.savez_compressed(save_path, **res_save)
        print(f"Saved npz results          → {save_path}")


if __name__ == "__main__":
    main()