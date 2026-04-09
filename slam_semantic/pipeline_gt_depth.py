"""
pipeline_gt_depth.py
====================
Semantic SLAM pipeline using GT camera poses AND GT depth maps.

Design rationale
----------------
pipeline_gt.py uses model-predicted 3D points, requiring scale estimation
(model-space → metric) that is noisy and causes point-cloud overlap.

This pipeline eliminates the scale problem entirely:
  - 3D positions come from GT depth + GT intrinsics + GT pose (already metric)
  - The model is used ONLY for semantic/instance feature extraction
  - Features are directly assigned to the corresponding GT point-cloud pixels

No scale estimation. No coordinate alignment. No GT-vs-predicted fusion.

Data flow
---------
  GT depth + intrinsics + c2w pose  →  world-space pts  (T, H, W, 3)
  model.forward(images)             →  semantic_feat     (T, C, H_f, W_f)
                                       instance_feat     (T, C, H_f, W_f)
  _push_to_voxel_map(gt_pts, feat, conf * depth_mask)  →  VoxelFeatureMap

Usage
-----
    from slam_semantic.pipeline_gt_depth import AMB3R_VO_GT_Depth
    from slam_semantic.datasets.scannet_slam_gt_depth import ScannetDemoDatasetWithDepth

    dataset = ScannetDemoDatasetWithDepth(ROOT, resolution=(518, 336))
    _, views_all = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))

    pipeline = AMB3R_VO_GT_Depth(model)
    memory = pipeline.run(
        images         = views_all['images'],                  # (1, T, 3, H, W)
        depths_gt      = views_all['depth'][0],               # (T, H, W) metres
        poses_gt       = views_all['camera_pose'][0],         # (T, 4, 4)
        intrinsics_gt  = views_all['camera_intrinsics'][0],   # (T, 3, 3)
    )
"""

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty')
)

from omegaconf import OmegaConf

from slam_semantic.memory import SLAMemory
from amb3r.tools.pose_dist import extrinsic_distance_batch_query
from amb3r.tools.keyframes import select_keyframes_iteratively


# ── Geometry helpers ──────────────────────────────────────────────────────────

def unproject_depth_to_world(
    depth: torch.Tensor,        # (H, W)  metres, cpu float32; 0 = invalid
    intrinsics: torch.Tensor,   # (3, 3)  cpu float32
    c2w: torch.Tensor,          # (4, 4)  cpu float32
) -> torch.Tensor:
    """
    Unproject a depth map to world-space 3D points.

    Invalid pixels (depth == 0) are set to (0, 0, 0) in the output;
    use the depth validity mask as a confidence weight when pushing to the
    voxel map so those pixels contribute nothing.

    Returns: (H, W, 3) float32 cpu.
    """
    H, W = depth.shape
    device = depth.device

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij',
    )                                           # (H, W)

    valid = depth > 0

    x_cam = torch.where(valid, (xs - cx) / fx * depth, torch.zeros_like(depth))
    y_cam = torch.where(valid, (ys - cy) / fy * depth, torch.zeros_like(depth))
    z_cam = depth.clone()
    z_cam[~valid] = 0.0

    pts_cam  = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (H, W, 3)
    pts_flat = pts_cam.reshape(-1, 3)                       # (H*W, 3)

    R = c2w[:3, :3].to(device)   # (3, 3)
    t = c2w[:3, 3].to(device)    # (3,)

    pts_world = (R @ pts_flat.T).T + t                      # (H*W, 3)

    # Zero out invalid pixels (R@0+t = camera centre, not meaningful)
    pts_world[~valid.reshape(-1)] = 0.0

    return pts_world.reshape(H, W, 3)


def unproject_batch(
    depths: torch.Tensor,       # (T, H, W) metres
    intrinsics: torch.Tensor,   # (T, 3, 3) or (3, 3)
    poses: torch.Tensor,        # (T, 4, 4)
) -> torch.Tensor:
    """Unproject a batch of depth maps. Returns (T, H, W, 3) cpu float32."""
    T = depths.shape[0]
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(T, -1, -1)

    pts_list = []
    for i in range(T):
        pts_list.append(
            unproject_depth_to_world(depths[i], intrinsics[i], poses[i])
        )
    return torch.stack(pts_list, dim=0)   # (T, H, W, 3)


def depth_validity_conf(
    depths: torch.Tensor,   # (T, H, W)
    feat_h: int,
    feat_w: int,
) -> torch.Tensor:
    """
    Downsample depth validity mask to feature spatial resolution.

    Returns (T, 1, feat_h, feat_w) float32 binary mask: 1 = valid, 0 = no depth.
    Uses 'area' pooling so a patch is valid only if its centre depth pixel is
    valid (threshold > 0).
    """
    valid = (depths > 0).float().unsqueeze(1)   # (T, 1, H, W)
    valid_down = F.interpolate(valid, size=(feat_h, feat_w), mode='nearest')
    return valid_down   # 1.0 / 0.0


# ── Pipeline ──────────────────────────────────────────────────────────────────

class AMB3R_VO_GT_Depth:
    """
    Semantic mapping with GT camera poses and GT depth maps.

    The model is invoked only for semantic/instance feature extraction;
    geometry comes entirely from GT data (no scale estimation required).

    Call:
        memory = AMB3R_VO_GT_Depth(model).run(images, depths_gt, poses_gt, intrinsics_gt)
    """

    def __init__(self, model, cfg_path='./slam_semantic/slam_config.yaml'):
        self.cfg   = OmegaConf.load(cfg_path)
        self.model = model.to(self.cfg.device)

    # ──────────────────────────────────────────────────────────────────────
    # Model inference — features only
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _forward_features(self, views_all: dict, init=False) -> dict:
        """
        Run model on a chunk of images; return prediction dict.

        images : (1, T_chunk, 3, H, W) in [-1, 1], on self.cfg.device.
        Returns predictions dict with keys:
            semantic_feat  (1, T_chunk, C,   H_f, W_f)
            semantic_conf  (1, T_chunk, 1,   H_f, W_f)
            instance_feat  (1, T_chunk, C_i, H_f, W_f)
            instance_conf  (1, T_chunk, 1,   H_f, W_f)
        """
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        #     preds = self.model.forward({'images': images})
        # return preds[0]   # strip list wrapper; batch dim still present
        res = self.local_mapping(views_all, self.cfg, init=init)
        return res

    @torch.no_grad()
    def local_mapping(self, views_all, cfg, init=False):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # res = self.model.run_amb3r_sem_vo(
            #     views_all,
            #     cfg,
            #     self.keyframe_memory if not init else None,
            # )
            res = self.model.run_amb3r_sem_feat(views_all, cfg)
        return res

    # ──────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def run(
        self,
        images:         torch.Tensor,   # (1, T, 3, H, W)  in [-1, 1]
        depths_gt:      torch.Tensor,   # (T, H, W)         metres, cpu float32
        poses_gt:       torch.Tensor,   # (T, 4, 4)         c2w,   cpu float32
        intrinsics_gt:  torch.Tensor,   # (T, 3, 3) or (3,3) depth cam intrinsics
    ) -> SLAMemory:
        """
        Build a semantic voxel map from GT geometry and model-predicted features.

        Returns SLAMemory with:
            .pts       (T, H, W, 3)  — GT world pts (from depth unprojection)
            .poses     (T, 4, 4)     — GT poses
            .conf      (T, H, W)     — depth validity mask (1 = valid, 0 = no depth)
            .iter      (T,)          — all 1
            .kf_idx / .cur_kf_idx   — keyframe indices from GT pose distances
            .semantic_voxel_map     — populated if model has semantic head
            .instance_voxel_map     — populated if model has instance head
        """
        assert images.min() >= -1.01 and images.max() <= 1.01, \
            "Images must be in [-1, 1]"

        Bs, T, _, H, W = images.shape
        assert Bs == 1, "Batch size must be 1"

        has_semantic = (
            hasattr(self.model, 'lseg')
            and self.model.front_end.model.semantic_head is not None
        )
        has_instance = self.model.front_end.model.instance_head is not None

        memory = SLAMemory(
            self.cfg, T, H, W,
            has_semantic=has_semantic,
            has_instance=has_instance,
            sem_dim=self.model.clip_dim if has_semantic else 0,
            ins_dim=self.model.ins_dim  if has_instance else 0,
        )
        self.keyframe_memory = memory

        # ── Step 1: Build GT geometry buffers from depth ──────────────────
        print("[GT-depth] Unprojecting GT depth maps to world pts...")
        gt_pts = unproject_batch(depths_gt, intrinsics_gt, poses_gt)  # (T, H, W, 3)

        memory.pts   = gt_pts                                  # (T, H, W, 3)
        memory.poses = poses_gt.float()                        # (T, 4, 4)
        memory.conf  = (depths_gt > 0).float()                 # (T, H, W)  validity
        memory.iter  = torch.ones(T)

        # ── Step 2: Keyframe selection from GT pose distances ─────────────
        dists = extrinsic_distance_batch_query(poses_gt, poses_gt)
        is_kf = select_keyframes_iteratively(
            dists, memory.conf, self.cfg.keyframe_threshold,
            keyframe_indices=[0],
        )
        kf_indices        = torch.nonzero(is_kf, as_tuple=False).squeeze(1)
        memory.kf_idx     = kf_indices
        memory.cur_kf_idx = kf_indices
        print(f"[GT-depth] {len(kf_indices)} keyframes selected out of {T} frames")

        # ── Step 3: Feature extraction + voxel map update (chunked) ──────
        # chunk_size balances GPU memory vs temporal context for the model.
        # Use map_init_window from config as the chunk size.
        chunk_size = int(getattr(self.cfg, 'map_init_window', 8))
        t0 = time.time()

        for chunk_start in range(0, T, chunk_size):
            chunk_end  = min(chunk_start + chunk_size, T)
            T_chunk    = chunk_end - chunk_start

            # chunk_imgs = images[:, chunk_start:chunk_end].to(self.cfg.device)
            # res = self._forward_features(chunk_imgs)
            views_all = {
                'images': images[:, chunk_start:chunk_end].to(self.cfg.device),
                'start_idx': chunk_start,
                'end_idx': chunk_end - 1,
            }
            res = self._forward_features(views_all, init=(chunk_start == 0))

            # GT world pts for this chunk (cpu)
            gt_pts_chunk = gt_pts[chunk_start:chunk_end]   # (T_chunk, H, W, 3)

            # ── Semantic features ─────────────────────────────────────────
            if has_semantic and 'semantic_feat' in res:
                sem_feat = res['semantic_feat'][0].float().cpu()  # (T_c, C, H_f, W_f)
                sem_conf = res['semantic_conf'][0].float().cpu()  # (T_c, 1, H_f, W_f)

                _, _, H_f, W_f = sem_feat.shape
                depth_mask = depth_validity_conf(
                    depths_gt[chunk_start:chunk_end], H_f, W_f
                )                                                  # (T_c, 1, H_f, W_f)
                sem_conf = sem_conf * depth_mask

                SLAMemory._push_to_voxel_map(
                    memory.semantic_voxel_map,
                    gt_pts_chunk,
                    sem_feat,
                    sem_conf,
                )

            # ── Instance features ─────────────────────────────────────────
            if has_instance and 'instance_feat' in res:
                ins_feat = res['instance_feat'][0].float().cpu()  # (T_c, C_i, H_f, W_f)
                ins_conf = res['instance_conf'][0].float().cpu()  # (T_c, 1,   H_f, W_f)

                _, _, H_f, W_f = ins_feat.shape
                depth_mask = depth_validity_conf(
                    depths_gt[chunk_start:chunk_end], H_f, W_f
                )
                ins_conf = ins_conf * depth_mask

                SLAMemory._push_to_voxel_map(
                    memory.instance_voxel_map,
                    gt_pts_chunk,
                    ins_feat,
                    ins_conf,
                )

            fps = chunk_end / (time.time() - t0)
            print(
                f"[GT-depth] frames {chunk_start}–{chunk_end - 1}/{T}  ({fps:.1f} fps)"
            )

        if memory.semantic_voxel_map is not None:
            print(f"[GT-depth] Semantic voxels: {memory.semantic_voxel_map.num_voxels:,}")
        if memory.instance_voxel_map is not None:
            print(f"[GT-depth] Instance voxels: {memory.instance_voxel_map.num_voxels:,}")

        return memory
