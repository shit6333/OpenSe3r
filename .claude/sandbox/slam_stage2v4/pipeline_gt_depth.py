"""
pipeline_gt_depth.py  —  GT-depth + GT-pose pipeline for AMB3RStage2V4.

Design (fixed in this version)
------------------------------
All geometry comes from GT:
    pts_world = unproject(gt_depth, gt_intrinsics, gt_pose)

CRITICAL FIX: GT world-space pts are now passed as `pts_for_query` to
model.forward_chunk().  This is how the model was TRAINED
(training_stage2v4.py passes `pts_for_query=gt_pts_chunk` on every chunk).
Previously inference used the model's own predicted `world_points`, which
is in a *chunk-local frame* that changes every chunk — voxel_store keys
were therefore inconsistent across chunks, every query missed, and every
fusion saw only null_token.  That is why instance-feature PCA collapsed
to a single colour.

With this fix:
    query  pts  ⇒ GT-world frame  (consistent across chunks)
    update pts  ⇒ same GT-world frame  (pts_flat returned from forward_chunk)
    voxel_store ⇒ consistently keyed in GT world frame

The model is queried frame-by-frame (keyframe context + new frames) for:
    1. semantic_feat / instance_feat → pushed to VoxelFeatureMap (GT-world pts)
    2. M_content → stored in voxel_store (GT-world pts_flat)

No coordinate_alignment, no scale estimation, no pose prediction, no blending.
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

from amb3r.model_stage2v4 import AMB3RStage2V4
from amb3r.tools.pose_dist import extrinsic_distance_batch_query
from amb3r.tools.keyframes import select_keyframes_iteratively
from slam_stage2v4.memory import SLAMemoryV4


# ── GT geometry helper ────────────────────────────────────────────────────────

def unproject_depth_to_world(
    depth:      torch.Tensor,   # (H, W)   metres, cpu float32; 0 = invalid
    intrinsics: torch.Tensor,   # (3, 3)   cpu float32
    c2w:        torch.Tensor,   # (4, 4)   cpu float32
) -> tuple:
    """
    Unproject a depth map to world-space 3D points.

    Returns:
        pts   : (H, W, 3) float32 cpu — world-space pts (invalid → (0,0,0))
        valid : (H, W)    bool    cpu — True where depth > 0
    """
    H, W   = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
    v = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)

    x_cam = (u - cx) / fx * depth
    y_cam = (v - cy) / fy * depth
    z_cam = depth

    pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)            # (H, W, 3)
    pts_flat = pts_cam.reshape(-1, 3)
    pts_world = (c2w[:3, :3] @ pts_flat.T).T + c2w[:3, 3]
    pts_world = pts_world.reshape(H, W, 3)

    valid = depth > 0
    pts_world[~valid] = 0.0
    return pts_world, valid


# ── Pipeline class ────────────────────────────────────────────────────────────

class AMB3RV4_VO_GT_Depth:
    """
    Stage-2 V4 SLAM using GT depth + GT poses for geometry AND memory query.

    Keyframe selection uses GT-pose distances.
    voxel_store is updated with GT-world `pts_flat` derived from GT depth.
    """

    def __init__(
        self,
        model: AMB3RStage2V4,
        cfg_path: str = './slam_stage2v4/slam_config_v4.yaml',
    ):
        self.cfg   = OmegaConf.load(cfg_path)
        self.model = model.to(self.cfg.device)

    # ── Core forward (with GT pts_for_query) ─────────────────────────────────

    @torch.no_grad()
    def _forward(self, views_all: dict, gt_pts_chunk: torch.Tensor,
                 init: bool = False) -> dict:
        """
        Run forward_chunk using GT world pts as the memory-query coordinates.

        gt_pts_chunk : (1, T, H, W, 3) float32 on device — GT world-frame pts
                        for every pixel of every frame in this chunk.
        """
        voxel_store = self.memory.voxel_store
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            frames = {'images': views_all['images']}
            preds, M_content, W_conf, pts_flat = self.model.forward_chunk(
                frames, voxel_store, pts_for_query=gt_pts_chunk,
            )
        preds['pts_flat'] = pts_flat         # in GT-world frame
        preds['W_conf']   = W_conf

        if not init and M_content is not None:
            AMB3RStage2V4.update_voxel_store(
                voxel_store, M_content, W_conf, pts_flat,
                mem_mode=self.model.mem_mode,
            )
        return preds

    # ── Unproject batch ───────────────────────────────────────────────────────

    @staticmethod
    def _unproject_batch(
        depths:      torch.Tensor,   # (T, H, W)
        poses_gt:    torch.Tensor,   # (all_frames, 4, 4)
        intrinsics:  torch.Tensor,   # (all_frames, 3, 3)
        map_idx:     torch.Tensor,   # (T,) global frame indices
    ):
        """
        Unproject GT depth for a batch of T frames.

        Returns:
            pts_world : (T, H, W, 3) float32 cpu
            valid     : (T, H, W)    bool    cpu
        """
        T, H, W = depths.shape
        pts_list, valid_list = [], []
        for local_i, global_i in enumerate(map_idx.tolist()):
            pts, valid = unproject_depth_to_world(
                depths[local_i],
                intrinsics[global_i],
                poses_gt[global_i],
            )
            pts_list.append(pts)
            valid_list.append(valid)
        return torch.stack(pts_list), torch.stack(valid_list)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_map(
        self,
        views_all:     dict,
        depths_gt:     torch.Tensor,  # (all_T, H, W)
        poses_gt:      torch.Tensor,  # (all_T, 4, 4)
        intrinsics_gt: torch.Tensor,  # (all_T, 3, 3)
    ):
        T_init = views_all['images'].shape[1]
        map_idx_init = torch.arange(T_init)

        # GT geometry for init chunk
        pts_world, valid = self._unproject_batch(
            depths_gt[:T_init], poses_gt, intrinsics_gt, map_idx_init,
        )
        conf_gt = valid.float()  # 1.0 where depth valid

        # GT pts for model query (add batch dim, move to device)
        gt_pts_chunk = pts_world.unsqueeze(0).to(self.cfg.device)   # (1, T, H, W, 3)

        res = self._forward(views_all, gt_pts_chunk, init=True)

        # Keyframe selection on GT poses
        c2w_gt_init = poses_gt[:T_init].float()
        dists = extrinsic_distance_batch_query(c2w_gt_init, c2w_gt_init)
        is_kf = select_keyframes_iteratively(
            dists, conf_gt, self.cfg.keyframe_threshold,
            keyframe_indices=[0],
        )
        kf_indices = torch.nonzero(is_kf, as_tuple=False).squeeze(1)

        self.memory.pts  [:T_init] = pts_world
        self.memory.conf [:T_init] = conf_gt
        self.memory.poses[:T_init] = c2w_gt_init
        self.memory.iter [:T_init] = 1
        self.memory.kf_idx         = kf_indices
        self.memory.cur_kf_idx     = kf_indices

        # Seed voxel_store with GT-world pts_flat
        if res['M_content'] is not None:
            AMB3RStage2V4.update_voxel_store(
                self.memory.voxel_store,
                res['M_content'], res['W_conf'], res['pts_flat'],
                mem_mode=self.model.mem_mode,
            )
            print(f"[GT-depth V4] voxel_store seeded: "
                  f"{len(self.memory.voxel_store):,} voxels")

        if self.memory.semantic_voxel_map is not None and 'semantic_feat' in res:
            SLAMemoryV4._push_to_voxel_map(
                self.memory.semantic_voxel_map, pts_world,
                res['semantic_feat'][0].float().cpu(),
                res['semantic_conf'][0].float().cpu(),
            )
            print(f"[GT-depth V4] Init semantic voxels: "
                  f"{self.memory.semantic_voxel_map.num_voxels:,}")

        if self.memory.instance_voxel_map is not None and 'instance_feat' in res:
            SLAMemoryV4._push_to_voxel_map(
                self.memory.instance_voxel_map, pts_world,
                res['instance_feat'][0].float().cpu(),
                res['instance_conf'][0].float().cpu(),
            )
            print(f"[GT-depth V4] Init instance voxels: "
                  f"{self.memory.instance_voxel_map.num_voxels:,}")

    # ── Incremental update ────────────────────────────────────────────────────

    def _update_map(
        self,
        views_all:     dict,
        depths_gt:     torch.Tensor,
        poses_gt:      torch.Tensor,
        intrinsics_gt: torch.Tensor,
        start_idx:     int,
        end_idx:       int,
    ):
        num_kf  = len(self.memory.cur_kf_idx)
        map_idx = torch.cat(
            [self.memory.cur_kf_idx,
             torch.arange(start_idx, end_idx + 1)], dim=0,
        )

        # GT geometry for [keyframes + new frames]
        pts_world, valid = self._unproject_batch(
            depths_gt[map_idx], poses_gt, intrinsics_gt, map_idx,
        )
        conf_new = valid.float()

        # GT pts for query (same ordering as views_all['images'])
        gt_pts_chunk = pts_world.unsqueeze(0).to(self.cfg.device)

        res = self._forward(views_all, gt_pts_chunk)

        # ── Geometry fusion (confidence-weighted running mean) ───────────
        conf_old = self.memory.conf[map_idx]
        iter_old = self.memory.iter[map_idx]

        conf_sum_old = conf_old * iter_old[:, None, None]
        self.memory.pts[map_idx] = (
            conf_sum_old[..., None] * self.memory.pts[map_idx]
            + conf_new[..., None]   * pts_world
        ) / (conf_sum_old[..., None] + conf_new[..., None] + 1e-8)
        self.memory.conf[map_idx] = (
            conf_sum_old + conf_new
        ) / (iter_old[:, None, None] + 1)
        self.memory.iter[map_idx] = iter_old + 1

        for global_i in map_idx.tolist():
            self.memory.poses[global_i] = poses_gt[global_i].float()

        fused = self.memory.pts[map_idx].cpu()

        # ── Semantic / instance map push (uses fused GT-world pts) ───────
        if self.memory.semantic_voxel_map is not None and 'semantic_feat' in res:
            SLAMemoryV4._push_to_voxel_map(
                self.memory.semantic_voxel_map, fused,
                res['semantic_feat'][0].float().cpu(),
                res['semantic_conf'][0].float().cpu(),
            )
        if self.memory.instance_voxel_map is not None and 'instance_feat' in res:
            SLAMemoryV4._push_to_voxel_map(
                self.memory.instance_voxel_map, fused,
                res['instance_feat'][0].float().cpu(),
                res['instance_conf'][0].float().cpu(),
            )

        # ── Keyframe discovery with GT poses ─────────────────────────────
        c2w_gt_map = poses_gt[map_idx].float()
        dists = extrinsic_distance_batch_query(c2w_gt_map, c2w_gt_map)
        is_kf = select_keyframes_iteratively(
            dists, conf_new, self.cfg.keyframe_threshold,
            keyframe_indices=list(range(num_kf)),
        )
        if is_kf[num_kf:].sum() > 0:
            new_kf = (
                torch.nonzero(is_kf[num_kf:], as_tuple=False).squeeze(1)
                + start_idx
            )
            self.memory.cur_kf_idx = torch.cat([self.memory.cur_kf_idx, new_kf])
            self.memory.kf_idx     = torch.cat([self.memory.kf_idx,     new_kf])
            self.memory.keyframe_management()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(
        self,
        images:        torch.Tensor,   # (1, T, 3, H, W) in [-1, 1]
        depths_gt:     torch.Tensor,   # (T, H, W) metres
        poses_gt:      torch.Tensor,   # (T, 4, 4)
        intrinsics_gt: torch.Tensor,   # (T, 3, 3)
    ) -> SLAMemoryV4:
        assert images.min() >= -1 and images.max() <= 1
        Bs, T, _, H, W = images.shape
        assert Bs == 1

        depths_gt     = depths_gt.float().cpu()
        poses_gt      = poses_gt.float().cpu()
        intrinsics_gt = intrinsics_gt.float().cpu()

        s1           = self.model.stage1
        has_semantic = (hasattr(s1, 'lseg')
                        and s1.front_end.model.semantic_head is not None)
        has_instance = s1.front_end.model.instance_head is not None

        self.memory = SLAMemoryV4(
            cfg=self.cfg, num_frames=T, H=H, W=W,
            model=self.model,
            has_semantic=has_semantic, has_instance=has_instance,
            sem_dim=getattr(s1, 'clip_dim', 512) if has_semantic else 0,
            ins_dim=getattr(s1, 'ins_dim',  16)  if has_instance else 0,
        )

        initialized = False
        last_mapped_idx = 0
        t0 = time.time()

        for idx in range(T):
            if idx < self.cfg.map_init_window - 1:
                continue

            if not initialized:
                views_all = {'images': images[:, :idx + 1].to(self.cfg.device)}
                self._init_map(views_all, depths_gt, poses_gt, intrinsics_gt)
                initialized     = True
                last_mapped_idx = idx
            else:
                if (idx - last_mapped_idx < self.cfg.map_every) and (idx < T - 1):
                    continue

                start_idx = idx + 1 - self.cfg.map_every
                views_to_map = {
                    'images': torch.cat(
                        [images[:, self.memory.cur_kf_idx],
                         images[:, start_idx:idx + 1]], dim=1,
                    ).to(self.cfg.device),
                    'start_idx': start_idx,
                    'end_idx':   idx,
                }
                self._update_map(
                    views_to_map, depths_gt, poses_gt, intrinsics_gt,
                    start_idx, idx,
                )
                last_mapped_idx = idx

            fps = (idx + 1) / max(time.time() - t0, 1e-6)
            print(f"[GT-depth V4] frame {idx + 1}/{T}  "
                  f"KF: {self.memory.cur_kf_idx.tolist()}  "
                  f"({fps:.1f} fps)")

        vs = self.memory.voxel_store
        print(f"\n[GT-depth V4] Final voxel_store : {len(vs):,} voxels "
              f"(size={vs.voxel_size}m, dim={vs.mem_dim})")
        return self.memory
