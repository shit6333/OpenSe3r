"""
pipeline_gt.py
==============
GT-pose semantic SLAM pipeline for AMB3R.

Identical SLAM loop structure to pipeline.py (keyframe management,
model called with keyframes + new frames for context).

Key differences from the standard SLAM pipeline
------------------------------------------------
1. Poses are always GT (never predicted).
2. At **initialization**, a metric scale is estimated from GT-pose
   translation distances vs predicted-pose translation distances, then
   pts are immediately stored in GT-metric space.
3. At each subsequent **update**, coordinate_alignment's scale step is
   replicated to find model_scale→metric_scale factor (works correctly
   because stored pts are already metric).  Per-frame:
       T_i = c2w_gt[map_idx[i]] @ inv(c2w_pred_scaled[i])
   transforms pts to GT-world.  GT poses are stored directly (no
   pose blending with predictions).
4. Keyframe selection uses GT poses for distance computation.

Why the previous version had heavy overlap
------------------------------------------
• init stored pts in model scale, then overrode poses to GT metric →
  inconsistent scale between pts and poses.
• _estimate_chunk_scale then compared model-scale pts to metric-scale
  poses in the transform step → wrong scale → overlap.

Usage in run.py
---------------
    from slam_semantic.pipeline_gt import AMB3R_VO_GT
    ...
    if args.use_gt_pose and args.demo_type == 'scannet':
        poses_gt = views_all['camera_pose'][0].cpu().float()  # (T, 4, 4)
        pipeline = AMB3R_VO_GT(model)
        memory   = pipeline.run(images, poses_gt)
    else:
        from slam_semantic.pipeline import AMB3R_VO
        pipeline = AMB3R_VO(model)
        memory   = pipeline.run(images)
"""

import os
import sys
import torch
import time

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))

from omegaconf import OmegaConf

from slam_semantic.memory import SLAMemory
from amb3r.tools.pts_align import (
    transform_pts_global_to_local,
    robust_scale_invariant_alignment,
)
from amb3r.tools.pose_dist import extrinsic_distance_batch_query
from amb3r.tools.keyframes import select_keyframes_iteratively


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scale_from_translations(c2w_gt: torch.Tensor,
                              c2w_pred: torch.Tensor) -> float:
    """
    Estimate metric scale from consecutive-frame translation distances.

    scale s  such that  c2w_pred[:, :3, 3] * s ≈ c2w_gt[:, :3, 3]  (in magnitude)

    Both inputs: (T, 4, 4).  Returns 1.0 if < 2 frames or all dists ≈ 0.
    """
    if c2w_gt.shape[0] < 2:
        return 1.0
    t_gt   = c2w_gt  [:, :3, 3]
    t_pred = c2w_pred[:, :3, 3]
    d_gt   = torch.norm(t_gt  [1:] - t_gt  [:-1], dim=-1)
    d_pred = torch.norm(t_pred[1:] - t_pred[:-1], dim=-1)
    valid  = d_pred > 1e-4
    if valid.sum() == 0:
        return 1.0
    scale = float(torch.median(d_gt[valid] / d_pred[valid]).item())
    return max(scale, 1e-4)


def _scale_from_pts(pts_local: torch.Tensor,
                    c2w_local: torch.Tensor,
                    conf_local: torch.Tensor,
                    pts_global: torch.Tensor,
                    c2w_global: torch.Tensor,
                    num_kf: int) -> float:
    """
    Estimate model_scale → stored_scale factor, mirroring the scale-estimation
    step inside coordinate_alignment().

    pts_global is already in the stored (GT-metric after first init) scale.
    pts_local  is in raw model-prediction scale.
    Returns float s such that pts_local * s ≈ pts_global (after alignment).
    """
    # Transform global pts into local frame (same as coordinate_alignment)
    pts_kf_local_from_global = transform_pts_global_to_local(
        pts_global, c2w_global
    )[:num_kf]                          # (num_kf, H, W, 3)

    pts_kf_local = pts_local[:num_kf]   # (num_kf, H, W, 3)
    conf_kf      = conf_local[:num_kf]  # (num_kf, H, W)

    T_kf, H, W, _ = pts_kf_local.shape
    valid = (
        conf_kf >= torch.quantile(conf_kf.contiguous(), 0.5)
    ).view(1, T_kf * H, W)

    _, scale = robust_scale_invariant_alignment(
        pts_kf_local.contiguous().view(1, T_kf * H, W, 3),
        pts_kf_local_from_global.contiguous().view(1, T_kf * H, W, 3),
        valid,
    )
    return float(scale)


def _apply_gt_transform(pts_local_scaled: torch.Tensor,
                        c2w_local_scaled: torch.Tensor,
                        poses_gt: torch.Tensor,
                        map_idx: torch.Tensor) -> torch.Tensor:
    """
    Per-frame transform: T_i = c2w_gt[map_idx[i]] @ inv(c2w_pred_scaled[i])
    Applied to pts_local_scaled[i] → GT-world pts.

    pts_local_scaled : (T, H, W, 3)  already scale-corrected
    c2w_local_scaled : (T, 4, 4)     already scale-corrected
    poses_gt         : (ALL_FRAMES, 4, 4)
    map_idx          : (T,) long  — global frame index for each local position
    """
    T, H, W, _ = pts_local_scaled.shape
    pts_out = torch.zeros_like(pts_local_scaled)

    for local_i, global_i in enumerate(map_idx.tolist()):
        T_i = (
            poses_gt[global_i].double()
            @ torch.inverse(c2w_local_scaled[local_i].double())
        ).float()                                       # (4, 4)

        pts_i   = pts_local_scaled[local_i].reshape(-1, 3)   # (H*W, 3)
        pts_out[local_i] = (
            (T_i[:3, :3] @ pts_i.T).T + T_i[:3, 3]
        ).reshape(H, W, 3)

    return pts_out


# ── Main pipeline class ───────────────────────────────────────────────────────

class AMB3R_VO_GT:
    """
    Semantic SLAM with GT camera poses.

    Drop-in replacement for AMB3R_VO when GT poses are available.
    Call:
        memory = AMB3R_VO_GT(model).run(images, poses_gt)
    """

    def __init__(self, model, cfg_path='./slam_semantic/slam_config.yaml'):
        self.cfg   = OmegaConf.load(cfg_path)
        self.model = model.to(self.cfg.device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forward(self, views_all, init=False):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            return self.model.run_amb3r_sem_vo(
                views_all, self.cfg,
                self.memory if not init else None,
            )

    # ------------------------------------------------------------------
    def _pts_key(self):
        return (
            'pts3d_by_unprojection'
            if getattr(self.cfg, 'pts_by_unprojection', False)
            else 'world_points'
        )

    # ------------------------------------------------------------------
    def _init_map(self, views_all, poses_gt_init: torch.Tensor):
        """
        Initialise the map from the first window.

        poses_gt_init : (T_init, 4, 4) GT poses for frames 0..T_init-1.

        Flow:
          1. Run model forward (model-scale output).
          2. Estimate metric scale from GT vs predicted translations.
          3. Scale pts + poses, then per-frame GT-transform → metric pts.
          4. Initialise SLAMemory with metric pts + GT poses.
          5. Update voxel maps.
        """
        res = self._forward(views_all, init=True)

        pts_local  = res[self._pts_key()][0].cpu()         # (T, H, W, 3)
        c2w_local  = res['pose'][0].cpu()                  # (T, 4, 4)
        conf_local = res['world_points_conf'][0].cpu()     # (T, H, W)
        T_init     = pts_local.shape[0]

        conf_sig = (conf_local - 1) / conf_local
        conf_sig[conf_sig == 0] = 1e-6

        # ── Metric scale from GT translations vs predicted translations ──
        self._metric_scale = _scale_from_translations(
            poses_gt_init[:T_init], c2w_local
        )
        print(f"[GT-pose] Init metric scale: {self._metric_scale:.4f}")

        c2w_local_scaled = c2w_local.clone()
        c2w_local_scaled[:, :3, 3] *= self._metric_scale
        pts_local_scaled = pts_local * self._metric_scale

        # map_idx at init: frames 0..T_init-1 in order
        map_idx_init = torch.arange(T_init)

        pts_metric = _apply_gt_transform(
            pts_local_scaled, c2w_local_scaled,
            poses_gt_init, map_idx_init,
        )

        # ── Keyframe selection using GT poses ────────────────────────────
        c2w_gt_init = poses_gt_init[:T_init].float()
        dists = extrinsic_distance_batch_query(c2w_gt_init, c2w_gt_init)
        is_kf = select_keyframes_iteratively(
            dists, conf_local, self.cfg.keyframe_threshold,
            keyframe_indices=[0],
        )
        kf_indices = torch.nonzero(is_kf, as_tuple=False).squeeze(1)

        # ── Store into memory ────────────────────────────────────────────
        self.memory.pts  [:T_init] = pts_metric
        self.memory.conf [:T_init] = conf_sig
        self.memory.poses[:T_init] = c2w_gt_init
        self.memory.iter [:T_init] = 1
        self.memory.kf_idx         = kf_indices
        self.memory.cur_kf_idx     = kf_indices

        # ── Voxel maps ───────────────────────────────────────────────────
        if self.memory.semantic_voxel_map is not None and 'semantic_feat' in res:
            SLAMemory._push_to_voxel_map(
                self.memory.semantic_voxel_map,
                pts_metric,
                res['semantic_feat'][0].float().cpu(),
                res['semantic_conf'][0].float().cpu(),
            )
            print(f"[GT-pose] Init semantic voxels: "
                  f"{self.memory.semantic_voxel_map.num_voxels:,}")

        if self.memory.instance_voxel_map is not None and 'instance_feat' in res:
            SLAMemory._push_to_voxel_map(
                self.memory.instance_voxel_map,
                pts_metric,
                res['instance_feat'][0].float().cpu(),
                res['instance_conf'][0].float().cpu(),
            )
            print(f"[GT-pose] Init instance voxels: "
                  f"{self.memory.instance_voxel_map.num_voxels:,}")

    # ------------------------------------------------------------------
    def _update_map(self, views_all, poses_gt: torch.Tensor,
                    start_idx: int, end_idx: int):
        """
        Incremental update step.

        Model input:
            [kf0_img, kf1_img, ..., new_frame0_img, new_frame1_img]
        map_idx (matches input order):
            [cur_kf_idx[0], cur_kf_idx[1], ..., start_idx, start_idx+1, ...]

        Flow:
          1. Model forward with keyframe context.
          2. Scale estimation: model-scale local pts vs metric stored pts
             (coordinate_alignment-style, replicated so we get the scale).
          3. Per-frame GT transform → metric pts.
          4. Confidence-weighted fusion with stored pts.
          5. GT poses stored (no blending).
          6. Voxel map update with fused metric pts.
          7. Keyframe selection with GT poses.
        """
        res = self._forward(views_all)

        pts_local  = res[self._pts_key()][0].cpu()
        conf_local = res['world_points_conf'][0].cpu()
        c2w_local  = res['pose'][0].cpu()

        conf_sig_local = (conf_local - 1) / conf_local
        conf_sig_local[conf_sig_local == 0] = 1e-6

        num_kf  = len(self.memory.cur_kf_idx)
        map_idx = torch.cat(
            [self.memory.cur_kf_idx,
             torch.arange(start_idx, end_idx + 1)], dim=0
        )                                       # (num_kf + num_new,)

        pts_global  = self.memory.pts  [map_idx]   # metric
        conf_global = self.memory.conf [map_idx]
        c2w_global  = self.memory.poses[map_idx]   # GT metric
        iter_global = self.memory.iter [map_idx]

        # ── Scale: model-scale local → stored metric scale ───────────────
        scale = _scale_from_pts(
            pts_local, c2w_local, conf_sig_local,
            pts_global, c2w_global, num_kf,
        )
        print(f"[GT-pose] chunk scale: {scale:.4f}")

        c2w_local_scaled = c2w_local.clone()
        c2w_local_scaled[:, :3, 3] *= scale
        pts_local_scaled = pts_local * scale

        # ── Per-frame GT-world transform ─────────────────────────────────
        pts_metric = _apply_gt_transform(
            pts_local_scaled, c2w_local_scaled,
            poses_gt, map_idx,
        )

        # ── Confidence-weighted fusion ───────────────────────────────────
        conf_global_sum = conf_global * iter_global[:, None, None]

        self.memory.pts[map_idx] = (
            conf_global_sum[..., None] * pts_global
            + conf_sig_local[..., None] * pts_metric
        ) / (conf_global_sum[..., None] + conf_sig_local[..., None])

        self.memory.conf[map_idx] = (
            conf_global_sum + conf_sig_local
        ) / (iter_global[:, None, None] + 1)
        self.memory.iter[map_idx] = iter_global + 1

        # ── GT poses (direct, no blending with predicted) ────────────────
        for global_i in map_idx.tolist():
            self.memory.poses[global_i] = poses_gt[global_i].float()

        # ── Voxel maps (use fused metric pts as anchors) ─────────────────
        fused = self.memory.pts[map_idx].cpu()

        if self.memory.semantic_voxel_map is not None and 'semantic_feat' in res:
            SLAMemory._push_to_voxel_map(
                self.memory.semantic_voxel_map, fused,
                res['semantic_feat'][0].float().cpu(),
                res['semantic_conf'][0].float().cpu(),
            )

        if self.memory.instance_voxel_map is not None and 'instance_feat' in res:
            SLAMemory._push_to_voxel_map(
                self.memory.instance_voxel_map, fused,
                res['instance_feat'][0].float().cpu(),
                res['instance_conf'][0].float().cpu(),
            )

        # ── Keyframe selection with GT poses ─────────────────────────────
        c2w_gt_map = poses_gt[map_idx].float()
        dists = extrinsic_distance_batch_query(c2w_gt_map, c2w_gt_map)
        is_kf = select_keyframes_iteratively(
            dists, conf_local, self.cfg.keyframe_threshold,
            keyframe_indices=list(range(num_kf)),
        )

        if is_kf[num_kf:].sum() > 0:
            new_kf = (
                torch.nonzero(is_kf[num_kf:], as_tuple=False).squeeze(1)
                + start_idx
            )
            self.memory.cur_kf_idx = torch.cat(
                [self.memory.cur_kf_idx, new_kf], dim=0
            )
            self.memory.kf_idx = torch.cat(
                [self.memory.kf_idx, new_kf], dim=0
            )
            self.memory.keyframe_management()

    # ------------------------------------------------------------------
    def run(self, images: torch.Tensor,
            poses_gt: torch.Tensor) -> SLAMemory:
        """
        Args:
            images  : (1, T, 3, H, W)  in [-1, 1]
            poses_gt: (T, 4, 4)        GT camera-to-world, metric, cpu float32

        Returns:
            SLAMemory with semantic/instance voxel maps populated.
        """
        assert images.min() >= -1 and images.max() <= 1
        cfg = self.cfg
        Bs, T, _, H, W = images.shape
        assert Bs == 1

        has_semantic = (
            hasattr(self.model, 'lseg')
            and self.model.front_end.model.semantic_head is not None
        )
        has_instance = self.model.front_end.model.instance_head is not None

        self.memory = SLAMemory(
            cfg, T, H, W,
            has_semantic=has_semantic,
            has_instance=has_instance,
            sem_dim=self.model.clip_dim if has_semantic else 0,
            ins_dim=self.model.ins_dim  if has_instance else 0,
        )

        initialized     = False
        last_mapped_idx = 0
        t0              = time.time()

        for idx in range(T):
            if idx < cfg.map_init_window - 1:
                continue

            # ── Init ─────────────────────────────────────────────────────
            if not initialized:
                views_all = {
                    'images': images[:, :idx + 1].to(cfg.device),
                }
                self._init_map(views_all, poses_gt)
                initialized     = True
                last_mapped_idx = idx

            # ── Incremental ───────────────────────────────────────────────
            else:
                if (idx - last_mapped_idx < cfg.map_every) and (idx < T - 1):
                    continue

                start_idx = idx + 1 - cfg.map_every

                views_to_map = {
                    'images': torch.cat(
                        [images[:, self.memory.cur_kf_idx],
                         images[:, start_idx: idx + 1]],
                        dim=1,
                    ).to(cfg.device),
                    'start_idx': start_idx,
                    'end_idx':   idx,
                }

                self._update_map(views_to_map, poses_gt, start_idx, idx)
                last_mapped_idx = idx

            fps = (idx + 1) / (time.time() - t0)
            print(f"[GT-pose] frame {idx + 1}/{T}  "
                  f"KF ids: {self.memory.cur_kf_idx.tolist()}  "
                  f"({fps:.1f} fps)")

        return self.memory
