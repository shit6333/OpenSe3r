"""
pipeline_gt_pose_method.py
==========================
Drop-in additions for slam_semantic/pipeline.py to support GT-pose semantic mapping.

Strategy
--------
Keep the original SLAM loop structure unchanged (keyframe management, model called
with keyframes + new frames for context).  For each update step:
  1. Call coordinate_alignment's SCALE estimation only  (replicated here so we
     don't need to modify pts_align.py).
  2. Scale local pts and poses with that per-chunk scale.
  3. Per-frame: transform scaled pts to GT-world using
         T_i = c2w_gt[i] @ inv(c2w_pred_scaled[i])
     instead of the SLAM's averaged rigid alignment.
  4. Weighted-fuse pts same as memory.update().
  5. Store GT poses directly (no blending with predicted).
  6. Update voxel maps with fused pts as anchors.
  7. Keyframe selection uses GT poses for distance computation.

Changes needed in pipeline.py
------------------------------
1.  run() signature:
        def run(self, images, poses_gt=None, use_gt_pose=False):
            if use_gt_pose and poses_gt is not None:
                return self._run_gt_pose_mode(images, poses_gt)
            if use_gt_pose and poses_gt is None:
                raise ValueError("use_gt_pose=True but no poses_gt supplied")
            # ... original SLAM code unchanged ...

2.  Add _estimate_chunk_scale(), _initialize_with_gt_pose(),
    _mapping_with_gt_pose(), and _run_gt_pose_mode() to AMB3R_VO.
"""

import torch
import torch.nn.functional as F

from slam_semantic.memory import SLAMemory
from amb3r.tools.pts_align import (
    transform_pts_global_to_local,
    robust_scale_invariant_alignment,
)
from amb3r.tools.pose_dist import extrinsic_distance_batch_query
from amb3r.tools.keyframes import select_keyframes_iteratively


# ── Scale estimation (mirrors coordinate_alignment internals) ─────────────────

def _estimate_chunk_scale(pts_local, c2w_local, conf_local,
                           pts_global, c2w_global,
                           num_kf: int, transform: bool) -> float:
    """
    Estimate the per-chunk scale factor that coordinate_alignment would use.

    Returns float s such that pts_local * s is in the same scale as pts_global.
    When transform=False (first chunk, global IS local), returns 1.0 because
    there is no existing global map to compare against.
    """
    if not transform:
        return 1.0

    pts_kf_local_from_global = transform_pts_global_to_local(
        pts_global, c2w_global
    )[:num_kf]                                   # (num_kf, H, W, 3)
    pts_kf_local = pts_local[:num_kf]            # (num_kf, H, W, 3)
    conf_kf      = conf_local[:num_kf]           # (num_kf, H, W)

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


# ── Initialise map with GT poses ──────────────────────────────────────────────

def _initialize_with_gt_pose(self, views_all, cfg, poses_gt):
    """
    Drop-in for initialize_map() when using GT poses.
    Runs the model forward, initialises memory geometry, then overrides
    stored poses with GT.
    """
    res = self.local_mapping(views_all, cfg, init=True)

    conf = res['world_points_conf']
    avg_confidence = conf[0].mean(dim=(-1, -2)).cpu().numpy()
    print("Average confidence values:", [f"{c:.2f}" for c in avg_confidence])

    # Standard memory init (stores predicted poses internally)
    self.keyframe_memory.initialize(res)

    # Override poses with GT for all initialised frames
    T_init = self.keyframe_memory.kf_idx.max().item() + 1
    for i in range(T_init):
        self.keyframe_memory.poses[i] = poses_gt[i].float()

    print(f"[GT-pose] Initialised with {T_init} frames, "
          f"KF ids: {self.keyframe_memory.cur_kf_idx.tolist()}")


# ── Per-chunk update with GT poses ────────────────────────────────────────────

def _mapping_with_gt_pose(self, views_all, cfg, poses_gt):
    """
    Drop-in for mapping() when using GT poses.

    Replicates memory.update() with two key differences:
      - Scale is estimated per-chunk but pts are transformed using GT poses.
      - Stored poses = GT poses (not blended with predicted).
    """
    pts_key = (
        'pts3d_by_unprojection'
        if getattr(cfg, 'pts_by_unprojection', False)
        else 'world_points'
    )

    res = self.local_mapping(views_all, cfg)

    pts_local  = res[pts_key][0].cpu()              # (T, H, W, 3)
    conf_local = res['world_points_conf'][0].cpu()  # (T, H, W)
    c2w_local  = res['pose'][0].cpu()               # (T, 4, 4)

    conf_sig_local = (conf_local - 1) / conf_local
    conf_sig_local[conf_sig_local == 0] = 1e-6

    start_idx = views_all['start_idx']
    end_idx   = views_all['end_idx']
    num_kf    = len(self.keyframe_memory.cur_kf_idx)

    map_idx     = torch.cat(
        [self.keyframe_memory.cur_kf_idx,
         torch.arange(start_idx, end_idx + 1)], dim=0
    )
    pts_global  = self.keyframe_memory.pts  [map_idx]
    conf_global = self.keyframe_memory.conf [map_idx]
    c2w_global  = self.keyframe_memory.poses[map_idx]
    iter_global = self.keyframe_memory.iter [map_idx]

    # ── Scale estimation ──────────────────────────────────────────────────
    transform = self.keyframe_memory.cur_kf_idx[0] != 0
    scale = _estimate_chunk_scale(
        pts_local, c2w_local, conf_sig_local,
        pts_global, c2w_global,
        num_kf, transform,
    )
    print(f"[GT-pose] chunk scale: {scale:.4f}")

    c2w_local_scaled = c2w_local.clone()
    c2w_local_scaled[:, :3, 3] *= scale
    pts_local_scaled = pts_local * scale

    # ── Per-frame GT-world pts ────────────────────────────────────────────
    T_len, Hf, Wf, _ = pts_local_scaled.shape
    pts_global_from_local = torch.zeros_like(pts_local_scaled)

    for local_i, global_i in enumerate(map_idx.tolist()):
        T_i = (
            poses_gt[global_i].double()
            @ torch.inverse(c2w_local_scaled[local_i].double())
        ).float()

        pts_i = pts_local_scaled[local_i].reshape(-1, 3)   # (H*W, 3)
        pts_global_from_local[local_i] = (
            (T_i[:3, :3] @ pts_i.T).T + T_i[:3, 3]
        ).reshape(Hf, Wf, 3)

    # ── Geometry weighted fusion (identical to memory.update) ─────────────
    conf_global_sum = conf_global * iter_global[:, None, None]

    self.keyframe_memory.pts[map_idx] = (
        conf_global_sum[..., None] * pts_global
        + conf_sig_local[..., None] * pts_global_from_local
    ) / (conf_global_sum[..., None] + conf_sig_local[..., None])

    self.keyframe_memory.conf[map_idx] = (
        conf_global_sum + conf_sig_local
    ) / (iter_global[:, None, None] + 1)
    self.keyframe_memory.iter[map_idx] = iter_global + 1

    # ── GT poses (no blending) ────────────────────────────────────────────
    for local_i, global_i in enumerate(map_idx.tolist()):
        self.keyframe_memory.poses[global_i] = poses_gt[global_i].float()

    # ── Voxel map updates ─────────────────────────────────────────────────
    fused_pts = self.keyframe_memory.pts[map_idx].cpu()

    if self.keyframe_memory.semantic_voxel_map is not None \
            and 'semantic_feat' in res:
        SLAMemory._push_to_voxel_map(
            self.keyframe_memory.semantic_voxel_map,
            fused_pts,
            res['semantic_feat'][0].float().cpu(),
            res['semantic_conf'][0].float().cpu(),
        )

    if self.keyframe_memory.instance_voxel_map is not None \
            and 'instance_feat' in res:
        SLAMemory._push_to_voxel_map(
            self.keyframe_memory.instance_voxel_map,
            fused_pts,
            res['instance_feat'][0].float().cpu(),
            res['instance_conf'][0].float().cpu(),
        )

    # ── Keyframe discovery (use GT poses for distance) ────────────────────
    c2w_gt_map = poses_gt[map_idx].float()
    dists = extrinsic_distance_batch_query(c2w_gt_map, c2w_gt_map)
    is_keyframe = select_keyframes_iteratively(
        dists, conf_local, cfg.keyframe_threshold,
        keyframe_indices=list(range(num_kf)),
    )

    if is_keyframe[num_kf:].sum() > 0:
        new_kf = (
            torch.nonzero(is_keyframe[num_kf:], as_tuple=False)
            .squeeze(1) + start_idx
        )
        self.keyframe_memory.cur_kf_idx = torch.cat(
            [self.keyframe_memory.cur_kf_idx, new_kf], dim=0
        )
        self.keyframe_memory.kf_idx = torch.cat(
            [self.keyframe_memory.kf_idx, new_kf], dim=0
        )
        self.keyframe_memory.keyframe_management()


# ── Main GT-pose run loop ─────────────────────────────────────────────────────

def _run_gt_pose_mode(self, images: torch.Tensor,
                      poses_gt: torch.Tensor) -> 'SLAMemory':
    """
    Semantic SLAM with GT camera poses.

    Structure is identical to run() (SLAM loop with keyframes) except:
      - initialize_map  →  _initialize_with_gt_pose
      - mapping         →  _mapping_with_gt_pose

    Args:
        images  : (1, T, 3, H, W)  in [-1, 1]
        poses_gt: (T, 4, 4)        metric camera-to-world GT poses  (cpu float32)
    """
    cfg = self.cfg
    Bs, T, _, H, W = images.shape
    assert Bs == 1, "GT-pose mode only supports batch size 1"

    has_semantic = (
        hasattr(self.model, 'lseg')
        and self.model.front_end.model.semantic_head is not None
    )
    has_instance = self.model.front_end.model.instance_head is not None

    self.keyframe_memory = SLAMemory(
        cfg, T, H, W,
        has_semantic=has_semantic,
        has_instance=has_instance,
        sem_dim=self.model.clip_dim if has_semantic else 0,
        ins_dim=self.model.ins_dim  if has_instance else 0,
    )

    initialized     = False
    last_mapped_idx = 0

    for idx in range(T):
        if idx < cfg.map_init_window - 1:
            continue

        # ── Initialisation ────────────────────────────────────────────────
        if not initialized:
            views_all = {
                'images': images[:, :idx + 1].to(cfg.device),
            }
            _initialize_with_gt_pose(self, views_all, cfg, poses_gt)
            initialized     = True
            last_mapped_idx = idx

        # ── Incremental mapping ───────────────────────────────────────────
        else:
            if (idx - last_mapped_idx < cfg.map_every) and (idx < T - 1):
                continue

            start_idx = idx + 1 - cfg.map_every

            views_to_map = {
                'images': torch.cat(
                    [images[:, self.keyframe_memory.cur_kf_idx],
                     images[:, start_idx: idx + 1]],
                    dim=1,
                ).to(cfg.device),
                'start_idx': start_idx,
                'end_idx':   idx,
            }

            _mapping_with_gt_pose(self, views_to_map, cfg, poses_gt)
            last_mapped_idx = idx

        print(f"[GT-pose] frame {idx + 1}/{T}  "
              f"KF ids: {self.keyframe_memory.cur_kf_idx.tolist()}")

    return self.keyframe_memory
