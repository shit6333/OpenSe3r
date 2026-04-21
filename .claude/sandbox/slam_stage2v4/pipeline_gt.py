"""
pipeline_gt.py  —  GT-pose SLAM pipeline for AMB3RStage2V4.

Identical loop structure to pipeline.py (AMB3RV4_VO) with these differences:

    1. Poses are always GT (never predicted / blended).
    2. At initialisation, a metric scale is estimated from GT vs predicted
       translation distances, pts are stored in GT-metric space.
    3. At each incremental step, model-scale local pts are scaled to metric
       via _scale_from_pts, then per-frame GT-transformed to world.
    4. Keyframe selection and distance computation use GT poses.
    5. voxel_store is updated normally (same as AMB3RV4_VO) — M_content is
       computed at the model's pts, which is fine for model conditioning.

Usage:
    from slam_stage2v4.pipeline_gt import AMB3RV4_VO_GT
    pipeline = AMB3RV4_VO_GT(model, cfg_path=...)
    memory   = pipeline.run(images, poses_gt)   # poses_gt: (T, 4, 4) cpu float32
"""

import os
import sys
import time
import torch

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty')
)

from omegaconf import OmegaConf

from amb3r.model_stage2v4 import AMB3RStage2V4
from amb3r.tools.pts_align import (
    transform_pts_global_to_local,
    robust_scale_invariant_alignment,
)
from amb3r.tools.pose_dist import extrinsic_distance_batch_query
from amb3r.tools.keyframes import select_keyframes_iteratively
from slam_stage2v4.memory import SLAMemoryV4


# ── Helpers (identical to slam_semantic/pipeline_gt.py) ──────────────────────

def _scale_from_translations(c2w_gt: torch.Tensor,
                              c2w_pred: torch.Tensor) -> float:
    """Estimate metric scale from GT vs predicted translation distances."""
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


def _scale_from_pts(pts_local, c2w_local, conf_local,
                    pts_global, c2w_global, num_kf) -> float:
    """Estimate model-scale → stored-scale factor (mirrors coordinate_alignment)."""
    pts_kf_local_from_global = transform_pts_global_to_local(
        pts_global, c2w_global
    )[:num_kf]
    pts_kf_local = pts_local[:num_kf]
    conf_kf      = conf_local[:num_kf]

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


def _apply_gt_transform(pts_local_scaled, c2w_local_scaled,
                        poses_gt, map_idx) -> torch.Tensor:
    """Per-frame: T_i = c2w_gt[map_idx[i]] @ inv(c2w_pred_scaled[i])."""
    T, H, W, _ = pts_local_scaled.shape
    pts_out = torch.zeros_like(pts_local_scaled)
    for local_i, global_i in enumerate(map_idx.tolist()):
        T_i = (
            poses_gt[global_i].double()
            @ torch.inverse(c2w_local_scaled[local_i].double())
        ).float()
        pts_i = pts_local_scaled[local_i].reshape(-1, 3)
        pts_out[local_i] = (
            (T_i[:3, :3] @ pts_i.T).T + T_i[:3, 3]
        ).reshape(H, W, 3)
    return pts_out


# ── Pipeline class ────────────────────────────────────────────────────────────

class AMB3RV4_VO_GT:
    """
    Stage-2 V4 SLAM with GT camera poses.

    Drop-in replacement for AMB3RV4_VO when GT poses are available.
    Geometry is in GT-metric space; the model's voxel_store memory is still
    updated from forward_chunk output for semantic conditioning.

    Usage:
        memory = AMB3RV4_VO_GT(model, cfg_path).run(images, poses_gt)
    """

    def __init__(
        self,
        model: AMB3RStage2V4,
        cfg_path: str = './slam_stage2v4/slam_config_v4.yaml',
    ):
        self.cfg   = OmegaConf.load(cfg_path)
        self.model = model.to(self.cfg.device)

    def _pts_key(self):
        return (
            'pts3d_by_unprojection'
            if getattr(self.cfg, 'pts_by_unprojection', False)
            else 'world_points'
        )

    # ── Core forward (same as AMB3RV4_VO.local_mapping) ──────────────────────

    @torch.no_grad()
    def _forward(self, views_all: dict, init: bool = False) -> dict:
        """
        Run forward_chunk with query-pts in GT-world frame.

        For non-init chunks we pass memory.pts[cur_kf_idx] (GT-world,
        metric) as pts_for_query so voxel_store lookups hit the same
        spatial lattice the store was written into. voxel_store UPDATE
        is deferred to _init_map / _update_map, where GT-world
        pts_metric is available and can be downsampled to patch
        resolution before writing — ensuring store keys live in the
        same frame as future queries.
        """
        voxel_store = self.memory.voxel_store
        pts_query   = None
        if not init:
            cur_kf_idx = self.memory.cur_kf_idx
            num_kf     = len(cur_kf_idx)
            _, T_chunk, _, H, W = views_all['images'].shape
            device     = views_all['images'].device
            pts_query  = torch.zeros(1, T_chunk, H, W, 3, device=device)
            pts_query[0, :num_kf] = self.memory.pts[cur_kf_idx].to(device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            frames = {'images': views_all['images']}
            preds, M_content, W_conf, pts_flat = self.model.forward_chunk(
                frames, voxel_store, pts_for_query=pts_query,
            )
        preds['pts_flat']  = pts_flat
        preds['W_conf']    = W_conf
        preds['M_content'] = M_content  # expose for downstream voxel_store update
        return preds

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_map(self, views_all: dict, poses_gt: torch.Tensor):
        """
        Bootstrap with metric-scale geometry from GT poses.

        1. forward_chunk (voxel_store not yet updated)
        2. Estimate metric scale from GT vs predicted translations
        3. Scale pts, then per-frame GT-transform → metric pts
        4. Init geometry buffers with metric pts + GT poses
        5. Seed voxel_store with M_content
        6. Push semantic/instance features to VoxelFeatureMap
        """
        res = self._forward(views_all, init=True)

        pts_local  = res[self._pts_key()][0].cpu()
        c2w_local  = res['pose'][0].cpu()
        conf_local = res['world_points_conf'][0].cpu()
        T_init     = pts_local.shape[0]

        conf_sig = (conf_local - 1) / conf_local
        conf_sig[conf_sig == 0] = 1e-6

        self._metric_scale = _scale_from_translations(
            poses_gt[:T_init], c2w_local
        )
        print(f"[GT-pose V4] Init metric scale: {self._metric_scale:.4f}")

        c2w_scaled = c2w_local.clone()
        c2w_scaled[:, :3, 3] *= self._metric_scale
        pts_scaled = pts_local * self._metric_scale

        map_idx_init = torch.arange(T_init)
        pts_metric   = _apply_gt_transform(
            pts_scaled, c2w_scaled, poses_gt, map_idx_init
        )

        # Keyframe selection using GT poses
        c2w_gt_init = poses_gt[:T_init].float()
        dists = extrinsic_distance_batch_query(c2w_gt_init, c2w_gt_init)
        is_kf = select_keyframes_iteratively(
            dists, conf_local, self.cfg.keyframe_threshold,
            keyframe_indices=[0],
        )
        kf_indices = torch.nonzero(is_kf, as_tuple=False).squeeze(1)

        self.memory.pts  [:T_init] = pts_metric
        self.memory.conf [:T_init] = conf_sig
        self.memory.poses[:T_init] = c2w_gt_init
        self.memory.iter [:T_init] = 1
        self.memory.kf_idx         = kf_indices
        self.memory.cur_kf_idx     = kf_indices

        # Seed voxel_store using GT-world pts_metric (patch-downsampled).
        # Writing with GT-world coords makes voxel_store keys consistent
        # with all future queries (which use memory.pts = GT-world).
        if res.get('M_content') is not None:
            T_len, H_full, W_full, _ = pts_metric.shape
            N_per_t = res['pts_flat'].shape[1] // T_len
            H_patch = H_full // 14   # VGGT patch stride
            W_patch = W_full // 14
            assert H_patch * W_patch == N_per_t, (
                f"Patch grid {H_patch}x{W_patch}={H_patch * W_patch} "
                f"!= per-frame flat count {N_per_t}"
            )
            pts_metric_patch = SLAMemoryV4._downsample_pts(
                pts_metric.float(), H_patch, W_patch
            )  # (T, Hp, Wp, 3)
            pts_flat_gtworld = pts_metric_patch.reshape(1, -1, 3).to(
                res['pts_flat'].device, res['pts_flat'].dtype
            )
            AMB3RStage2V4.update_voxel_store(
                self.memory.voxel_store,
                res['M_content'], res['W_conf'], pts_flat_gtworld,
                mem_mode=self.model.mem_mode,
            )
            print(f"[GT-pose V4] voxel_store seeded (GT-world frame): "
                  f"{len(self.memory.voxel_store):,} voxels")

        if self.memory.semantic_voxel_map is not None and 'semantic_feat' in res:
            SLAMemoryV4._push_to_voxel_map(
                self.memory.semantic_voxel_map,
                pts_metric,
                res['semantic_feat'][0].float().cpu(),
                res['semantic_conf'][0].float().cpu(),
            )
            print(f"[GT-pose V4] Init semantic voxels: "
                  f"{self.memory.semantic_voxel_map.num_voxels:,}")

        if self.memory.instance_voxel_map is not None and 'instance_feat' in res:
            SLAMemoryV4._push_to_voxel_map(
                self.memory.instance_voxel_map,
                pts_metric,
                res['instance_feat'][0].float().cpu(),
                res['instance_conf'][0].float().cpu(),
            )
            print(f"[GT-pose V4] Init instance voxels: "
                  f"{self.memory.instance_voxel_map.num_voxels:,}")

    # ── Incremental update ────────────────────────────────────────────────────

    def _update_map(self, views_all: dict, poses_gt: torch.Tensor,
                    start_idx: int, end_idx: int):
        """
        Incremental step with GT poses.

        Geometry: scale-corrected model pts → GT-transform → metric world pts
                  → confidence-weighted fusion with stored metric pts.
        Poses:    stored directly from poses_gt (no blending).
        Semantic: VoxelFeatureMap updated with fused metric pts.
        voxel_store: updated here with GT-world pts_metric (patch-downsampled).
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
        )

        pts_global  = self.memory.pts  [map_idx]
        conf_global = self.memory.conf [map_idx]
        c2w_global  = self.memory.poses[map_idx]
        iter_global = self.memory.iter [map_idx]

        scale = _scale_from_pts(
            pts_local, c2w_local, conf_sig_local,
            pts_global, c2w_global, num_kf,
        )
        print(f"[GT-pose V4] chunk scale: {scale:.4f}")

        c2w_scaled = c2w_local.clone()
        c2w_scaled[:, :3, 3] *= scale
        pts_scaled = pts_local * scale

        pts_metric = _apply_gt_transform(pts_scaled, c2w_scaled, poses_gt, map_idx)

        # ── voxel_store update (model conditioning memory) ────────────────
        # Update with GT-world pts_metric (patch-downsampled) so that
        # storage and query both live in GT-world. Without this, every
        # chunk writes into a different chunk-local lattice and fused
        # features collapse to null_token.
        if res.get('M_content') is not None:
            T_len, H_full, W_full, _ = pts_metric.shape
            N_per_t = res['pts_flat'].shape[1] // T_len
            H_patch = H_full // 14
            W_patch = W_full // 14
            assert H_patch * W_patch == N_per_t, (
                f"Patch grid {H_patch}x{W_patch}={H_patch * W_patch} "
                f"!= per-frame flat count {N_per_t}"
            )
            pts_metric_patch = SLAMemoryV4._downsample_pts(
                pts_metric.float(), H_patch, W_patch
            )
            pts_flat_gtworld = pts_metric_patch.reshape(1, -1, 3).to(
                res['pts_flat'].device, res['pts_flat'].dtype
            )
            AMB3RStage2V4.update_voxel_store(
                self.memory.voxel_store,
                res['M_content'], res['W_conf'], pts_flat_gtworld,
                mem_mode=self.model.mem_mode,
            )

        conf_global_sum = conf_global * iter_global[:, None, None]
        self.memory.pts[map_idx] = (
            conf_global_sum[..., None] * pts_global
            + conf_sig_local[..., None] * pts_metric
        ) / (conf_global_sum[..., None] + conf_sig_local[..., None])
        self.memory.conf[map_idx] = (
            conf_global_sum + conf_sig_local
        ) / (iter_global[:, None, None] + 1)
        self.memory.iter[map_idx] = iter_global + 1

        for global_i in map_idx.tolist():
            self.memory.poses[global_i] = poses_gt[global_i].float()

        fused = self.memory.pts[map_idx].cpu()

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
            self.memory.cur_kf_idx = torch.cat([self.memory.cur_kf_idx, new_kf])
            self.memory.kf_idx     = torch.cat([self.memory.kf_idx,     new_kf])
            self.memory.keyframe_management()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, images: torch.Tensor,
            poses_gt: torch.Tensor) -> SLAMemoryV4:
        """
        Parameters
        ----------
        images   : (1, T, 3, H, W) float32 in [-1, 1]
        poses_gt : (T, 4, 4)  GT camera-to-world, metric, cpu float32

        Returns
        -------
        SLAMemoryV4
        """
        assert images.min() >= -1 and images.max() <= 1
        Bs, T, _, H, W = images.shape
        assert Bs == 1

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
                self._init_map(views_all, poses_gt)
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
                self._update_map(views_to_map, poses_gt, start_idx, idx)
                last_mapped_idx = idx

            fps = (idx + 1) / max(time.time() - t0, 1e-6)
            print(f"[GT-pose V4] frame {idx + 1}/{T}  "
                  f"KF: {self.memory.cur_kf_idx.tolist()}  "
                  f"({fps:.1f} fps)")

        return self.memory
