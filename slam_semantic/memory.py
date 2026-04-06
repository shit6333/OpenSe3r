"""
SLAMemory  (semantic + instance variant)
========================================
Geometry buffers (pts, conf, poses, iter) are kept exactly as the original.
Semantic and instance features are stored in VoxelFeatureMaps instead of
per-frame tensors, eliminating the  num_frames × C × H × W  memory spike.

Voxel update strategy
---------------------
Model output:
    pts_local:     (T, H, W, 3)          full-resolution world-space pts
    semantic_feat: (T, C, H_feat, W_feat) patch-resolution features (H_feat ≈ H/14)
    semantic_conf: (T, 1, H_feat, W_feat)

We downsample pts to (T, H_feat, W_feat, 3) via bilinear interpolation so that
each feature pixel has a matching 3-D location, then flatten and push to
VoxelFeatureMap.update().

Per-step cost:  T × H_feat × W_feat  ≈  8 × 37 × 28 ≈  8 300 points → trivial.
Total memory:   V × C × 4 bytes  where V = occupied voxels (≈50K–200K indoors).
"""

import torch
import torch.nn.functional as F

from amb3r.tools.pose_interp import interpolate_poses
from amb3r.tools.pts_align import coordinate_alignment
from amb3r.tools.keyframes import select_keyframes_iteratively
from amb3r.tools.pose_dist import extrinsic_distance_batch_query
from slam_semantic.semantic_voxel_map import VoxelFeatureMap


class SLAMemory():
    def __init__(self, cfg, num_frames: int, H: int, W: int,
                 has_semantic: bool = False,
                 has_instance: bool = False,
                 sem_dim: int = 512,
                 ins_dim: int = 16):
        """
        Params:
            cfg          — OmegaConf slam config
            num_frames   — total frames in the sequence
            H, W         — full image resolution
            has_semantic — create a semantic VoxelFeatureMap
            has_instance — create an instance VoxelFeatureMap
            sem_dim      — semantic feature channels (e.g. 512 for CLIP)
            ins_dim      — instance feature channels (e.g. 16)
        """
        self.cfg          = cfg
        self.num_frames   = num_frames
        self.has_semantic = has_semantic
        self.has_instance = has_instance
        self.sem_dim      = sem_dim
        self.ins_dim      = ins_dim

        # ── Geometry buffers (unchanged) ──────────────────────────────────
        self.pts   = torch.zeros((num_frames, H, W, 3))
        self.conf  = torch.zeros((num_frames, H, W))
        self.poses = torch.zeros((num_frames, 4, 4))
        self.iter  = torch.zeros((num_frames,))

        self.kf_idx     = None
        self.cur_kf_idx = None

        # ── Voxel feature maps ────────────────────────────────────────────
        # voxel_size defaults to 0.05 m if not in config
        voxel_size = float(getattr(cfg, 'semantic_voxel_size', 0.05))

        if has_semantic and sem_dim > 0:
            self.semantic_voxel_map = VoxelFeatureMap(
                voxel_size=voxel_size,
                feat_dim=sem_dim,
            )
        else:
            self.semantic_voxel_map = None

        if has_instance and ins_dim > 0:
            self.instance_voxel_map = VoxelFeatureMap(
                voxel_size=voxel_size,
                feat_dim=ins_dim,
            )
        else:
            self.instance_voxel_map = None

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _downsample_pts(pts: torch.Tensor,
                        target_h: int, target_w: int) -> torch.Tensor:
        """(T, H, W, 3) → (T, target_h, target_w, 3) via bilinear."""
        T = pts.shape[0]
        pts_chw = pts.permute(0, 3, 1, 2).float()   # (T, 3, H, W)
        pts_dn  = F.interpolate(
            pts_chw, size=(target_h, target_w),
            mode='bilinear', align_corners=False,
        )                                             # (T, 3, h, w)
        return pts_dn.permute(0, 2, 3, 1)            # (T, h, w, 3)

    @staticmethod
    def _push_to_voxel_map(voxel_map: VoxelFeatureMap,
                            pts_world: torch.Tensor,   # (T, H, W, 3)  cpu
                            feat:      torch.Tensor,   # (T, C, H_f, W_f) cpu
                            conf:      torch.Tensor,   # (T, 1, H_f, W_f) cpu
                            ):
        """
        Downsample pts to match the feature spatial resolution,
        flatten everything, and push to the voxel map.
        """
        T, C, H_f, W_f = feat.shape

        pts_down  = SLAMemory._downsample_pts(pts_world.float(), H_f, W_f)
        pts_flat  = pts_down.reshape(-1, 3)                          # (T*H_f*W_f, 3)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C).float() # (T*H_f*W_f, C)
        conf_flat = conf.squeeze(1).reshape(-1).float()              # (T*H_f*W_f,)

        voxel_map.update(pts_flat, feat_flat, conf_flat)

    # ──────────────────────────────────────────────────────────────────────
    # initialize
    # ──────────────────────────────────────────────────────────────────────

    def initialize(self, res: dict):
        """Initialise map from the first batch of model predictions."""
        conf    = res['world_points_conf'][0].cpu()      # (T, H, W)
        pts_key = (
            'pts3d_by_unprojection'
            if getattr(self.cfg, 'pts_by_unprojection', False)
            and 'pts3d_by_unprojection' in res
            else 'world_points'
        )
        pts  = res[pts_key][0].cpu()                     # (T, H, W, 3)
        c2w  = res['pose'][0].cpu()                      # (T, 4, 4)
        T_len = len(pts)

        conf_sig = (conf - 1) / conf
        conf_sig[conf_sig == 0] = 1e-6

        dists       = extrinsic_distance_batch_query(c2w, c2w)
        is_keyframe = select_keyframes_iteratively(
            dists, conf, self.cfg.keyframe_threshold,
            keyframe_indices=[0],
        )
        kf_indices = torch.nonzero(is_keyframe, as_tuple=False).squeeze(1)

        self.pts[:T_len]   = pts
        self.conf[:T_len]  = conf_sig
        self.poses[:T_len] = c2w
        self.iter[:T_len]  = 1
        self.kf_idx        = kf_indices
        self.cur_kf_idx    = kf_indices

        # ── Semantic voxel map ─────────────────────────────────────────────
        if self.semantic_voxel_map is not None and 'semantic_feat' in res:
            self._push_to_voxel_map(
                self.semantic_voxel_map,
                pts,                                    # already cpu
                res['semantic_feat'][0].float().cpu(),  # (T, C, H_f, W_f)
                res['semantic_conf'][0].float().cpu(),  # (T, 1, H_f, W_f)
            )
            print(f"[SLAMemory] Semantic voxel map init: "
                  f"{self.semantic_voxel_map.num_voxels} voxels")

        # ── Instance voxel map ─────────────────────────────────────────────
        if self.instance_voxel_map is not None and 'instance_feat' in res:
            self._push_to_voxel_map(
                self.instance_voxel_map,
                pts,
                res['instance_feat'][0].float().cpu(),
                res['instance_conf'][0].float().cpu(),
            )
            print(f"[SLAMemory] Instance voxel map init: "
                  f"{self.instance_voxel_map.num_voxels} voxels")

    # ──────────────────────────────────────────────────────────────────────
    # Keyframe management  (unchanged from original)
    # ──────────────────────────────────────────────────────────────────────

    def resample_keyframes(self, current_kf_indices, current_poses, num_to_keep,
                           all_mapped_poses, num_top_k=3, sum_min=True):
        if len(current_kf_indices) <= num_to_keep:
            return current_kf_indices

        dists        = extrinsic_distance_batch_query(current_poses, current_poses)
        position_map = {
            global_idx.item(): pos
            for pos, global_idx in enumerate(current_kf_indices)
        }

        newest_kf_global_idx = current_kf_indices[-1].item()
        resampled_indices    = [newest_kf_global_idx]

        initial_candidates = current_kf_indices[:-1]
        if len(initial_candidates) > 0:
            newest_kf_pos   = position_map[newest_kf_global_idx]
            ic_positions    = [position_map[idx.item()] for idx in initial_candidates]
            dists_to_newest = dists[ic_positions, newest_kf_pos]
            sorted_idx      = torch.argsort(dists_to_newest)
            sorted_cands    = initial_candidates[sorted_idx]

            for kf in sorted_cands:
                if len(resampled_indices) >= num_top_k + 1:
                    break
                gidx = kf.item()
                if abs(gidx - newest_kf_global_idx) > 200:
                    resampled_indices.append(gidx)
                    continue
                pos = position_map[gidx]
                if all(
                    dists[pos, position_map[s]] > self.cfg.keyframe_resample_threshold_min
                    for s in resampled_indices
                ):
                    resampled_indices.append(gidx)

        candidate_pool = sorted([
            idx.item() for idx in current_kf_indices
            if idx.item() not in resampled_indices
        ])

        num_to_fill = num_to_keep - self.cfg.bridge_keyframes
        while len(resampled_indices) < num_to_fill:
            best = None
            for gidx in candidate_pool:
                pos  = position_map[gidx]
                dmin = min(dists[pos, position_map[s]] for s in resampled_indices)
                dmax = max(dists[pos, position_map[s]] for s in resampled_indices)
                if (dmin <= self.cfg.keyframe_resample_threshold
                        and dmax <= self.cfg.keyframe_resample_threshold_max_b):
                    best = gidx
                    break
            if best is None:
                break
            resampled_indices.append(best)
            candidate_pool.remove(best)

        bridge_frames_to_add = torch.tensor([], dtype=torch.long)
        if self.cfg.bridge_keyframes > 0:
            resampled_set = set(resampled_indices)
            mid_pool      = [
                idx.item() for idx in current_kf_indices
                if idx.item() not in resampled_set
            ]
            if len(mid_pool) > 0:
                mid_poses = all_mapped_poses[mid_pool]
                res_poses = all_mapped_poses[resampled_indices]
                d_mid     = extrinsic_distance_batch_query(mid_poses, res_poses)
                k         = min(self.cfg.bridge_keyframes, len(mid_pool))
                if not sum_min:
                    min_d, _ = torch.min(d_mid, dim=1)
                    _, top_k = torch.topk(min_d, k=k, largest=False)
                else:
                    sum_d    = torch.sum(d_mid, dim=1)
                    _, top_k = torch.topk(sum_d, k=k, largest=False)
                bridge_frames_to_add = torch.tensor(mid_pool)[top_k]

        return torch.tensor(
            bridge_frames_to_add.tolist() + sorted(list(set(resampled_indices))),
            dtype=torch.long,
        )

    def keyframe_management(self):
        last_kf_idx   = self.cur_kf_idx[-1]
        last_kf_pose  = self.poses[last_kf_idx][None]
        other_kf_idx  = self.cur_kf_idx[self.cur_kf_idx != last_kf_idx]
        other_kf_pose = self.poses[other_kf_idx]

        max_dist = torch.max(
            extrinsic_distance_batch_query(last_kf_pose, other_kf_pose)
        )
        if (max_dist < self.cfg.keyframe_resample_threshold_max_f
                and len(self.cur_kf_idx) <= self.cfg.max_keyframes):
            return

        print(f"Active keyframes ({len(self.cur_kf_idx)}) > max "
              f"({self.cfg.max_keyframes}). Resampling...")
        resampled = self.resample_keyframes(
            self.kf_idx, self.poses[self.kf_idx],
            num_to_keep=self.cfg.min_keyframes,
            all_mapped_poses=self.poses,
            num_top_k=self.cfg.top_k_keyframes,
        )
        print(f"Resampled to {len(resampled)} KFs: {resampled.tolist()}")
        self.cur_kf_idx = resampled

    # ──────────────────────────────────────────────────────────────────────
    # update
    # ──────────────────────────────────────────────────────────────────────

    def update(self, res: dict, start_idx: int, end_idx: int):
        """
        Update geometry + voxel feature maps from new model predictions.

        Geometry is fused with the original confidence-weighted running average.
        After fusion, the best available 3-D positions (fused pts) are used as
        anchors for pushing features into the voxel maps.
        """
        pts_key = (
            'pts3d_by_unprojection'
            if getattr(self.cfg, 'pts_by_unprojection', False)
            and 'pts3d_by_unprojection' in res
            else 'world_points'
        )

        pts_local  = res[pts_key][0].cpu()               # (T, H, W, 3)
        conf_local = res['world_points_conf'][0].cpu()   # (T, H, W)
        c2w_local  = res['pose'][0].cpu()                # (T, 4, 4)

        conf_sig_local = (conf_local - 1) / conf_local
        conf_sig_local[conf_sig_local == 0] = 1e-6

        map_idx     = torch.cat(
            [self.cur_kf_idx, torch.arange(start_idx, end_idx + 1)], dim=0
        )
        pts_global  = self.pts[map_idx]
        conf_global = self.conf[map_idx]
        c2w_global  = self.poses[map_idx]
        iter_global = self.iter[map_idx]

        # ── Coordinate alignment ──────────────────────────────────────────
        pts_global_from_local, c2w_global_from_local = coordinate_alignment(
            pts_local, c2w_local, conf_sig_local,
            pts_global, c2w_global, conf_global,
            len(self.cur_kf_idx),
            transform=self.cur_kf_idx[0] != 0,
            scale=None,
        )

        # ── Geometry weighted fusion ──────────────────────────────────────
        conf_global_sum = conf_global * iter_global[:, None, None]

        self.pts[map_idx] = (
            conf_global_sum[..., None] * pts_global
            + conf_sig_local[..., None] * pts_global_from_local
        ) / (conf_global_sum[..., None] + conf_sig_local[..., None])

        self.conf[map_idx] = (conf_global_sum + conf_sig_local) / (
            iter_global[:, None, None] + 1
        )
        self.iter[map_idx] = iter_global + 1

        c2w_global_merged = interpolate_poses(
            c2w_global, c2w_global_from_local,
            conf_global, conf_sig_local,
            interpolate=True,
        )
        self.poses[map_idx] = c2w_global_merged

        # ── Semantic voxel map update ─────────────────────────────────────
        # Use the fused pts (most accurate world-space coords) as anchors.
        if self.semantic_voxel_map is not None and 'semantic_feat' in res:
            self._push_to_voxel_map(
                self.semantic_voxel_map,
                self.pts[map_idx].cpu(),               # fused geometry
                res['semantic_feat'][0].float().cpu(), # (T, C, H_f, W_f)
                res['semantic_conf'][0].float().cpu(), # (T, 1, H_f, W_f)
            )

        # ── Instance voxel map update ─────────────────────────────────────
        if self.instance_voxel_map is not None and 'instance_feat' in res:
            self._push_to_voxel_map(
                self.instance_voxel_map,
                self.pts[map_idx].cpu(),
                res['instance_feat'][0].float().cpu(),
                res['instance_conf'][0].float().cpu(),
            )

        # ── Keyframe discovery ────────────────────────────────────────────
        dists = extrinsic_distance_batch_query(c2w_global_merged, c2w_global_merged)
        is_keyframe = select_keyframes_iteratively(
            dists, conf_local, self.cfg.keyframe_threshold,
            keyframe_indices=list(range(len(self.cur_kf_idx))),
        )

        if is_keyframe[len(self.cur_kf_idx):].sum() > 0:
            new_kf = (
                torch.nonzero(is_keyframe[len(self.cur_kf_idx):], as_tuple=False)
                .squeeze(1) + start_idx
            )
            self.cur_kf_idx = torch.cat([self.cur_kf_idx, new_kf], dim=0)
            self.kf_idx     = torch.cat([self.kf_idx,     new_kf], dim=0)
            self.keyframe_management()