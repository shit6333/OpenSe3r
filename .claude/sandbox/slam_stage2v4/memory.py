"""
SLAMemoryV4  —  Scene memory for Stage-2 V4 SLAM.

Two distinct memory structures coexist:

    voxel_store  (DetachedVoxelStore / DifferentiableVoxelMap)
        Model conditioning memory.  Stores compact features at patch
        resolution and is queried by AMB3RStage2V4.forward_chunk() on
        every SLAM step.  Lives at mem_voxel_size resolution.

    semantic_voxel_map / instance_voxel_map  (VoxelFeatureMap)
        Final 3-D semantic / instance map for export.  Stores high-dim
        task features (CLIP 512-dim, ins 16-dim) accumulated over the
        whole sequence.  Lives at semantic_voxel_size resolution.

Geometry buffers (pts, conf, poses, iter) and keyframe management are
identical to the original SLAMemory.
"""

import torch
import torch.nn.functional as F

from amb3r.tools.pose_interp import interpolate_poses
from amb3r.tools.pts_align import coordinate_alignment
from amb3r.tools.keyframes import select_keyframes_iteratively
from amb3r.tools.pose_dist import extrinsic_distance_batch_query
from slam_semantic.semantic_voxel_map import VoxelFeatureMap


class SLAMemoryV4:
    def __init__(
        self,
        cfg,
        num_frames: int,
        H: int,
        W: int,
        model,               # AMB3RStage2V4 — used to create voxel_store
        has_semantic: bool = True,
        has_instance: bool = True,
        sem_dim: int = 512,
        ins_dim: int = 16,
    ):
        """
        Parameters
        ----------
        cfg          OmegaConf slam config (slam_config_v4.yaml)
        num_frames   Total frames in the sequence
        H, W         Full image resolution
        model        AMB3RStage2V4 instance — calls model.make_voxel_store()
        has_semantic Create semantic VoxelFeatureMap
        has_instance Create instance VoxelFeatureMap
        sem_dim      Semantic feature dimension (e.g. 512 for CLIP)
        ins_dim      Instance feature dimension (e.g. 16)
        """
        self.cfg          = cfg
        self.num_frames   = num_frames
        self.has_semantic = has_semantic
        self.has_instance = has_instance
        self.sem_dim      = sem_dim
        self.ins_dim      = ins_dim

        # ── Geometry buffers (identical to SLAMemory) ─────────────────────
        self.pts   = torch.zeros(num_frames, H, W, 3)
        self.conf  = torch.zeros(num_frames, H, W)
        self.poses = torch.zeros(num_frames, 4, 4)
        self.iter  = torch.zeros(num_frames)

        self.kf_idx     = None
        self.cur_kf_idx = None

        # ── Model conditioning memory (voxel_store) ────────────────────────
        # Created via model.make_voxel_store() so that mem_mode,
        # mem_voxel_size, store_ema_lambda, etc. are inherited from the model.
        self.voxel_store = model.make_voxel_store()
        self._model_mem_mode = model.mem_mode

        # ── Final semantic / instance 3-D maps ───────────────────────────
        sem_vox = float(getattr(cfg, 'semantic_voxel_size', 0.05))

        self.semantic_voxel_map = (
            VoxelFeatureMap(voxel_size=sem_vox, feat_dim=sem_dim)
            if has_semantic and sem_dim > 0 else None
        )
        self.instance_voxel_map = (
            VoxelFeatureMap(voxel_size=sem_vox, feat_dim=ins_dim)
            if has_instance and ins_dim > 0 else None
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _downsample_pts(pts: torch.Tensor, th: int, tw: int) -> torch.Tensor:
        """(T, H, W, 3) → (T, th, tw, 3) via bilinear."""
        T = pts.shape[0]
        p = pts.permute(0, 3, 1, 2).float()
        p = F.interpolate(p, size=(th, tw), mode='bilinear', align_corners=False)
        return p.permute(0, 2, 3, 1)

    @staticmethod
    def _push_to_voxel_map(
        voxel_map: VoxelFeatureMap,
        pts_world: torch.Tensor,   # (T, H, W, 3)
        feat:      torch.Tensor,   # (T, C, H_f, W_f)
        conf:      torch.Tensor,   # (T, 1, H_f, W_f)
    ):
        T, C, H_f, W_f = feat.shape
        pts_d  = SLAMemoryV4._downsample_pts(pts_world.float(), H_f, W_f)
        pts_f  = pts_d.reshape(-1, 3)
        feat_f = feat.permute(0, 2, 3, 1).reshape(-1, C).float()
        conf_f = conf.squeeze(1).reshape(-1).float()
        voxel_map.update(pts_f, feat_f, conf_f)

    # ── initialize ───────────────────────────────────────────────────────────

    def initialize(self, res: dict):
        """
        Initialise map from the first batch of predictions (forward_chunk output).

        Geometry: fills pts / conf / poses / iter for the init window.
        Semantic / instance voxel maps: pushed from res features.
        voxel_store: updated by the caller (AMB3RV4_VO.initialize_map) so that
                     subsequent forward_chunk calls see non-empty memory.
        """
        pts_key = (
            'pts3d_by_unprojection'
            if getattr(self.cfg, 'pts_by_unprojection', False)
               and 'pts3d_by_unprojection' in res
            else 'world_points'
        )

        pts  = res[pts_key][0].cpu()             # (T, H, W, 3)
        conf = res['world_points_conf'][0].cpu()  # (T, H, W)
        c2w  = res['pose'][0].cpu()               # (T, 4, 4)
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

        if self.semantic_voxel_map is not None and 'semantic_feat' in res:
            self._push_to_voxel_map(
                self.semantic_voxel_map,
                pts,
                res['semantic_feat'][0].float().cpu(),
                res['semantic_conf'][0].float().cpu(),
            )
            print(f"[SLAMemoryV4] Semantic map init: "
                  f"{self.semantic_voxel_map.num_voxels:,} voxels")

        if self.instance_voxel_map is not None and 'instance_feat' in res:
            self._push_to_voxel_map(
                self.instance_voxel_map,
                pts,
                res['instance_feat'][0].float().cpu(),
                res['instance_conf'][0].float().cpu(),
            )
            print(f"[SLAMemoryV4] Instance map init: "
                  f"{self.instance_voxel_map.num_voxels:,} voxels")

    # ── Keyframe management (identical to SLAMemory) ─────────────────────────

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

    # ── update ───────────────────────────────────────────────────────────────

    def update(self, res: dict, start_idx: int, end_idx: int):
        """
        Update geometry, voxel_store, and semantic/instance maps from new
        forward_chunk predictions.

        Steps:
            1. Geometry: coordinate alignment + confidence-weighted fusion
               (identical to SLAMemory.update)
            2. voxel_store: update with M_content at fused pts
               (provides conditioning for the next forward_chunk call)
            3. Semantic / instance VoxelFeatureMaps: push task features
               using fused full-resolution pts as anchors
            4. Keyframe discovery
        """
        from amb3r.model_stage2v4 import AMB3RStage2V4

        pts_key = (
            'pts3d_by_unprojection'
            if getattr(self.cfg, 'pts_by_unprojection', False)
               and 'pts3d_by_unprojection' in res
            else 'world_points'
        )

        pts_local  = res[pts_key][0].cpu()               # (T, H, W, 3)
        conf_local = res['world_points_conf'][0].cpu()    # (T, H, W)
        c2w_local  = res['pose'][0].cpu()                 # (T, 4, 4)

        conf_sig_local = (conf_local - 1) / conf_local
        conf_sig_local[conf_sig_local == 0] = 1e-6

        map_idx     = torch.cat(
            [self.cur_kf_idx, torch.arange(start_idx, end_idx + 1)], dim=0
        )
        pts_global  = self.pts[map_idx]
        conf_global = self.conf[map_idx]
        c2w_global  = self.poses[map_idx]
        iter_global = self.iter[map_idx]

        # ── 1. Coordinate alignment ───────────────────────────────────────
        pts_global_from_local, c2w_global_from_local = coordinate_alignment(
            pts_local, c2w_local, conf_sig_local,
            pts_global, c2w_global, conf_global,
            len(self.cur_kf_idx),
            transform=self.cur_kf_idx[0] != 0,
            scale=None,
        )

        # ── 2. Geometry weighted fusion ───────────────────────────────────
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

        # ── 3. voxel_store update (model conditioning memory) ─────────────
        # res['pts_flat'] from forward_chunk is in the CHUNK-LOCAL
        # ("camera-0 = identity") frame the model internally assumes.
        # voxel_store is a global hash-keyed store, so writing with
        # chunk-local coords produces keys on a different lattice than
        # queries (which are made in sequence-global frame via
        # pts_for_query = memory.pts[cur_kf_idx]). That would make hit
        # rate on subsequent chunks ≈ 0 and collapse fused features to
        # null_token. Fix: use pts_global_from_local (already in
        # sequence-global frame from coordinate_alignment above) and
        # downsample to patch resolution before writing.
        if res.get('M_content') is not None:
            T_len, H_full, W_full, _ = pts_local.shape
            N_per_t = res['pts_flat'].shape[1] // T_len
            # VGGT patch stride is 14
            H_patch = H_full // 14
            W_patch = W_full // 14
            assert H_patch * W_patch == N_per_t, (
                f"Patch grid {H_patch}x{W_patch}={H_patch * W_patch} "
                f"!= per-frame flat count {N_per_t}"
            )
            pts_global_patch = self._downsample_pts(
                pts_global_from_local.float(), H_patch, W_patch
            )  # (T, Hp, Wp, 3)
            pts_flat_global = pts_global_patch.reshape(1, -1, 3).to(
                res['pts_flat'].device, res['pts_flat'].dtype
            )
            AMB3RStage2V4.update_voxel_store(
                self.voxel_store,
                res['M_content'],
                res.get('W_conf'),
                pts_flat_global,
                mem_mode=self._model_mem_mode,
            )

        # ── 4. Semantic / instance VoxelFeatureMap update ─────────────────
        # Uses fused full-resolution pts for the highest-accuracy anchors.
        if self.semantic_voxel_map is not None and 'semantic_feat' in res:
            self._push_to_voxel_map(
                self.semantic_voxel_map,
                self.pts[map_idx].cpu(),               # fused pts
                res['semantic_feat'][0].float().cpu(),
                res['semantic_conf'][0].float().cpu(),
            )

        if self.instance_voxel_map is not None and 'instance_feat' in res:
            self._push_to_voxel_map(
                self.instance_voxel_map,
                self.pts[map_idx].cpu(),
                res['instance_feat'][0].float().cpu(),
                res['instance_conf'][0].float().cpu(),
            )

        # ── 5. Keyframe discovery ─────────────────────────────────────────
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
