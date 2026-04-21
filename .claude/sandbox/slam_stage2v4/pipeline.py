"""
AMB3RV4_VO  —  Incremental SLAM pipeline for AMB3RStage2V4 (predicted poses).

Coordinate system
-----------------
Non-GT mode anchors "global frame" to frame-0-at-identity (defined by
the init chunk's model output).  All subsequent chunks are aligned to
this frame via coordinate_alignment in memory.update.

voxel_store now lives entirely in this global frame:
    query  — pts_for_query = memory.pts[cur_kf_idx] (global) for keyframes,
             zeros for new frames (→ voxel_store miss → null token, safe)
    update — pts_flat is recomputed from coordinate_alignment output
             (pts_global_from_local, full-res) inside memory.update
             so that every update key is globally consistent.

This matches the training-time invariant where `pts_for_query` is always
in the same frame as where `voxel_store` was previously populated.
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
from slam_stage2v4.memory import SLAMemoryV4
from benchmark.tools.pose_eval import evaluate_evo


PATCH_SIZE = 14


class AMB3RV4_VO:
    def __init__(
        self,
        model: AMB3RStage2V4,
        cfg_path: str = './slam_stage2v4/slam_config_v4.yaml',
    ):
        self.cfg   = OmegaConf.load(cfg_path)
        self.model = model.to(self.cfg.device)

    # ── Core forward pass ────────────────────────────────────────────────────

    @torch.no_grad()
    def local_mapping(self, views_all: dict, init: bool = False) -> dict:
        """
        One SLAM step.

        Init  chunk (init=True):
            pts_for_query = None → model uses its own world_points (the init
            chunk defines "global frame" for all subsequent chunks).
            voxel_store update is DEFERRED to initialize_map() where we
            can feed the fused full-res pts.

        Later chunks (init=False):
            pts_for_query is built from memory.pts[cur_kf_idx] (global frame
            known for keyframes).  New-frame slots are zero (→ voxel_store
            miss, null_token returned, fusion safe).
            voxel_store update happens in SLAMemoryV4.update using the
            coordinate_alignment-corrected pts.
        """
        voxel_store = self.keyframe_memory.voxel_store

        # Build pts_for_query in global frame for keyframe slots
        pts_query = None
        if not init:
            cur_kf_idx = self.keyframe_memory.cur_kf_idx
            num_kf     = len(cur_kf_idx)
            _, T_chunk, _, H, W = views_all['images'].shape
            device  = views_all['images'].device
            pts_query = torch.zeros(1, T_chunk, H, W, 3, device=device)
            pts_query[0, :num_kf] = (
                self.keyframe_memory.pts[cur_kf_idx].to(device)
            )
            # new-frame slots remain 0 → voxel_store query misses, fusion
            # sees null_token for those patches (matches "no-memory" training)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            frames = {'images': views_all['images']}
            preds, M_content, W_conf, pts_flat = self.model.forward_chunk(
                frames, voxel_store, pts_for_query=pts_query,
            )

        preds['pts_flat']       = pts_flat      # in global frame for KF slots
        preds['W_conf']         = W_conf
        preds['_num_kf']        = (len(self.keyframe_memory.cur_kf_idx)
                                    if not init else 0)

        # voxel_store update is done later (in initialize_map or memory.update)
        # with fused full-res global-frame pts for maximum consistency.
        return preds

    # ── Initialisation ───────────────────────────────────────────────────────

    def initialize_map(self, views_all: dict):
        """
        Seed global frame and voxel_store from the first window.

        Since init chunk's model output defines "global frame", we can use
        res['world_points'] directly (no alignment needed).  voxel_store is
        seeded with the full-res pts downsampled to patch resolution.
        """
        import torch.nn.functional as F

        res  = self.local_mapping(views_all, init=True)
        conf = res['world_points_conf']
        avg_conf = conf[0].mean(dim=(-1, -2)).cpu().numpy()
        print("Average confidence values:", [f"{c:.2f}" for c in avg_conf])

        self.keyframe_memory.initialize(res)

        # Seed voxel_store with init-window features
        # Use res['pts_flat'] which is the patch-downsampled world_points
        # (= global frame by definition of init chunk).
        if res['M_content'] is not None:
            AMB3RStage2V4.update_voxel_store(
                self.keyframe_memory.voxel_store,
                res['M_content'], res['W_conf'], res['pts_flat'],
                mem_mode=self.model.mem_mode,
            )
            print(f"[AMB3RV4_VO] voxel_store seeded: "
                  f"{len(self.keyframe_memory.voxel_store):,} voxels "
                  f"(global = frame-0-identity)")

    # ── Incremental mapping ──────────────────────────────────────────────────

    def mapping(self, views_all: dict) -> tuple:
        t0  = time.time()
        res = self.local_mapping(views_all)
        t_ff = time.time() - t0

        t1 = time.time()
        self.keyframe_memory.update(
            res,
            start_idx=views_all['start_idx'],
            end_idx=views_all['end_idx'],
        )
        t_mem = time.time() - t1
        return t_ff, t_mem

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self, images: torch.Tensor, poses_gt=None) -> SLAMemoryV4:
        assert images.min() >= -1 and images.max() <= 1

        Bs, T, _, H, W = images.shape

        s1 = self.model.stage1
        has_semantic = (
            hasattr(s1, 'lseg')
            and s1.front_end.model.semantic_head is not None
        )
        has_instance = s1.front_end.model.instance_head is not None

        sem_dim = getattr(s1, 'clip_dim', 512) if has_semantic else 0
        ins_dim = getattr(s1, 'ins_dim',  16)  if has_instance else 0

        self.keyframe_memory = SLAMemoryV4(
            cfg=self.cfg, num_frames=T, H=H, W=W,
            model=self.model,
            has_semantic=has_semantic, has_instance=has_instance,
            sem_dim=sem_dim, ins_dim=ins_dim,
        )

        initialized = False
        last_mapped_idx = -1
        t_ff_acc = t_mem_acc = 0.0

        for idx in range(T):
            if idx < self.cfg.map_init_window - 1:
                continue

            if not initialized:
                views_all = {
                    'images': images[:, :idx + 1].to(self.cfg.device),
                }
                self.initialize_map(views_all)
                initialized     = True
                last_mapped_idx = idx
                print(f"Processed frame {idx + 1}/{T}, "
                      f"KF ids: {self.keyframe_memory.cur_kf_idx.tolist()}")
                continue

            if (idx - last_mapped_idx < self.cfg.map_every) and (idx < T - 1):
                continue

            start_idx = idx + 1 - self.cfg.map_every
            views_to_map = {
                'images': torch.cat(
                    [images[:, self.keyframe_memory.cur_kf_idx],
                     images[:, start_idx:idx + 1]],
                    dim=1,
                ).to(self.cfg.device),
                'start_idx': start_idx,
                'end_idx':   idx,
            }

            t_ff, t_mem = self.mapping(views_to_map)
            t_ff_acc  += t_ff
            t_mem_acc += t_mem
            last_mapped_idx = idx

            print(f"Processed frame {idx + 1}/{T}, "
                  f"KF ids: {self.keyframe_memory.cur_kf_idx.tolist()}")

            try:
                evaluate_evo(
                    poses_gt[:idx + 1] if poses_gt is not None else None,
                    self.keyframe_memory.poses[:idx + 1].numpy(),
                    None, None, monocular=True, plot=False,
                )
            except Exception:
                pass

        n_mapped = max(T - self.cfg.map_init_window, 1)
        print(f"\nAvg feedforward  : {t_ff_acc / n_mapped:.3f}s  "
              f"({n_mapped / max(t_ff_acc, 1e-9):.1f} fps)")
        print(f"Avg memory update: {t_mem_acc / n_mapped:.3f}s  "
              f"({n_mapped / max(t_mem_acc, 1e-9):.1f} fps)")

        vs = self.keyframe_memory.voxel_store
        print(f"voxel_store size : {len(vs):,} voxels")
        if self.keyframe_memory.semantic_voxel_map is not None:
            print(f"semantic map     : "
                  f"{self.keyframe_memory.semantic_voxel_map.num_voxels:,} voxels")
        if self.keyframe_memory.instance_voxel_map is not None:
            print(f"instance map     : "
                  f"{self.keyframe_memory.instance_voxel_map.num_voxels:,} voxels")

        return self.keyframe_memory
