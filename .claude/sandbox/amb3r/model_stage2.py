"""
AMB3RStage2
============
Wraps a frozen AMB3RStage1FullFT with two powerful DPT-based memory heads:

    SemanticMemoryHead  → MemoryConditionedDPTHead
    InstanceMemoryHead  → MemoryConditionedDPTHead

Data-flow
---------
                 ┌── Stage 1 (fully frozen, no_grad) ──────────────────────┐
  frames ──────► │  encode_patch_tokens                                      │
                 │  → clip_patch_fusion (CLIP injected into patch tokens)    │
                 │  → decode_patch_tokens  → aggregated_tokens_list  ────────┼──► (detached)
                 │  → decode_heads (sem/ins/geo)                             │
                 │    → world_points, sem_s1, ins_s1  [B,T,C,H,W]           │
                 └────────────────────────────────────────────────────────────┘
                                    │
                      pts (globally-aligned) downsampled to patch res (H/14)
                                    │
                         VoxelMemory.query()  @ patch res
                                    │
                         mem_sem/mem_ins/mem_mask  (B*T, C, Hp, Wp)
                                    │
                         upsample back to (H, W)
                                    │
                 ┌── Stage 2 (trainable) ──────────────────────────────────┐
                 │                                                           │
                 │  sem_cond = cat[mem_sem_up, sem_s1, mem_mask_up]        │
                 │  ins_cond = cat[mem_ins_up, sem_s1, ins_s1, mem_mask_up]│
                 │  all at (H, W) — DPT head pools internally to Hp×Wp     │
                 │                                                           │
                 │  SemanticMemoryDPTHead(aggregated_tokens_list, sem_cond) │
                 │  → refined_sem  [B,T,C_sem,H,W], sem_conf               │
                 │                                                           │
                 │  InstanceMemoryDPTHead(aggregated_tokens_list, ins_cond) │
                 │  → refined_ins  [B,T,C_ins,H,W], ins_conf               │
                 └────────────────────────────────────────────────────────────┘
                                    │
                         update VoxelMemory @ patch res (stop-grad)

Output resolution
-----------------
Stage 1 heads (PatchConditionedDPTHead with down_ratio=1) output at FULL image
resolution (H, W).  Stage 2 heads inherit the same architecture and also output
at (H, W) — pixel-aligned with the input images / depth maps / point cloud.

Voxel memory query / update is done at PATCH resolution (H//14, W//14) for
efficiency.  Memory features are upsampled back to (H, W) before being used as
conditioning inside the Stage-2 DPT heads.

Coordinate alignment
--------------------
The voxel memory is indexed in a globally-consistent coordinate frame.

* Training  : caller passes pts_for_query = GT world pts (B, T, H, W, 3)
  from the dataset (absolute ScanNet++ poses → absolute world pts).
* Inference : caller passes pts_for_query = coordinate-aligned predicted pts
  via SLAMemory.update() → coordinate_alignment() in the SLAM pipeline.
"""

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from amb3r.stage2_heads import build_semantic_memory_head, build_instance_memory_head

PATCH_SIZE = 14   # DINOv2 / VGGT patch size


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resize_pts(pts: torch.Tensor, H_out: int, W_out: int) -> torch.Tensor:
    """(B, T, H, W, 3) → (B*T, H_out, W_out, 3) via bilinear."""
    B, T, H, W, _ = pts.shape
    p = pts.flatten(0, 1).permute(0, 3, 1, 2).float()   # (B*T, 3, H, W)
    p = F.interpolate(p, size=(H_out, W_out), mode='bilinear', align_corners=False)
    return p.permute(0, 2, 3, 1)                          # (B*T, H_out, W_out, 3)


@torch.no_grad()
def _query_voxel_maps(pts_bt, sem_map, ins_map, device, dtype, N, Hp, Wp, ins_dim):
    """
    Query sem and ins voxel maps at patch resolution.

    pts_bt   : (N, Hp, Wp, 3) float32 cpu   N = B*T
    returns  : mem_sem  (N, C_sem, Hp, Wp)
               mem_ins  (N, C_ins, Hp, Wp)
               mem_mask (N, 1,     Hp, Wp)   soft confidence in [0,1]
    all on `device`, dtype `dtype`.
    """
    pts_flat = pts_bt.reshape(-1, 3)              # (N*Hp*Wp, 3)

    sem_flat  = sem_map.query(pts_flat)           # (N*Hp*Wp, C_sem) cpu
    ins_flat  = ins_map.query(pts_flat)           # (N*Hp*Wp, C_ins) cpu
    mask_flat = sem_map.query_conf(pts_flat)      # (N*Hp*Wp, 1)     cpu [0,1]

    C_s = sem_flat.shape[-1]
    C_i = ins_flat.shape[-1]

    mem_sem  = sem_flat.reshape(N, Hp, Wp, C_s).permute(0, 3, 1, 2).to(device=device, dtype=dtype)
    mem_ins  = ins_flat.reshape(N, Hp, Wp, C_i).permute(0, 3, 1, 2).to(device=device, dtype=dtype)
    mem_mask = mask_flat.reshape(N, 1, Hp, Wp).to(device=device, dtype=dtype)
    return mem_sem, mem_ins, mem_mask


# ---------------------------------------------------------------------------
# AMB3RStage2
# ---------------------------------------------------------------------------

class AMB3RStage2(nn.Module):
    """
    Stage-2 model = frozen Stage-1 + two memory-conditioned DPT heads.

    Parameters
    ----------
    stage1_model  : AMB3RStage1FullFT  (or compatible)
    sem_dim       : 512
    ins_dim       : 16
    sem_proj_dim  : projection dim inside DPT head for semantic conditioning (128)
    ins_proj_dim  : projection dim inside DPT head for instance conditioning (128)
    """

    def __init__(
        self,
        stage1_model: nn.Module,
        sem_dim: int = 512,
        ins_dim: int = 16,
        sem_proj_dim: int = 128,
        ins_proj_dim: int = 128,
    ):
        super().__init__()

        # ── Stage-1 (fully frozen) ─────────────────────────────────────────
        self.stage1 = stage1_model
        for p in self.stage1.parameters():
            p.requires_grad_(False)

        # ── Semantic memory DPT head ───────────────────────────────────────
        # cond channels at (H, W): mem_sem(sem_dim) + sem_s1(sem_dim) + mem_mask(1)
        self.semantic_memory_head = build_semantic_memory_head(
            sem_dim=sem_dim,
            dim_in=2 * 1024,
            semantic_proj_dim=sem_proj_dim,
        )

        # ── Instance memory DPT head ───────────────────────────────────────
        # cond channels at (H, W): mem_ins(ins_dim) + sem_s1(sem_dim) + ins_s1(ins_dim) + mem_mask(1)
        self.instance_memory_head = build_instance_memory_head(
            ins_dim=ins_dim,
            sem_dim=sem_dim,
            dim_in=2 * 1024,
            semantic_proj_dim=ins_proj_dim,
        )

        self.sem_dim = sem_dim
        self.ins_dim = ins_dim

    # ------------------------------------------------------------------
    # Stage-1 step-by-step forward (captures aggregated_tokens_list)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_stage1(self, frames: dict):
        """
        Run Stage-1 and return both the predictions dict AND the decoded
        intermediate features (which contain aggregated_tokens_list).

        Returns
        -------
        images              : (B, T, 3, H, W) in [0,1]
        predictions         : dict  (all feat maps at full (H, W) resolution)
        aggregated_tokens_list : list of tensors  (detached, no grad)
        patch_start_idx     : int
        """
        s1 = self.stage1

        # 1. DINOv2 patch embedding (frozen)
        images, patch_tokens = s1.front_end.encode_patch_tokens(frames)
        B, T = images.shape[:2]
        H, W = images.shape[-2:]

        # 2. LSeg features (frozen)
        lseg_feat = s1._extract_lseg(images)              # (B, T, 512, H, W)
        lseg_flat = lseg_feat.flatten(0, 1)               # (B*T, 512, H, W)

        # 3. CLIP-Patch cross-attention fusion (frozen)
        patch_tokens = s1.clip_patch_fusion(patch_tokens, lseg_flat, H, W)

        # 4. VGGT decode — call decode_patch_tokens to capture aggregated_tokens_list
        decoded_features = s1.front_end.decode_patch_tokens(patch_tokens, images)

        # 5. Heads decode — sem_feat / ins_feat output at full (H, W) resolution
        predictions = s1.front_end.decode_heads(
            images, decoded_features,
            semantic_feats=lseg_feat,
            has_backend=True,
        )
        predictions['_clip_feat_gt'] = lseg_feat

        if 'semantic_feat' in predictions:
            predictions['semantic_feat_expanded'] = predictions['semantic_feat']

        # 6. Detach token list so Stage-2 grads never flow back into Stage-1
        agg_tokens = [t.detach() for t in decoded_features['aggregated_tokens_list']]
        ps_idx     = decoded_features['patch_start_idx']

        return images, predictions, agg_tokens, ps_idx

    # ------------------------------------------------------------------
    # weight helpers
    # ------------------------------------------------------------------

    def load_stage1_weights(self, path: str, data_type: str = 'bf16', strict: bool = False):
        self.stage1.load_weights(path, data_type=data_type, strict=strict)

    def load_stage2_weights(self, path: str, strict: bool = True):
        state = torch.load(path, map_location='cpu')
        if 'model' in state:
            state = state['model']
        stage2_state = {
            k: v for k, v in state.items()
            if k.startswith('semantic_memory_head.') or
               k.startswith('instance_memory_head.')
        }
        missing, unexpected = self.load_state_dict(stage2_state, strict=False)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Stage-2 weight mismatch:\n  missing={missing[:10]}\n  "
                f"unexpected={unexpected[:10]}"
            )
        print(f"[Stage2] Loaded {len(stage2_state)} tensors from {path}")

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        frames: dict,
        sem_voxel_map=None,
        ins_voxel_map=None,
        pts_for_query: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Parameters
        ----------
        frames        : dict with 'images' (B, T, 3, H, W) in [0, 1]
        sem_voxel_map : VoxelFeatureMapV2 | None  — None on first chunk
        ins_voxel_map : VoxelFeatureMapV2 | None
        pts_for_query : (B, T, H, W, 3) | None
            Globally-aligned world points used to index the voxel memory.
            Pass GT pts3d during training; pass coordinate-aligned predicted
            pts during SLAM inference.
            Falls back to predictions['world_points'] if None (safe only for
            single-chunk inference where no cross-chunk consistency needed).

        Returns
        -------
        dict with all Stage-1 keys plus:
            semantic_feat        : (B, T, C_sem, H, W) — refined, pixel-aligned
            semantic_conf        : (B, T, 1,     H, W)
            instance_feat        : (B, T, C_ins, H, W) — refined, pixel-aligned
            instance_conf        : (B, T, 1,     H, W)
            sem_feat_s1          : raw Stage-1 semantic (B, T, C_sem, H, W)
            ins_feat_s1          : raw Stage-1 instance (B, T, C_ins, H, W)
            mem_mask             : (B, T, 1, H, W) soft confidence (upsampled)
            mem_sem_for_loss     : (B, T, C_sem, H, W) | None  (upsampled)
            mem_ins_for_loss     : (B, T, C_ins, H, W) | None  (upsampled)
        """
        # ── Stage-1 (no grad) ──────────────────────────────────────────────
        images, preds, agg_tokens, ps_idx = self._run_stage1(frames)

        sem_s1    = preds['semantic_feat']      # (B, T, C_sem, H, W)
        ins_s1    = preds['instance_feat']      # (B, T, C_ins, H, W)
        world_pts = preds['world_points']       # (B, T, H, W, 3) — local chunk frame

        B, T, C_sem, H, W = sem_s1.shape
        device = sem_s1.device
        dtype  = sem_s1.dtype
        N      = B * T

        # Patch resolution for voxel ops (200× fewer lookups than full res)
        Hp, Wp = H // PATCH_SIZE, W // PATCH_SIZE

        sem_s1_flat = sem_s1.flatten(0, 1)      # (B*T, C_sem, H, W)
        ins_s1_flat = ins_s1.flatten(0, 1)      # (B*T, C_ins, H, W)

        # ── Memory query @ patch resolution ────────────────────────────────
        # pts_for_query must be globally-aligned so all chunks index the same
        # voxel coordinate frame.
        query_pts = pts_for_query if pts_for_query is not None else world_pts

        if sem_voxel_map is not None and ins_voxel_map is not None:
            pts_patch = _resize_pts(query_pts, Hp, Wp)   # (B*T, Hp, Wp, 3)
            mem_sem_p, mem_ins_p, mem_mask_p = _query_voxel_maps(
                pts_patch.cpu(), sem_voxel_map, ins_voxel_map,
                device=device, dtype=dtype, N=N, Hp=Hp, Wp=Wp,
                ins_dim=self.ins_dim,
            )
            has_memory = True
        else:
            # First chunk — memory is empty; zeros at patch res
            mem_sem_p  = torch.zeros(N, self.sem_dim, Hp, Wp, device=device, dtype=dtype)
            mem_ins_p  = torch.zeros(N, self.ins_dim, Hp, Wp, device=device, dtype=dtype)
            mem_mask_p = torch.zeros(N, 1, Hp, Wp, device=device, dtype=dtype)
            has_memory = False

        # ── Upsample memory from patch res → image res ────────────────────
        # So we can cat directly with pixel-aligned sem_s1 / ins_s1.
        # mem_mask: nearest to preserve binary-ish character.
        mem_sem  = F.interpolate(mem_sem_p,  size=(H, W), mode='bilinear', align_corners=False)
        mem_ins  = F.interpolate(mem_ins_p,  size=(H, W), mode='bilinear', align_corners=False)
        mem_mask = F.interpolate(mem_mask_p, size=(H, W), mode='nearest')

        # ── Build full-resolution conditioning signals ────────────────────
        # PatchConditionedDPTHead.forward() will adaptive_avg_pool2d these
        # down to (Hp, Wp) internally before fusing with patch tokens.

        # Semantic cond: [mem_sem, sem_s1, mem_mask]  channels = C_sem+C_sem+1
        sem_cond_flat = torch.cat([mem_sem, sem_s1_flat, mem_mask], dim=1)
        sem_cond = sem_cond_flat.view(B, T, sem_cond_flat.shape[1], H, W)

        # Instance cond: [mem_ins, sem_s1, ins_s1, mem_mask]
        # channels = C_ins + C_sem + C_ins + 1
        ins_cond_flat = torch.cat([mem_ins, sem_s1_flat, ins_s1_flat, mem_mask], dim=1)
        ins_cond = ins_cond_flat.view(B, T, ins_cond_flat.shape[1], H, W)

        # ── Stage-2 DPT heads (trainable, with grad) ──────────────────────
        # Outputs at full pixel resolution (H, W) — same as Stage-1 heads.
        refined_sem, sem_conf = self.semantic_memory_head(
            agg_tokens, images,
            semantic_cond=sem_cond,
            patch_start_idx=ps_idx,
        )
        # refined_sem: (B, T, C_sem, H, W)

        refined_ins, ins_conf = self.instance_memory_head(
            agg_tokens, images,
            semantic_cond=ins_cond,
            patch_start_idx=ps_idx,
        )
        # refined_ins: (B, T, C_ins, H, W)

        # Reshape memory tensors back to (B, T, C, H, W) for loss / output
        mem_sem_bt  = mem_sem.view(B, T, self.sem_dim, H, W)
        mem_ins_bt  = mem_ins.view(B, T, self.ins_dim, H, W)
        mem_mask_bt = mem_mask.view(B, T, 1, H, W)

        # ── Return ────────────────────────────────────────────────────────
        return {
            **preds,
            # refined outputs (override Stage-1), pixel-aligned (H, W)
            'semantic_feat'          : refined_sem,
            'semantic_conf'          : sem_conf,
            'instance_feat'          : refined_ins,
            'instance_conf'          : ins_conf,
            'semantic_feat_expanded' : refined_sem,
            # raw Stage-1 for logging / ablation
            'sem_feat_s1'            : sem_s1,
            'ins_feat_s1'            : ins_s1,
            # memory metadata (upsampled to H, W)
            'mem_mask'               : mem_mask_bt,
            # queried memory for loss (None on first chunk)
            'mem_sem_for_loss'       : mem_sem_bt  if has_memory else None,
            'mem_ins_for_loss'       : mem_ins_bt  if has_memory else None,
        }

    # ------------------------------------------------------------------
    def prepare(self, data_type: str = 'bf16'):
        self.stage1.prepare(data_type)

    @property
    def trainable_params(self):
        return (list(self.semantic_memory_head.parameters()) +
                list(self.instance_memory_head.parameters()))
