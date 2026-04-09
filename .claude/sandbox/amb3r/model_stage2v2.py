"""
AMB3RStage2V2
=============
Stage-2 via PTv3 backend with memory conditioning, injected into frozen Stage-1.

Key design vs. AMB3RStage2 (v1):
  - Trainable component : PTv3 BackEnd (same arch as original AMB3R backend)
                          with expanded conditioning dims for memory fusion.
  - Memory conditioning : cat[mem_sem(512), sem_s1(512)] → 1024-dim semantic
                          cat[mem_ins(16),  ins_s1(16), mem_mask(1)] → 33-dim instance
  - Output              : Stage-1 is *re-decoded* with zero-conv voxel injection
                          → re-predicts depth, pose, pts, semantic, instance.
  - Geometry is supervised in Stage-2 → PTv3 also improves geometry.

Data flow
---------
Input: K frames + VoxelMemory M_{t-1}

 Stage-1 first pass  (frozen, no_grad)
 ───────────────────
  encode_patch_tokens + clip_patch_fusion
  → decode_patch_tokens  (VGGT frame/global blocks, run ONCE)
  → decode_heads (has_backend=False)  → enc, dec, world_pts, depth, pose
  → semantic_head / instance_head     → sem_s1, ins_s1

 Memory query  (no_grad)
 ────────────
  pts_patch = resize(world_pts or gt_pts, H//14)
  mem_sem_p, mem_ins_p, mem_mask_p = query_voxel_maps(pts_patch)

 Build conditioning
 ──────────────────
  sem_cond = cat[mem_sem, sem_s1]  (1024-dim, at patch res)
  ins_cond = cat[mem_ins, ins_s1, mem_mask]  (33-dim, at patch res)

 Stage-2 PTv3 Backend  (trainable, WITH grad)
 ─────────────────────
  feat = cat[enc, dec].detach()  →  aligner  →  feat_geo (1024-dim)
  resize pts & feat_geo to H//7 ("upsampled patch" level, same as original)
  sem_cond & ins_cond upsampled to H//7 and flattened → fed to backend.forward()
  PTv3 voxelizes and refines  →  voxel_feat_aligned
  downsample → (B*T, 1024, Hp, Wp)
  zero_conv  → voxel_feat_aligned_vis  (B, T, Hp, Wp, 1024)
  voxel_layer_list  for per-VGGT-layer conditioning

 Stage-1 second decode  (frozen weights, grad via voxel injection)
 ──────────────────────
  patch_tokens["x_norm_patchtokens"] += voxel_feat_aligned_vis   (new tensor, not in-place)
  decode_patch_tokens_and_heads(images, patched_tokens,
                                voxel_feat=voxel_feat_aligned,
                                voxel_layer_list=voxel_layer_list,
                                semantic_feats=lseg_feat,
                                has_backend=True)
  → res_s2: depth, pose, pts, semantic_feat, instance_feat

 Update VoxelMemory  (stop-grad, after each chunk)
 ──────────────────
  update(pts_patch, refined_sem.detach(), refined_ins.detach(), conf.detach())

Trainable parameters: backend.* only  (~same count as original BackEnd)
Frozen            : everything in stage1 (VGGT, LSeg, CLIP, DPT heads)
"""

import os
import sys
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from amb3r.backend_semantic import BackEnd

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
def _query_voxel_maps(pts_bt, sem_map, ins_map, device, dtype,
                      N, Hp, Wp, sem_dim, ins_dim):
    """
    Query sem and ins voxel maps at patch resolution.
    pts_bt : (N, Hp, Wp, 3)  float32 cpu
    Returns mem_sem (N, sem_dim, Hp, Wp), mem_ins (N, ins_dim, Hp, Wp),
            mem_mask (N, 1, Hp, Wp) – soft confidence in [0,1].
    All on `device` / `dtype`.
    """
    pts_flat = pts_bt.reshape(-1, 3)              # (N*Hp*Wp, 3)

    sem_flat  = sem_map.query(pts_flat)           # (N*Hp*Wp, sem_dim) cpu
    ins_flat  = ins_map.query(pts_flat)           # (N*Hp*Wp, ins_dim) cpu
    mask_flat = sem_map.query_conf(pts_flat)      # (N*Hp*Wp, 1)       cpu [0,1]

    mem_sem  = sem_flat.reshape(N, Hp, Wp, sem_dim).permute(0, 3, 1, 2).to(device=device, dtype=dtype)
    mem_ins  = ins_flat.reshape(N, Hp, Wp, ins_dim).permute(0, 3, 1, 2).to(device=device, dtype=dtype)
    mem_mask = mask_flat.reshape(N, 1, Hp, Wp).to(device=device, dtype=dtype)
    return mem_sem, mem_ins, mem_mask


def _resize_bt(x: torch.Tensor, target_size) -> torch.Tensor:
    """
    Bilinear resize for (B, T, H, W, C) tensors.
    Returns (B, T, target_H, target_W, C).
    """
    B, T, H, W, C = x.shape
    y = F.interpolate(
        x.permute(0, 1, 4, 2, 3).flatten(0, 1).float(),   # (B*T, C, H, W)
        size=target_size, mode='bilinear', align_corners=False
    )                                                        # (B*T, C, th, tw)
    return y.permute(0, 2, 3, 1).view(B, T, target_size[0], target_size[1], C)


# ---------------------------------------------------------------------------
# AMB3RStage2V2
# ---------------------------------------------------------------------------

class AMB3RStage2V2(nn.Module):
    """
    Stage-2 model = frozen Stage-1 + trainable PTv3 BackEnd with memory conditioning.

    Parameters
    ----------
    stage1_model      : AMB3RStage1FullFT (or compatible)
    sem_dim           : 512  – Stage-1 semantic feature dim
    ins_dim           : 16   – Stage-1 instance feature dim
    voxel_resolutions : list[float]  – voxel sizes passed to backend (default [0.01])
    interp_v2         : bool – coordinate alignment mode (match Stage-1 ckpt)
    """

    def __init__(
        self,
        stage1_model: nn.Module,
        sem_dim: int = 512,
        ins_dim: int = 16,
        voxel_resolutions: Optional[List[float]] = None,
        interp_v2: bool = False,
    ):
        super().__init__()

        # ── Stage-1 (fully frozen) ─────────────────────────────────────────
        self.stage1 = stage1_model
        for p in self.stage1.parameters():
            p.requires_grad_(False)

        # ── Stage-2 PTv3 BackEnd (trainable) ──────────────────────────────
        # sem_cond = cat[mem_sem(sem_dim), sem_s1(sem_dim)]  → sem_dim*2 = 1024
        # ins_cond = cat[mem_ins(ins_dim), ins_s1(ins_dim), mem_mask(1)] → ins_dim*2+1 = 33
        self.backend = BackEnd(
            in_dim=2048 + 1024,           # same as original AMB3R (enc+dec → aligner)
            out_dim=1024,
            k_neighbors=16,
            interp_v2=interp_v2,
            sem_dim=sem_dim * 2,          # 1024: cat[mem_sem, sem_s1]
            ins_dim=ins_dim * 2 + 1,      # 33:   cat[mem_ins, ins_s1, mem_mask]
        )

        self.sem_dim = sem_dim
        self.ins_dim = ins_dim
        self.voxel_resolutions = voxel_resolutions or [0.01]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_stage1(self, frames):
        """
        Run frozen Stage-1 encoding:
          DINOv2 patch embed  →  LSeg features  →  CLIP-patch fusion.
        Must be called inside torch.no_grad() (or wraps itself in no_grad).

        Returns
        -------
        images      : (B, T, 3, H, W) in [0, 1]
        patch_tokens: dict  (x_norm_patchtokens, …)  — CLIP-fused, requires_grad=False
        lseg_feat   : (B, T, 512, H, W)              — requires_grad=False
        """
        s1 = self.stage1
        images, patch_tokens = s1.front_end.encode_patch_tokens(frames)
        B, T = images.shape[:2]
        H, W = images.shape[-2:]

        with torch.no_grad():
            lseg_feat = s1._extract_lseg(images)          # (B, T, 512, H, W)
            lseg_flat = lseg_feat.flatten(0, 1)            # (B*T, 512, H, W)
            patch_tokens = s1.clip_patch_fusion(patch_tokens, lseg_flat, H, W)

        return images, patch_tokens, lseg_feat

    def _get_voxel_feat(self, res_geo, sem_cond_sp, ins_cond_sp, B, T, H, W):
        """
        Run the Stage-2 PTv3 backend and produce voxel injection signals.

        Parameters
        ----------
        res_geo       : Stage-1 first-pass predictions (enc, dec, world_points)
        sem_cond_sp   : (B*T, Hp, Wp, sem_dim*2) – cat[mem_sem, sem_s1] spatial
        ins_cond_sp   : (B*T, Hp, Wp, ins_dim*2+1) – cat[mem_ins, ins_s1, mem_mask]
        B, T, H, W    : batch / time / spatial dims of the original input

        Returns
        -------
        voxel_feat_aligned     : (B*T, 1024, Hp, Wp)  – raw, for metric scale + layer cond
        voxel_feat_aligned_vis : (B, T, Hp, Wp, 1024) – after zero_conv, for patch token add
        voxel_layer_list       : list of dicts for per-VGGT-layer injection
        """
        Hp, Wp = H // PATCH_SIZE, W // PATCH_SIZE
        Hs, Ws = H // 7, W // 7        # "upsampled-patch" resolution, same as original backend

        # ── Geo features ──────────────────────────────────────────────────
        # enc: (B, T, Hp*Wp, C_enc)  dec: (B, T, Hp*Wp, C_dec)
        feat = torch.cat([res_geo['enc'], res_geo['dec']], dim=-1).detach()  # (B, T, N, C)
        pts  = res_geo['world_points'].detach()                               # (B, T, H, W, 3)

        # Reshape to spatial and apply aligner (trainable)
        feat_sp = feat.view(B, T, Hp, Wp, feat.shape[-1])  # (B, T, Hp, Wp, C)
        feat_sp = self.backend.aligner(feat_sp)             # (B, T, Hp, Wp, 1024)

        # Upsample to (Hs, Ws)
        pts_up  = _resize_bt(pts,     (Hs, Ws))   # (B, T, Hs, Ws, 3)
        feat_up = _resize_bt(feat_sp, (Hs, Ws))   # (B, T, Hs, Ws, 1024)

        # ── Conditioning (sem + ins) ───────────────────────────────────────
        # sem_cond_sp: (B*T, Hp, Wp, 1024) → upsample → (B*T, 1024, Hs, Ws) → flat
        N_bt = B * T
        sem_up = F.interpolate(
            sem_cond_sp.permute(0, 3, 1, 2).float(),   # (B*T, 1024, Hp, Wp)
            size=(Hs, Ws), mode='bilinear', align_corners=False,
        ).to(feat_up.dtype)   # (B*T, 1024, Hs, Ws)
        sem_flat = sem_up.permute(0, 2, 3, 1).reshape(-1, sem_up.shape[1])  # (B*T*Hs*Ws, 1024)

        ins_up = F.interpolate(
            ins_cond_sp.permute(0, 3, 1, 2).float(),   # (B*T, 33, Hp, Wp)
            size=(Hs, Ws), mode='bilinear', align_corners=False,
        ).to(feat_up.dtype)   # (B*T, 33, Hs, Ws)
        ins_flat = ins_up.permute(0, 2, 3, 1).reshape(-1, ins_up.shape[1])  # (B*T*Hs*Ws, 33)

        # ── PTv3 backend forward (trainable) ──────────────────────────────
        with torch.amp.autocast('cuda', enabled=True):
            voxel_out = self.backend.forward(
                pts_up,    # (B, T, Hs, Ws, 3)
                feat_up,   # (B, T, Hs, Ws, 1024)
                voxel_sizes=self.voxel_resolutions,
                semantic_feats=sem_flat,    # (B*T*Hs*Ws, 1024)
                instance_feats=ins_flat,    # (B*T*Hs*Ws, 33)
            )

        # voxel_out[-1]: (B*T*Hs*Ws, 1024) — interpolated back to original pts
        voxel_fine = voxel_out[-1].reshape(B, T, Hs, Ws, -1)  # (B, T, Hs, Ws, 1024)

        # Downsample from (Hs, Ws) → (Hp, Wp)
        voxel_feat_aligned = self.backend.downsample(
            voxel_fine.permute(0, 1, 4, 2, 3).flatten(0, 1)   # (B*T, 1024, Hs, Ws)
        )  # (B*T, 1024, Hp, Wp)

        # Zero-conv for patch token injection  (trainable gate)
        voxel_feat_aligned_vis = (
            self.backend.zero_conv(voxel_feat_aligned) * self.backend.gate_scale
        )  # (B*T, 1024, Hp, Wp)

        # Reshape to (B, T, Hp, Wp, 1024) for add_voxel_feat_to_patch_tokens
        voxel_feat_aligned_vis = (
            voxel_feat_aligned_vis.view(B, T, -1, Hp, Wp).permute(0, 1, 3, 4, 2)
        )  # (B, T, Hp, Wp, 1024)

        # Per-VGGT-layer conditioning (48 zero-conv layers, one per decoder block)
        voxel_layer_list = [
            {'layer': layer, 'H': Hp, 'W': Wp, 'gate_scale': self.backend.gate_scales[i]}
            for i, layer in enumerate(self.backend.zero_conv_layers)
        ]

        return voxel_feat_aligned, voxel_feat_aligned_vis, voxel_layer_list

    # ------------------------------------------------------------------
    # weight helpers
    # ------------------------------------------------------------------

    def load_stage1_weights(self, path: str, data_type: str = 'bf16',
                            strict: bool = False):
        self.stage1.load_weights(path, data_type=data_type, strict=strict)

    def load_stage2_weights(self, path: str, strict: bool = True):
        state = torch.load(path, map_location='cpu')
        if 'model' in state:
            state = state['model']
        s2_state = {k: v for k, v in state.items() if k.startswith('backend.')}
        missing, unexpected = self.load_state_dict(s2_state, strict=False)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Stage-2V2 weight mismatch:\n"
                f"  missing={missing[:10]}\n  unexpected={unexpected[:10]}"
            )
        print(f"[Stage2V2] Loaded {len(s2_state)} tensors from {path}")

    def prepare(self, data_type: str = 'bf16'):
        dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16,
                     'fp32': torch.float32}
        self.stage1.prepare(data_type)
        # Keep backend in fp32 for stability (PTv3 uses fp32 internally)

    @property
    def trainable_params(self):
        return list(self.backend.parameters())

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
        frames         : dict with 'images' (B, T, 3, H, W)  — values in [-1, 1]
        sem_voxel_map  : VoxelFeatureMapV2 | None  — None on first chunk
        ins_voxel_map  : VoxelFeatureMapV2 | None
        pts_for_query  : (B, T, H, W, 3) | None
            Globally-aligned world points used to index the voxel memory.
            Pass GT pts3d during training; pass coordinate-aligned predicted
            pts at inference.  Falls back to Stage-1 world_points if None.

        Returns
        -------
        dict with all Stage-1 keys (depth, pose, world_points, semantic_feat,
        instance_feat, …) plus memory metadata:
            sem_feat_s1       : (B, T, sem_dim, H, W)  — raw Stage-1 semantic
            ins_feat_s1       : (B, T, ins_dim, H, W)  — raw Stage-1 instance
            mem_mask          : (B, T, 1, Hp, Wp)
            mem_sem_for_loss  : (B, T, sem_dim, Hp, Wp) | None  (None on chunk 0)
            mem_ins_for_loss  : (B, T, ins_dim, Hp, Wp) | None
        """
        s1 = self.stage1

        # ── Step 1: Stage-1 first pass (fully no_grad) ─────────────────
        with torch.no_grad():
            images, patch_tokens, lseg_feat = self._encode_stage1(frames)
            B, T = images.shape[:2]
            H, W = images.shape[-2:]
            Hp, Wp = H // PATCH_SIZE, W // PATCH_SIZE
            N = B * T
            device, dtype = images.device, images.dtype

            # VGGT decode — single expensive pass
            decoded = s1.front_end.decode_patch_tokens(patch_tokens, images)

            # Geometry predictions + save enc/dec for backend conditioning
            res_geo = s1.front_end.decode_heads(images, decoded, has_backend=False)
            # res_geo keys: enc, dec, world_points, depth, pose, aggregated_tokens_list, …

            # Stage-1 semantic / instance features for conditioning
            sem_feat_s1, _ = s1.front_end.model.semantic_head(
                res_geo['aggregated_tokens_list'],
                images=images,
                semantic_cond=lseg_feat,
                patch_start_idx=res_geo['patch_start_idx'],
            )  # (B, T, sem_dim, H, W)

            ins_feat_s1, _ = s1.front_end.model.instance_head(
                res_geo['aggregated_tokens_list'],
                images=images,
                semantic_cond=None,
                patch_start_idx=res_geo['patch_start_idx'],
            )  # (B, T, ins_dim, H, W)

        # ── Step 2: Memory query (no_grad) ───────────────────────────────
        query_pts = pts_for_query if pts_for_query is not None else res_geo['world_points']

        if sem_voxel_map is not None and ins_voxel_map is not None:
            pts_patch = _resize_pts(query_pts, Hp, Wp)    # (B*T, Hp, Wp, 3)
            mem_sem_p, mem_ins_p, mem_mask_p = _query_voxel_maps(
                pts_patch.cpu(), sem_voxel_map, ins_voxel_map,
                device=device, dtype=dtype, N=N, Hp=Hp, Wp=Wp,
                sem_dim=self.sem_dim, ins_dim=self.ins_dim,
            )
            has_memory = True
        else:
            mem_sem_p  = torch.zeros(N, self.sem_dim, Hp, Wp, device=device, dtype=dtype)
            mem_ins_p  = torch.zeros(N, self.ins_dim, Hp, Wp, device=device, dtype=dtype)
            mem_mask_p = torch.zeros(N, 1, Hp, Wp,    device=device, dtype=dtype)
            has_memory = False

        # ── Step 3: Build conditioning tensors ────────────────────────────
        # Stage-1 semantic/instance at patch resolution  (no_grad values)
        sem_s1_p = F.adaptive_avg_pool2d(
            sem_feat_s1.flatten(0, 1).float(), (Hp, Wp)
        ).to(dtype)  # (B*T, sem_dim, Hp, Wp)

        ins_s1_p = F.adaptive_avg_pool2d(
            ins_feat_s1.flatten(0, 1).float(), (Hp, Wp)
        ).to(dtype)  # (B*T, ins_dim, Hp, Wp)

        # cat[mem_sem, sem_s1]: (B*T, sem_dim*2, Hp, Wp) → spatial (B*T, Hp, Wp, sem_dim*2)
        sem_cond_sp = (
            torch.cat([mem_sem_p, sem_s1_p], dim=1).permute(0, 2, 3, 1)
        )  # (B*T, Hp, Wp, 1024)

        # cat[mem_ins, ins_s1, mem_mask]: (B*T, ins_dim*2+1, Hp, Wp) → spatial
        ins_cond_sp = (
            torch.cat([mem_ins_p, ins_s1_p, mem_mask_p], dim=1).permute(0, 2, 3, 1)
        )  # (B*T, Hp, Wp, 33)

        # ── Step 4: Stage-2 PTv3 backend (trainable, WITH grad) ───────────
        voxel_feat_aligned, voxel_feat_aligned_vis, voxel_layer_list = \
            self._get_voxel_feat(res_geo, sem_cond_sp, ins_cond_sp, B, T, H, W)
        # voxel_feat_aligned     : (B*T, 1024, Hp, Wp)   requires_grad=True
        # voxel_feat_aligned_vis : (B, T, Hp, Wp, 1024)  requires_grad=True

        # ── Step 5: Stage-1 second decode (frozen weights, grad via injection) ──
        # Create a new patch token dict so we don't modify the original in-place.
        # The addition below creates a new tensor in the autograd graph:
        #   patch_tokens["x_norm_patchtokens"].requires_grad = False  (from no_grad encode)
        #   vf_flat.requires_grad = True  (from Stage-2 backend)
        #   → result has requires_grad = True  → gradient flows to Stage-2 backend
        vf_flat = voxel_feat_aligned_vis.flatten(0, 1).flatten(1, 2)  # (B*T, Hp*Wp, 1024)
        patch_tokens_cond = dict(patch_tokens)
        patch_tokens_cond["x_norm_patchtokens"] = (
            patch_tokens["x_norm_patchtokens"] + vf_flat
        )

        # Re-decode Stage-1 with voxel conditioning
        res_s2 = s1.front_end.decode_patch_tokens_and_heads(
            images, patch_tokens_cond,
            voxel_feat=voxel_feat_aligned,
            voxel_layer_list=voxel_layer_list,
            semantic_feats=lseg_feat,
            has_backend=True,
        )

        # ── Enrich return dict ──────────────────────────────────────────────
        res_s2['_clip_feat_gt']       = lseg_feat
        res_s2['semantic_feat_expanded'] = res_s2.get('semantic_feat')
        res_s2['sem_feat_s1']         = sem_feat_s1     # (B, T, sem_dim, H, W)
        res_s2['ins_feat_s1']         = ins_feat_s1     # (B, T, ins_dim, H, W)
        res_s2['mem_mask']            = mem_mask_p.view(B, T, 1, Hp, Wp)
        res_s2['mem_sem_for_loss']    = mem_sem_p.view(B, T, self.sem_dim, Hp, Wp) if has_memory else None
        res_s2['mem_ins_for_loss']    = mem_ins_p.view(B, T, self.ins_dim, Hp, Wp) if has_memory else None

        return res_s2
