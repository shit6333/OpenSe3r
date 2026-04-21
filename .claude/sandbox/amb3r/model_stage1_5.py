"""
model_stage1_5.py — Stage 1.5: VGGT Feature Autoencoder.

Compresses concat(atl[layer_early], atl[layer_late]) from 4096-dim down to a
compact bottleneck, then decodes back to 4096-dim.  The decoded features are
injected into the frozen DPT heads (semantic_head, instance_head) and
supervised against CLIP GT (semantic) and instance masks (instance contrastive).

After training, the *encoder* is used in Stage 2 (mem_mode=3) to produce
compact memory tokens that carry both semantic and instance information.

Architecture:
    concat(atl[4], atl[23])  [B*T, N, 4096]
        → Encoder: Linear(4096,1024) → GELU → LN → Linear(1024, bottleneck)
        → Decoder: Linear(bottleneck,1024) → GELU → LN → Linear(1024, 4096)
    split decoded → (2048, 2048) → inject into modified_atl → frozen heads
"""

import os
import sys
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

PATCH_SIZE = 14
C_VGGT = 2048          # per-layer dim (frame 1024 + global 1024)
C_CONCAT = C_VGGT * 2  # 4096 — concat of two DPT layers
DPT_LAYERS = [4, 11, 17, 23]


# ─────────────────────────────────────────────────────────────────────────────
# VGGTAutoencoder
# ─────────────────────────────────────────────────────────────────────────────

class VGGTAutoencoder(nn.Module):
    """
    Autoencoder for concatenated VGGT layer features.

    Input:  concat(atl[early], atl[late])  →  [*, 4096]
    Output: reconstructed [*, 4096]

    The encoder bottleneck is what Stage 2 stores as memory.
    """

    def __init__(
        self,
        input_dim: int = C_CONCAT,   # 4096
        bottleneck_dim: int = 256,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.GELU(),
            nn.LayerNorm(input_dim//2),
            nn.Linear(input_dim//2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim//2),
            nn.GELU(),
            nn.LayerNorm(input_dim//2),
            nn.Linear(input_dim//2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[*, input_dim] → [*, bottleneck_dim]"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[*, bottleneck_dim] → [*, input_dim]"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstructed, bottleneck)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ─────────────────────────────────────────────────────────────────────────────
# Stage1_5 — wraps frozen Stage-1 + trainable autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class Stage1_5(nn.Module):
    """
    Stage 1.5: frozen Stage-1 backbone + trainable VGGTAutoencoder.

    Forward flow:
        1. Frozen Stage-1: images → patch_tokens → decode → atl, lseg_feat
        2. Extract X_early = atl[first_dpt_idx], X_late = atl[last_dpt_idx]
        3. X_concat = cat(X_early, X_late)  [B*T, N, 4096]
        4. X_hat, Z = autoencoder(X_concat)
        5. Split X_hat → (X_hat_early, X_hat_late) each [B*T, N, 2048]
        6. Inject into modified_atl, run frozen sem_head + ins_head
        7. Return predictions with sem_feat, ins_feat for loss computation

    Only the autoencoder parameters are trained.
    """

    def __init__(
        self,
        stage1_model: nn.Module,
        bottleneck_dim: int = 256,
        ae_hidden_dim: int = 1024,
    ):
        super().__init__()

        # ── Frozen Stage-1 ───────────────────────────────────────────────
        self.stage1 = stage1_model
        for p in self.stage1.parameters():
            p.requires_grad_(False)

        # ── Trainable autoencoder ────────────────────────────────────────
        self.autoencoder = VGGTAutoencoder(
            input_dim=C_CONCAT,
            bottleneck_dim=bottleneck_dim,
            hidden_dim=ae_hidden_dim,
        )

        self.bottleneck_dim = bottleneck_dim

        # Lazily resolved DPT layer indices
        self._dpt_indices: Optional[Tuple[int, int]] = None

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def _get_dpt_indices(self) -> Tuple[int, int]:
        """Return (first_idx, last_idx) of DPT intermediate layers."""
        if self._dpt_indices is not None:
            return self._dpt_indices
        try:
            sem_head = self.stage1.front_end.model.semantic_head
            layers = sem_head.intermediate_layer_idx
            first_idx, last_idx = min(layers), max(layers)
        except AttributeError:
            first_idx, last_idx = DPT_LAYERS[0], DPT_LAYERS[-1]
        self._dpt_indices = (first_idx, last_idx)
        return self._dpt_indices

    def load_stage1_weights(self, path: str, data_type: str = 'bf16',
                            strict: bool = False):
        self.stage1.load_weights(path, data_type=data_type, strict=strict)

    def prepare(self, data_type: str = 'bf16'):
        self.stage1.prepare(data_type)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, frames: dict) -> dict:
        """
        Full forward pass through frozen Stage-1 + trainable autoencoder.

        Returns dict with:
            semantic_feat, instance_feat        — from decoded AE features
            semantic_feat_original, instance_feat_original  — from original atl (for reference)
            _clip_feat_gt                       — frozen LSeg features
            bottleneck                          — AE bottleneck [B*T, N, bottleneck_dim]
            + all geometry predictions (depth, pose, world_points, etc.)
        """
        s1 = self.stage1

        # ── Step 1: Stage-1 backbone (frozen) ────────────────────────────
        with torch.no_grad():
            images, patch_tokens = s1.front_end.encode_patch_tokens(frames)
            B, T = images.shape[:2]
            H, W = images.shape[-2:]
            Hp, Wp = H // PATCH_SIZE, W // PATCH_SIZE
            N_patch = Hp * Wp

            lseg_feat = s1._extract_lseg(images)
            lseg_flat = lseg_feat.flatten(0, 1)
            patch_tokens = s1.clip_patch_fusion(patch_tokens, lseg_flat, H, W)

            decoded = s1.front_end.decode_patch_tokens(patch_tokens, images)
            res_geo = s1.front_end.decode_heads(images, decoded, has_backend=False)

        # ── Step 2: Extract patch tokens from DPT layers ─────────────────
        atl = res_geo['aggregated_tokens_list']
        ps_idx = res_geo['patch_start_idx']
        first_dpt_idx, last_dpt_idx = self._get_dpt_indices()

        early_layer = atl[first_dpt_idx]   # [B, T, ps_idx+N, C]
        last_layer = atl[last_dpt_idx]
        C = last_layer.shape[-1]  # C_VGGT = 2048

        X_early = early_layer[:, :, ps_idx:, :].reshape(B * T, N_patch, C)
        X_late = last_layer[:, :, ps_idx:, :].reshape(B * T, N_patch, C)

        # ── Step 3: Autoencoder (TRAINABLE) ──────────────────────────────
        # Concat the two layers → 4096-dim
        X_concat = torch.cat([X_early, X_late], dim=-1)  # [B*T, N, 4096]

        with torch.amp.autocast("cuda", enabled=False):
            X_hat, Z = self.autoencoder(X_concat.float())  # [B*T, N, 4096], [B*T, N, bottleneck]

        # Split back to two 2048-dim halves
        X_hat_early = X_hat[:, :, :C].to(X_early.dtype)
        X_hat_late = X_hat[:, :, C:].to(X_late.dtype)

        # ── Step 4: Inject decoded features into modified_atl ────────────
        prefix_early = early_layer[:, :, :ps_idx, :]
        hat_early_4d = X_hat_early.reshape(B, T, N_patch, C)
        modified_early = torch.cat([prefix_early, hat_early_4d], dim=2)

        prefix_last = last_layer[:, :, :ps_idx, :]
        hat_late_4d = X_hat_late.reshape(B, T, N_patch, C)
        modified_last = torch.cat([prefix_last, hat_late_4d], dim=2)

        modified_atl = list(atl)
        modified_atl[first_dpt_idx] = modified_early
        modified_atl[last_dpt_idx] = modified_last

        # ── Step 5: Frozen DPT heads on decoded features ─────────────────
        sem_feat, sem_conf = s1.front_end.model.semantic_head(
            modified_atl, images=images,
            semantic_cond=lseg_feat, patch_start_idx=ps_idx,
        )
        ins_feat, ins_conf = s1.front_end.model.instance_head(
            modified_atl, images=images,
            semantic_cond=None, patch_start_idx=ps_idx,
        )

        # ── Step 6: Also get original (unmodified) features for reference ──
        with torch.no_grad():
            sem_feat_orig, _ = s1.front_end.model.semantic_head(
                atl, images=images,
                semantic_cond=lseg_feat, patch_start_idx=ps_idx,
            )
            ins_feat_orig, _ = s1.front_end.model.instance_head(
                atl, images=images,
                semantic_cond=None, patch_start_idx=ps_idx,
            )

        # ── Assemble predictions ─────────────────────────────────────────
        # sem_expander for semantic loss compatibility
        sem_feat_expanded = s1.sem_expander(sem_feat)

        predictions = {
            # From decoded AE
            'semantic_feat': sem_feat,                # [B,T,C_sem,Hp,Wp]
            'semantic_conf': sem_conf,
            'semantic_feat_expanded': sem_feat_expanded,
            'instance_feat': ins_feat,                # [B,T,C_ins,Hp,Wp]
            # Original (frozen baseline)
            'semantic_feat_original': sem_feat_orig,
            'instance_feat_original': ins_feat_orig,
            # Autoencoder internals
            'bottleneck': Z,                          # [B*T, N, bottleneck_dim]
            # GT and metadata
            '_clip_feat_gt': lseg_feat,
            # Geometry (from frozen Stage-1)
            'depth': res_geo['depth'],
            'depth_conf': res_geo['depth_conf'],
            'pose_enc': res_geo['pose_enc'],
            'pose_enc_list': res_geo.get('pose_enc_list', [res_geo['pose_enc']]),
            'world_points': res_geo['world_points'],
            'extrinsic': res_geo['extrinsic'],
            'intrinsic': res_geo['intrinsic'],
            'images': images,
        }
        return predictions
