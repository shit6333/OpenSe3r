"""
CLIPPatchFusion: Cross-attention module that injects LSeg/CLIP dense features
into VGGT DINOv2 patch tokens before the frame/global attention blocks.

Architecture:
    Q  = DINOv2 patch tokens  [N, P, 1024]
    K  = projected CLIP feat  [N, P, 1024]
    V  = projected CLIP feat  [N, P, 1024]
    out = x + zero_proj(cross_attn(norm_q, norm_kv, norm_kv))

The zero-init output projection (ControlNet-style) ensures the module
behaves as identity at the start of training, preserving the pre-trained
VGGT geometry capability while allowing gradual semantic injection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPPatchFusion(nn.Module):
    """
    Fuses LSeg/CLIP dense features into VGGT DINOv2 patch tokens via cross-attention.

    Inputs (at forward):
        patch_tokens : dict returned by aggregator.encode_patch_tokens()
                       Must contain key "x_norm_patchtokens"  [N, P, patch_dim]
        clip_feat    : LSeg/CLIP dense features  [N, clip_dim, H, W]
        H, W         : image spatial dimensions (used to derive patch grid H/patch_size)

    Output:
        patch_tokens dict with "x_norm_patchtokens" replaced by the fused version.
        All other keys (x_norm_clstoken, x_norm_regtokens, ...) are passed through.
    """

    def __init__(
        self,
        patch_dim: int = 1024,
        clip_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        patch_size: int = 14,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.clip_dim = clip_dim
        self.patch_size = patch_size

        # Project CLIP features from clip_dim -> patch_dim for K/V
        self.clip_proj = nn.Linear(clip_dim, patch_dim)

        # Pre-norm for Q and K/V (pre-norm style is more stable)
        self.norm_q  = nn.LayerNorm(patch_dim)
        self.norm_kv = nn.LayerNorm(patch_dim)

        # Cross-attention: Q from patch tokens, K/V from projected CLIP
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=patch_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Zero-init output projection (identity at init -> no disruption to VGGT)
        self.zero_proj = nn.Linear(patch_dim, patch_dim)
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        patch_tokens: dict,
        clip_feat: torch.Tensor,
        H: int,
        W: int,
    ) -> dict:
        """
        Args:
            patch_tokens : dict with key "x_norm_patchtokens" [N, P, patch_dim]
            clip_feat    : [N, clip_dim, H, W]  (float32, LSeg output resized to img H×W)
            H, W         : image height / width (to compute patch grid size)

        Returns:
            Updated patch_tokens dict. "x_norm_patchtokens" is replaced with the
            cross-attention fused version; all other dict keys are unchanged.
        """
        x = patch_tokens["x_norm_patchtokens"]          # [N, P, patch_dim]
        N, P, D = x.shape
        assert D == self.patch_dim, f"patch_dim mismatch: got {D}, expected {self.patch_dim}"

        # ---- Pool CLIP to patch resolution --------------------------------
        H_p = H // self.patch_size
        W_p = W // self.patch_size
        # [N, clip_dim, H, W] -> [N, clip_dim, H_p, W_p]
        clip_pooled = F.adaptive_avg_pool2d(clip_feat.float(), (H_p, W_p))
        # [N, H_p*W_p, clip_dim]
        clip_flat = clip_pooled.flatten(2).transpose(1, 2)
        assert clip_flat.shape[1] == P, (
            f"patch count mismatch: clip gives {clip_flat.shape[1]} patches, "
            f"but patch_tokens has {P}. Check patch_size={self.patch_size}."
        )

        # ---- Project CLIP to patch_dim ------------------------------------
        # [N, P, patch_dim]
        clip_kv = self.clip_proj(clip_flat)

        # ---- Cross-attention (pre-norm) -----------------------------------
        q  = self.norm_q(x)
        kv = self.norm_kv(clip_kv)
        attn_out, _ = self.cross_attn(q, kv, kv)        # [N, P, patch_dim]

        # ---- Zero-init residual ------------------------------------------
        delta = self.zero_proj(attn_out)                 # near-zero at init

        # Cast back to the original dtype of x (may be bfloat16 in mixed precision)
        fused = x + delta.to(dtype=x.dtype)

        # ---- Return updated dict (shallow copy to avoid mutating original) -
        out = dict(patch_tokens)
        out["x_norm_patchtokens"] = fused
        return out
