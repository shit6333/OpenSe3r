"""
AMB3RStage1FullFT: Stage-1 model for full fine-tuning (no LoRA).

Compared to AMB3RStage1 (model_stage1.py):
  - Uses FrontEnd (from amb3r.frontend_semantic) directly — no LoRA injection.
  - frame_blocks and global_blocks are fully trainable (all weights, not just rank-r adapters).
  - DINOv2 patch_embed stays FROZEN (same as LoRA version).

Trainable components (Stage 1, full-FT):
  clip_patch_fusion        cross-attention CLIP->patch injection (new)
  front_end.model.aggregator.frame_blocks   full fine-tune
  front_end.model.aggregator.global_blocks  full fine-tune
  semantic_head            PatchConditionedDPTHead
  instance_head            PatchConditionedDPTHead
  sem_expander             FeatureExpander
  instance_token           learnable token

Frozen:
  front_end.model.aggregator.patch_embed   DINOv2 weights
  All LSeg / CLIP parameters

---- How to switch in train_stage1.py ----------------------------------------
Replace:
    from amb3r.model_stage1 import AMB3RStage1
    model = AMB3RStage1(lora_r=..., lora_last_n=..., ...)

With:
    from amb3r.model_stage1_wo_lora import AMB3RStage1FullFT
    model = AMB3RStage1FullFT(...)

And change TRAINABLE_KEYWORDS to:
    TRAINABLE_KEYWORDS = [
        'clip_patch_fusion',
        'frame_blocks',
        'global_blocks',
        'semantic_head',
        'instance_head',
        'sem_expander',
        'instance_token',
    ]

Note: 'patch_embed' is intentionally absent — DINOv2 weights stay frozen.
-------------------------------------------------------------------------------
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))


class AMB3RStage1FullFT(nn.Module):
    """
    Stage-1 model with full fine-tuning of VGGT frame/global attention blocks.

    Architecture is identical to AMB3RStage1 except the backbone uses the
    standard FrontEnd (VGGT, no LoRA). The frame_blocks and global_blocks
    are left fully trainable by the training script's TRAINABLE_KEYWORDS.

    Forward returns [predictions] — single-element list, same interface as
    AMB3RStage1 and AMB3R.
    """

    def __init__(
        self,
        metric_scale: bool = True,
        clip_dim: int = 512,
        sem_dim: int = 512,
        ins_dim: int = 16,
        # Paths
        vggt_ckpt: str = './checkpoints/VGGT.pt',
        lseg_ckpt: str = './checkpoints/demo_e200.ckpt',
    ):
        super().__init__()

        from amb3r.frontend_semantic import FrontEnd
        from amb3r.dpt_head_patch_cond import PatchConditionedDPTHead
        from amb3r.blocks import FeatureExpander
        from amb3r.lseg import LSegFeatureExtractor
        from .clip_patch_fusion import CLIPPatchFusion

        # ---- Backbone: standard VGGT (no LoRA) ----------------------------
        # FrontEnd loads the pretrained VGGT checkpoint internally.
        self.front_end = FrontEnd(
            ckpt_path=vggt_ckpt,
            metric_scale=metric_scale,
            clip_dim=clip_dim,
            sem_dim=sem_dim,
        )

        # ---- CLIP-Patch cross-attention fusion ----------------------------
        self.clip_patch_fusion = CLIPPatchFusion(
            patch_dim=1024,
            clip_dim=clip_dim,
            num_heads=8,
            patch_size=14,
        )

        # ---- Semantic head -----------------------------------------------
        self.front_end.model.semantic_head = PatchConditionedDPTHead(
            output_channels=sem_dim,
            semantic_dim=clip_dim,
            semantic_proj_dim=128,
            dim_in=2 * 1024,
            feature_only=True,
            input_identity=True,
            patch_size=14,
        )

        # ---- Instance head -----------------------------------------------
        self.front_end.model.instance_head = PatchConditionedDPTHead(
            output_channels=ins_dim,
            semantic_dim=clip_dim,
            semantic_proj_dim=128,
            dim_in=2 * 1024,
            feature_only=True,
            input_identity=True,
            patch_size=14,
        )

        # ---- Semantic expander -------------------------------------------
        self.sem_expander = FeatureExpander(in_dim=sem_dim, out_dim=clip_dim)

        # ---- Learnable instance token -----------------------------------
        self.instance_token = nn.Parameter(torch.zeros(1, 1, 1, 1, ins_dim))
        nn.init.trunc_normal_(self.instance_token, std=0.02)

        # ---- LSeg (frozen) -----------------------------------------------
        self.lseg = LSegFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path=lseg_ckpt,
            half_res=True,
        )
        for p in self.lseg.parameters():
            p.requires_grad_(False)

        self.clip_dim = clip_dim
        self.ins_dim  = ins_dim

    # ------------------------------------------------------------------
    def _extract_lseg(self, images: torch.Tensor) -> torch.Tensor:
        """Extract LSeg features. Identical to AMB3RStage1._extract_lseg."""
        Bs, T = images.shape[:2]
        H_orig, W_orig = images.shape[-2:]
        imgs_flat = images.flatten(0, 1)

        lseg_size = 384
        scale  = lseg_size / max(H_orig, W_orig)
        H_lseg = int(round(H_orig * scale / 32)) * 32
        W_lseg = int(round(W_orig * scale / 32)) * 32

        imgs_lseg = F.interpolate(
            imgs_flat, size=(H_lseg, W_lseg),
            mode='bilinear', align_corners=False,
        )
        lseg_feat = self.lseg.extract_features(imgs_lseg)
        lseg_feat = F.interpolate(
            lseg_feat, size=(H_orig, W_orig),
            mode='bilinear', align_corners=False,
        )
        return lseg_feat.view(Bs, T, *lseg_feat.shape[1:])

    # ------------------------------------------------------------------
    def forward(self, frames: dict) -> list:
        """
        Identical forward logic to AMB3RStage1.forward().

        1. DINOv2 patch encoding (patch_embed — frozen)
        2. LSeg feature extraction (frozen)
        3. CLIPPatchFusion: inject CLIP into patch tokens (trainable)
        4. VGGT frame/global blocks (fully trainable)
        5. DPT heads: geometry + semantic + instance
        """
        images, patch_tokens = self.front_end.encode_patch_tokens(frames)
        Bs, T = images.shape[:2]
        H, W  = images.shape[-2:]

        with torch.no_grad():
            lseg_feat = self._extract_lseg(images)

        lseg_flat    = lseg_feat.flatten(0, 1)
        patch_tokens = self.clip_patch_fusion(patch_tokens, lseg_flat, H, W)

        predictions = self.front_end.decode_patch_tokens_and_heads(
            images, patch_tokens,
            semantic_feats=lseg_feat,
            has_backend=True,
        )

        predictions['_clip_feat_gt'] = lseg_feat
        if 'semantic_feat' in predictions:
            predictions['semantic_feat_expanded'] = predictions['semantic_feat']

        return [predictions]

    # ------------------------------------------------------------------
    def prepare(self, data_type: str = 'bf16'):
        dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16,
                     'fp32': torch.float32}
        self.front_end.model.aggregator.to(dtype_map[data_type])

    def load_weights(self, path: str, data_type: str = 'bf16', strict: bool = False):
        state = torch.load(path, map_location='cpu')
        if 'model' in state:
            state = state['model']
        self.load_state_dict(state, strict=strict)
        self.prepare(data_type)


    @torch.inference_mode()
    def run_amb3r_sem_vo(self, frames, cfg, keyframe_memory):
        """
        Stage-1 inference interface for semantic SLAM.
        No backend → single forward pass; no blending.
        """
        predictions = self.forward(frames)[0]
        return predictions

    @torch.inference_mode()
    def run_amb3r_sem_feat(self, frames, cfg=None):
        """
        Feature-only inference path for GT-depth semantic pipeline.
        No VO post-processing, no blending, no keyframe-memory logic.
        """
        predictions = self.forward(frames)[0]
        return predictions