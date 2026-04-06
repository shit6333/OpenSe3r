"""
AMB3RStage1: Stage-1 model for semantic/instance feature learning.

Design:
  - Uses VGGTwLoRA (DINOv2 + LoRA on frame/global blocks) as the backbone.
  - Inserts CLIPPatchFusion after DINOv2 patch embed, before VGGT attention blocks.
  - Runs a single forward pass (no backend / PointTransformerV3).
  - Produces semantic_feat, instance_feat alongside geometry predictions.
  - Returns a 1-element list [predictions] to be compatible with the existing
    MultitaskLoss and training loop interface.

Trainable components (Stage 1):
  - clip_patch_fusion   (new cross-attention module)
  - lora_*              (LoRA adapters injected into VGGT frame/global blocks)
  - semantic_head       (PatchConditionedDPTHead)
  - instance_head       (PatchConditionedDPTHead)
  - sem_expander        (FeatureExpander)
  - instance_token      (learnable token, kept for API compatibility)

Frozen:
  - DINOv2 patch_embed
  - VGGT frame/global block base weights
  - All LSeg/CLIP parameters
  - Geometry heads (camera_head, depth_head, point_head)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))


class FrontEndLoRA(nn.Module):
    """
    Drop-in replacement for FrontEnd that uses VGGTwLoRA as the backbone.

    Replicates FrontEnd.__init__ but instantiates VGGTwLoRA instead of VGGT,
    then loads the pretrained VGGT checkpoint with strict=False so that LoRA
    parameters (lora_A / lora_B) keep their initialisation values while all
    base model weights are restored from the checkpoint.

    All forward methods (encode_patch_tokens, decode_patch_tokens,
    decode_heads, decode_patch_tokens_and_heads) are inherited by delegating
    to the same self.model.aggregator interface used by FrontEnd.
    """

    def __init__(
        self,
        ckpt_path: str = './checkpoints/VGGT.pt',
        metric_scale: bool = True,
        clip_dim: int = 512,
        sem_dim: int = 512,
        # LoRA hyper-parameters
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_last_n: int = 8,
        lora_on_qkv: bool = True,
        lora_on_proj: bool = True,
        lora_on_mlp: bool = False,
        lora_on_frame_blocks: bool = True,
        lora_on_global_blocks: bool = True,
    ):
        super().__init__()
        self.metric_scale = metric_scale

        from amb3r.vggt_w_lora import VGGTwLoRA
        from amb3r.blocks import ScaleProjector

        # Build VGGTwLoRA (injects LoRA on construction)
        self.model = VGGTwLoRA(
            return_depth_feat=metric_scale,
            use_lora=True,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_last_n=lora_last_n,
            lora_on_qkv=lora_on_qkv,
            lora_on_proj=lora_on_proj,
            lora_on_mlp=lora_on_mlp,
            lora_on_frame_blocks=lora_on_frame_blocks,
            lora_on_global_blocks=lora_on_global_blocks,
        )

        # Load pretrained VGGT weights (strict=False: LoRA keys are new/missing)
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            msg = self.model.load_state_dict(ckpt, strict=False)
            lora_missing  = [k for k in msg.missing_keys  if 'lora_' in k]
            other_missing = [k for k in msg.missing_keys  if 'lora_' not in k]
            print(f"[FrontEndLoRA] VGGT loaded. "
                  f"LoRA missing (expected): {len(lora_missing)}  "
                  f"Other missing: {len(other_missing)}  "
                  f"Unexpected: {len(msg.unexpected_keys)}")
            if other_missing:
                print("  Non-LoRA missing keys:", other_missing[:10])
        else:
            print(f"[FrontEndLoRA] WARNING: checkpoint not found at {ckpt_path}. "
                  "Starting from random weights.")

        # Same auxiliary modules as FrontEnd
        self.metric_scale_projector = ScaleProjector(depth_feat_channels=128)
        self.sem_shortcut = nn.Sequential(
            nn.Conv2d(clip_dim, sem_dim, kernel_size=1),
            nn.GELU(),
        )

    # ------------------------------------------------------------------
    # Forward helpers (delegated to self.model — same interface as FrontEnd)
    # ------------------------------------------------------------------

    def encode_patch_tokens(self, frames):
        """images [0,1] + DINOv2 patch tokens."""
        images = frames['images']
        images = (images + 1.0) / 2.0
        assert images.min() >= 0.0 and images.max() <= 1.0
        return images, self.model.aggregator.encode_patch_tokens(images)

    def add_voxel_feat_to_patch_tokens(self, patch_tokens, voxel_feat):
        patch_tokens["x_norm_patchtokens"] += voxel_feat.flatten(0, 1).flatten(1, 2)
        return patch_tokens

    def decode_patch_tokens(self, patch_tokens, images,
                            voxel_feat=None, voxel_layer_list=None,
                            detach=False):
        return self.model.aggregator.decode_patch_tokens(
            patch_tokens, images,
            voxel_feat=voxel_feat,
            voxel_layer_list=voxel_layer_list,
            detach=detach,
            mem_eff=True,
        )

    def decode_heads(self, images, decoded_features,
                     semantic_feats=None, instance_feats=None,
                     has_backend=False):
        """Thin wrapper — identical logic to FrontEnd.decode_heads."""
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import (
            unproject_depth_map_to_point_map_torch,
            closed_form_inverse_se3,
        )

        aggregated_tokens_list = decoded_features['aggregated_tokens_list']
        ps_idx       = decoded_features['patch_start_idx']
        patch_tokens = decoded_features['patch_tokens']
        cls_token    = decoded_features['cls_token']
        reg_token    = decoded_features['reg_token']

        predictions = {}

        with torch.amp.autocast("cuda", enabled=False):
            if self.model.camera_head is not None:
                pose_enc_list = self.model.camera_head(aggregated_tokens_list)
                predictions["pose_enc"]      = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list

            if self.model.depth_head is not None:
                if self.metric_scale:
                    depth, depth_conf, depth_feat = self.model.depth_head(
                        aggregated_tokens_list, images=images,
                        patch_start_idx=ps_idx,
                    )
                    Bs, nimgs, H, W, _ = depth.shape
                    enc_tokens = torch.cat(
                        [patch_tokens, cls_token[:, None], reg_token], dim=1
                    )
                    median_z_log = self.metric_scale_projector(
                        depth_feat, enc_tokens
                    ).view(Bs, nimgs, 1)
                else:
                    depth, depth_conf = self.model.depth_head(
                        aggregated_tokens_list, images=images,
                        patch_start_idx=ps_idx,
                    )
                predictions["depth"]      = depth
                predictions["depth_conf"] = depth_conf

            if self.model.point_head is not None:
                pts3d, pts3d_conf = self.model.point_head(
                    aggregated_tokens_list, images=images,
                    patch_start_idx=ps_idx,
                )
                predictions["world_points"]      = pts3d
                predictions["world_points_conf"] = pts3d_conf

        predictions["images"] = images

        if self.metric_scale:
            median_pred_flat, _ = torch.median(
                predictions["depth"].view(Bs * nimgs, H * W), dim=1
            )
            median_pred_flat = median_pred_flat.view(Bs, nimgs, 1)
            predictions["median_metric_z"] = median_z_log.exp()
            metric_scale_ = predictions["median_metric_z"] / (
                median_pred_flat + 1e-8
            )
            metric_scale_median, _ = torch.median(metric_scale_, dim=1, keepdim=True)
            predictions["depth_metric"] = (
                depth * metric_scale_median.view(Bs, 1, 1, 1, 1)
            )

        Bs, nimgs, H, W, _ = depth.shape
        if not has_backend:
            predictions["enc"] = patch_tokens.view(
                Bs, nimgs, patch_tokens.shape[-2], patch_tokens.shape[-1]
            )
            predictions["dec"] = (
                aggregated_tokens_list[-1]
                .view(Bs, nimgs, aggregated_tokens_list[-1].shape[-2],
                      aggregated_tokens_list[-1].shape[-1])[..., ps_idx:, :]
            )

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )
        point_map = unproject_depth_map_to_point_map_torch(
            predictions["depth"].view(-1, H, W, 1),
            extrinsic.view(-1, 3, 4),
            intrinsic.view(-1, 3, 3),
        )
        predictions["pts3d_by_unprojection"] = point_map.view(Bs, nimgs, H, W, 3)
        predictions['extrinsic'] = extrinsic.view(Bs, nimgs, 3, 4)
        predictions['intrinsic'] = intrinsic.view(Bs, nimgs, 3, 3)
        predictions['pose']      = closed_form_inverse_se3(
            predictions['extrinsic'].view(-1, 3, 4)
        ).view(Bs, nimgs, 4, 4)
        predictions['model'] = 'vggt'

        if not has_backend:
            predictions['aggregated_tokens_list'] = aggregated_tokens_list
            predictions['cls_token']              = cls_token
            predictions['reg_token']              = reg_token
        predictions['patch_start_idx'] = ps_idx

        # Semantic / instance heads (always active in Stage 1)
        if has_backend:
            if self.model.semantic_head is not None:
                sem_feat, sem_conf = self.model.semantic_head(
                    aggregated_tokens_list,
                    images=images,
                    semantic_cond=semantic_feats,
                    patch_start_idx=ps_idx,
                )
                predictions["semantic_feat"] = sem_feat
                predictions["semantic_conf"] = sem_conf

            if self.model.instance_head is not None:
                ins_feat, ins_conf = self.model.instance_head(
                    aggregated_tokens_list,
                    images=images,
                    semantic_cond=None,
                    patch_start_idx=ps_idx,
                )
                predictions["instance_feat"] = ins_feat
                predictions["instance_conf"] = ins_conf

        return predictions

    def decode_patch_tokens_and_heads(
        self, images, patch_tokens,
        voxel_feat=None, voxel_layer_list=None,
        semantic_feats=None, instance_feats=None,
        has_backend=False, detach=False,
    ):
        decoded = self.decode_patch_tokens(
            patch_tokens, images,
            voxel_feat=voxel_feat,
            voxel_layer_list=voxel_layer_list,
            detach=detach,
        )
        return self.decode_heads(
            images, decoded,
            semantic_feats=semantic_feats,
            instance_feats=instance_feats,
            has_backend=has_backend,
        )


# ---------------------------------------------------------------------------

class AMB3RStage1(nn.Module):
    """
    Stage-1 training model: LoRA-adapted VGGT + CLIP-Patch fusion +
    semantic/instance DPT heads. No backend (PointTransformerV3).

    Forward returns [predictions] — a single-element list — so the caller
    can use the same loop interface as AMB3R (which returns [pred_0, pred_1]).
    """

    def __init__(
        self,
        metric_scale: bool = True,
        clip_dim: int = 512,
        sem_dim: int = 512,
        ins_dim: int = 16,
        # LoRA
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_last_n: int = 8,
        lora_on_qkv: bool = True,
        lora_on_proj: bool = True,
        lora_on_mlp: bool = False,
        lora_on_frame_blocks: bool = True,
        lora_on_global_blocks: bool = True,
        # Paths
        vggt_ckpt: str = './checkpoints/VGGT.pt',
        lseg_ckpt: str = '/mnt/HDD3/ricky/slam/Uni3R/checkpoints/pretrained_models/demo_e200.ckpt',
    ):
        super().__init__()

        from amb3r.dpt_head_patch_cond import PatchConditionedDPTHead
        from amb3r.blocks import FeatureExpander
        from amb3r.lseg import LSegFeatureExtractor
        from .clip_patch_fusion import CLIPPatchFusion

        # ---- Backbone (VGGTwLoRA) ----------------------------------------
        self.front_end = FrontEndLoRA(
            ckpt_path=vggt_ckpt,
            metric_scale=metric_scale,
            clip_dim=clip_dim,
            sem_dim=sem_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_last_n=lora_last_n,
            lora_on_qkv=lora_on_qkv,
            lora_on_proj=lora_on_proj,
            lora_on_mlp=lora_on_mlp,
            lora_on_frame_blocks=lora_on_frame_blocks,
            lora_on_global_blocks=lora_on_global_blocks,
        )

        # ---- CLIP-Patch cross-attention fusion ---------------------------
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

        # ---- Semantic expander (dim compatibility with CLIP for loss) -----
        self.sem_expander = FeatureExpander(in_dim=sem_dim, out_dim=clip_dim)

        # ---- Learnable instance token (kept for API compat.) -------------
        self.instance_token = nn.Parameter(torch.zeros(1, 1, 1, 1, ins_dim))
        nn.init.trunc_normal_(self.instance_token, std=0.02)

        # ---- LSeg / CLIP feature extractor (frozen) ----------------------
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
        """
        Extract LSeg features from images.

        Args:
            images: [B, T, 3, H, W] in [0, 1]

        Returns:
            lseg_feat: [B, T, 512, H, W]  (resized back to image resolution)
        """
        Bs, T = images.shape[:2]
        H_orig, W_orig = images.shape[-2:]
        imgs_flat = images.flatten(0, 1)       # [B*T, 3, H, W]

        lseg_size = 384
        scale  = lseg_size / max(H_orig, W_orig)
        H_lseg = int(round(H_orig * scale / 32)) * 32
        W_lseg = int(round(W_orig * scale / 32)) * 32

        imgs_lseg = F.interpolate(
            imgs_flat, size=(H_lseg, W_lseg),
            mode='bilinear', align_corners=False,
        )
        lseg_feat = self.lseg.extract_features(imgs_lseg)   # [B*T, 512, H_l/2, W_l/2]
        lseg_feat = F.interpolate(
            lseg_feat, size=(H_orig, W_orig),
            mode='bilinear', align_corners=False,
        )
        return lseg_feat.view(Bs, T, *lseg_feat.shape[1:])  # [B, T, 512, H, W]

    # ------------------------------------------------------------------
    def forward(self, frames: dict) -> list:
        """
        Stage-1 forward pass.

        Steps:
          1. DINOv2 patch encoding
          2. LSeg feature extraction (no_grad, frozen)
          3. CLIPPatchFusion: inject CLIP into patch tokens
          4. VGGT frame/global blocks (LoRA enabled)
          5. DPT heads: geometry + semantic + instance
          6. Expand semantic_feat for loss

        Returns:
            [predictions]  — single-element list compatible with the
                             existing MultitaskLoss / training loop.
        """
        # 1. Encode patch tokens via DINOv2
        images, patch_tokens = self.front_end.encode_patch_tokens(frames)
        Bs, T = images.shape[:2]
        H, W  = images.shape[-2:]

        # 2. Extract LSeg features (frozen)
        with torch.no_grad():
            lseg_feat = self._extract_lseg(images)     # [B, T, 512, H, W]

        # 3. CLIP-Patch fusion (trainable cross-attention)
        lseg_flat   = lseg_feat.flatten(0, 1)          # [B*T, 512, H, W]
        patch_tokens = self.clip_patch_fusion(patch_tokens, lseg_flat, H, W)

        # 4+5. Decode through VGGT blocks + DPT heads
        predictions = self.front_end.decode_patch_tokens_and_heads(
            images, patch_tokens,
            semantic_feats=lseg_feat,  # PatchConditionedDPTHead conditioning
            has_backend=True,          # activates semantic_head + instance_head
        )

        # 6. Semantic feature expansion + GT CLIP for loss
        predictions['_clip_feat_gt'] = lseg_feat
        if 'semantic_feat' in predictions:
            # sem_expander: [B*T, C_sem] -> [B*T, clip_dim]  (kept for compat)
            # In Stage 1 sem_dim == clip_dim == 512, so this is identity-like,
            # but we keep it to match the AMB3R interface.
            predictions['semantic_feat_expanded'] = predictions['semantic_feat']

        return [predictions]

    # ------------------------------------------------------------------
    def prepare(self, data_type: str = 'bf16'):
        """Cast aggregator to target dtype (mirrors AMB3R.prepare)."""
        dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16,
                     'fp32': torch.float32}
        self.front_end.model.aggregator.to(dtype_map[data_type])

    def load_weights(self, path: str, data_type: str = 'bf16', strict: bool = False):
        """Load a Stage-1 checkpoint."""
        state = torch.load(path, map_location='cpu')
        if 'model' in state:
            state = state['model']
        self.load_state_dict(state, strict=strict)
        self.prepare(data_type)
