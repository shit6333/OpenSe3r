import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))


# from vggt.models.vggt import VGGT
from .vggt_w_lora import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map_torch, closed_form_inverse_se3

from .blocks import ScaleProjector
from .dpt_head_patch_cond import PatchConditionedDPTHead

class FrontEnd(nn.Module):
    def __init__(self, ckpt_path='./checkpoints/VGGT.pt', metric_scale=True, clip_dim=512, sem_dim=64):
        super().__init__()
        self.metric_scale=metric_scale
        # self.model = VGGT(return_depth_feat=metric_scale)
        self.model = VGGT(
            return_depth_feat=metric_scale,
            use_lora=True,
            lora_r=4,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_last_n=4,
            lora_on_qkv=True,
            lora_on_proj=True,
            lora_on_mlp=False,
            lora_on_frame_blocks=True,
            lora_on_global_blocks=True,
        )

        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            # self.model.load_state_dict(checkpoint, strict=False)
            self.msg = self.model.load_state_dict(checkpoint, strict=False)
            print("missing keys:", self.msg.missing_keys[:50])
            print("unexpected keys:", self.msg.unexpected_keys[:50])
            print("num missing:", len(self.msg.missing_keys))
            print("num unexpected:", len(self.msg.unexpected_keys))
            # print("num unexpected:", len(msg.unexpected_keys))
        
        self.metric_scale_projector = ScaleProjector(depth_feat_channels=128)
        self.sem_shortcut = nn.Sequential(
            nn.Conv2d(clip_dim, sem_dim, kernel_size=1),
            nn.GELU(),
        )
    
    def encode_patch_tokens(self, frames):
        '''
        Encode patch tokens from frames.

        Params:
            - frames: dict containing 'images' (B, nimgs, 3, H, W) in [-1, 1] range
        
        Returns:
            - images: normalized images in [0, 1] range
            - patch_tokens: encoded patch tokens
        '''
        images = frames['images']  # B, nimgs, 3, H, W
        images = (images + 1.0) / 2.0  # Normalize to [0, 1] range if needed
        assert images.min() >= 0.0 and images.max() <= 1.0, "Images should be in the range [0, 1], got min {}, max {}".format(images.min(), images.max())
        return images, self.model.aggregator.encode_patch_tokens(images)
    

    def add_voxel_feat_to_patch_tokens(self, patch_tokens, voxel_feat):
        '''
        Add voxel features to patch tokens.

        Params:
            - patch_tokens: dict containing patch tokens
            - voxel_feat: voxel features to be added
        
        Returns:
            - patch_tokens: updated patch tokens with voxel features added
        '''

        patch_tokens["x_norm_patchtokens"] += voxel_feat.flatten(0, 1).flatten(1, 2)  # Add voxel features to patch tokens
        return patch_tokens

    
    def decode_patch_tokens(self, patch_tokens, images, voxel_feat=None, voxel_layer_list=None, detach=False):
        '''
        Decode patch tokens using VGGT decoder.
        
        Params:
            - patch_tokens: dict containing patch tokens
            - images: input images (B, nimgs, 3, H, W) in [0, 1] range
            - voxel_feat: encoder voxel features
            - voxel_layer_list: list of voxel features for each decoder layer
        
        Returns:
            - decoded_features: dict containing decoded features
        '''
        
        return self.model.aggregator.decode_patch_tokens(patch_tokens, images, voxel_feat=voxel_feat, 
                                                         voxel_layer_list=voxel_layer_list, detach=detach, mem_eff=True)
    

    def decode_heads(self, images, decoded_features, semantic_feats=None, instance_feats=None, has_backend=False):

        '''
        Decode feature using VGGT heads to get depth, camera pose, and 3D points.
        
        Params:
            - images: input images (B, nimgs, 3, H, W) in [0, 1] range
            - decoded_features: dict containing decoded features from the decoder
        
        Returns:
            - predictions: dict containing depth, camera pose, 3D points, camera poses, etc.
        '''
        
        aggregated_tokens_list = decoded_features['aggregated_tokens_list']
        ps_idx = decoded_features['patch_start_idx']
        patch_tokens = decoded_features['patch_tokens']
        cls_token = decoded_features['cls_token']
        reg_token = decoded_features['reg_token']        

        predictions = {}

        with torch.amp.autocast("cuda", enabled=False):
            # Camere pose prediction
            if self.model.camera_head is not None:
                pose_enc_list = self.model.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list  # all pose encodings

            # Depth prediction
            if self.model.depth_head is not None:
                if self.metric_scale:
                    depth, depth_conf, depth_feat = self.model.depth_head(
                        aggregated_tokens_list, images=images, patch_start_idx=ps_idx
                    )
                    Bs, nimgs, H, W, _ = depth.shape
                    enc_tokens = torch.cat([patch_tokens, cls_token[:, None], reg_token], dim=1)
                    median_z_log = self.metric_scale_projector(depth_feat, enc_tokens).view(Bs, nimgs, 1)
                else:
                    depth, depth_conf = self.model.depth_head(
                        aggregated_tokens_list, images=images, patch_start_idx=ps_idx
                    )
                
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
                    
            # Point prediction
            if self.model.point_head is not None:
                pts3d, pts3d_conf = self.model.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=ps_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf
            
            

        predictions["images"] = images

        if self.metric_scale:
            median_pred_values_flat, _ = torch.median(predictions["depth"].view(Bs * nimgs, H * W), dim=1)
            median_pred_values_flat = median_pred_values_flat.view(Bs, nimgs, 1)
            predictions["median_metric_z"] = median_z_log.exp()

            metric_scale = predictions["median_metric_z"] / (median_pred_values_flat + 1e-8) # Bs, nimgs, 1
            metric_scale_median, _ = torch.median(metric_scale, dim=1, keepdim=True)  # Bs, 1, 1
            predictions["depth_metric"] = depth * metric_scale_median.view(Bs, 1, 1, 1, 1)

        Bs, nimgs, H, W, one_ = depth.shape
        predictions["enc"] = patch_tokens.view(Bs, nimgs, patch_tokens.shape[-2], patch_tokens.shape[-1])
        predictions["dec"] = aggregated_tokens_list[-1].view(Bs, nimgs, aggregated_tokens_list[-1].shape[-2], aggregated_tokens_list[-1].shape[-1])[..., ps_idx:, :]

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        point_map_by_unprojection = unproject_depth_map_to_point_map_torch(predictions["depth"].view(-1, H, W, 1), 
                                                                extrinsic.view(-1, 3, 4),
                                                                intrinsic.view(-1, 3, 3))
        
        predictions["pts3d_by_unprojection"] = point_map_by_unprojection.view(Bs, nimgs, H, W, 3)

        predictions['extrinsic'] = extrinsic.view(Bs, nimgs, 3, 4)
        predictions['intrinsic'] = intrinsic.view(Bs, nimgs, 3, 3)
        predictions['pose'] = closed_form_inverse_se3(predictions['extrinsic'].view(-1, 3, 4)).view(Bs, nimgs, 4, 4)


        predictions['model'] = 'vggt'
        predictions['aggregated_tokens_list'] = aggregated_tokens_list
        predictions['patch_start_idx'] = ps_idx
        predictions['cls_token'] = cls_token
        predictions['reg_token'] = reg_token

        # Process semantic and instance features if available and if the backend is present
        if has_backend:
            if self.model.semantic_head is not None:
                sem_feat, sem_conf = self.model.semantic_head(
                    aggregated_tokens_list,
                    images=images,
                    semantic_cond=semantic_feats,   # PatchConditionedDPTHead 的參數名
                    patch_start_idx=ps_idx,
                )
                # sem_feat: (B*T, C_out, H, W) → reshape
                
                s, S = images.shape[:2]
                H_out, W_out = sem_feat.shape[-2], sem_feat.shape[-1]
                
                # sc = semantic_feats.reshape(Bs * S, *semantic_feats.shape[2:])  # (B*S, 512, H, W)
                # sc = F.interpolate(sc, size=(H_out, W_out), mode='bilinear', align_corners=False)
                # sc = self.sem_shortcut(sc)  # (B*S, C_sem, H_out, W_out)
                # sc = sc.view(Bs, S, -1, H_out, W_out)
                # predictions["semantic_feat"] = sem_feat + sc
                # predictions["semantic_conf"] = sem_conf
                
                predictions["semantic_feat"] = sem_feat # (B, S, C_out, H_out, W_out)
                predictions["semantic_conf"] = sem_conf # (B, S, 1, H_out, W_out)
                # print(f'Semantic Head Process...')

            if self.model.instance_head is not None:
                ins_feat, ins_conf = self.model.instance_head(
                    aggregated_tokens_list,
                    images=images,
                    semantic_cond=None,   # instance head 不需要 lseg cond
                    patch_start_idx=ps_idx,
                )
                predictions["instance_feat"] = ins_feat
                predictions["instance_conf"] = ins_conf
                # print(f'Instance Head Process...')
            
        return predictions


    def decode_patch_tokens_and_heads(self, images, patch_tokens, voxel_feat=None, voxel_layer_list=None, 
                                      semantic_feats=None, instance_feats=None, has_backend=False, detach=False):
        '''
        Decode patch tokens and heads to get final predictions.
        Params:
            - images: input images (B, nimgs, 3, H, W) in [0, 1]
            - patch_tokens: dict containing patch tokens
            - voxel_feat: encoder voxel features
            - voxel_layer_list: list of voxel features for each decoder layer
        Returns:
            - predictions: dict containing depth, camera pose, 3D points, camera poses,
        '''
        
        decoded_features = self.decode_patch_tokens(patch_tokens, images, voxel_feat=voxel_feat, voxel_layer_list=voxel_layer_list, detach=detach)
        predictions = self.decode_heads(images, decoded_features, semantic_feats=semantic_feats, instance_feats=instance_feats, has_backend=has_backend)

        return predictions
    
   