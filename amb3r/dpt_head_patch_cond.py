import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from vggt.heads.dpt_head import DPTHead as VGGTDPTHead
from vggt.heads.dpt_head import custom_interpolate

class PatchConditionedDPTHead(VGGTDPTHead):
    def __init__(
        self,
        semantic_dim: int = 512,  # LSeg 或其他 Semantic prior 的輸入通道數 (請依你的模型修改)
        semantic_proj_dim: int = 256, # 降維後的通道數，減少運算量
        input_identity: bool = False,
        output_channels: int = None,
        *args,
        **kwargs
    ):
        # 初始化父類別 (包含 feature_only, input_identity 等設定)
        super().__init__(*args, **kwargs)
        if input_identity:
            self.input_merger = nn.Sequential(
                nn.Conv2d(3, self.scratch.output_conv1.in_channels, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
            )
        self.null_sem_token = nn.Parameter(torch.zeros(1, semantic_proj_dim, 1, 1))
        # ==========================================
        # 1. Semantic Condition 下採樣與特徵編碼器
        # ==========================================
        # 將高解析度或任意尺寸的 Semantic 特徵，映射到我們指定的投影維度
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_dim, semantic_proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(semantic_proj_dim),
            nn.GELU(),
            nn.Conv2d(semantic_proj_dim, semantic_proj_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(semantic_proj_dim),
            nn.GELU()
        )

        # ==========================================
        # 2. Token 層級融合模組 (Patch Fusion)
        # ==========================================
        # DPT 會抽取多層 (預設4層) 的 token。我們為每一層建立一個 1x1 卷積，
        # 負責將 [VGGT_dim + Semantic_dim] 降維回原本的 VGGT_dim
        num_layers = len(self.intermediate_layer_idx)
        dim_in = kwargs.get('dim_in', 768) # VGGT Token 的維度，通常是 768
        
        self.fusion_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim_in + semantic_proj_dim, dim_in, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim_in),
                nn.GELU()
            ) for _ in range(num_layers)
        ])

        features_dim = self.scratch.output_conv1.in_channels # default is 256

        # Output head
        self.output_channels = output_channels
        if output_channels is not None and output_channels != features_dim:
            self.output_proj = nn.Sequential(
                nn.Conv2d(features_dim, output_channels, kernel_size=1),
                nn.GELU(),
            )
        else:
            self.output_proj = None

        # Confidence Head
        self.conf_head = nn.Sequential(
            nn.Conv2d(features_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Softplus()   # -> (0, +inf)
        )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        semantic_cond: torch.Tensor, # <--- 新增的 LSeg 語義特徵 [B, S, C_sem, H_sem, W_sem]
        patch_start_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        if semantic_cond is None:
            # --- cond=None -> learnable token (broadcast to patch grid) ---
            sem_patch_feat = self.null_sem_token.to(images.device, images.dtype).expand(B*S, -1, patch_h, patch_w)
        else:
            if semantic_cond.dim() == 5:
                # [B, S, C, H, W] -> [B*S, C, H, W]
                C_sem = semantic_cond.shape[2]
                sem_flat = semantic_cond.reshape(B*S, C_sem, semantic_cond.shape[3], semantic_cond.shape[4])
            elif semantic_cond.dim() == 4:
                # [B*S, C, H, W]
                C_sem = semantic_cond.shape[1]
                sem_flat = semantic_cond
            else:
                raise ValueError(f"semantic_cond must be 4D/5D or None, got {semantic_cond.shape}")

            # 對齊到 patch grid + 投影
            sem_patch = F.adaptive_avg_pool2d(sem_flat, (patch_h, patch_w))
            sem_patch_feat = self.semantic_proj(sem_patch)

        out = []
        dpt_idx = 0
        
        # --- 2. 在每一層特徵抽取時，進行 Token Fusion ---
        for layer_idx in self.intermediate_layer_idx:
            # 取出 VGGT 該層的 token 並變換為 2D 空間特徵
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            x = x.view(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((B * S, x.shape[-1], patch_h, patch_w)) # [B*S, 768, patch_h, patch_w]
            
            # 【關鍵融合步驟】Concat VGGT Token 與 Semantic Patch Token
            fused_x = torch.cat([x, sem_patch_feat], dim=1) # [B*S, 768 + 256, patch_h, patch_w]
            
            # 將維度投影回 768，這樣就可以無縫接軌原本的 DPT 架構
            x = self.fusion_projs[dpt_idx](fused_x)
            
            # 繼續原本 DPT 的空間卷積與降維
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            
            out.append(x)
            dpt_idx += 1

        # --- 3. DPT 解碼端整合 (Scratch) ---
        out = self.scratch_forward(out) # [B*S, 256, h, w]

        # --- 4. 內插回對應的解析度 ---  
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio),
             int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        # 處理 input_identity (維持你原本設定的 True)
        if hasattr(self, 'input_merger'):
            out = out + self.input_merger(images.view(B * S, *images.shape[2:]))
        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        # --- 5. 輸出特徵與信心度 (Confidence) ---
        # 即使 feature_only=True，我們也額外計算並回傳 conf
        conf = self.conf_head(out) + 1e-6 # 產生 [B*S, 1, H_out, W_out] 的信心度圖
        if self.output_proj is not None:
            out = self.output_proj(out)

        # 將 Batch 與 Sequence 維度切分回來
        out = out.view(B, S, *out.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])

        # 回傳你想要的特徵 (準備給 gs_feat_proj) 與 信心度 (準備算 Loss)
        return out, conf