"""
model_stage2v4.py — Neural network modules for Stage-2 V4.

Updated with:
    - mem_mode: 0=tokenizer(BPTT), 1=detached sem⊕ins, 2=full VGGT, 3=AE bottleneck
    - Separate fusion paths for semantic and instance (4 MemoryFusionModules)
    - MemoryTokenizer only used in mem_mode=0
    - DetachedVoxelStore for mem_mode 1-3

Data flow (per chunk of T frames):
    1. Stage-1 (frozen):  images → VGGT → depth, pose, world_points, atl
    2. Memory retrieval:  pts → voxel_store.query() → X_mem
    3. Separate fusion (trainable):
         Semantic path:
           X_fuse_sem_early = fusion_sem_early(atl[4],  X_mem)
           X_fuse_sem_late  = fusion_sem_late(atl[23], X_mem)
         Instance path:
           X_fuse_ins_early = fusion_ins_early(atl[4],  X_mem)
           X_fuse_ins_late  = fusion_ins_late(atl[23], X_mem)
    4. Task decoding (frozen params, grad ENABLED):
         sem_feat = sem_head(modified_atl_sem)
         ins_feat = ins_head(modified_atl_ins)
    5. Memory content (depends on mem_mode):
         mode 0: MemoryTokenizer(avg(X_fuse_sem, X_fuse_ins)) → M, W
         mode 1: detached sem_feat ⊕ ins_feat
         mode 2: detached X_vggt_late
         mode 3: frozen AE encoder(concat(X_early, X_late))
"""

import os
import sys
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
import torch.utils.checkpoint as ckpt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from amb3r.memory_stage2v4 import (
    PATCH_SIZE, C_VGGT, MEM_DIM, HIDDEN_DIM, DPT_LAYERS,
    LearnableNullToken, DifferentiableVoxelMap, CrossChunkFeatureBuffer,
    DetachedVoxelStore,
)


# ─────────────────────────────────────────────────────────────────────────────
# MemoryFusionModule
# ─────────────────────────────────────────────────────────────────────────────

class MemoryFusionModule(nn.Module):
    """
    Two-stage fusion: cross-attention (VGGT ← memory) then self-attention.

    Zero-init on cross_out_proj.weight for identity mapping at start.
    """

    def __init__(
        self,
        c_vggt: int    = C_VGGT,
        c_mem: int     = MEM_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int  = 4,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        # ── Cross-attention ───────────────────────────────────────────────
        self.W_q = nn.Linear(c_vggt, hidden_dim, bias=False)
        self.W_k = nn.Linear(c_mem, hidden_dim, bias=False)
        self.W_v = nn.Linear(c_mem, hidden_dim, bias=False)
        self.cross_attn     = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True)
        self.cross_out_proj = nn.Linear(hidden_dim, c_vggt, bias=False)
        nn.init.zeros_(self.cross_out_proj.weight)

        # ── Self-attention ────────────────────────────────────────────────
        self.self_norm     = nn.LayerNorm(c_vggt)
        self.self_proj_in  = nn.Linear(c_vggt, hidden_dim, bias=False)
        self.self_attn     = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True)
        self.self_out_proj = nn.Linear(hidden_dim, c_vggt, bias=False)
        nn.init.zeros_(self.self_out_proj.weight)  # identity at init

    def forward(self, X_vggt: torch.Tensor, X_mem: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X_vggt : [B*T, N, c_vggt]
            X_mem  : [B*T, N, c_mem]
        Returns:
            X_fuse : [B*T, N, c_vggt]
        """
        with torch.amp.autocast("cuda", enabled=False):
            vggt_f = X_vggt.float()
            mem_f  = X_mem.float()

            Q = self.W_q(vggt_f)
            K = self.W_k(mem_f)
            V = self.W_v(mem_f)
            cross_out, _ = self.cross_attn(Q, K, V)
            cross_delta  = self.cross_out_proj(cross_out)
            X_cross = vggt_f + cross_delta

            X_norm = self.self_norm(X_cross)
            q = k = v = self.self_proj_in(X_norm)
            self_out, _ = self.self_attn(q, k, v)
            self_delta  = self.self_out_proj(self_out)
            X_fuse = X_cross + self_delta

        return X_fuse.to(dtype=X_vggt.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# MemoryTokenizer (only for mem_mode=0)
# ─────────────────────────────────────────────────────────────────────────────

class MemoryTokenizer(nn.Module):
    """
    Compress fused VGGT features → compact memory token + confidence score.
    Only used in mem_mode=0 (trainable BPTT memory).
    """

    def __init__(self, c_vggt: int = C_VGGT, mem_dim: int = MEM_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c_vggt, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, mem_dim + 1),
        )
        self.mem_dim = mem_dim

    def forward(self, X_fuse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(X_fuse)
        M_readout = out[..., :self.mem_dim]
        W_conf    = torch.sigmoid(out[..., self.mem_dim:])
        return M_readout, W_conf


# ─────────────────────────────────────────────────────────────────────────────
# AMB3RStage2V4
# ─────────────────────────────────────────────────────────────────────────────

class AMB3RStage2V4(nn.Module):
    """
    Stage-2 V4: Decoupled Spatial-Memory Attention with multiple memory modes
    and separate fusion paths for semantic vs instance.

    Memory modes (--mem_mode):
        0: Trainable MemoryTokenizer → BPTT through DifferentiableVoxelMap
        1: Detached sem_feat ⊕ ins_feat → DetachedVoxelStore
        2: Detached X_vggt_late (full 2048-dim) → DetachedVoxelStore
        3: Frozen AE encoder(concat(X_early, X_late)) → DetachedVoxelStore

    Fusion: 4 separate MemoryFusionModules
        fusion_sem_early, fusion_sem_late   → modified_atl_sem → sem_head
        fusion_ins_early, fusion_ins_late   → modified_atl_ins → ins_head
    """

    def __init__(
        self,
        stage1_model: nn.Module,
        mem_mode: int      = 2,
        mem_dim: int       = MEM_DIM,
        vggt_dim: int      = C_VGGT,
        hidden_dim: int    = HIDDEN_DIM,
        num_heads: int     = 4,
        mem_voxel_size: float  = 0.05,
        feat_voxel_size: float = 0.01,
        ema_alpha: float   = 0.5,
        use_checkpoint: bool = False,
        # Stage 1.5 autoencoder (for mem_mode=3)
        ae_encoder: nn.Module = None,
        ae_bottleneck_dim: int = 256,
        finetune_head: bool = False,
    ):
        super().__init__()

        self.mem_mode = mem_mode
        self.finetune_head = finetune_head

        # ── Frozen Stage-1 ───────────────────────────────────────────────
        self.stage1 = stage1_model
        for p in self.stage1.parameters():
            p.requires_grad_(False)

        # ── Optionally unfreeze sem/ins heads ────────────────────────────
        if finetune_head:
            for p in self.stage1.front_end.model.semantic_head.parameters():
                p.requires_grad_(True)
            for p in self.stage1.front_end.model.instance_head.parameters():
                p.requires_grad_(True)

        # ── Determine effective mem_dim based on mode ────────────────────
        if mem_mode == 0:
            effective_mem_dim = mem_dim               # MemoryTokenizer output
        elif mem_mode == 1:
            effective_mem_dim = 512 + 16              # sem_dim + ins_dim (default)
        elif mem_mode == 2:
            effective_mem_dim = vggt_dim              # 2048
        elif mem_mode == 3:
            effective_mem_dim = ae_bottleneck_dim     # AE bottleneck dim
        else:
            raise ValueError(f'Unknown mem_mode: {mem_mode}')

        self.effective_mem_dim = effective_mem_dim

        # ── Trainable: null token + 4 fusion modules ─────────────────────
        self.null_token = LearnableNullToken(effective_mem_dim)

        # Separate fusion for semantic path
        self.fusion_sem_early = MemoryFusionModule(vggt_dim, effective_mem_dim, hidden_dim, num_heads)
        self.fusion_sem_late  = MemoryFusionModule(vggt_dim, effective_mem_dim, hidden_dim, num_heads)

        # Separate fusion for instance path
        self.fusion_ins_early = MemoryFusionModule(vggt_dim, effective_mem_dim, hidden_dim, num_heads)
        self.fusion_ins_late  = MemoryFusionModule(vggt_dim, effective_mem_dim, hidden_dim, num_heads)

        # ── MemoryTokenizer (only for mem_mode=0) ────────────────────────
        if mem_mode == 0:
            self.memory_tokenizer = MemoryTokenizer(vggt_dim, mem_dim)
        else:
            self.memory_tokenizer = None

        # ── Frozen AE encoder (only for mem_mode=3) ──────────────────────
        if mem_mode == 3:
            assert ae_encoder is not None, 'mem_mode=3 requires ae_encoder'
            self.ae_encoder = ae_encoder
            for p in self.ae_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.ae_encoder = None

        self.mem_dim        = mem_dim
        self.vggt_dim       = vggt_dim
        self.mem_voxel_size = mem_voxel_size
        self.feat_voxel_size = feat_voxel_size
        self.ema_alpha      = ema_alpha
        self.use_checkpoint = use_checkpoint

        self._dpt_indices: Optional[Tuple[int, int]] = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def train(self, mode: bool = True):
        """Override to keep frozen modules in eval mode (critical for BatchNorm)."""
        super().train(mode)
        # Frozen Stage-1 must use eval-mode BatchNorm (running stats, no dropout)
        self.stage1.eval()
        # If finetuning heads, put them back in train mode (BN uses batch stats)
        if self.finetune_head:
            self.stage1.front_end.model.semantic_head.train(mode)
            self.stage1.front_end.model.instance_head.train(mode)
        if self.ae_encoder is not None:
            self.ae_encoder.eval()
        return self

    @property
    def trainable_params(self) -> List[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def head_params(self) -> List[torch.nn.Parameter]:
        """Parameters of the (unfrozen) DPT heads — use with a lower LR."""
        if not self.finetune_head:
            return []
        params = []
        for p in self.stage1.front_end.model.semantic_head.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.stage1.front_end.model.instance_head.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    @property
    def non_head_params(self) -> List[torch.nn.Parameter]:
        """Trainable params excluding DPT heads (fusion modules, null token, etc.)."""
        head_set = set(id(p) for p in self.head_params)
        return [p for p in self.parameters()
                if p.requires_grad and id(p) not in head_set]

    def _get_dpt_indices(self) -> Tuple[int, int]:
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

    # ── factories ─────────────────────────────────────────────────────────────

    def make_voxel_store(self):
        """Create appropriate voxel store based on mem_mode."""
        if self.mem_mode == 0:
            return DifferentiableVoxelMap(
                voxel_size=self.mem_voxel_size,
                mem_dim=self.mem_dim,
                ema_alpha=self.ema_alpha,
            )
        else:
            return DetachedVoxelStore(
                voxel_size=self.mem_voxel_size,
                mem_dim=self.effective_mem_dim,
            )

    # Keep old name for backward compat
    def make_voxel_map(self):
        return self.make_voxel_store()

    def make_feat_buffer(
        self, sem_dim: int = 512, ins_dim: int = 16,
    ) -> CrossChunkFeatureBuffer:
        return CrossChunkFeatureBuffer(
            voxel_size=self.feat_voxel_size,
            sem_dim=sem_dim, ins_dim=ins_dim,
        )

    def load_stage1_weights(self, path: str, data_type: str = 'bf16',
                            strict: bool = False):
        self.stage1.load_weights(path, data_type=data_type, strict=strict)

    def prepare(self, data_type: str = 'bf16'):
        self.stage1.prepare(data_type)

    # ── fusion (checkpoint-friendly) ─────────────────────────────────────────

    def _fuse_sem_path(self, X_early, X_late, X_mem):
        """Semantic fusion path → (X_fuse_sem_early, X_fuse_sem_late)"""
        return (
            self.fusion_sem_early(X_early, X_mem),
            self.fusion_sem_late(X_late, X_mem),
        )

    def _fuse_ins_path(self, X_early, X_late, X_mem):
        """Instance fusion path → (X_fuse_ins_early, X_fuse_ins_late)"""
        return (
            self.fusion_ins_early(X_early, X_mem),
            self.fusion_ins_late(X_late, X_mem),
        )

    # ── compute memory content for storage ───────────────────────────────────

    def _compute_memory_content(
        self,
        X_vggt_early: torch.Tensor,
        X_vggt_late: torch.Tensor,
        X_fuse_sem_late: torch.Tensor,
        sem_feat: torch.Tensor,
        ins_feat: torch.Tensor,
        B: int, T: int, Hp: int, Wp: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute memory content + confidence based on mem_mode.

        Returns:
            M_content : [B*T*N, mem_dim] or None
            W_conf    : [B*T*N, 1] or None  (only for mode 0)
        """
        N_patch = Hp * Wp

        # Trainable MemoryTokenizer
        if self.mem_mode == 0:
            # Trainable tokenizer on semantic fusion output
            M_readout, W_conf = self.memory_tokenizer(X_fuse_sem_late)
            M_content = M_readout.reshape(B * T * N_patch, self.mem_dim)
            W_conf_flat = W_conf.reshape(B * T * N_patch, 1)
            return M_content, W_conf_flat

        # Detached sem_feat ⊕ ins_feat at patch resolution
        elif self.mem_mode == 1:
            # Detached sem ⊕ ins at patch resolution
            sem_flat = sem_feat.flatten(0, 1)  # [B*T, C_sem, H, W]
            ins_flat = ins_feat.flatten(0, 1)
            sem_p = F.adaptive_avg_pool2d(sem_flat, (Hp, Wp)).permute(0, 2, 3, 1).reshape(B * T * N_patch, -1)
            ins_p = F.adaptive_avg_pool2d(ins_flat, (Hp, Wp)).permute(0, 2, 3, 1).reshape(B * T * N_patch, -1)
            M_content = torch.cat([sem_p, ins_p], dim=-1).detach()
            return M_content, None

        # Use full late VGGT features (2048-dim) as memory
        elif self.mem_mode == 2:
            # Detached full VGGT late features
            M_content = X_vggt_late.reshape(B * T * N_patch, self.vggt_dim).detach()
            return M_content, None

        # AE bottleneck features from concatenated early+late VGGT (frozen encoder)
        elif self.mem_mode == 3:
            # Frozen AE encoder on concat(early, late)
            with torch.no_grad():
                X_concat = torch.cat([X_vggt_early, X_vggt_late], dim=-1)
                Z = self.ae_encoder(X_concat.float())
            M_content = Z.reshape(B * T * N_patch, -1).detach()
            return M_content, None

    # ── per-chunk forward ─────────────────────────────────────────────────────

    def forward_chunk(
        self,
        frames: dict,
        voxel_store,  # DifferentiableVoxelMap or DetachedVoxelStore
        pts_for_query: Optional[torch.Tensor] = None,
    ) -> Tuple[dict, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Process one chunk of T frames with memory conditioning.

        Returns:
            predictions  : dict
            M_content    : [B*T*N, mem_dim] or None — memory content for update
            W_conf       : [B*T*N, 1] or None — confidence (mode 0 only)
            pts_flat     : [B*T*N, 3]
        """
        s1 = self.stage1

        # ── Step 1: Stage-1 backbone (frozen) ────────────────────────────
        with torch.no_grad():
            images, patch_tokens = s1.front_end.encode_patch_tokens(frames)
            B, T = images.shape[:2]
            H, W = images.shape[-2:]
            Hp, Wp = H // PATCH_SIZE, W // PATCH_SIZE
            N_patch = Hp * Wp
            device = images.device

            lseg_feat = s1._extract_lseg(images)
            lseg_flat = lseg_feat.flatten(0, 1)
            patch_tokens = s1.clip_patch_fusion(patch_tokens, lseg_flat, H, W)

            decoded = s1.front_end.decode_patch_tokens(patch_tokens, images)
            res_geo = s1.front_end.decode_heads(images, decoded, has_backend=False)

        # ── Step 2: Extract X_vggt from both DPT layers ─────────────────
        atl = res_geo['aggregated_tokens_list']
        ps_idx = res_geo['patch_start_idx']
        first_dpt_idx, last_dpt_idx = self._get_dpt_indices()

        early_layer = atl[first_dpt_idx]
        last_layer = atl[last_dpt_idx]
        C = last_layer.shape[-1]

        X_vggt_early = early_layer[:, :, ps_idx:, :].reshape(B * T, N_patch, C)
        X_vggt_late  = last_layer[:, :, ps_idx:, :].reshape(B * T, N_patch, C)

        # ── Step 3: World-space coordinates at patch resolution ──────────
        query_pts = (pts_for_query if pts_for_query is not None
                     else res_geo['world_points'])

        pts_patch = F.interpolate(
            query_pts.flatten(0, 1).permute(0, 3, 1, 2).float(),
            size=(Hp, Wp), mode='bilinear', align_corners=False,
        ).permute(0, 2, 3, 1)
        pts_flat = pts_patch.reshape(B * T * N_patch, 3)

        # ── Step 4: Memory retrieval ─────────────────────────────────────
        X_mem_flat, mask_flat = voxel_store.query(pts_flat, self.null_token.token)
        X_mem = X_mem_flat.reshape(B * T, N_patch, self.effective_mem_dim)
        mem_mask = mask_flat.reshape(B, T, 1, Hp, Wp)
        has_memory = (mask_flat.sum() > 0).item()

        # ── Step 5: Separate fusion (TRAINABLE) ─────────────────────────
        if self.use_checkpoint:
            X_fuse_sem_early, X_fuse_sem_late = ckpt.checkpoint(
                self._fuse_sem_path, X_vggt_early, X_vggt_late, X_mem,
                use_reentrant=False)
            X_fuse_ins_early, X_fuse_ins_late = ckpt.checkpoint(
                self._fuse_ins_path, X_vggt_early, X_vggt_late, X_mem,
                use_reentrant=False)
        else:
            X_fuse_sem_early, X_fuse_sem_late = self._fuse_sem_path(
                X_vggt_early, X_vggt_late, X_mem)
            X_fuse_ins_early, X_fuse_ins_late = self._fuse_ins_path(
                X_vggt_early, X_vggt_late, X_mem)

        # ── Step 6: Inject into separate modified_atl and decode ─────────
        prefix_early = early_layer[:, :, :ps_idx, :]
        prefix_last  = last_layer[:, :, :ps_idx, :]

        # Semantic modified_atl
        modified_atl_sem = list(atl)
        modified_atl_sem[first_dpt_idx] = torch.cat([
            prefix_early,
            X_fuse_sem_early.reshape(B, T, N_patch, C),
        ], dim=2)
        modified_atl_sem[last_dpt_idx] = torch.cat([
            prefix_last,
            X_fuse_sem_late.reshape(B, T, N_patch, C),
        ], dim=2)

        # Instance modified_atl
        modified_atl_ins = list(atl)
        modified_atl_ins[first_dpt_idx] = torch.cat([
            prefix_early,
            X_fuse_ins_early.reshape(B, T, N_patch, C),
        ], dim=2)
        modified_atl_ins[last_dpt_idx] = torch.cat([
            prefix_last,
            X_fuse_ins_late.reshape(B, T, N_patch, C),
        ], dim=2)

        # Decode through frozen DPT heads
        sem_feat, sem_conf = s1.front_end.model.semantic_head(
            modified_atl_sem, images=images,
            semantic_cond=lseg_feat, patch_start_idx=ps_idx,
        )
        ins_feat, ins_conf = s1.front_end.model.instance_head(
            modified_atl_ins, images=images,
            semantic_cond=None, patch_start_idx=ps_idx,
        )

        # ── Step 7: Compute memory content for voxel update ──────────────
        M_content, W_conf = self._compute_memory_content(
            X_vggt_early, X_vggt_late,
            X_fuse_sem_late,
            sem_feat, ins_feat,
            B, T, Hp, Wp,
        )

        # ── Patch-level features for cross-view buffer ───────────────────
        sem_flat = sem_feat.flatten(0, 1)
        ins_flat = ins_feat.flatten(0, 1)
        sem_feat_patch = (F.adaptive_avg_pool2d(sem_flat, (Hp, Wp))
                          .permute(0, 2, 3, 1)
                          .reshape(B * T * N_patch, -1))
        ins_feat_patch = (F.adaptive_avg_pool2d(ins_flat, (Hp, Wp))
                          .permute(0, 2, 3, 1)
                          .reshape(B * T * N_patch, -1))

        # ── Assemble predictions ─────────────────────────────────────────
        predictions = {
            'semantic_feat':    sem_feat,
            'instance_feat':    ins_feat,
            'semantic_conf':    sem_conf,
            'instance_conf':   ins_conf,
            'sem_feat_patch':   sem_feat_patch,
            'ins_feat_patch':   ins_feat_patch,
            '_clip_feat_gt':    lseg_feat,
            'mem_mask':         mem_mask,
            'M_content':        M_content,
            'M_queried':        X_mem if has_memory else None,
            'depth':            res_geo['depth'],
            'depth_conf':       res_geo['depth_conf'],
            'pose_enc':         res_geo['pose_enc'],
            'world_points':     res_geo['world_points'],
            'extrinsic':        res_geo['extrinsic'],
            'intrinsic':        res_geo['intrinsic'],
            'images':           images,
        }
        # Backward compat aliases
        predictions['M_readout'] = M_content
        return predictions, M_content, W_conf, pts_flat

    # ── VoxelMap update (static) ──────────────────────────────────────────────

    @staticmethod
    def update_voxel_store(
        voxel_store,
        M_content: torch.Tensor,
        W_conf: Optional[torch.Tensor],
        pts_flat: torch.Tensor,
        mem_mode: int = 0,
    ):
        """
        Update the voxel store with new memory content.

        For mem_mode=0: confidence-weighted scatter + EMA (grad-preserving)
        For mem_mode 1-3: simple first-visit detached update
        """
        if mem_mode == 0:
            # BPTT mode — DifferentiableVoxelMap
            voxel_keys, voxel_idx = voxel_store.compute_voxel_assignments(pts_flat)
            num_voxels = len(voxel_keys)
            idx_dev = voxel_idx.to(M_content.device)

            weighted = M_content * W_conf
            sum_wf = scatter_sum(weighted, idx_dev, dim=0, dim_size=num_voxels)
            sum_w = scatter_sum(W_conf, idx_dev, dim=0, dim_size=num_voxels)
            voxel_new = sum_wf / (sum_w + 1e-6)

            voxel_store.update(voxel_keys, voxel_new)
        else:
            # Detached mode — DetachedVoxelStore
            voxel_store.update(pts_flat, M_content.detach())

    # Keep old name for backward compat
    @staticmethod
    def update_voxel_map(voxel_map, M_readout, W_conf, pts_flat):
        """Legacy interface — assumes mem_mode=0."""
        AMB3RStage2V4.update_voxel_store(voxel_map, M_readout, W_conf, pts_flat, mem_mode=0)
