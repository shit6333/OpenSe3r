"""
AMB3RStage2V4: Decoupled Spatial-Memory Attention (Confidence-Weighted Edition)
=================================================================================

Key design vs. Stage-2 V2 (PTv3 backend):
  - No PTv3 / PointTransformerV3 backend at all.
  - Trainable: MemoryFusionModule (cross-attention), MemoryTokenizer (compact MLP).
  - Implicit spatial memory: DifferentiableVoxelMap — stores [mem_dim] tensors
    that KEEP their grad_fn, enabling Chunked Unrolled BPTT across chunks.
  - Memory aggregation: confidence-weighted intra-frame (scatter_sum) + EMA inter-frame.
  - Geometry predictions: taken straight from frozen Stage-1 (no re-prediction).

Data flow (per chunk of T frames):
┌──────────────────────────────────────────────────────────────────────┐
│  1. Stage-1 backbone (frozen, no_grad)                               │
│     images → DINOv2 + LoRA + VGGT frame/global blocks               │
│     → aggregated_tokens_list, depth, pose, world_points              │
│                                                                      │
│  2. Memory retrieval (query DifferentiableVoxelMap)                  │
│     pts_patch → voxel_map.query() → X_mem [B*T, N_patch, mem_dim]   │
│     (LearnableNullToken for unseen voxels)                           │
│                                                                      │
│  3. Memory fusion (TRAINABLE)                                        │
│     X_fuse = X_vggt + γ * OutProj(Attn(Q_vggt, K_mem, V_mem))       │
│                                                                      │
│  4. Task decoding (frozen params, grad ENABLED)                      │
│     modified_atl[-1]: replace patch tokens with X_fuse               │
│     sem_head(modified_atl) → sem_feat  [grad flows through frozen op]│
│     ins_head(modified_atl) → ins_feat                                │
│                                                                      │
│  5. Memory tokenizer (TRAINABLE)                                     │
│     M_readout, W_conf = MLP(X_fuse)                                  │
│     scatter_sum(M*W) / scatter_sum(W) → voxel_new [V, mem_dim]      │
│     voxel_map.update(EMA)  ← grad_fn preserved for BPTT             │
└──────────────────────────────────────────────────────────────────────┘

Training: Chunked Unrolled BPTT
  - Init voxel_map at start of sequence (e.g. 16 frames).
  - Process in chunks (e.g. 4 frames each); accumulate chunk losses.
  - NO .detach() on memory tensors between chunks within a sequence.
  - Single loss.backward() + optimizer.step() after all chunks.
  - Gradient checkpointing optional (wrap fusion+tokenizer).

Trainable parameters:
  null_token                     LearnableNullToken
  memory_fusion.*                MemoryFusionModule
  memory_tokenizer.*             MemoryTokenizer

Frozen (requires_grad=False):
  stage1.*                       VGGT backbone, LSeg, CLIP-patch fusion, DPT heads
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

PATCH_SIZE = 14          # DINOv2 / VGGT patch size
C_VGGT    = 2048         # VGGT aggregated token dim = 2 × embed_dim (frame + global)
MEM_DIM   = 128          # Implicit memory token dim
HIDDEN_DIM = 256         # MemoryFusionModule hidden dim
# DPT intermediate layer indices (from vggt/heads/dpt_head.py default, 0-indexed)
DPT_LAYERS = [4, 11, 17, 23]


# ─────────────────────────────────────────────────────────────────────────────
# A. LearnableNullToken
# ─────────────────────────────────────────────────────────────────────────────

class LearnableNullToken(nn.Module):
    """
    Learnable fallback token for unvisited voxels (cold-start / unseen areas).

    At init → values ~ N(0, 0.02²), small so it does not disrupt frozen features
    before training starts.
    """

    def __init__(self, mem_dim: int = MEM_DIM):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, mem_dim) * 0.02)

    def forward(self) -> torch.Tensor:   # [1, mem_dim]
        return self.token


# ─────────────────────────────────────────────────────────────────────────────
# B. MemoryFusionModule
# ─────────────────────────────────────────────────────────────────────────────

class MemoryFusionModule(nn.Module):
    """
    Cross-attention fusion: current VGGT features + retrieved memory tokens.

    Forward equation:
        Q = W_q(X_vggt)
        K = W_k(X_mem),  V = W_v(X_mem)
        X_fuse = X_vggt + γ · OutProj(MultiheadAttn(Q, K, V))

    γ (gamma) is nn.Parameter initialized strictly to 0.0 (ControlNet-style).
    At the start of training the module acts as identity → VGGT features are
    completely unchanged, matching the pre-trained Stage-1 performance.

    Args:
        c_vggt    : VGGT token dim (2048)
        c_mem     : Memory token dim (128)
        hidden_dim: Cross-attention hidden dim (256)
        num_heads : Attention heads (4 or 8; must divide hidden_dim)
    """

    def __init__(
        self,
        c_vggt: int = C_VGGT,
        c_mem: int = MEM_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = 4,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        # Query projection from VGGT space → hidden
        self.W_q = nn.Linear(c_vggt, hidden_dim, bias=False)
        # Key / Value projections from memory space → hidden
        self.W_k = nn.Linear(c_mem, hidden_dim, bias=False)
        self.W_v = nn.Linear(c_mem, hidden_dim, bias=False)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        # Project back from hidden → VGGT dim
        self.out_proj = nn.Linear(hidden_dim, c_vggt, bias=False)

        # Zero-init residual gate — crucial for stable training initialisation
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, X_vggt: torch.Tensor, X_mem: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X_vggt : [B*T, N_patch, c_vggt]  — frozen backbone features (no grad)
            X_mem  : [B*T, N_patch, c_mem]   — memory features (may have grad_fn)
        Returns:
            X_fuse : [B*T, N_patch, c_vggt]  — grad flows through γ, W_*, attn, out_proj
        """
        # Run in float32 for numerical stability, regardless of outer autocast
        with torch.amp.autocast("cuda", enabled=False):
            vggt_f = X_vggt.float()
            mem_f  = X_mem.float()

            Q = self.W_q(vggt_f)         # [B*T, N, hidden]
            K = self.W_k(mem_f)           # [B*T, N, hidden]
            V = self.W_v(mem_f)           # [B*T, N, hidden]

            attn_out, _ = self.attn(Q, K, V)        # [B*T, N, hidden]
            delta = self.out_proj(attn_out)          # [B*T, N, c_vggt]

        # Cast back to input dtype (bf16 during mixed-precision training)
        X_fuse = X_vggt + self.gamma * delta.to(dtype=X_vggt.dtype)
        return X_fuse


# ─────────────────────────────────────────────────────────────────────────────
# C. MemoryTokenizer
# ─────────────────────────────────────────────────────────────────────────────

class MemoryTokenizer(nn.Module):
    """
    Compresses fused features → compact memory token + confidence score.

    Architecture:
        Linear(c_vggt, 512) → GELU → LayerNorm(512) → Linear(512, mem_dim + 1)

    Output split:
        M_readout = out[..., :mem_dim]            — 128-dim implicit feature
        W_conf    = sigmoid(out[..., mem_dim:])   — confidence ∈ (0, 1)

    Args:
        c_vggt  : 2048 — input feature dimension
        mem_dim : 128  — output memory token dimension
    """

    def __init__(self, c_vggt: int = C_VGGT, mem_dim: int = MEM_DIM):
        super().__init__()
        self.mem_dim = mem_dim
        self.mlp = nn.Sequential(
            nn.Linear(c_vggt, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, mem_dim + 1),
        )

    def forward(
        self, X_fuse: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X_fuse   : [B*T, N_patch, c_vggt]
        Returns:
            M_readout: [B*T, N_patch, mem_dim]
            W_conf   : [B*T, N_patch, 1]
        """
        with torch.amp.autocast("cuda", enabled=False):
            out = self.mlp(X_fuse.float())              # [B*T, N, mem_dim+1]

        M_readout = out[..., :self.mem_dim].to(X_fuse.dtype)       # [B*T, N, mem_dim]
        W_conf    = torch.sigmoid(out[..., self.mem_dim:]).to(X_fuse.dtype)  # [B*T, N, 1]
        return M_readout, W_conf


# ─────────────────────────────────────────────────────────────────────────────
# DifferentiableVoxelMap
# ─────────────────────────────────────────────────────────────────────────────

class DifferentiableVoxelMap:
    """
    Implicit spatial memory that preserves grad_fn for Unrolled BPTT.

    Stores: (ix, iy, iz) → [mem_dim] CUDA tensor with grad_fn.

    Query:
        Hit  → returns stored tensor (grad_fn from previous MemoryTokenizer call)
        Miss → returns null_token (learnable, requires_grad=True)

    Update (EMA, grad_fn preserved):
        Existing: store[k] = (1 - α) * old_tensor + α * voxel_new[vi]
        New:      store[k] = voxel_new[vi]

    CRITICAL for BPTT: do NOT call .detach() on any tensor between chunks
    within the same training sequence. The grad_fn chain through _store
    connects future chunk losses to past chunk MemoryFusion/Tokenizer.
    """

    def __init__(
        self,
        voxel_size: float = 0.05,
        mem_dim: int = MEM_DIM,
        ema_alpha: float = 0.5,
    ):
        self.voxel_size = voxel_size
        self.mem_dim    = mem_dim
        self.alpha      = ema_alpha
        self._store: dict = {}   # (ix,iy,iz) → [mem_dim] tensor on CUDA, keeps grad_fn

    # ── helpers ──────────────────────────────────────────────────────────────

    def _quantize(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz [N, 3] (any device/dtype) → [N, 3] int64 on CPU."""
        return (xyz.detach().float().cpu() / self.voxel_size).floor().long()

    # ── voxel assignment (for scatter aggregation) ───────────────────────────

    def compute_voxel_assignments(
        self, xyz: torch.Tensor
    ) -> Tuple[List[tuple], torch.Tensor]:
        """
        Map N world-space points to unique voxel indices.

        Args:
            xyz : [N, 3] — world coordinates (any device)
        Returns:
            voxel_keys: list of K unique (ix,iy,iz) tuples
            voxel_idx : [N] int64 tensor on xyz.device — index into voxel_keys
        """
        vcoords = self._quantize(xyz)          # [N, 3] CPU int64
        key_to_idx: dict = {}
        voxel_keys: List[tuple] = []
        idx_list: List[int] = []

        for i in range(len(vcoords)):
            k = (vcoords[i, 0].item(),
                 vcoords[i, 1].item(),
                 vcoords[i, 2].item())
            if k not in key_to_idx:
                key_to_idx[k] = len(voxel_keys)
                voxel_keys.append(k)
            idx_list.append(key_to_idx[k])

        voxel_idx = torch.tensor(idx_list, dtype=torch.long, device=xyz.device)
        return voxel_keys, voxel_idx

    # ── query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        xyz: torch.Tensor,       # [N, 3] world coords (any device)
        null_token: torch.Tensor,  # [1, mem_dim] learnable (model device)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve memory features for N points.

        Returns:
            features : [N, mem_dim] — grad_fn from stored tensors or null_token
            mask     : [N, 1]  float — 1.0 if voxel was previously visited
        """
        vcoords = self._quantize(xyz)          # [N, 3] CPU int64
        null_vec = null_token.squeeze(0)       # [mem_dim], requires_grad=True

        feats: List[torch.Tensor] = []
        hits: List[float] = []

        for i in range(len(vcoords)):
            k = (vcoords[i, 0].item(),
                 vcoords[i, 1].item(),
                 vcoords[i, 2].item())
            if k in self._store:
                feats.append(self._store[k])   # CUDA tensor, grad_fn preserved
                hits.append(1.0)
            else:
                feats.append(null_vec)          # learnable null token
                hits.append(0.0)

        features = torch.stack(feats, dim=0)   # [N, mem_dim] — autograd-aware
        mask = torch.tensor(
            hits, dtype=features.dtype, device=xyz.device
        ).unsqueeze(1)                          # [N, 1]
        return features, mask

    # ── EMA update ───────────────────────────────────────────────────────────

    def update(
        self,
        voxel_keys: List[tuple],
        voxel_new: torch.Tensor,   # [V, mem_dim] with grad_fn
    ):
        """
        Apply EMA update for each unique voxel.

        IMPORTANT: voxel_new MUST retain its grad_fn.
        The chain: store[k] = (1-α)*old + α*new preserves grad through both
        old (previous chunk) and new (current chunk), enabling BPTT.

        Args:
            voxel_keys : list of V unique (ix,iy,iz) tuples
            voxel_new  : [V, mem_dim] confidence-weighted new features
        """
        for vi, key in enumerate(voxel_keys):
            feat_vi = voxel_new[vi]   # [mem_dim], grad_fn from MemoryTokenizer
            if key in self._store:
                old = self._store[key]
                # EMA — new tensor connecting old and new computation graphs
                self._store[key] = (1.0 - self.alpha) * old + self.alpha * feat_vi
            else:
                self._store[key] = feat_vi

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def clear(self):
        """Reset at the start of each new sequence."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# ─────────────────────────────────────────────────────────────────────────────
# Stage2V4Loss
# ─────────────────────────────────────────────────────────────────────────────

class Stage2V4Loss(nn.Module):
    """
    Combined loss for Stage-2 V4:
      1. Semantic alignment  : sem_feat ↔ LSeg/CLIP ground-truth (cosine)
      2. Instance contrastive: ins_feat with instance_mask labels
      3. Memory consistency  : M_readout ↔ previously stored M_queried (optional)

    Note: memory consistency uses .detach() on M_queried for training stability.
    The primary BPTT signal flows through sem_align as described in the module
    docstring: loss → sem_feat → frozen DPT → X_fuse → MemoryFusion → X_mem
    → VoxelHashMap → previous chunk's MemoryTokenizer.
    """

    def __init__(
        self,
        w_sem_align:    float = 0.5,
        w_ins_contrast: float = 1.0,
        w_mem_consist:  float = 0.1,
    ):
        super().__init__()
        self.w_sem_align    = w_sem_align
        self.w_ins_contrast = w_ins_contrast
        self.w_mem_consist  = w_mem_consist

        # Reuse the same contrastive loss helper as Stage-1 / Stage-2 V1
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
        from amb3r.tools.contrastive_loss import ContrastiveLoss
        self.contrastive_fn = ContrastiveLoss(
            inter_mode='hinge', inter_margin=0.2, normalize_feats=True
        )

    def forward(self, predictions: dict, batch: dict) -> dict:
        sem_feat  = predictions['semantic_feat']   # [B, T, C_sem, H, W]
        ins_feat  = predictions['instance_feat']   # [B, T, C_ins, H, W]
        lseg_gt   = predictions['_clip_feat_gt']   # [B, T, 512, H, W]
        mem_mask  = predictions['mem_mask']        # [B, T, 1, Hp, Wp]
        M_readout = predictions['M_readout']       # [B*T, N, mem_dim]
        M_queried = predictions.get('M_queried')   # [B*T, N, mem_dim] | None
        inst_mask = batch.get('instance_mask')     # [B, T, H, W] long

        all_losses = {}
        objective  = sem_feat.new_tensor(0.0)

        # ── 1. Semantic alignment ─────────────────────────────────────────
        if inst_mask is not None and self.w_sem_align > 0:
            from amb3r.loss_stage2 import compute_sem_align_loss
            d = compute_sem_align_loss(sem_feat, lseg_gt, inst_mask,
                                       weight=self.w_sem_align)
            all_losses.update(d)
            objective = objective + d['loss_sem_align']

        # ── 2. Instance contrastive ───────────────────────────────────────
        if inst_mask is not None and self.w_ins_contrast > 0:
            from amb3r.loss_stage2 import compute_ins_contrastive_loss
            d = compute_ins_contrastive_loss(ins_feat, inst_mask,
                                             self.contrastive_fn,
                                             weight=self.w_ins_contrast)
            all_losses.update(d)
            objective = objective + d['loss_ins_contrast']

        # ── 3. Memory token consistency (optional) ────────────────────────
        # Pulls current M_readout toward previously stored tokens.
        # M_queried is DETACHED to avoid double-gradient instability.
        # Primary BPTT flows via sem_align → X_mem → VoxelHashMap (see docs).
        if (M_queried is not None and self.w_mem_consist > 0
                and mem_mask.sum() > 0):
            B, T, _, Hp, Wp = mem_mask.shape
            N = Hp * Wp
            # Patch-aligned mem_mask [B*T, N, 1]
            mask_flat = mem_mask.reshape(B * T, N, 1)

            cur  = F.normalize(M_readout.float(), dim=-1)
            tgt  = F.normalize(M_queried.detach().float(), dim=-1)
            cos_dist = 1.0 - (cur * tgt).sum(-1, keepdim=True)   # [B*T, N, 1]
            mem_loss = (cos_dist * mask_flat).sum() / mask_flat.sum().clamp(1)
            mem_loss = mem_loss.to(sem_feat.dtype)

            all_losses['loss_mem_consist']     = self.w_mem_consist * mem_loss
            all_losses['loss_mem_consist_det'] = mem_loss.detach()
            objective = objective + all_losses['loss_mem_consist']

        all_losses['objective'] = objective
        return all_losses


# ─────────────────────────────────────────────────────────────────────────────
# AMB3RStage2V4
# ─────────────────────────────────────────────────────────────────────────────

class AMB3RStage2V4(nn.Module):
    """
    Stage-2 V4: Decoupled Spatial-Memory Attention.

    Wraps a frozen Stage-1 model with lightweight trainable modules:
      - LearnableNullToken      (1 × mem_dim parameter)
      - MemoryFusionModule      (cross-attention, ~3M params at default dims)
      - MemoryTokenizer         (MLP, ~1M params)

    Usage (training):
        voxel_map = model.make_voxel_map()   # once per sequence
        total_loss = 0
        for chunk in chunks:
            preds, M_readout, W_conf, pts_flat = model.forward_chunk(
                chunk_frames, voxel_map, gt_pts_chunk)
            loss = criterion(preds, chunk)
            total_loss += loss['objective']
            # Update memory while KEEPING grad_fn (no detach!)
            AMB3RStage2V4.update_voxel_map(voxel_map, M_readout, W_conf, pts_flat)
        total_loss.backward()    # single backward over full sequence
        optimizer.step()
        voxel_map.clear()        # discard graph

    Parameters
    ----------
    stage1_model  : AMB3RStage1 / AMB3RStage1FullFT (or compatible)
    mem_dim       : 128  — implicit memory token dimension
    vggt_dim      : 2048 — VGGT aggregated token dim (frame+global)
    hidden_dim    : 256  — MemoryFusionModule cross-attention hidden dim
    num_heads     : 4    — attention heads (must divide hidden_dim)
    voxel_size    : 0.05 — voxel grid cell size (metres)
    ema_alpha     : 0.5  — inter-frame EMA weight for memory update
    use_checkpoint: False — wrap fusion+tokenizer with gradient checkpointing
    """

    def __init__(
        self,
        stage1_model: nn.Module,
        mem_dim: int       = MEM_DIM,
        vggt_dim: int      = C_VGGT,
        hidden_dim: int    = HIDDEN_DIM,
        num_heads: int     = 4,
        voxel_size: float  = 0.05,
        ema_alpha: float   = 0.5,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        # ── Frozen Stage-1 ───────────────────────────────────────────────
        self.stage1 = stage1_model
        for p in self.stage1.parameters():
            p.requires_grad_(False)

        # ── Trainable Stage-2 modules ────────────────────────────────────
        self.null_token       = LearnableNullToken(mem_dim)
        self.memory_fusion    = MemoryFusionModule(vggt_dim, mem_dim, hidden_dim, num_heads)
        self.memory_tokenizer = MemoryTokenizer(vggt_dim, mem_dim)

        self.mem_dim       = mem_dim
        self.vggt_dim      = vggt_dim
        self.voxel_size    = voxel_size
        self.ema_alpha     = ema_alpha
        self.use_checkpoint = use_checkpoint

        # Cache last-layer DPT index (resolved lazily at first forward)
        self._last_dpt_idx: Optional[int] = None

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def trainable_params(self) -> List[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def _get_last_dpt_idx(self) -> int:
        """
        Return the index of the last DPT intermediate layer used by semantic_head.
        Resolved once and cached; falls back to -1 (last element) if not found.
        """
        if self._last_dpt_idx is not None:
            return self._last_dpt_idx
        try:
            sem_head = self.stage1.front_end.model.semantic_head
            idx = max(sem_head.intermediate_layer_idx)
        except AttributeError:
            idx = -1   # last element of aggregated_tokens_list
        self._last_dpt_idx = idx
        return idx

    def make_voxel_map(self) -> DifferentiableVoxelMap:
        """Factory — create a fresh DifferentiableVoxelMap for a new sequence."""
        return DifferentiableVoxelMap(
            voxel_size=self.voxel_size,
            mem_dim=self.mem_dim,
            ema_alpha=self.ema_alpha,
        )

    def load_stage1_weights(self, path: str, data_type: str = 'bf16',
                            strict: bool = False):
        self.stage1.load_weights(path, data_type=data_type, strict=strict)

    def prepare(self, data_type: str = 'bf16'):
        self.stage1.prepare(data_type)

    # ── fusion + tokenizer (optionally checkpointed) ──────────────────────────

    def _fusion_and_tokenize(
        self, X_vggt: torch.Tensor, X_mem: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute X_fuse, M_readout, W_conf.  Checkpoint-friendly."""
        X_fuse              = self.memory_fusion(X_vggt, X_mem)
        M_readout, W_conf   = self.memory_tokenizer(X_fuse)
        return X_fuse, M_readout, W_conf

    # ── per-chunk forward ─────────────────────────────────────────────────────

    def forward_chunk(
        self,
        frames: dict,
        voxel_map: DifferentiableVoxelMap,
        pts_for_query: Optional[torch.Tensor] = None,
    ) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one chunk of T frames with memory conditioning.

        Parameters
        ----------
        frames        : dict with 'images' [B, T, 3, H, W] in [-1, 1]
        voxel_map     : DifferentiableVoxelMap — shared across ALL chunks in the
                        current sequence (cleared only between sequences).
        pts_for_query : [B, T, H, W, 3] GT world points for stable voxel queries.
                        Falls back to Stage-1 world_points if None.

        Returns
        -------
        predictions   : dict — semantic/instance features, geometry (no_grad),
                        memory metadata for loss computation.
        M_readout     : [B*T, N_patch, mem_dim] — memory tokens WITH grad_fn
        W_conf        : [B*T, N_patch, 1]       — confidence scores WITH grad_fn
        pts_flat      : [B*T*N_patch, 3]        — query world-space coords
        """
        s1 = self.stage1

        # ── Step 1: Stage-1 backbone (fully no_grad) ────────────────────
        with torch.no_grad():
            images, patch_tokens = s1.front_end.encode_patch_tokens(frames)
            B, T = images.shape[:2]
            H, W = images.shape[-2:]
            Hp, Wp  = H // PATCH_SIZE, W // PATCH_SIZE
            N_patch = Hp * Wp
            device, dtype = images.device, images.dtype

            # Frozen LSeg features + CLIP-patch fusion
            lseg_feat   = s1._extract_lseg(images)             # [B, T, 512, H, W]
            lseg_flat   = lseg_feat.flatten(0, 1)              # [B*T, 512, H, W]
            patch_tokens = s1.clip_patch_fusion(patch_tokens, lseg_flat, H, W)

            # Full VGGT decode (24 frame/global blocks, frozen)
            decoded = s1.front_end.decode_patch_tokens(patch_tokens, images)

            # Geometry heads (depth, pose, world_points) — no grad
            res_geo = s1.front_end.decode_heads(images, decoded, has_backend=False)
            # res_geo['aggregated_tokens_list'] available because has_backend=False

        # ── Step 2: Extract X_vggt from last DPT intermediate layer ─────
        atl    = res_geo['aggregated_tokens_list']   # list of [B, T, P, 2C] or None
        ps_idx = res_geo['patch_start_idx']          # int: skip camera+register tokens
        last_dpt_idx = self._get_last_dpt_idx()      # e.g. 23

        last_layer = atl[last_dpt_idx]               # [B, T, P, C_vggt] — no grad
        C = last_layer.shape[-1]                     # should be C_VGGT = 2048
        # Patch tokens at the last DPT layer
        X_vggt = last_layer[:, :, ps_idx:, :].reshape(B * T, N_patch, C)  # [B*T, N, C]

        # ── Step 3: World-space coordinates at patch resolution ──────────
        query_pts = pts_for_query if pts_for_query is not None \
                    else res_geo['world_points']        # [B, T, H, W, 3]

        pts_patch = F.interpolate(
            query_pts.flatten(0, 1).permute(0, 3, 1, 2).float(),  # [B*T, 3, H, W]
            size=(Hp, Wp), mode='bilinear', align_corners=False,
        ).permute(0, 2, 3, 1)                                      # [B*T, Hp, Wp, 3]
        pts_flat = pts_patch.reshape(B * T * N_patch, 3)           # [B*T*N, 3]

        # ── Step 4: Memory retrieval from DifferentiableVoxelMap ────────
        # X_mem_flat may contain:
        #   - stored tensors WITH grad_fn from previous chunk's MemoryTokenizer
        #   - null_token (learnable, requires_grad=True)
        X_mem_flat, mask_flat = voxel_map.query(pts_flat, self.null_token.token)
        # X_mem_flat : [B*T*N, mem_dim]  mask_flat : [B*T*N, 1]

        X_mem        = X_mem_flat.reshape(B * T, N_patch, self.mem_dim)   # [B*T, N, 128]
        mem_mask_spat = mask_flat.reshape(B, T, 1, Hp, Wp)                # [B, T, 1, Hp, Wp]
        has_memory   = (mask_flat.sum() > 0).item()

        # ── Step 5: Fusion + Tokenizer (TRAINABLE) ───────────────────────
        # Gradient checkpointing optional — saves peak memory at the cost of
        # recomputing the forward pass during backward.
        if self.use_checkpoint:
            X_fuse, M_readout, W_conf = ckpt.checkpoint(
                self._fusion_and_tokenize, X_vggt, X_mem, use_reentrant=False
            )
        else:
            X_fuse, M_readout, W_conf = self._fusion_and_tokenize(X_vggt, X_mem)
        # X_fuse   : [B*T, N, C_vggt]  — grad flows from γ, W_*, attn
        # M_readout: [B*T, N, mem_dim] — for VoxelMap update
        # W_conf   : [B*T, N, 1]       — confidence for weighted aggregation

        # ── Step 6: Task decoding (frozen params, grad ENABLED) ──────────
        # Build modified aggregated_tokens_list:
        #   Replace patch-token slice of last_dpt_idx with X_fuse.
        #   This creates a new tensor with grad_fn, connecting the DPT head
        #   output back to X_fuse → MemoryFusion → X_mem → VoxelHashMap.
        fused_patch   = X_fuse.reshape(B, T, N_patch, C)           # [B, T, N, C]
        prefix        = last_layer[:, :, :ps_idx, :]               # [B, T, ps, C] no grad
        modified_last = torch.cat([prefix, fused_patch], dim=2)    # [B, T, P, C] has grad

        modified_atl = list(atl)
        modified_atl[last_dpt_idx] = modified_last

        # Call frozen DPT heads — their parameters have requires_grad=False,
        # but the COMPUTATION runs in grad mode, so grad flows through the ops
        # back to modified_last → X_fuse.
        sem_feat, sem_conf = s1.front_end.model.semantic_head(
            modified_atl,
            images=images,
            semantic_cond=lseg_feat,
            patch_start_idx=ps_idx,
        )   # [B, T, sem_dim, H, W]

        ins_feat, ins_conf = s1.front_end.model.instance_head(
            modified_atl,
            images=images,
            semantic_cond=None,
            patch_start_idx=ps_idx,
        )   # [B, T, ins_dim, H, W]

        # ── Assemble predictions ─────────────────────────────────────────
        predictions = {
            # Task outputs (gradients flow through these)
            'semantic_feat':    sem_feat,          # [B, T, sem_dim, H, W]
            'instance_feat':    ins_feat,           # [B, T, ins_dim, H, W]
            'semantic_conf':    sem_conf,           # [B, T, 1, H, W]
            # Ground-truth CLIP for alignment loss
            '_clip_feat_gt':    lseg_feat,          # [B, T, 512, H, W]
            # Memory metadata
            'mem_mask':         mem_mask_spat,      # [B, T, 1, Hp, Wp]
            # M_readout for memory consistency loss:
            #   current token vs what was retrieved (= previous chunk's output)
            'M_readout': M_readout,                 # [B*T, N, mem_dim], has grad
            'M_queried':  X_mem if has_memory else None,  # None on first chunk
            # Geometry from frozen Stage-1 (informational, no grad needed)
            'depth':                res_geo['depth'],
            'depth_conf':           res_geo['depth_conf'],
            'pose_enc':             res_geo['pose_enc'],
            'world_points':         res_geo['world_points'],
            'pts3d_by_unprojection': res_geo.get('pts3d_by_unprojection'),
            'extrinsic':            res_geo['extrinsic'],
            'intrinsic':            res_geo['intrinsic'],
            'images':               images,
        }

        return predictions, M_readout, W_conf, pts_flat

    # ── VoxelMap update (static, called from training loop) ───────────────────

    @staticmethod
    def update_voxel_map(
        voxel_map: DifferentiableVoxelMap,
        M_readout: torch.Tensor,   # [B*T, N_patch, mem_dim] WITH grad_fn
        W_conf:    torch.Tensor,   # [B*T, N_patch, 1]       WITH grad_fn
        pts_flat:  torch.Tensor,   # [B*T*N_patch, 3]
    ):
        """
        Confidence-weighted intra-frame aggregation then EMA inter-frame update.

        Step A (intra-frame — torch_scatter):
            weighted_feats = M_readout * W_conf                [N, mem_dim]
            sum_wf = scatter_sum(weighted_feats, voxel_idx)   [V, mem_dim]
            sum_w  = scatter_sum(W_conf,         voxel_idx)   [V, 1]
            voxel_new = sum_wf / (sum_w + 1e-6)               [V, mem_dim]

        Step B (inter-frame — EMA in DifferentiableVoxelMap.update):
            existing: store[k] = (1-α)*old + α*voxel_new[vi]
            new:      store[k] = voxel_new[vi]

        CRITICAL: Do NOT call .detach() on M_readout within a training sequence.
        The grad_fn of voxel_new connects to current MemoryTokenizer.
        The EMA chain connects to previous chunks via old tensors.
        """
        N_total  = pts_flat.shape[0]
        M_flat   = M_readout.reshape(N_total, voxel_map.mem_dim)  # [N, mem_dim]
        W_flat   = W_conf.reshape(N_total, 1)                      # [N, 1]

        # Compute voxel assignment on CPU (integer arithmetic), idx on device
        voxel_keys, voxel_idx = voxel_map.compute_voxel_assignments(pts_flat)
        num_voxels = len(voxel_keys)
        idx_dev    = voxel_idx.to(M_flat.device)

        # Intra-frame confidence-weighted aggregation
        weighted  = M_flat * W_flat                        # [N, mem_dim], grad OK
        sum_wf    = scatter_sum(weighted, idx_dev, dim=0,
                                dim_size=num_voxels)       # [V, mem_dim]
        sum_w     = scatter_sum(W_flat,   idx_dev, dim=0,
                                dim_size=num_voxels)       # [V, 1]
        voxel_new = sum_wf / (sum_w + 1e-6)               # [V, mem_dim], grad OK

        # Inter-frame EMA update (grad_fn preserved)
        voxel_map.update(voxel_keys, voxel_new)
