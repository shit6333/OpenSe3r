"""
memory_stage2v4.py — Spatial memory data structures for Stage-2 V4.

Contains:
    LearnableNullToken        Learnable fallback for unvisited voxels
    DifferentiableVoxelMap    Grad-preserving voxel hash map (for BPTT)
    CrossChunkFeatureBuffer   Detached per-voxel anchor store (for cross-view loss)

Design principles:
    - DifferentiableVoxelMap keeps grad_fn on stored tensors, enabling Chunked
      Unrolled BPTT across chunks within a training sequence.
    - CrossChunkFeatureBuffer stores DETACHED first-visit anchor features as
      targets for the cross-view consistency loss.
    - Both are created fresh per sequence and discarded after backward().
"""

from typing import List, Tuple

import torch
import torch.nn as nn

PATCH_SIZE = 14
C_VGGT     = 2048       # frame(1024) + global(1024)
MEM_DIM    = 128
HIDDEN_DIM = 256
DPT_LAYERS = [4, 11, 17, 23]


# ─────────────────────────────────────────────────────────────────────────────
# LearnableNullToken
# ─────────────────────────────────────────────────────────────────────────────

class LearnableNullToken(nn.Module):
    """
    Learnable fallback token for unvisited voxels (cold-start / unseen areas).

    Initialized to small random values so it does not disrupt frozen VGGT
    features before training begins.
    """

    def __init__(self, mem_dim: int = MEM_DIM):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, mem_dim) * 0.02)

    def forward(self) -> torch.Tensor:
        return self.token   # [1, mem_dim]


# ─────────────────────────────────────────────────────────────────────────────
# DifferentiableVoxelMap
# ─────────────────────────────────────────────────────────────────────────────

class DifferentiableVoxelMap:
    """
    Spatial hash map: (ix, iy, iz) → [mem_dim] CUDA tensor **with grad_fn**.

    Key properties for BPTT:
        - query() returns stored tensors that carry grad_fn from the
          MemoryTokenizer that wrote them in a previous chunk.
        - update() uses EMA: store[k] = (1-α)*old + α*new, which creates a
          new tensor connecting BOTH the old (previous chunk) and new (current
          chunk) computation graphs.
        - No .detach() is ever called on stored tensors during a training
          sequence, enabling full Chunked Unrolled BPTT.

    Lifecycle:
        vmap = DifferentiableVoxelMap(...)   # start of sequence
        for chunk in chunks:
            feats, mask = vmap.query(pts, null_token)
            ...
            vmap.update(keys, new_feats)     # grad_fn preserved
        loss.backward()                       # BPTT across all chunks
        vmap.clear()                          # discard graph
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
        self._store: dict = {}   # (ix,iy,iz) → Tensor[mem_dim] with grad_fn

    # ── helpers ───────────────────────────────────────────────────────────────

    def _quantize(self, xyz: torch.Tensor) -> torch.Tensor:
        """[N, 3] float → [N, 3] int64 voxel indices (CPU)."""
        return (xyz.detach().cpu() / self.voxel_size).long()

    def compute_voxel_assignments(
        self, xyz: torch.Tensor,
    ) -> Tuple[List[tuple], torch.Tensor]:
        """
        Map N world-space points to unique voxel keys + per-point indices.

        Returns:
            voxel_keys : list of V unique (ix, iy, iz) tuples
            voxel_idx  : [N] long tensor — voxel_idx[i] ∈ [0, V)
        """
        vcoords   = self._quantize(xyz)
        key_to_idx: dict       = {}
        voxel_keys: List[tuple] = []
        idx_list:   List[int]   = []

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
        xyz: torch.Tensor,        # [N, 3] world coords
        null_token: torch.Tensor,  # [1, mem_dim] learnable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve memory features for N points.

        Returns:
            features : [N, mem_dim] — grad_fn from stored tensors or null_token
            mask     : [N, 1] float — 1.0 if voxel was previously visited
        """
        vcoords  = self._quantize(xyz)
        null_vec = null_token.squeeze(0)   # [mem_dim]

        feats: List[torch.Tensor] = []
        hits:  List[float] = []

        for i in range(len(vcoords)):
            k = (vcoords[i, 0].item(),
                 vcoords[i, 1].item(),
                 vcoords[i, 2].item())
            if k in self._store:
                feats.append(self._store[k])
                hits.append(1.0)
            else:
                feats.append(null_vec)
                hits.append(0.0)

        features = torch.stack(feats, dim=0)
        mask = torch.tensor(
            hits, dtype=features.dtype, device=xyz.device,
        ).unsqueeze(1)
        return features, mask

    # ── EMA update ────────────────────────────────────────────────────────────

    def update(
        self,
        voxel_keys: List[tuple],
        voxel_new: torch.Tensor,   # [V, mem_dim] WITH grad_fn
    ):
        """
        EMA update: store[k] = (1-α)*old + α*new.

        CRITICAL: voxel_new must retain its grad_fn — do NOT detach.
        The EMA chain preserves gradient flow across chunks for BPTT.
        """
        for vi, key in enumerate(voxel_keys):
            feat_vi = voxel_new[vi]
            if key in self._store:
                old = self._store[key]
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
# CrossChunkFeatureBuffer  (vectorized — no per-pixel Python loops)
# ─────────────────────────────────────────────────────────────────────────────

class CrossChunkFeatureBuffer:
    """
    Per-voxel sem/ins feature store for cross-view consistency supervision.

    ALL stored tensors are DETACHED.  This buffer provides TARGETS for the
    cross-view loss, not a BPTT gradient channel.

    Policy: FIRST-VISIT ANCHORING
        Only the features from the first chunk that visits a voxel are stored.

    Stores are maintained as SORTED tensors on CPU for vectorized
    searchsorted-based query (no per-pixel Python loops).

    Also supports instance-ID-keyed storage via update_by_id / query_by_id.

    Lifecycle:
        buf = CrossChunkFeatureBuffer(...)
        for c, chunk in enumerate(chunks):
            if c > 0:
                sem_s, ins_s, mask = buf.query(pts)
                loss += cross_view_loss(...)
            buf.update(pts, sem.detach(), ins.detach())
            buf.update_by_id(inst_mask, ins.detach(), sem.detach())
        buf.clear()
    """

    _P = 1_000_003  # prime for spatial hash: key = ix*P^2 + iy*P + iz

    def __init__(self, voxel_size: float, sem_dim: int, ins_dim: int,
                 ignore_id: int = 0):
        self.voxel_size = voxel_size
        self.sem_dim    = sem_dim
        self.ins_dim    = ins_dim
        self.ignore_id  = ignore_id

        # ── voxel-keyed stores (sorted tensors on CPU) ──
        self._voxel_keys: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._voxel_sem:  torch.Tensor = torch.empty(0, sem_dim)
        self._voxel_ins:  torch.Tensor = torch.empty(0, ins_dim)
        self._voxel_key_set: set       = set()   # for O(1) first-visit check

        # ── instance-ID-keyed stores (sorted tensors on CPU) ──
        self._id_keys:    torch.Tensor = torch.empty(0, dtype=torch.long)
        self._id_sem:     torch.Tensor = torch.empty(0, sem_dim)
        self._id_ins:     torch.Tensor = torch.empty(0, ins_dim)
        self._id_key_set: set          = set()

    # ── spatial hash ─────────────────────────────────────────────────────────

    def _encode_keys(self, xyz: torch.Tensor) -> torch.Tensor:
        """Vectorized (ix,iy,iz) → int64.  Input [N,3], output [N] on CPU.

        Offset by _P//2 so that all components are non-negative before
        packing into a single int64.  This makes the encoding bijective
        for coordinates in [-P//2, P//2).
        """
        vc = (xyz.detach().cpu().float() / self.voxel_size).long()
        P  = self._P
        H  = P // 2  # offset to make coords non-negative
        vx = vc[:, 0] + H
        vy = vc[:, 1] + H
        vz = vc[:, 2] + H
        return vx * (P * P) + vy * P + vz

    # ── voxel query (vectorized) ─────────────────────────────────────────────

    def query(
        self, pts_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            sem_stored : [N, sem_dim]  (zeros for misses)
            ins_stored : [N, ins_dim]  (zeros for misses)
            mask       : [N, 1] float  (1.0 = hit)
        """
        device = pts_flat.device
        N      = pts_flat.shape[0]

        if self._voxel_keys.shape[0] == 0:
            return (torch.zeros(N, self.sem_dim, device=device),
                    torch.zeros(N, self.ins_dim, device=device),
                    torch.zeros(N, 1, device=device))

        qk   = self._encode_keys(pts_flat)          # [N] CPU int64
        sk   = self._voxel_keys                      # [V] CPU sorted
        idx  = torch.searchsorted(sk, qk)            # [N]
        idx  = idx.clamp(max=sk.shape[0] - 1)
        hit  = (sk[idx] == qk)                        # [N] bool CPU

        sem_stored = torch.zeros(N, self.sem_dim, device=device)
        ins_stored = torch.zeros(N, self.ins_dim, device=device)

        if hit.any():
            matched_idx = idx[hit]
            sem_stored[hit.to(device)] = self._voxel_sem[matched_idx].to(device)
            ins_stored[hit.to(device)] = self._voxel_ins[matched_idx].to(device)

        mask = hit.float().unsqueeze(1).to(device)
        return sem_stored, ins_stored, mask

    # ── voxel update (vectorized scatter_mean, first-visit) ──────────────────

    def update(
        self,
        pts_flat:  torch.Tensor,   # [N, 3]
        sem_patch: torch.Tensor,   # [N, sem_dim]  DETACHED
        ins_patch: torch.Tensor,   # [N, ins_dim]  DETACHED
    ):
        from torch_scatter import scatter_mean

        keys = self._encode_keys(pts_flat)                      # [N] CPU
        unique_keys, inverse = torch.unique(keys, return_inverse=True)
        V = unique_keys.shape[0]

        inv_dev = inverse.to(sem_patch.device)
        sem_v = scatter_mean(sem_patch.float(), inv_dev, dim=0, dim_size=V)
        ins_v = scatter_mean(ins_patch.float(), inv_dev, dim=0, dim_size=V)

        # First-visit filter (loop over unique voxels only — typically a few thousand)
        new_mask = torch.tensor(
            [unique_keys[i].item() not in self._voxel_key_set for i in range(V)],
            dtype=torch.bool)

        if not new_mask.any():
            return

        new_keys = unique_keys[new_mask]
        new_sem  = sem_v[new_mask].detach().cpu()
        new_ins  = ins_v[new_mask].detach().cpu()

        for k in new_keys.tolist():
            self._voxel_key_set.add(k)

        # Append + re-sort for searchsorted
        all_keys = torch.cat([self._voxel_keys, new_keys.cpu()])
        all_sem  = torch.cat([self._voxel_sem,  new_sem])
        all_ins  = torch.cat([self._voxel_ins,  new_ins])
        order = all_keys.argsort()
        self._voxel_keys = all_keys[order]
        self._voxel_sem  = all_sem[order]
        self._voxel_ins  = all_ins[order]

    # ── instance-ID query (vectorized) ───────────────────────────────────────

    def query_by_id(
        self,
        instance_mask_flat: torch.Tensor,  # [N] long
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            sem_stored : [N, sem_dim]
            ins_stored : [N, ins_dim]
            mask       : [N] bool
        """
        N = instance_mask_flat.shape[0]

        if self._id_keys.shape[0] == 0:
            return (torch.zeros(N, self.sem_dim, device=device),
                    torch.zeros(N, self.ins_dim, device=device),
                    torch.zeros(N, dtype=torch.bool, device=device))

        qk  = instance_mask_flat.cpu().long()        # [N]
        sk  = self._id_keys                           # [K] sorted
        idx = torch.searchsorted(sk, qk)
        idx = idx.clamp(max=sk.shape[0] - 1)
        hit = (sk[idx] == qk) & (qk != self.ignore_id)

        sem_stored = torch.zeros(N, self.sem_dim, device=device)
        ins_stored = torch.zeros(N, self.ins_dim, device=device)

        if hit.any():
            matched_idx = idx[hit]
            sem_stored[hit.to(device)] = self._id_sem[matched_idx].to(device)
            ins_stored[hit.to(device)] = self._id_ins[matched_idx].to(device)

        return sem_stored, ins_stored, hit.to(device)

    # ── instance-ID update (first-visit anchor) ─────────────────────────────

    def update_by_id(
        self,
        instance_mask_flat: torch.Tensor,  # [N] long
        ins_feat_flat:      torch.Tensor,  # [N, ins_dim] DETACHED
        sem_feat_flat:      torch.Tensor = None,  # [N, sem_dim] DETACHED
    ):
        from torch_scatter import scatter_mean

        ids_cpu = instance_mask_flat.cpu().long()
        unique_ids = ids_cpu.unique()

        # Filter: skip ignore_id and already-stored
        new_ids = [iid.item() for iid in unique_ids
                   if iid.item() != self.ignore_id and iid.item() not in self._id_key_set]
        if not new_ids:
            return

        new_keys_list, new_sem_list, new_ins_list = [], [], []
        for iid in new_ids:
            px = (ids_cpu == iid)
            self._id_key_set.add(iid)
            new_keys_list.append(iid)
            new_ins_list.append(ins_feat_flat[px].float().mean(0).detach().cpu())
            if sem_feat_flat is not None:
                new_sem_list.append(sem_feat_flat[px].float().mean(0).detach().cpu())
            else:
                new_sem_list.append(torch.zeros(self.sem_dim))

        new_keys = torch.tensor(new_keys_list, dtype=torch.long)
        new_sem  = torch.stack(new_sem_list)
        new_ins  = torch.stack(new_ins_list)

        # Append + re-sort
        all_keys = torch.cat([self._id_keys, new_keys])
        all_sem  = torch.cat([self._id_sem, new_sem])
        all_ins  = torch.cat([self._id_ins, new_ins])
        order = all_keys.argsort()
        self._id_keys = all_keys[order]
        self._id_sem  = all_sem[order]
        self._id_ins  = all_ins[order]

    # ── lifecycle ────────────────────────────────────────────────────────────

    def clear(self):
        self._voxel_keys = torch.empty(0, dtype=torch.long)
        self._voxel_sem  = torch.empty(0, self.sem_dim)
        self._voxel_ins  = torch.empty(0, self.ins_dim)
        self._voxel_key_set.clear()
        self._id_keys = torch.empty(0, dtype=torch.long)
        self._id_sem  = torch.empty(0, self.sem_dim)
        self._id_ins  = torch.empty(0, self.ins_dim)
        self._id_key_set.clear()

    def __len__(self) -> int:
        return self._voxel_keys.shape[0]

    @property
    def n_instance_ids(self) -> int:
        return self._id_keys.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# DetachedVoxelStore — simple first-visit store for non-BPTT memory modes
# ─────────────────────────────────────────────────────────────────────────────

class DetachedVoxelStore:
    """
    Simple detached voxel store for memory modes that don't need BPTT
    (mem_mode 1: sem⊕ins, mem_mode 2: full VGGT, mem_mode 3: AE bottleneck).

    Key differences from DifferentiableVoxelMap:
        - All stored tensors are DETACHED (no grad_fn)
        - First-visit anchoring (no EMA — first observation wins)
        - Vectorized query via searchsorted (same pattern as CrossChunkFeatureBuffer)

    Lifecycle:
        store = DetachedVoxelStore(voxel_size, mem_dim)
        for chunk in chunks:
            X_mem, mask = store.query(pts, null_token)  # [N, mem_dim]
            ...
            store.update(pts, features.detach())         # first-visit only
        store.clear()
    """

    _P = 1_000_003

    def __init__(self, voxel_size: float, mem_dim: int):
        self.voxel_size = voxel_size
        self.mem_dim = mem_dim

        self._voxel_keys: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._voxel_feats: torch.Tensor = torch.empty(0, mem_dim)
        self._voxel_key_set: set = set()

    def _encode_keys(self, xyz: torch.Tensor) -> torch.Tensor:
        """[N,3] → [N] int64 on CPU."""
        vc = (xyz.detach().cpu().float() / self.voxel_size).long()
        P = self._P
        H = P // 2
        vx = vc[:, 0] + H
        vy = vc[:, 1] + H
        vz = vc[:, 2] + H
        return vx * (P * P) + vy * P + vz

    def query(
        self,
        xyz: torch.Tensor,       # [N, 3]
        null_token: torch.Tensor, # [1, mem_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            features : [N, mem_dim] — stored features or null_token for misses
            mask     : [N, 1] float — 1.0 if voxel previously visited
        """
        device = xyz.device
        N = xyz.shape[0]

        if self._voxel_keys.shape[0] == 0:
            null_expanded = null_token.squeeze(0).expand(N, -1)
            return null_expanded, torch.zeros(N, 1, device=device)

        qk = self._encode_keys(xyz)               # [N] CPU
        sk = self._voxel_keys                       # [V] CPU sorted
        idx = torch.searchsorted(sk, qk)            # [N]
        idx = idx.clamp(max=sk.shape[0] - 1)
        hit = (sk[idx] == qk)                        # [N] bool CPU

        # Start with null_token for all
        features = null_token.squeeze(0).expand(N, -1).clone().to(device)

        if hit.any():
            matched_idx = idx[hit]
            hit_dev = hit.to(device)
            features[hit_dev] = self._voxel_feats[matched_idx].to(device)

        mask = hit.float().unsqueeze(1).to(device)
        return features, mask

    def update(
        self,
        xyz: torch.Tensor,      # [N, 3]
        features: torch.Tensor,  # [N, mem_dim]  DETACHED
    ):
        """First-visit update with scatter_mean aggregation per voxel."""
        from torch_scatter import scatter_mean

        keys = self._encode_keys(xyz)                           # [N] CPU
        unique_keys, inverse = torch.unique(keys, return_inverse=True)
        V = unique_keys.shape[0]

        inv_dev = inverse.to(features.device)
        feats_v = scatter_mean(features.float(), inv_dev, dim=0, dim_size=V)

        # First-visit filter
        new_mask = torch.tensor(
            [unique_keys[i].item() not in self._voxel_key_set for i in range(V)],
            dtype=torch.bool)

        if not new_mask.any():
            return

        new_keys = unique_keys[new_mask]
        new_feats = feats_v[new_mask].detach().cpu()

        for k in new_keys.tolist():
            self._voxel_key_set.add(k)

        # Append + re-sort for searchsorted
        all_keys = torch.cat([self._voxel_keys, new_keys.cpu()])
        all_feats = torch.cat([self._voxel_feats, new_feats])
        order = all_keys.argsort()
        self._voxel_keys = all_keys[order]
        self._voxel_feats = all_feats[order]

    def clear(self):
        self._voxel_keys = torch.empty(0, dtype=torch.long)
        self._voxel_feats = torch.empty(0, self.mem_dim)
        self._voxel_key_set.clear()

    def __len__(self) -> int:
        return self._voxel_keys.shape[0]
