"""
memory_stage2v4.py — Spatial memory data structures for Stage-2 V4.

Contains:
    LearnableNullToken        Learnable fallback for unvisited voxels
    DifferentiableVoxelMap    Grad-preserving voxel hash map (for BPTT)
    CrossChunkFeatureBuffer   Detached per-voxel anchor store (for cross-view loss)

Design principles:
    - DifferentiableVoxelMap keeps grad_fn on stored tensors, enabling Chunked
      Unrolled BPTT across chunks within a training sequence.
    - CrossChunkFeatureBuffer stores DETACHED anchor features as targets for the
      cross-view consistency loss.  Anchors are updated every chunk via
      confidence-weighted EMA:
          F_store = (1 - λ·conf_new) · F_store + λ·conf_new · F_new
      This lets high-confidence observations gradually refine the anchor while
      low-confidence observations (occlusion, blur) contribute minimally.
    - Both are created fresh per sequence and discarded after backward().
"""

from typing import List, Optional, Tuple

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

    Policy: CONFIDENCE-WEIGHTED EMA ANCHORING
        Every chunk updates existing voxels via:
            F_store = (1 - λ·conf_new) · F_store + λ·conf_new · F_new
        where λ = ema_lambda (default 0.1).  New voxels are initialized on
        first visit.  High-confidence observations pull the anchor more; low-
        confidence observations (occlusion, blur, uncertain predictions)
        contribute minimally.  sem and ins use their respective conf scores.

        When conf is not supplied, λ is used directly as a fixed EMA weight
        (equivalent to conf=1.0 everywhere).

    Stores are maintained as SORTED tensors on CPU for vectorized
    searchsorted-based query (no per-pixel Python loops).

    Also supports instance-ID-keyed storage via update_by_id / query_by_id.

    Lifecycle:
        buf = CrossChunkFeatureBuffer(...)
        for c, chunk in enumerate(chunks):
            if c > 0:
                sem_s, ins_s, mask = buf.query(pts)
                loss += cross_view_loss(...)
            buf.update(pts, sem.detach(), ins.detach(),
                       sem_conf=sem_c.detach(), ins_conf=ins_c.detach())
            buf.update_by_id(inst_mask, ins.detach(), sem.detach(),
                             ins_conf_flat=ins_c.detach(),
                             sem_conf_flat=sem_c.detach())
        buf.clear()
    """

    _P = 1_000_003  # prime for spatial hash: key = ix*P^2 + iy*P + iz

    def __init__(self, voxel_size: float, sem_dim: int, ins_dim: int,
                 ignore_id: int = 0, ema_lambda: float = 0.1):
        self.voxel_size  = voxel_size
        self.sem_dim     = sem_dim
        self.ins_dim     = ins_dim
        self.ignore_id   = ignore_id
        self._lambda     = ema_lambda

        # ── voxel-keyed stores (sorted tensors on CPU) ──
        self._voxel_keys: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._voxel_sem:  torch.Tensor = torch.empty(0, sem_dim)
        self._voxel_ins:  torch.Tensor = torch.empty(0, ins_dim)

        # ── instance-ID-keyed stores (sorted tensors on CPU) ──
        self._id_keys: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._id_sem:  torch.Tensor = torch.empty(0, sem_dim)
        self._id_ins:  torch.Tensor = torch.empty(0, ins_dim)

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

    # ── voxel update (conf-weighted EMA, fully vectorized) ───────────────────

    def update(
        self,
        pts_flat:  torch.Tensor,             # [N, 3]
        sem_patch: torch.Tensor,             # [N, sem_dim]  DETACHED
        ins_patch: torch.Tensor,             # [N, ins_dim]  DETACHED
        sem_conf:  Optional[torch.Tensor] = None,  # [N] or [N,1]  in [0,1]
        ins_conf:  Optional[torch.Tensor] = None,  # [N] or [N,1]  in [0,1]
    ):
        """
        Conf-weighted EMA update for every voxel touched this chunk.

        Existing voxels:
            F_store = (1 - λ·conf_v) · F_store + λ·conf_v · F_new
        New voxels: initialized directly from F_new (first-visit).

        sem_conf / ins_conf are scatter-averaged per voxel before use.
        If None, λ is used as a fixed EMA weight (conf treated as 1.0).
        """
        from torch_scatter import scatter_mean

        keys = self._encode_keys(pts_flat)                  # [N] CPU
        unique_keys, inverse = torch.unique(keys, return_inverse=True)
        V = unique_keys.shape[0]

        dev       = sem_patch.device
        inv_dev   = inverse.to(dev)

        sem_v = scatter_mean(sem_patch.float(), inv_dev, dim=0, dim_size=V)  # [V, sem_dim]
        ins_v = scatter_mean(ins_patch.float(), inv_dev, dim=0, dim_size=V)  # [V, ins_dim]

        # Per-voxel averaged confidence — shape [V, 1]
        if sem_conf is not None:
            sc = sem_conf.float().view(-1, 1).to(dev)
            sem_conf_v = scatter_mean(sc, inv_dev, dim=0, dim_size=V).cpu()  # [V, 1]
        else:
            sem_conf_v = torch.ones(V, 1)

        if ins_conf is not None:
            ic = ins_conf.float().view(-1, 1).to(dev)
            ins_conf_v = scatter_mean(ic, inv_dev, dim=0, dim_size=V).cpu()  # [V, 1]
        else:
            ins_conf_v = torch.ones(V, 1)

        sem_v_cpu      = sem_v.detach().cpu()
        ins_v_cpu      = ins_v.detach().cpu()
        unique_keys_cpu = unique_keys.cpu()

        # ── cold store: initialise all voxels directly ────────────────────────
        if self._voxel_keys.shape[0] == 0:
            order = unique_keys_cpu.argsort()
            self._voxel_keys = unique_keys_cpu[order]
            self._voxel_sem  = sem_v_cpu[order]
            self._voxel_ins  = ins_v_cpu[order]
            return

        # ── split into existing (EMA update) and new (append) ─────────────────
        idx_in_store = torch.searchsorted(self._voxel_keys, unique_keys_cpu)   # [V]
        idx_clamped  = idx_in_store.clamp(max=self._voxel_keys.shape[0] - 1)
        existing     = (self._voxel_keys[idx_clamped] == unique_keys_cpu)      # [V] bool

        if existing.any():
            pos     = idx_clamped[existing]                          # positions in store
            s_alpha = (self._lambda * sem_conf_v[existing]).clamp(0.0, 1.0)  # [E, 1]
            i_alpha = (self._lambda * ins_conf_v[existing]).clamp(0.0, 1.0)  # [E, 1]
            self._voxel_sem[pos] = ((1.0 - s_alpha) * self._voxel_sem[pos]
                                    + s_alpha * sem_v_cpu[existing])
            self._voxel_ins[pos] = ((1.0 - i_alpha) * self._voxel_ins[pos]
                                    + i_alpha * ins_v_cpu[existing])

        new_mask = ~existing
        if new_mask.any():
            new_keys = unique_keys_cpu[new_mask]
            new_sem  = sem_v_cpu[new_mask]
            new_ins  = ins_v_cpu[new_mask]
            all_keys = torch.cat([self._voxel_keys, new_keys])
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

    # ── instance-ID update (conf-weighted EMA, vectorized) ───────────────────

    def update_by_id(
        self,
        instance_mask_flat: torch.Tensor,             # [N] long
        ins_feat_flat:      torch.Tensor,             # [N, ins_dim] DETACHED
        sem_feat_flat:      Optional[torch.Tensor] = None,  # [N, sem_dim] DETACHED
        ins_conf_flat:      Optional[torch.Tensor] = None,  # [N] or [N,1] in [0,1]
        sem_conf_flat:      Optional[torch.Tensor] = None,  # [N] or [N,1] in [0,1]
    ):
        """
        Conf-weighted EMA update keyed by instance ID.

        Existing IDs:
            F_store = (1 - λ·conf_id) · F_store + λ·conf_id · F_new
        New IDs: initialized from the per-ID mean feature (first-visit).

        Confidence is averaged over all pixels belonging to each ID.
        If conf is None, λ is used as a fixed EMA weight.
        """
        from torch_scatter import scatter_mean

        ids_cpu    = instance_mask_flat.cpu().long()
        valid_mask = ids_cpu != self.ignore_id
        if not valid_mask.any():
            return

        ids_valid = ids_cpu[valid_mask]
        dev       = ins_feat_flat.device

        unique_ids, inverse = torch.unique(ids_valid, return_inverse=True)
        K       = unique_ids.shape[0]
        inv_dev = inverse.to(dev)

        ins_valid = ins_feat_flat[valid_mask]
        ins_k = scatter_mean(ins_valid.float(), inv_dev, dim=0, dim_size=K)  # [K, ins_dim]

        if sem_feat_flat is not None:
            sem_valid = sem_feat_flat[valid_mask]
            sem_k = scatter_mean(sem_valid.float(), inv_dev, dim=0, dim_size=K)
        else:
            sem_k = torch.zeros(K, self.sem_dim, device=dev)

        if ins_conf_flat is not None:
            ic_valid = ins_conf_flat.float().view(-1, 1)[valid_mask].to(dev)
            ins_conf_k = scatter_mean(ic_valid, inv_dev, dim=0, dim_size=K).cpu()
        else:
            ins_conf_k = torch.ones(K, 1)

        if sem_conf_flat is not None:
            sc_valid = sem_conf_flat.float().view(-1, 1)[valid_mask].to(dev)
            sem_conf_k = scatter_mean(sc_valid, inv_dev, dim=0, dim_size=K).cpu()
        else:
            sem_conf_k = torch.ones(K, 1)

        ins_k_cpu      = ins_k.detach().cpu()
        sem_k_cpu      = sem_k.detach().cpu()
        unique_ids_cpu = unique_ids.cpu()

        # ── cold store ────────────────────────────────────────────────────────
        if self._id_keys.shape[0] == 0:
            order = unique_ids_cpu.argsort()
            self._id_keys = unique_ids_cpu[order]
            self._id_sem  = sem_k_cpu[order]
            self._id_ins  = ins_k_cpu[order]
            return

        # ── split into existing (EMA) and new (append) ────────────────────────
        idx_in_store = torch.searchsorted(self._id_keys, unique_ids_cpu)
        idx_clamped  = idx_in_store.clamp(max=self._id_keys.shape[0] - 1)
        existing     = (self._id_keys[idx_clamped] == unique_ids_cpu)

        if existing.any():
            pos     = idx_clamped[existing]
            s_alpha = (self._lambda * sem_conf_k[existing]).clamp(0.0, 1.0)
            i_alpha = (self._lambda * ins_conf_k[existing]).clamp(0.0, 1.0)
            self._id_sem[pos] = ((1.0 - s_alpha) * self._id_sem[pos]
                                 + s_alpha * sem_k_cpu[existing])
            self._id_ins[pos] = ((1.0 - i_alpha) * self._id_ins[pos]
                                 + i_alpha * ins_k_cpu[existing])

        new_mask = ~existing
        if new_mask.any():
            new_keys = unique_ids_cpu[new_mask]
            new_sem  = sem_k_cpu[new_mask]
            new_ins  = ins_k_cpu[new_mask]
            all_keys = torch.cat([self._id_keys, new_keys])
            all_sem  = torch.cat([self._id_sem,  new_sem])
            all_ins  = torch.cat([self._id_ins,  new_ins])
            order = all_keys.argsort()
            self._id_keys = all_keys[order]
            self._id_sem  = all_sem[order]
            self._id_ins  = all_ins[order]

    # ── lifecycle ────────────────────────────────────────────────────────────

    def clear(self):
        self._voxel_keys = torch.empty(0, dtype=torch.long)
        self._voxel_sem  = torch.empty(0, self.sem_dim)
        self._voxel_ins  = torch.empty(0, self.ins_dim)
        self._id_keys    = torch.empty(0, dtype=torch.long)
        self._id_sem     = torch.empty(0, self.sem_dim)
        self._id_ins     = torch.empty(0, self.ins_dim)

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
