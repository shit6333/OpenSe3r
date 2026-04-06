"""
VoxelFeatureMap
===============
Sparse voxel accumulator for large-channel feature vectors (e.g. CLIP-dim=512,
instance-dim=16).

Memory model
------------
Instead of storing features per-frame (num_frames × H × W × C), we bucket all
3-D points into axis-aligned voxels and keep a confidence-weighted running
average per voxel.

Memory cost:  V × C × 4 bytes   where V = number of *occupied* voxels
              (typically 50K–200K for an indoor scene vs >1M frame pixels)

Performance notes
-----------------
* update() processes T×H_feat×W_feat points per SLAM step — small (< 15K/step).
* query() over N points (export stage, up to 6M) uses int64 key encoding +
  numpy dict lookup: ~1.5 s for 6 M points.
* scatter_add_ handles the per-step accumulation in one GPU/CPU kernel call.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Key encoding helpers  (int64, 4× faster than tuple keys)
# ---------------------------------------------------------------------------

_OFFSET = 150_000          # supports ±150 000 voxels per axis
_SCALE  = 300_001          # 2 * _OFFSET + 1

def _encode_coords(coords: torch.Tensor) -> torch.Tensor:
    """
    Encode (N, 3) int64 voxel coordinates to (N,) int64 scalar keys.

    Assumes each coordinate axis is in [-150000, 150000], i.e. the scene fits
    inside a cube of side  150 000 × voxel_size metres.  At voxel_size=0.05 m
    that is ±7 500 m — more than enough for any indoor/outdoor SLAM scenario.
    """
    c = coords + _OFFSET          # shift to non-negative
    return c[:, 0] * (_SCALE * _SCALE) + c[:, 1] * _SCALE + c[:, 2]


# ---------------------------------------------------------------------------
# VoxelFeatureMap
# ---------------------------------------------------------------------------

class VoxelFeatureMap:
    """
    Sparse, growing voxel map that accumulates confidence-weighted features.

    Params:
        voxel_size:        side length of each voxel (metres)
        feat_dim:          channel dimension C of the feature vectors
        initial_capacity:  pre-allocated number of voxel rows (grows as needed)
    """

    def __init__(self, voxel_size: float, feat_dim: int,
                 initial_capacity: int = 100_000):
        self.voxel_size       = float(voxel_size)
        self.feat_dim         = int(feat_dim)

        # int64 scalar key → row index in dense buffers
        self._key_to_idx: dict = {}
        self._num_voxels       = 0

        # Dense buffers (CPU float32, grown via doubling)
        cap                  = initial_capacity
        self._feat_sum        = torch.zeros(cap, feat_dim, dtype=torch.float32)
        self._conf_sum        = torch.zeros(cap,            dtype=torch.float32)
        self._capacity        = cap

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _quantize(self, pts: torch.Tensor) -> torch.Tensor:
        """pts (N,3) float → (N,3) int64 voxel coords."""
        return (pts.float() / self.voxel_size).floor().to(torch.int64)

    def _ensure_capacity(self, needed: int):
        if needed <= self._capacity:
            return
        new_cap  = max(needed, self._capacity * 2)
        new_feat = torch.zeros(new_cap, self.feat_dim, dtype=torch.float32)
        new_conf = torch.zeros(new_cap,                dtype=torch.float32)
        new_feat[:self._num_voxels] = self._feat_sum[:self._num_voxels]
        new_conf[:self._num_voxels] = self._conf_sum[:self._num_voxels]
        self._feat_sum  = new_feat
        self._conf_sum  = new_conf
        self._capacity  = new_cap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self,
               pts:  torch.Tensor,   # (N, 3) float — world-space coords
               feat: torch.Tensor,   # (N, C) float — feature vectors
               conf: torch.Tensor,   # (N,)   float — confidence weights
               ):
        """
        Accumulate features into voxels.  All inputs are moved to CPU float32.

        The update rule per voxel v is:
            feat_sum[v] += conf[i] * feat[i]   for all points i that map to v
            conf_sum[v] += conf[i]
        Averaged feature: feat_sum[v] / conf_sum[v]
        """
        pts  = pts.detach().float().cpu()
        feat = feat.detach().float().cpu()
        conf = conf.detach().float().cpu().view(-1)    # ensure (N,)

        if pts.shape[0] == 0:
            return

        coords  = self._quantize(pts)                  # (N, 3)
        keys    = _encode_coords(coords)               # (N,) int64 tensor
        keys_np = keys.numpy()                         # numpy for dict ops

        # ---- Discover new voxels -----------------------------------------
        new_in_batch: dict = {}                        # key -> tentative idx
        for k in keys_np:
            k = int(k)
            if k not in self._key_to_idx and k not in new_in_batch:
                new_in_batch[k] = self._num_voxels + len(new_in_batch)

        self._ensure_capacity(self._num_voxels + len(new_in_batch))
        self._key_to_idx.update(new_in_batch)
        self._num_voxels += len(new_in_batch)

        # ---- Map each point to its row index ---------------------------------
        row_idx = torch.tensor(
            [self._key_to_idx[int(k)] for k in keys_np], dtype=torch.long
        )                                              # (N,)

        # ---- Weighted accumulation via scatter_add_ -------------------------
        # feat_sum[v] += conf[i] * feat[i]
        conf_w = conf.unsqueeze(-1) * feat             # (N, C)
        self._feat_sum.scatter_add_(
            0,
            row_idx.unsqueeze(-1).expand(-1, self.feat_dim),
            conf_w,
        )
        self._conf_sum.scatter_add_(0, row_idx, conf)

    @torch.no_grad()
    def query(self, pts: torch.Tensor) -> torch.Tensor:
        """
        For each query point, return the confidence-averaged feature of the
        voxel it falls into.  Points with no matching voxel get zero vectors.

        pts:     (N, 3) float
        returns: (N, C) float32 (cpu)
        """
        if self._num_voxels == 0:
            return torch.zeros(len(pts), self.feat_dim, dtype=torch.float32)

        pts    = pts.detach().float().cpu()
        coords = self._quantize(pts)
        keys   = _encode_coords(coords)
        keys_np = keys.numpy()

        # Averaged features over occupied voxels
        feat_avg = (
            self._feat_sum[:self._num_voxels]
            / (self._conf_sum[:self._num_voxels, None] + 1e-8)
        )                                              # (V, C)

        # Vectorized lookup — int key dict is 4× faster than tuple
        row_idx = np.array(
            [self._key_to_idx.get(int(k), -1) for k in keys_np],
            dtype=np.int64,
        )
        row_tensor = torch.from_numpy(row_idx)         # (N,)

        out   = torch.zeros(len(pts), self.feat_dim, dtype=torch.float32)
        valid = row_tensor >= 0
        if valid.any():
            out[valid] = feat_avg[row_tensor[valid]]
        return out

    # ------------------------------------------------------------------
    def get_all(self) -> tuple:
        """
        Return all occupied voxels as (centers, features).

        centers:  (V, 3) float32  — voxel centre world coordinates
        features: (V, C) float32  — confidence-averaged feature vectors
        """
        if self._num_voxels == 0:
            return (
                torch.zeros(0, 3,            dtype=torch.float32),
                torch.zeros(0, self.feat_dim, dtype=torch.float32),
            )

        # Keys are stored in insertion order (Python ≥ 3.7)
        # _key_to_idx[k] == row index, so sort by value to match feat_sum rows
        sorted_items = sorted(self._key_to_idx.items(), key=lambda x: x[1])
        raw_keys = torch.tensor([item[0] for item in sorted_items], dtype=torch.int64)  # (V,)

        # Decode int64 keys back to (V, 3) integer coords
        z = raw_keys % _SCALE - _OFFSET
        y = (raw_keys // _SCALE) % _SCALE - _OFFSET
        x = raw_keys // (_SCALE * _SCALE) - _OFFSET
        int_coords = torch.stack([x, y, z], dim=1).float()               # (V, 3)
        centers    = (int_coords + 0.5) * self.voxel_size                 # voxel centres

        feat_avg = (
            self._feat_sum[:self._num_voxels]
            / (self._conf_sum[:self._num_voxels, None] + 1e-8)
        )

        return centers, feat_avg

    @property
    def num_voxels(self) -> int:
        return self._num_voxels

    def __repr__(self) -> str:
        return (
            f"VoxelFeatureMap(voxel_size={self.voxel_size}, "
            f"feat_dim={self.feat_dim}, "
            f"num_voxels={self._num_voxels})"
        )