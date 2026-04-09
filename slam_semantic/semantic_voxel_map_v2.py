"""
VoxelFeatureMapV2
=================
Extends VoxelFeatureMap (slam_semantic/semantic_voxel_map.py) with:
    query_conf(pts, conf_scale) → (N, 1) soft confidence mask in [0, 1]

Used by Stage-2 to tell the memory heads whether a voxel has been observed.
"""

import sys
import os
import torch
import numpy as np

# allow importing from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from slam_semantic.semantic_voxel_map import VoxelFeatureMap, _encode_coords


class VoxelFeatureMapV2(VoxelFeatureMap):
    """
    Adds query_conf() on top of the base VoxelFeatureMap.

    conf_scale : float
        The accumulated confidence at which the soft mask saturates to 1.
        Voxels with conf_sum >= conf_scale will return mask ≈ 1.0.
        Unseen voxels always return 0.0.
    """

    def __init__(self, voxel_size: float, feat_dim: int,
                 initial_capacity: int = 100_000,
                 conf_scale: float = 1.0):
        super().__init__(voxel_size=voxel_size, feat_dim=feat_dim,
                         initial_capacity=initial_capacity)
        self.conf_scale = float(conf_scale)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def query_conf(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Return soft confidence mask for each query point.

            mask[i] = clamp( conf_sum[voxel(pts[i])] / conf_scale, 0, 1 )

        Points whose voxel has never been updated get mask = 0.0.

        pts    : (N, 3) float
        returns: (N, 1) float32 cpu, values in [0, 1]
        """
        if self._num_voxels == 0:
            return torch.zeros(len(pts), 1, dtype=torch.float32)

        pts_cpu = pts.detach().float().cpu()
        coords  = self._quantize(pts_cpu)
        keys    = _encode_coords(coords)
        keys_np = keys.numpy()

        row_idx = np.array(
            [self._key_to_idx.get(int(k), -1) for k in keys_np],
            dtype=np.int64,
        )
        row_tensor = torch.from_numpy(row_idx)    # (N,)

        out   = torch.zeros(len(pts), 1, dtype=torch.float32)
        valid = row_tensor >= 0
        if valid.any():
            conf_vals = self._conf_sum[row_tensor[valid]]      # (M,)
            out[valid, 0] = (conf_vals / self.conf_scale).clamp(0.0, 1.0)
        return out
