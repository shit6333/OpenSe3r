"""
voxel_store_vis.py — PCA visualization export for voxel_store.

Decodes the hashed voxel keys in a DetachedVoxelStore back to 3-D voxel
centres, runs PCA on the stored features, and writes a coloured PLY.

Works with:
    - DetachedVoxelStore  (hash-keyed; used by mem_mode 1, 2, 3)
    - DifferentiableVoxelMap  (get_all() interface; mem_mode 0)
"""

import os
import numpy as np
import torch
import trimesh
from sklearn.decomposition import PCA


# Must match DetachedVoxelStore._P in amb3r/memory_stage2v4.py
_P_DEFAULT = 1_000_003


def _decode_voxel_keys(keys: torch.Tensor,
                        voxel_size: float,
                        P: int = _P_DEFAULT) -> torch.Tensor:
    """
    Inverse of DetachedVoxelStore._encode_keys().

    Encoding:   key = (vx+H)*P*P + (vy+H)*P + (vz+H),   H = P // 2
    Decoding:   vz = key % P - H
                vy = (key // P) % P - H
                vx = (key // (P*P)) - H
    Centre:     (vc + 0.5) * voxel_size
    """
    H = P // 2
    k = keys.long().cpu()
    vz = (k % P) - H
    vy = ((k // P) % P) - H
    vx = (k // (P * P)) - H
    vc = torch.stack([vx, vy, vz], dim=-1).float()
    return (vc + 0.5) * voxel_size


def feature_to_pca_color(feat: torch.Tensor) -> np.ndarray:
    """[N, C] → [N, 3] uint8 via PCA + min-max normalisation."""
    if feat.ndim != 2 or feat.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    x = feat.detach().float().cpu().numpy()
    n = x.shape[0]
    n_comp = 3 if n >= 3 else 1
    pca = PCA(n_components=n_comp)
    z = pca.fit_transform(x)
    if n_comp == 1:
        z = np.tile(z, (1, 3))
    lo = z.min(axis=0, keepdims=True)
    hi = z.max(axis=0, keepdims=True)
    rng = np.maximum(hi - lo, 1e-6)
    return np.clip((z - lo) / rng * 255.0, 0, 255).astype(np.uint8)


def _apply_transform(points: torch.Tensor,
                     transformation=None) -> torch.Tensor:
    """Apply optional 4x4 similarity to [N, 3] points."""
    if transformation is None:
        return points
    if isinstance(transformation, np.ndarray):
        T = torch.from_numpy(transformation).float()
    else:
        T = transformation.float()
    ones = torch.ones(points.shape[0], 1, dtype=points.dtype)
    ph = torch.cat([points.float(), ones], dim=1)
    return (T @ ph.T).T[:, :3]


def export_voxel_store_pca_ply(
    voxel_store,
    save_file: str,
    transformation=None,     # optional (4, 4) numpy or torch
):
    """
    Export voxel_store as a PCA-coloured point cloud PLY.

    Parameters
    ----------
    voxel_store    DetachedVoxelStore or DifferentiableVoxelMap
    save_file      output path (.ply)
    transformation optional (4, 4) similarity applied to voxel centres
                   (use this e.g. to align with T_voxel in run.py)
    """
    # ── Extract centres + features ───────────────────────────────────────
    # DetachedVoxelStore path: reconstruct centres from hash
    if hasattr(voxel_store, '_voxel_keys') and hasattr(voxel_store, '_voxel_feats'):
        keys  = voxel_store._voxel_keys
        feats = voxel_store._voxel_feats
        if keys.shape[0] == 0:
            print(f"[voxel_store_vis] store is empty, skipping {save_file}")
            return
        centers = _decode_voxel_keys(keys, voxel_store.voxel_size)

    # DifferentiableVoxelMap path: get_all()
    elif hasattr(voxel_store, 'get_all'):
        centers, feats = voxel_store.get_all()
        if centers.shape[0] == 0:
            print(f"[voxel_store_vis] store is empty, skipping {save_file}")
            return

    else:
        raise ValueError(f'Unsupported voxel_store type: {type(voxel_store)}')

    # ── Transform & colour ───────────────────────────────────────────────
    centers = _apply_transform(centers, transformation)
    colors  = feature_to_pca_color(feats)

    # ── Write PLY ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
    pc = trimesh.points.PointCloud(
        centers.numpy().astype(np.float32),
        colors=colors,
    )
    pc.export(save_file)
    print(f"[voxel_store_vis] saved {centers.shape[0]:,} voxels "
          f"(feat_dim={feats.shape[-1]}) → {save_file}")
