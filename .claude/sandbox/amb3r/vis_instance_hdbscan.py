"""
vis_instance_hdbscan.py
=======================
In-memory HDBSCAN clustering helpers for training-time visualization.

Designed to be called inside test_one_epoch() alongside the existing
export_back_semantic_pca_ply() calls.  Accepts torch tensors or numpy arrays
directly — no file I/O required for the features.

Voxel downsampling
------------------
With 500k–1M points HDBSCAN is far too slow.  The pipeline is:

  1. Voxelize  : snap each point to a voxel grid (voxel_size metres).
  2. Aggregate : average features and positions within each voxel
                 (fully vectorized via np.unique + np.bincount — no Python loops).
  3. HDBSCAN   : cluster on the ~10k–50k voxel centroids.
  4. Project   : assign every original point the label of its voxel.

Main entry point
----------------
    export_instance_hdbscan_ply(
        points_xyz,      # [N, 3]  tensor or ndarray
        instance_feat,   # [N, C]  tensor or ndarray  (e.g. C=16)
        save_file,       # str — output .ply path
        voxel_size=0.05, # metres; None to skip downsampling
        **kwargs,        # min_cluster_size, min_samples, normalize, use_pca, ...
    )

Usage in train_stage1.py test_one_epoch vis block
-------------------------------------------------
    from amb3r.vis_instance_hdbscan import export_instance_hdbscan_ply

    if 'instance_feat' in pred:
        pts_flat = pred['world_points'].reshape(-1, 3)
        ins_feat = (pred['instance_feat']
                    .permute(0, 1, 3, 4, 2)
                    .reshape(-1, pred['instance_feat'].shape[2]))
        # existing PCA vis
        export_back_semantic_pca_ply(
            points_xyz=pts_flat, semantic_feat=ins_feat,
            save_file=os.path.join(save_path, f'{dataset_name}_ins_pca_{i}.ply'),
        )
        # HDBSCAN clustering vis (with voxel downsampling)
        export_instance_hdbscan_ply(
            points_xyz=pts_flat, instance_feat=ins_feat,
            save_file=os.path.join(save_path, f'{dataset_name}_ins_hdbscan_{i}.ply'),
            voxel_size=0.05,
            verbose=True,
        )
"""

import colorsys
import warnings
import numpy as np


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def _labels_to_colors(
    labels: np.ndarray,
    noise_color: tuple = (80, 80, 80),
) -> np.ndarray:
    """
    Map HDBSCAN integer labels -> uint8 RGB colors.

    label == -1  (noise)  ->  noise_color
    label >= 0   (cluster) -> golden-angle hue cycling for visual separation
    """
    golden = 0.6180339887
    colors = np.full((len(labels), 3), noise_color, dtype=np.uint8)
    unique_ids = np.unique(labels)
    cluster_ids = unique_ids[unique_ids >= 0]
    for rank, cid in enumerate(cluster_ids):
        hue = (rank * golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors[labels == cid] = [int(r * 255), int(g * 255), int(b * 255)]
    return colors


# ---------------------------------------------------------------------------
# Voxel downsampling  (fully vectorized — no Python loops over voxels)
# ---------------------------------------------------------------------------

def voxel_downsample(
    points_xyz: np.ndarray,
    feat: np.ndarray,
    voxel_size: float,
) -> tuple:
    """
    Voxel-grid downsample points and features.

    Algorithm
    ---------
    1. Quantise each point to a voxel cell: coord = floor(xyz / voxel_size).
    2. Encode (cx, cy, cz) into a single int64 key.
    3. np.unique(keys, return_inverse=True)  ->  voxel index per point.
    4. np.bincount(inv, weights=...)         ->  vectorised sum per voxel.
    5. Divide by count -> mean xyz and mean feature per voxel.

    No Python-level loops — scales to 1M+ points in < 1 second.

    Args:
        points_xyz : [N, 3]  float32
        feat       : [N, C]  float32
        voxel_size : cell edge length (same units as points_xyz)

    Returns:
        vox_xyz   : [M, 3]  float32  — mean position per voxel
        vox_feat  : [M, C]  float32  — mean feature per voxel
        point2vox : [N]     int32    — voxel index for each original point
    """
    N, C = feat.shape

    # 1. quantise
    coords = np.floor(points_xyz / voxel_size).astype(np.int64)   # [N, 3]

    # 2. shift to non-negative and encode to a single key
    min_c          = coords.min(axis=0)
    coords_shifted = coords - min_c                                # [N, 3]
    max_dim        = coords_shifted.max(axis=0) + 1               # [3]
    stride         = np.array([max_dim[1] * max_dim[2], max_dim[2], 1], dtype=np.int64)
    keys           = coords_shifted @ stride                       # [N]

    # 3. unique -> voxel id per point
    _, inv, cnt_per_vox = np.unique(keys, return_inverse=True, return_counts=True)
    # inv[i]  = voxel index of point i
    # cnt_per_vox[v] = number of points in voxel v
    M = len(cnt_per_vox)
    inv = inv.astype(np.int32)

    # 4. vectorised sum via bincount
    # xyz  (3 channels)
    vox_sum_xyz = np.stack([
        np.bincount(inv, weights=points_xyz[:, d], minlength=M)
        for d in range(3)
    ], axis=1)   # [M, 3]

    # features (C channels) — stack bincount over each channel
    vox_sum_feat = np.stack([
        np.bincount(inv, weights=feat[:, c], minlength=M)
        for c in range(C)
    ], axis=1)   # [M, C]

    # 5. mean
    w = cnt_per_vox[:, None].astype(np.float64)
    vox_xyz  = (vox_sum_xyz  / w).astype(np.float32)
    vox_feat = (vox_sum_feat / w).astype(np.float32)

    return vox_xyz, vox_feat, inv


# ---------------------------------------------------------------------------
# Core clustering
# ---------------------------------------------------------------------------

def hdbscan_on_features(
    feat: np.ndarray,
    min_cluster_size: int = 30,
    min_samples: int = None,
    cluster_selection_epsilon: float = 0.0,
    normalize: bool = True,
    use_pca: int = None,
) -> np.ndarray:
    """
    Run HDBSCAN on feature matrix.

    Args:
        feat                      : [N, C] float32
        min_cluster_size          : minimum points to form a cluster
        min_samples               : HDBSCAN min_samples (defaults to min_cluster_size)
        cluster_selection_epsilon : merge clusters within this distance
        normalize                 : L2-normalise features before clustering
        use_pca                   : if set, reduce to this many dims via PCA first

    Returns:
        labels: [N] int  (-1 = noise, 0..K-1 = cluster id)
    """
    feat = feat.astype(np.float32)

    if normalize:
        norms = np.linalg.norm(feat, axis=1, keepdims=True).clip(min=1e-8)
        feat  = feat / norms

    if use_pca is not None and use_pca < feat.shape[1]:
        try:
            from sklearn.decomposition import PCA
            n_comp = min(use_pca, feat.shape[1], feat.shape[0])
            feat   = PCA(n_components=n_comp, random_state=42).fit_transform(feat)
        except ImportError:
            warnings.warn("sklearn not available — skipping PCA reduction.")

    _min_samples = min_samples if min_samples is not None else min_cluster_size

    # Try sklearn.cluster.HDBSCAN (sklearn >= 1.3) first, then hdbscan package
    try:
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=_min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            n_jobs=-1,
        )
        labels = clusterer.fit_predict(feat)
    except (ImportError, AttributeError):
        try:
            import hdbscan as hdbscan_pkg
            clusterer = hdbscan_pkg.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=_min_samples,
                core_dist_n_jobs=-1,
            )
            labels = clusterer.fit_predict(feat)
        except ImportError:
            warnings.warn(
                "Neither sklearn>=1.3 HDBSCAN nor hdbscan package found. "
                "All points labelled as noise (-1)."
            )
            labels = np.full(len(feat), -1, dtype=np.int32)

    return labels.astype(np.int32)


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_instance_hdbscan_ply(
    points_xyz,
    instance_feat,
    save_file: str,
    voxel_size: float = 0.05,
    min_cluster_size: int = 30,
    min_samples: int = None,
    cluster_selection_epsilon: float = 0.0,
    normalize: bool = True,
    use_pca: int = None,
    noise_color: tuple = (80, 80, 80),
    verbose: bool = False,
) -> dict:
    """
    Cluster instance features with HDBSCAN and save a colored PLY.

    With large point clouds (500k–1M pts) HDBSCAN is too slow to run directly.
    When voxel_size is set, the pipeline voxelizes the cloud, runs HDBSCAN on
    the voxel centroids (~10k–50k), then projects labels back to every original
    point (each point gets the color of its voxel).

    Args:
        points_xyz      : [N, 3]  torch.Tensor or np.ndarray — 3-D positions
        instance_feat   : [N, C]  torch.Tensor or np.ndarray — instance features
        save_file       : output .ply path
        voxel_size      : voxel grid cell size (same units as points_xyz, e.g. 0.05 m).
                          Set to None to skip downsampling (only for small clouds).
        min_cluster_size: minimum voxels per cluster after downsampling
        min_samples     : HDBSCAN core-point threshold (None -> same as min_cluster_size)
        cluster_selection_epsilon: merge nearby clusters (0 = disabled)
        normalize       : L2-normalise features before clustering
        use_pca         : optional PCA pre-reduction dimension (None = skip)
        noise_color     : RGB tuple for noise points (label -1)
        verbose         : print clustering summary

    Returns:
        info dict  {num_clusters, num_noise, noise_ratio, N_original, N_voxels}
    """
    import trimesh

    # ---- Convert to numpy -------------------------------------------------
    try:
        import torch
        if isinstance(points_xyz, torch.Tensor):
            points_xyz = points_xyz.detach().float().cpu().numpy()
        if isinstance(instance_feat, torch.Tensor):
            instance_feat = instance_feat.detach().float().cpu().numpy()
    except ImportError:
        pass

    points_xyz    = np.asarray(points_xyz,    dtype=np.float32)
    instance_feat = np.asarray(instance_feat, dtype=np.float32)

    N = len(points_xyz)
    assert len(instance_feat) == N, (
        f"points_xyz has {N} rows but instance_feat has {len(instance_feat)}"
    )

    if N == 0:
        warnings.warn(f"export_instance_hdbscan_ply: empty input, skipping {save_file}")
        return {"num_clusters": 0, "num_noise": 0, "noise_ratio": 0.0,
                "N_original": 0, "N_voxels": 0}

    # ---- Optional voxel downsampling --------------------------------------
    point2vox  = None
    N_voxels   = N

    if voxel_size is not None and voxel_size > 0:
        vox_xyz, vox_feat, point2vox = voxel_downsample(
            points_xyz, instance_feat, voxel_size
        )
        N_voxels     = len(vox_xyz)
        cluster_feat = vox_feat
        if verbose:
            print(f"[HDBSCAN] voxel downsample: {N:,} pts -> {N_voxels:,} voxels "
                  f"(voxel_size={voxel_size})")
    else:
        cluster_feat = instance_feat

    # ---- HDBSCAN on (downsampled) features --------------------------------
    vox_labels = hdbscan_on_features(
        cluster_feat,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        normalize=normalize,
        use_pca=use_pca,
    )

    # ---- Project voxel labels back to original points ---------------------
    labels = vox_labels[point2vox] if point2vox is not None else vox_labels

    # ---- Summary ----------------------------------------------------------
    unique_labels = np.unique(labels)
    num_clusters  = int((unique_labels >= 0).sum())
    num_noise     = int((labels == -1).sum())
    noise_ratio   = num_noise / max(N, 1)

    if verbose:
        print(f"[HDBSCAN] {save_file}")
        print(f"  original={N:,}  voxels={N_voxels:,}  "
              f"clusters={num_clusters}  noise={num_noise:,} ({noise_ratio*100:.1f}%)")

    # ---- Colors for original points ---------------------------------------
    colors = _labels_to_colors(labels, noise_color=noise_color)

    # ---- Save PLY ---------------------------------------------------------
    trimesh.points.PointCloud(
        points_xyz, colors=colors
    ).export(save_file)

    return {
        "num_clusters": num_clusters,
        "num_noise":    num_noise,
        "noise_ratio":  noise_ratio,
        "N_original":   N,
        "N_voxels":     N_voxels,
    }
