"""
instance_hdbscan_cluster.py
===========================
HDBSCAN clustering on instance voxel features exported from AMB3R semantic SLAM.

Clustering is done at **voxel level** (fast), but the final coloured output is
rendered at **per-point level** by mapping each geometry point to its nearest
voxel via a KD-tree.

Input files:
    <feats_npz>   — scene_<name>_instance_voxel_feats.npz
                    keys: voxel_centers (V,3), voxel_features (V,16)
    <results_npz> — scene_<name>_results.npz
                    keys: pts (T,H,W,3), conf (T,H,W), kf_idx (K,), sky_mask (T,H,W)

Outputs:
    instance_hdbscan_perpoint.ply  — per-point cloud coloured by instance label
    instance_hdbscan_voxels.ply    — voxel-centre cloud coloured by instance label
    instance_hdbscan_result.json   — clustering analysis

Usage:
    python instance_hdbscan_cluster.py \\
        --feats_npz   /path/to/scene_X_instance_voxel_feats.npz \\
        --results_npz /path/to/scene_X_results.npz \\
        --output      /path/to/output_dir \\
        [--conf_threshold 0.0] \\
        [--min_cluster_size 50] \\
        [--min_samples 10] \\
        [--use_pca 8]
"""

import argparse
import json
import os
import time

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def distinct_colors(n: int) -> np.ndarray:
    """Return (n, 3) float32 colors in [0,1] using HSV wheel."""
    import colorsys
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append([r, g, b])
    return np.array(colors, dtype=np.float32)


def save_ply(path: str, points: np.ndarray, colors: np.ndarray):
    """Write a PLY point cloud (open3d if available, else ASCII fallback)."""
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        o3d.io.write_point_cloud(path, pcd)
    except ImportError:
        N = len(points)
        with open(path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {N}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            rgb255 = (colors * 255).clip(0, 255).astype(np.uint8)
            for i in range(N):
                x, y, z = points[i]
                r, g, b = rgb255[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def load_per_point_geometry(results_npz: str,
                            conf_threshold: float = 0.0) -> np.ndarray:
    """
    Reconstruct the keyframe per-point set from results.npz.

    Returns pts_kf: (N, 3) float32 — masked keyframe points, same space as
    the voxel centers stored in the feature npz.
    """
    data     = np.load(results_npz)
    pts      = data['pts']        # (T, H, W, 3)
    conf     = data['conf']       # (T, H, W)
    kf_idx   = data['kf_idx']     # (K,) int

    pts_kf   = pts[kf_idx].reshape(-1, 3).astype(np.float32)
    conf_kf  = conf[kf_idx].reshape(-1)
    mask     = conf_kf >= conf_threshold

    if 'sky_mask' in data:
        sky_kf = data['sky_mask'][kf_idx].reshape(-1).astype(bool)
        mask   = mask & (~sky_kf)

    return pts_kf[mask]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HDBSCAN instance clustering — per-point output"
    )
    parser.add_argument('--feats_npz',   required=True,
                        help='scene_X_instance_voxel_feats.npz  (voxel_centers + voxel_features)')
    parser.add_argument('--results_npz', required=True,
                        help='scene_X_results.npz  (pts, conf, kf_idx, sky_mask)')
    parser.add_argument('--output',      default='.',
                        help='Output directory')
    parser.add_argument('--conf_threshold', type=float, default=0.0,
                        help='Confidence threshold for per-point reconstruction (default: 0.0)')
    parser.add_argument('--min_cluster_size', type=int, default=50)
    parser.add_argument('--min_samples',      type=int, default=None)
    parser.add_argument('--cluster_selection_epsilon', type=float, default=0.0)
    parser.add_argument('--use_pca',     type=int, default=None,
                        help='PCA dim reduction before clustering (e.g. 8)')
    parser.add_argument('--normalize',   action='store_true', default=True,
                        help='L2-normalise features before clustering (default: True)')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.add_argument('--max_points', type=int, default=6_000_000,
                        help='Max per-point output count; random-sampled if exceeded (default: 6 000 000)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Load voxel features ───────────────────────────────────────────────────
    print("Loading voxel features …")
    feats_data      = np.load(args.feats_npz)
    voxel_centers   = feats_data['voxel_centers'].astype(np.float32)   # (V, 3)
    voxel_features  = feats_data['voxel_features'].astype(np.float32)  # (V, C)
    V, C = voxel_features.shape
    print(f"  Voxels: {V:,}   Feature dim: {C}")

    if V == 0:
        print("No voxels — nothing to cluster.")
        return

    # ── Load per-point geometry ───────────────────────────────────────────────
    print("Loading per-point geometry from results.npz …")
    pts_kf = load_per_point_geometry(args.results_npz, args.conf_threshold)
    print(f"  Per-point count: {len(pts_kf):,}")

    # ── Downsample per-point cloud if needed ─────────────────────────────────
    if args.max_points > 0 and len(pts_kf) > args.max_points:
        print(f"  Downsampling: {len(pts_kf):,} → {args.max_points:,} (random)")
        rng = np.random.default_rng(seed=42)
        keep = rng.choice(len(pts_kf), args.max_points, replace=False)
        keep.sort()
        pts_kf = pts_kf[keep]

    # ── Pre-processing ────────────────────────────────────────────────────────
    feat = voxel_features.copy()

    if args.normalize:
        norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        feat  = feat / norms
        print("  L2-normalised features.")

    if args.use_pca is not None and args.use_pca < C:
        from sklearn.decomposition import PCA
        n_comp = min(args.use_pca, C, V)
        print(f"  PCA: {C} → {n_comp} dims …")
        pca  = PCA(n_components=n_comp, random_state=42)
        feat = pca.fit_transform(feat).astype(np.float32)
        explained = pca.explained_variance_ratio_.sum() * 100
        print(f"  PCA explained variance: {explained:.1f}%")

    # ── HDBSCAN on voxels ─────────────────────────────────────────────────────
    min_samples = args.min_samples if args.min_samples is not None else args.min_cluster_size

    try:
        from sklearn.cluster import HDBSCAN
        print(f"\nRunning HDBSCAN (sklearn) — min_cluster_size={args.min_cluster_size}, "
              f"min_samples={min_samples}, eps={args.cluster_selection_epsilon} …")
        t0 = time.time()
        clusterer = HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            n_jobs=-1,
        )
        voxel_labels = clusterer.fit_predict(feat)
    except (ImportError, AttributeError):
        import hdbscan as hdbscan_pkg
        print(f"\nRunning HDBSCAN (hdbscan pkg) — min_cluster_size={args.min_cluster_size}, "
              f"min_samples={min_samples} …")
        t0 = time.time()
        clusterer = hdbscan_pkg.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=min_samples,
            core_dist_n_jobs=-1,
        )
        voxel_labels = clusterer.fit_predict(feat)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s")

    # ── Cluster summary ───────────────────────────────────────────────────────
    unique_labels = np.unique(voxel_labels)
    instance_ids  = unique_labels[unique_labels >= 0]
    noise_mask_v  = voxel_labels == -1
    num_instances = len(instance_ids)
    num_noise_v   = int(noise_mask_v.sum())
    print(f"\nClustering result (voxels):")
    print(f"  Instances found : {num_instances}")
    print(f"  Noise voxels    : {num_noise_v:,}  ({100*num_noise_v/V:.1f}%)")

    # ── Map voxel labels → per-point via KD-tree ──────────────────────────────
    print("\nMapping voxel labels to per-point geometry via KD-tree …")
    t1 = time.time()
    try:
        from sklearn.neighbors import KDTree
        tree      = KDTree(voxel_centers)
        nn_idx    = tree.query(pts_kf, k=1, return_distance=False).ravel()  # (N,)
    except ImportError:
        from scipy.spatial import cKDTree
        tree   = cKDTree(voxel_centers)
        _, nn_idx = tree.query(pts_kf, k=1)
    point_labels = voxel_labels[nn_idx]   # (N,) — inherits voxel label
    print(f"  KD-tree query done in {time.time()-t1:.2f}s")

    # ── Build per-instance statistics (voxel level) ───────────────────────────
    palette      = distinct_colors(max(num_instances, 1))
    voxel_colors = np.ones((V, 3), dtype=np.float32) * 0.3   # noise → dark grey
    point_colors = np.ones((len(pts_kf), 3), dtype=np.float32) * 0.3

    instances_info = []
    for rank, iid in enumerate(instance_ids):
        vmask = voxel_labels == iid
        pmask = point_labels == iid
        pts_i = voxel_centers[vmask]

        center_xyz = pts_i.mean(axis=0).tolist()
        bbox_min   = pts_i.min(axis=0).tolist()
        bbox_max   = pts_i.max(axis=0).tolist()
        bbox_size  = (pts_i.max(axis=0) - pts_i.min(axis=0)).tolist()
        volume_m3  = float(np.prod(bbox_size))
        mean_feat  = voxel_features[vmask].mean(axis=0)

        color = palette[rank % len(palette)]
        voxel_colors[vmask] = color
        point_colors[pmask] = color

        instances_info.append({
            "instance_id"    : int(iid),
            "num_voxels"     : int(vmask.sum()),
            "num_points"     : int(pmask.sum()),
            "center_xyz"     : [round(v, 4) for v in center_xyz],
            "bbox_min"       : [round(v, 4) for v in bbox_min],
            "bbox_max"       : [round(v, 4) for v in bbox_max],
            "bbox_size_xyz"  : [round(v, 4) for v in bbox_size],
            "bbox_volume_m3" : round(volume_m3, 6),
            "mean_feature"   : [round(float(v), 6) for v in mean_feat.tolist()],
            "color_rgb"      : [round(float(v), 4) for v in color.tolist()],
        })

    instances_info.sort(key=lambda x: x['num_points'], reverse=True)
    for rank, inst in enumerate(instances_info):
        inst['rank_by_points'] = rank + 1

    # ── PLY outputs ───────────────────────────────────────────────────────────
    # Per-point (primary output)
    pp_ply = os.path.join(args.output, "instance_hdbscan_perpoint.ply")
    print(f"\nSaving per-point PLY  → {pp_ply}  ({len(pts_kf):,} pts)")
    save_ply(pp_ply, pts_kf, point_colors)

    # Voxel-level (for reference)
    vox_ply = os.path.join(args.output, "instance_hdbscan_voxels.ply")
    print(f"Saving voxel PLY      → {vox_ply}  ({V:,} voxels)")
    save_ply(vox_ply, voxel_centers, voxel_colors)

    # ── JSON output ───────────────────────────────────────────────────────────
    noise_pts = int((point_labels == -1).sum())
    summary = {
        "num_voxels_total"  : int(V),
        "num_points_total"  : int(len(pts_kf)),
        "feature_dim"       : int(C),
        "num_instances"     : num_instances,
        "num_noise_voxels"  : num_noise_v,
        "num_noise_points"  : noise_pts,
        "noise_point_ratio" : round(noise_pts / max(len(pts_kf), 1), 4),
        "hdbscan_params"    : {
            "min_cluster_size"          : args.min_cluster_size,
            "min_samples"               : min_samples,
            "cluster_selection_epsilon" : args.cluster_selection_epsilon,
            "normalize_features"        : args.normalize,
            "pca_dims"                  : args.use_pca,
            "conf_threshold"            : args.conf_threshold,
        },
        "instances"         : instances_info,
    }

    json_path = os.path.join(args.output, "instance_hdbscan_result.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saving JSON           → {json_path}")

    # ── Print top-10 ─────────────────────────────────────────────────────────
    print(f"\nTop instances by point count:")
    print(f"  {'rank':>4}  {'id':>6}  {'voxels':>8}  {'points':>8}  {'center (x,y,z)':>32}")
    for inst in instances_info[:10]:
        cx, cy, cz = inst['center_xyz']
        print(f"  {inst['rank_by_points']:>4}  {inst['instance_id']:>6}  "
              f"{inst['num_voxels']:>8,}  {inst['num_points']:>8,}  "
              f"({cx:7.3f}, {cy:7.3f}, {cz:7.3f})")

    print("\nDone.")


if __name__ == '__main__':
    main()
