"""
eval_semantic_slam.py
---------------------
Evaluate semantic SLAM results on ScanNet data.

Metrics:
  - Camera pose : ATE RMSE (via evo, Umeyama scale alignment, monocular)
  - 3D semantic : mIoU / mAcc / freq-mIoU  (KD-Tree GT->pred, k=5 majority vote)

Scale alignment:
  The pred map is already coordinate-aligned to GT via the first-frame pose
  (done in slam_semantic/run.py). Scale is still off (non-metric model).
  We estimate scale via Umeyama alignment of pred vs GT camera trajectories,
  then apply the full similarity transform (s, R, t) to the pred semantic
  point cloud before 3D evaluation.

  Use --metric_scale to skip alignment when pred is already metric.

Requirements:
  - slam_semantic/run.py must have been run with --save_semantic --demo_type scannet
  - amb3r/tools/semantic_vis_utils.py must be updated to export a 'label' field
    in the semantic PLYs (pred_idx + 1, scannet20 sequential 1-indexed 1~20)

Usage:
    python eval_semantic_slam.py \
        --results_path  outputs/slam/scene0050_00 \
        --scene_name    scene0050_00 \
        --data_path     /data/ScanNet/scene0050_00 \
        --gt_labels_ply /data/ScanNet/scene0050_00/scene0050_00_vh_clean_2.labels.ply \
        [--metric_scale]         # skip scale alignment (pred is already metric)
        [--use_voxel_ply]        # use semantic_voxels.ply instead of per_point
        [--k 5]                  # KD-Tree neighbours (default: 5)
        [--max_dist 0.5]         # ignore GT pts farther than this (metres)
        [--no_pose_eval]         # skip pose eval
        [--save_csv results.csv]
"""

import sys
import os
import argparse
import numpy as np
import plyfile
from scipy.spatial import cKDTree
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ── ScanNet20 class names (1-indexed; 0 = unlabeled / ignored) ──────────────
SCANNET20_NAMES = [
    "unlabeled",        # 0  → ignored
    "wall",             # 1
    "floor",            # 2
    "cabinet",          # 3
    "bed",              # 4
    "chair",            # 5
    "sofa",             # 6
    "table",            # 7
    "door",             # 8
    "window",           # 9
    "bookshelf",        # 10
    "picture",          # 11
    "counter",          # 12
    "desk",             # 13
    "curtain",          # 14
    "refrigerator",     # 15
    "shower curtain",   # 16
    "toilet",           # 17
    "sink",             # 18
    "bathtub",          # 19
    "otherfurniture",   # 20
]

# ── NYU40 → ScanNet20 sequential (1-20) ─────────────────────────────────────
NYU40_TO_SCANNET20 = {
    1:  1,   # wall
    2:  2,   # floor
    3:  3,   # cabinet
    4:  4,   # bed
    5:  5,   # chair
    6:  6,   # sofa
    7:  7,   # table
    8:  8,   # door
    9:  9,   # window
    10: 10,  # bookshelf
    11: 11,  # picture
    12: 12,  # counter
    14: 13,  # desk
    16: 14,  # curtain
    24: 15,  # refrigerator
    28: 16,  # shower curtain
    33: 17,  # toilet
    34: 18,  # sink
    36: 19,  # bathtub
    39: 20,  # otherfurniture
}


# ── I/O ──────────────────────────────────────────────────────────────────────

def nyu40_to_scannet20(labels: np.ndarray) -> np.ndarray:
    """Remap NYU40 label array to ScanNet20 sequential IDs (1-20; others→0)."""
    out = np.zeros_like(labels, dtype=np.int64)
    for nyu_id, sc20_id in NYU40_TO_SCANNET20.items():
        out[labels == nyu_id] = sc20_id
    return out


def read_ply_xyz_label(path: str, label_field: str = "label") -> tuple:
    """
    Read PLY file; return (xyz float64 (N,3), labels int64 (N,)).
    Auto-detects label field if the named one is missing.
    """
    ply = plyfile.PlyData.read(path)
    vtx = ply["vertex"]
    xyz = np.stack([vtx["x"], vtx["y"], vtx["z"]], axis=1).astype(np.float64)

    available = {p.name for p in vtx.properties}
    if label_field not in available:
        candidates = [n for n in available
                      if any(k in n.lower() for k in ("label", "semantic", "class"))]
        if not candidates:
            raise ValueError(
                f"Label field '{label_field}' not found in:\n  {path}\n"
                f"Available fields: {sorted(available)}\n"
                "Make sure semantic_vis_utils.py is updated to write the 'label' field."
            )
        label_field = candidates[0]
        print(f"  [WARN] auto-selecting label field: '{label_field}'")

    labels = np.array(vtx[label_field], dtype=np.int64)
    return xyz, labels


# ── Colourmap & coloured PCD export (reused from eval_3d_semantic_reference) ─

def build_scannet20_colormap() -> np.ndarray:
    """
    Returns (21, 3) uint8 array; index = ScanNet20 ID (0~20).
    Index 1~20 use Tab20 colormap. Index 0 (unlabeled) = grey (128,128,128).
    """
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab20', 20)
    colors = np.zeros((21, 3), dtype=np.uint8)
    colors[0] = [128, 128, 128]
    for i in range(20):
        r, g, b, _ = cmap(i)
        colors[i + 1] = [int(r * 255), int(g * 255), int(b * 255)]
    return colors


def save_colored_pcd(save_path: str, xyz: np.ndarray,
                     sc20_labels: np.ndarray, colormap: np.ndarray) -> None:
    """
    Save a point cloud PLY coloured by ScanNet20 label.
    xyz        : (N, 3) float
    sc20_labels: (N,)   int, ScanNet20 sequential IDs (0~20)
    colormap   : (21,3) uint8, from build_scannet20_colormap()
    """
    n = len(xyz)
    vertex = np.zeros(n, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertex['x'] = xyz[:, 0].astype(np.float32)
    vertex['y'] = xyz[:, 1].astype(np.float32)
    vertex['z'] = xyz[:, 2].astype(np.float32)
    safe_labels     = np.clip(sc20_labels, 0, 20).astype(np.int64)
    rgb             = colormap[safe_labels]
    vertex['red']   = rgb[:, 0]
    vertex['green'] = rgb[:, 1]
    vertex['blue']  = rgb[:, 2]
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    el = plyfile.PlyElement.describe(vertex, 'vertex')
    plyfile.PlyData([el], text=False).write(save_path)
    print(f"  Saved coloured PCD → {save_path}")


# ── Umeyama scale alignment (pure numpy, no evo version dependency) ──────────

def umeyama_alignment(src_pts: np.ndarray, dst_pts: np.ndarray,
                      with_scale: bool = True):
    """
    Umeyama similarity transform: dst ≈ s * R @ src + t

    src_pts, dst_pts : (N, 3) camera positions (translation part of c2w poses)
    Returns:
        s   : float   — scale factor
        R   : (3, 3)  — rotation matrix
        t   : (3,)    — translation vector
    """
    n, d = src_pts.shape
    mu_src = src_pts.mean(axis=0)
    mu_dst = dst_pts.mean(axis=0)

    src_c = src_pts - mu_src
    dst_c = dst_pts - mu_dst

    var_src = np.mean(np.sum(src_c ** 2, axis=1))

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)

    # Flip last singular vector if needed to ensure det(R) = +1
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[d - 1, d - 1] = -1

    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_src) if (with_scale and var_src > 0) else 1.0
    t = mu_dst - s * R @ mu_src

    return s, R, t


def compute_ate_rmse(pos_pred: np.ndarray, pos_gt: np.ndarray,
                     s: float, R: np.ndarray, t: np.ndarray) -> float:
    """ATE RMSE after applying similarity transform to pred positions."""
    aligned = s * (R @ pos_pred.T).T + t
    errors  = np.linalg.norm(aligned - pos_gt, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def compute_similarity_from_poses(poses_pred_c2w: np.ndarray,
                                   poses_gt_c2w:   np.ndarray):
    """
    Estimate similarity transform (s, R, t) from camera trajectories.
    Both arrays are (N, 4, 4) c2w. Uses translation vectors only.

    Returns s, R_a (3,3), t_a (3,), ape_rmse (float).
    After transform: p_metric = s * R_a @ p + t_a
    """
    pos_pred = poses_pred_c2w[:, :3, 3]   # (N, 3) camera positions
    pos_gt   = poses_gt_c2w[:, :3, 3]

    s, R_a, t_a = umeyama_alignment(pos_pred, pos_gt, with_scale=True)
    ape_rmse    = compute_ate_rmse(pos_pred, pos_gt, s, R_a, t_a)

    return s, R_a, t_a, ape_rmse


def apply_similarity(xyz: np.ndarray, s: float,
                     R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """p_out = s * R @ p + t  — (N,3) → (N,3)"""
    return s * (R @ xyz.T).T + t


# ── Pose evaluation via evo ───────────────────────────────────────────────────

def eval_poses_evo(poses_gt_c2w, poses_pred_c2w, plot_dir, label, monocular):
    """
    Evaluate camera poses using evo ATE RMSE.
    Returns (ape_rmse, info) or (None, None) on failure.
    """
    try:
        from benchmark.tools.pose_eval import evaluate_evo
        ape_rmse, info = evaluate_evo(
            poses_gt_c2w, poses_pred_c2w,
            plot_dir, label,
            monocular=monocular, plot=True,
        )
        return ape_rmse, info
    except Exception as e:
        print(f"  [WARN] evo pose eval failed: {e}")
        return None, None


# ── KD-Tree matching ──────────────────────────────────────────────────────────

def match_pred_to_gt(gt_xyz, pred_xyz, pred_labels, k=5, max_dist=None):
    """
    For each GT vertex find k nearest pred points, majority-vote label.
    Unmatched pts (> max_dist) get label -1.
    """
    print(f"  Building KD-Tree  ({len(pred_xyz):,} pred points)...")
    tree = cKDTree(pred_xyz)
    print(f"  Querying          ({len(gt_xyz):,} GT vertices, k={k})...")
    dists, idxs = tree.query(gt_xyz, k=k, workers=-1)

    if k == 1:
        dists = dists[:, np.newaxis]
        idxs  = idxs[:, np.newaxis]

    neighbor_labels = pred_labels[idxs]
    matched = stats.mode(neighbor_labels, axis=1, keepdims=False).mode.astype(np.int64)

    if max_dist is not None:
        bad = dists[:, 0] > max_dist
        matched[bad] = -1
        n_bad = int(bad.sum())
        print(f"  max_dist={max_dist}m → {n_bad:,}/{len(gt_xyz):,} "
              f"({n_bad/len(gt_xyz)*100:.1f}%) ignored")
    else:
        d = dists[:, 0]
        print(f"  NN dist  max={d.max():.4f}  mean={d.mean():.4f}  "
              f"median={np.median(d):.4f}")
    return matched


# ── Confusion matrix & metrics ────────────────────────────────────────────────

def build_confmat(gt_labels, pred_labels, num_classes=20, ignore_ids=(0,)):
    ignore_set = set(ignore_ids)
    valid = pred_labels != -1
    for ig in ignore_set:
        valid &= (gt_labels != ig)
        valid &= (pred_labels != ig)
    gt_v, pred_v = gt_labels[valid], pred_labels[valid]
    in_range = ((gt_v   >= 0) & (gt_v   <= num_classes) &
                (pred_v >= 0) & (pred_v <= num_classes))
    gt_v, pred_v = gt_v[in_range], pred_v[in_range]
    conf = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    np.add.at(conf, (gt_v, pred_v), 1)
    return conf, int(in_range.sum())


def _get_iou(label_id, conf):
    tp  = int(conf[label_id, label_id])
    fn  = int(conf[label_id, :].sum()) - tp
    fp  = int(conf[:, label_id].sum()) - tp
    d   = tp + fp + fn
    return (tp / d, tp / max(tp + fn, 1e-6)) if d > 0 else (float("nan"), float("nan"))


def compute_metrics(conf, num_classes=20, ignore_ids=(0,)):
    ignore_set = set(ignore_ids)
    ious, accs, weights, eval_cls = [], [], [], []
    for c in range(num_classes + 1):
        if c in ignore_set:
            continue
        iou, acc = _get_iou(c, conf)
        ious.append(iou); accs.append(acc)
        weights.append(int(conf[c, :].sum()))
        eval_cls.append(c)
    ious = np.array(ious); accs = np.array(accs); weights = np.array(weights, dtype=float)
    iou_mask = ~np.isnan(ious); acc_mask = ~np.isnan(accs)
    miou = float(np.mean(ious[iou_mask])) if iou_mask.any() else float("nan")
    macc = float(np.mean(accs[acc_mask])) if acc_mask.any() else float("nan")
    w_i  = weights[iou_mask]
    fiou = float(np.sum(ious[iou_mask] * w_i) / w_i.sum()) if w_i.sum() > 0 else float("nan")
    w_a  = weights[acc_mask]
    facc = float(np.sum(accs[acc_mask] * w_a) / w_a.sum()) if w_a.sum() > 0 else float("nan")
    overall = float(conf.diagonal().sum() / conf.sum()) if conf.sum() > 0 else float("nan")
    return {
        "miou": miou, "macc": macc, "fiou": fiou, "facc": facc,
        "overall_acc": overall,
        "per_class_iou": {c: ious[i] for i, c in enumerate(eval_cls)},
        "per_class_acc": {c: accs[i] for i, c in enumerate(eval_cls)},
        "eval_classes":  eval_cls,
    }


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_results(results, n_valid, n_total, class_names=None):
    cw = 22
    print("\n" + "=" * 70)
    print(f"{'Class':<{cw}} {'ID':>4}  {'IoU':>8}  {'Acc':>8}")
    print("-" * 70)
    for c in results["eval_classes"]:
        name  = (class_names[c] if class_names and c < len(class_names) else str(c))
        iou_s = (f"{results['per_class_iou'][c]*100:.2f}%"
                 if not np.isnan(results['per_class_iou'][c]) else "   N/A")
        acc_s = (f"{results['per_class_acc'][c]*100:.2f}%"
                 if not np.isnan(results['per_class_acc'][c]) else "   N/A")
        print(f"{name:<{cw}} {c:>4}  {iou_s:>8}  {acc_s:>8}")
    print("=" * 70)
    print(f"{'mIoU':<28} {results['miou']*100:.2f}%")
    print(f"{'mAcc':<28} {results['macc']*100:.2f}%")
    print(f"{'freq-mIoU':<28} {results['fiou']*100:.2f}%")
    print(f"{'freq-mAcc':<28} {results['facc']*100:.2f}%")
    print(f"{'Overall Acc':<28} {results['overall_acc']*100:.2f}%")
    print(f"{'Valid / GT pts':<28} {n_valid:,} / {n_total:,}")
    print("=" * 70 + "\n")


def save_csv(path, results, n_valid, n_total,
             ape_rmse=None, scale=None, class_names=None):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "iou(%)", "acc(%)"])
        for c in results["eval_classes"]:
            name = (class_names[c] if class_names and c < len(class_names) else str(c))
            w.writerow([c, name,
                        f"{results['per_class_iou'][c]*100:.4f}",
                        f"{results['per_class_acc'][c]*100:.4f}"])
        w.writerow(["", "mIoU",        f"{results['miou']*100:.4f}",        ""])
        w.writerow(["", "mAcc",        f"{results['macc']*100:.4f}",        ""])
        w.writerow(["", "freq-mIoU",   f"{results['fiou']*100:.4f}",        ""])
        w.writerow(["", "freq-mAcc",   f"{results['facc']*100:.4f}",        ""])
        w.writerow(["", "Overall Acc", f"{results['overall_acc']*100:.4f}", ""])
        w.writerow(["", "valid_pts",   str(n_valid),                        ""])
        w.writerow(["", "total_gt_pts",str(n_total),                        ""])
        if ape_rmse is not None:
            w.writerow(["", "ATE_RMSE(m)", f"{ape_rmse:.4f}", ""])
        if scale is not None:
            w.writerow(["", "scale_factor", f"{scale:.6f}", ""])
    print(f"CSV saved: {path}")


# ── Argument parser ───────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate semantic SLAM on ScanNet: pose ATE + 3D mIoU/mAcc",
    )
    p.add_argument("--results_path",  required=True,
                   help="Dir where slam_semantic/run.py saved results")
    p.add_argument("--scene_name",    required=True,
                   help="Scene name used as --demo_name in run.py (e.g. scene0050_00)")
    p.add_argument("--data_path",     required=True,
                   help="ScanNet scene root containing poses.npy")
    p.add_argument("--gt_labels_ply", required=True,
                   help="GT mesh PLY with NYU40 labels "
                        "(e.g. scene0050_00_vh_clean_2.labels.ply)")
    p.add_argument("--metric_scale",  action="store_true",
                   help="Skip Umeyama scale alignment — use when pred is already metric scale")
    p.add_argument("--use_voxel_ply", action="store_true",
                   help="Use semantic_voxels.ply instead of semantic_per_point.ply")
    p.add_argument("--no_pose_eval",  action="store_true",
                   help="Skip camera pose evaluation")
    p.add_argument("--k",             type=int,   default=5,
                   help="KD-Tree neighbours for majority-vote label assignment")
    p.add_argument("--max_dist",      type=float, default=None,
                   help="Ignore GT points farther than this (metres) from any pred point")
    p.add_argument("--num_classes",   type=int,   default=20)
    p.add_argument("--ignore_ids",    type=int,   nargs="+", default=[0])
    p.add_argument("--save_csv",      default=None,
                   help="Save per-class + summary results to CSV")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    npz_path      = os.path.join(args.results_path,
                                 f"scene_{args.scene_name}_results.npz")
    sem_ply_name  = (f"scene_{args.scene_name}_semantic_voxels.ply"
                     if args.use_voxel_ply
                     else f"scene_{args.scene_name}_semantic_per_point.ply")
    sem_ply_path  = os.path.join(args.results_path, sem_ply_name)
    gt_poses_path = os.path.join(args.data_path, "poses.npy")
    plot_dir      = args.results_path

    # ── [1] Load pred poses ──────────────────────────────────────────────────
    print(f"\n[1/6] Loading pred SLAM results: {npz_path}")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"NPZ not found: {npz_path}\n"
            "Run slam_semantic/run.py with --save_res (default=True) first."
        )
    npz = np.load(npz_path, allow_pickle=True)
    poses_pred_c2w = npz["pose"].astype(np.float64)   # (N, 4, 4), T_voxel aligned
    print(f"  pred poses shape : {poses_pred_c2w.shape}")

    # ── [2] Load GT poses ────────────────────────────────────────────────────
    print(f"\n[2/6] Loading GT poses: {gt_poses_path}")
    if not os.path.isfile(gt_poses_path):
        raise FileNotFoundError(f"GT poses not found: {gt_poses_path}")
    poses_gt_c2w = np.load(gt_poses_path).astype(np.float64)   # (N, 4, 4)
    print(f"  GT poses shape   : {poses_gt_c2w.shape}")

    if len(poses_pred_c2w) != len(poses_gt_c2w):
        raise ValueError(
            f"Frame count mismatch: pred={len(poses_pred_c2w)}, gt={len(poses_gt_c2w)}."
        )

    # ── [3] Scale alignment ──────────────────────────────────────────────────
    if args.metric_scale:
        print("\n[3/6] Scale alignment — skipped (--metric_scale)")
        scale, R_a, t_a, ape_after = 1.0, np.eye(3), np.zeros(3), None
    else:
        print("\n[3/6] Scale alignment (Umeyama, pure numpy)...")
        scale, R_a, t_a, ape_after = compute_similarity_from_poses(
            poses_pred_c2w, poses_gt_c2w
        )
        print(f"  Estimated scale  : {scale:.6f}")
        print(f"  ATE RMSE (after) : {ape_after:.4f} m")
        print(f"  R correction det : {np.linalg.det(R_a):.6f}")
        print(f"  t correction     : {t_a}")

    # ── [4] Pose evaluation ──────────────────────────────────────────────────
    ape_full = None
    if not args.no_pose_eval:
        print(f"\n[4/6] Pose evaluation (ATE RMSE via evo)...")
        ape_full, info = eval_poses_evo(
            poses_gt_c2w, poses_pred_c2w,
            plot_dir, f"pose_{args.scene_name}",
            monocular=not args.metric_scale,
        )
        if ape_full is not None:
            print(f"  ATE RMSE  : {ape_full:.4f} m  (scale={info['s']:.4f})")
    else:
        print("\n[4/6] Pose evaluation — skipped (--no_pose_eval)")

    # ── [5] Load pred semantic PLY & apply similarity transform ─────────────
    print(f"\n[5/6] Loading pred semantic PLY: {sem_ply_path}")
    if not os.path.isfile(sem_ply_path):
        raise FileNotFoundError(
            f"Semantic PLY not found: {sem_ply_path}\n"
            "Run slam_semantic/run.py with --save_semantic --demo_type scannet."
        )
    pred_xyz, pred_labels = read_ply_xyz_label(sem_ply_path, label_field="label")
    print(f"  pred points  : {len(pred_xyz):,}")
    print(f"  label range  : [{pred_labels.min()}, {pred_labels.max()}]  "
          f"unique={sorted(np.unique(pred_labels).tolist())}")

    print(f"  Applying similarity transform (s={scale:.4f})...")
    pred_xyz_metric = apply_similarity(pred_xyz, scale, R_a, t_a)

    # ── [6] Load GT semantic mesh, eval, save visualisations ─────────────────
    print(f"\n[6/6] Loading GT semantic PLY: {args.gt_labels_ply}")
    gt_xyz, gt_labels_nyu40 = read_ply_xyz_label(args.gt_labels_ply, label_field="label")
    print(f"  GT vertices   : {len(gt_xyz):,}")
    print(f"  NYU40 range   : [{gt_labels_nyu40.min()}, {gt_labels_nyu40.max()}]")

    gt_labels = nyu40_to_scannet20(gt_labels_nyu40)
    print(f"  Scannet20 unique: {sorted(np.unique(gt_labels).tolist())}")

    # Save coloured PCDs for visual inspection (same colormap for both)
    colormap = build_scannet20_colormap()

    gt_vis_path = os.path.join(args.results_path,
                               f"scene_{args.scene_name}_gt_colored.ply")
    save_colored_pcd(gt_vis_path, gt_xyz, gt_labels, colormap)

    pred_vis_path = os.path.join(args.results_path,
                                 f"scene_{args.scene_name}_pred_colored_metric.ply")
    save_colored_pcd(pred_vis_path, pred_xyz_metric, pred_labels, colormap)

    # KD-Tree matching
    print(f"\n  KD-Tree matching (k={args.k}, max_dist={args.max_dist})...")
    matched_pred = match_pred_to_gt(
        gt_xyz, pred_xyz_metric, pred_labels,
        k=args.k, max_dist=args.max_dist,
    )

    conf, n_valid = build_confmat(
        gt_labels, matched_pred,
        num_classes=args.num_classes,
        ignore_ids=args.ignore_ids,
    )
    results = compute_metrics(conf, args.num_classes, args.ignore_ids)
    print_results(results, n_valid, len(gt_labels), SCANNET20_NAMES)

    if ape_full is not None:
        print(f"Pose ATE RMSE : {ape_full:.4f} m")

    if args.save_csv:
        save_csv(
            args.save_csv, results, n_valid, len(gt_labels),
            ape_rmse=ape_full, scale=scale,
            class_names=SCANNET20_NAMES,
        )


if __name__ == "__main__":
    main()
