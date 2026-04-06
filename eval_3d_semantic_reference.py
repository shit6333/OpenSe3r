"""
eval_3d_semantic.py
--------------------
計算 3D 語意分割的 mIoU / mAcc / freq-mIoU。

對齊 OVO eval_utils.py 的實作：
  - KD-Tree 方向：對每個 GT mesh 頂點查詢最近的 k 個預測點（GT → pred）
  - k=5 最近鄰 + majority vote（與 OVO match_labels_to_vtx 一致）
  - Confusion matrix → mIoU, mAcc, freq-mIoU（與 OVO iou_acc_from_confmat 一致）
  - GT label NYU40 → ScanNet20 remap（與 OVO evaluate_scan map_gt_ids 一致）

用法：
    python eval_3d_semantic.py \
      --gt_ply   scene0000_00_vh_clean_2.labels.ply \
      --pred_ply my_result.ply \
      --pred_label_format nyu40
      
    python eval_3d_semantic.py \
        --gt_ply  scene0000_00_vh_clean_2.labels.ply \
        --pred_ply semantic_pcd_accumulated_gaussians.ply \
        --pred_label_format nyu40 \
        --save_gt_pcd ./output_dir/gt_colored.ply # save gt mesh to pcd .ply
        
    python eval_3d_semantic.py \
        --gt_ply  /mnt/HDD4/ricky/online_slam/LoopSplat/data/ScanNet/scene0000_00/scene0000_00_vh_clean_2.labels.ply \
        --pred_ply semantic_pcd_accumulated_gaussians.ply \
        --pred_label_format nyu40 \
        --save_gt_pcd ./output_dir # save gt mesh to pcd .ply
"""

import argparse
import numpy as np
import plyfile
from scipy.spatial import cKDTree
from scipy import stats

# ── ScanNet20 class names（1-indexed，0 = unlabeled/ignored）─────────────────
SCANNET20_NAMES = [
    "unlabeled",       # 0  → ignored
    "wall",            # 1
    "floor",           # 2
    "cabinet",         # 3
    "bed",             # 4
    "chair",           # 5
    "sofa",            # 6
    "table",           # 7
    "door",            # 8
    "window",          # 9
    "bookshelf",       # 10
    "picture",         # 11
    "counter",         # 12
    "desk",            # 13
    "curtain",         # 14
    "refrigerator",    # 15
    "shower curtain",  # 16
    "toilet",          # 17
    "sink",            # 18
    "bathtub",         # 19
    "otherfurniture",  # 20
]

# ── NYU40 ID → ScanNet20 ID mapping ──────────────────────────────────────────
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

def nyu40_to_scannet20(labels: np.ndarray) -> np.ndarray:
    """把 NYU40 label array remap 成 ScanNet20 (1-20，其餘→0)。"""
    out = np.zeros_like(labels, dtype=np.int64)
    for nyu_id, sc20_id in NYU40_TO_SCANNET20.items():
        out[labels == nyu_id] = sc20_id
    return out


# ── I/O ───────────────────────────────────────────────────────────────────────

def read_ply(path: str, label_field: str):
    """讀取 ply，回傳 (N,3) xyz float64 和 (N,) label int64。"""
    ply = plyfile.PlyData.read(path)
    vtx = ply["vertex"]
    xyz = np.stack([vtx["x"], vtx["y"], vtx["z"]], axis=1).astype(np.float64)

    available = {prop.name for prop in vtx.properties}
    if label_field not in available:
        candidates = [n for n in available
                      if any(k in n.lower() for k in ("label", "semantic", "class"))]
        if not candidates:
            raise ValueError(f"找不到欄位 '{label_field}'，ply 中有：{sorted(available)}")
        label_field = candidates[0]
        print(f"  [警告] 自動改用欄位: '{label_field}'")

    labels = np.array(vtx[label_field], dtype=np.int64)
    return xyz, labels


# ── Matching（對齊 OVO match_labels_to_vtx）──────────────────────────────────

def match_pred_to_gt(gt_xyz, pred_xyz, pred_labels, k: int = 5, max_dist=None):
    """
    對每個 GT 頂點，找最近 k 個預測點，majority vote 決定 label。
    對齊 OVO: KDTree(points_3d).query(mesh_vtx, k=5) + torch.mode

    max_dist: 若最近鄰距離超過此值，該 GT 頂點 label 設為 -1（ignore）。
              None = 不限制。
    """
    print(f"  建立 KD-Tree（{len(pred_xyz):,} 個預測點）...")
    tree = cKDTree(pred_xyz)

    print(f"  查詢 {len(gt_xyz):,} 個 GT 頂點的 k={k} 最近鄰...")
    dists, idxs = tree.query(gt_xyz, k=k, workers=-1)   # (N_gt, k)

    # k=1 時 dists/idxs 是 1D，統一成 2D
    if k == 1:
        dists = dists[:, np.newaxis]
        idxs  = idxs[:, np.newaxis]

    # 取 k 個鄰居的 label → majority vote（對齊 OVO torch.mode）
    neighbor_labels = pred_labels[idxs]                  # (N_gt, k)
    matched_labels  = stats.mode(neighbor_labels, axis=1, keepdims=False).mode  # (N_gt,)
    matched_labels  = matched_labels.astype(np.int64)

    # 用最近鄰距離（k 個中最小的，即 dists[:,0]）做 max_dist 過濾
    if max_dist is not None:
        unmatched = dists[:, 0] > max_dist
        matched_labels[unmatched] = -1
        n_un = int(unmatched.sum())
        print(f"  距離超過 {max_dist}m: {n_un:,} / {len(gt_xyz):,} "
              f"({n_un/len(gt_xyz)*100:.1f}%) → ignore")
    else:
        d = dists[:, 0]
        print(f"  最近鄰距離 — max: {d.max():.4f}m  "
              f"mean: {d.mean():.4f}m  median: {np.median(d):.4f}m")

    return matched_labels


# ── Metrics（對齊 OVO get_iou / iou_acc_from_confmat）───────────────────────

def build_confmat(gt_labels, pred_labels, num_classes, ignore_ids):
    """建立 (num_classes+1) x (num_classes+1) confusion matrix。"""
    ignore_set = set(ignore_ids)

    valid = (pred_labels != -1)
    for ig in ignore_set:
        valid &= (gt_labels != ig)
        valid &= (pred_labels != ig)

    gt_v   = gt_labels[valid]
    pred_v = pred_labels[valid]

    in_range = ((gt_v   >= 0) & (gt_v   <= num_classes) &
                (pred_v >= 0) & (pred_v <= num_classes))
    gt_v   = gt_v[in_range]
    pred_v = pred_v[in_range]

    conf = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    np.add.at(conf, (gt_v, pred_v), 1)
    return conf, int(in_range.sum())


def get_iou(label_id, confusion):
    """對齊 OVO get_iou。"""
    tp    = int(confusion[label_id, label_id])
    fn    = int(confusion[label_id, :].sum()) - tp
    fp    = int(confusion[:, label_id].sum()) - tp
    denom = tp + fp + fn
    if denom == 0:
        return float("nan"), float("nan")
    iou = tp / denom
    acc = tp / max(tp + fn, 1e-6)
    return iou, acc


def compute_metrics(conf, num_classes, ignore_ids, mask_nan=True):
    """對齊 OVO iou_acc_from_confmat + eval_semantics 的 freq-mIoU。"""
    ignore_set = set(ignore_ids)

    ious, accs, weights = [], [], []
    eval_classes = []
    for c in range(num_classes + 1):
        if c in ignore_set:
            continue
        iou, acc = get_iou(c, conf)
        ious.append(iou)
        accs.append(acc)
        weights.append(int(conf[c, :].sum()))   # TP + FN（GT 點數）
        eval_classes.append(c)

    ious    = np.array(ious,    dtype=float)
    accs    = np.array(accs,    dtype=float)
    weights = np.array(weights, dtype=float)

    if mask_nan:
        iou_mask = ~np.isnan(ious)
        acc_mask = ~np.isnan(accs)
    else:
        iou_mask = np.ones(len(ious), dtype=bool)
        acc_mask = np.ones(len(accs), dtype=bool)

    miou = float(np.mean(ious[iou_mask])) if iou_mask.any() else float("nan")
    macc = float(np.mean(accs[acc_mask])) if acc_mask.any() else float("nan")

    # frequency-weighted mIoU（對齊 OVO eval_semantics f-mIoU）
    w_valid = weights[iou_mask]
    fiou = (float(np.sum(ious[iou_mask] * w_valid) / w_valid.sum())
            if w_valid.sum() > 0 else float("nan"))
    w_valid_acc = weights[acc_mask]
    facc = (float(np.sum(accs[acc_mask] * w_valid_acc) / w_valid_acc.sum())
            if w_valid_acc.sum() > 0 else float("nan"))

    overall_acc = (float(conf.diagonal().sum() / conf.sum())
                   if conf.sum() > 0 else float("nan"))

    return {
        "miou": miou, "macc": macc,
        "fiou": fiou, "facc": facc,
        "overall_acc": overall_acc,
        "per_class_iou": {c: ious[i] for i, c in enumerate(eval_classes)},
        "per_class_acc": {c: accs[i] for i, c in enumerate(eval_classes)},
        "eval_classes": eval_classes,
        "confusion_matrix": conf,
    }


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_results(results, n_valid, n_total, class_names=None):
    eval_cls = results["eval_classes"]
    per_iou  = results["per_class_iou"]
    per_acc  = results["per_class_acc"]

    col_w = 22
    print("\n" + "=" * 70)
    print(f"{'Class':<{col_w}} {'ID':>4}  {'IoU':>8}  {'Acc':>8}")
    print("-" * 70)
    for c in eval_cls:
        name  = (class_names[c] if class_names and c < len(class_names) else str(c))
        iou_s = f"{per_iou[c]*100:.2f}%" if not np.isnan(per_iou[c]) else "   N/A"
        acc_s = f"{per_acc[c]*100:.2f}%" if not np.isnan(per_acc[c]) else "   N/A"
        print(f"{name:<{col_w}} {c:>4}  {iou_s:>8}  {acc_s:>8}")
    print("=" * 70)
    print(f"{'mIoU':<28} {results['miou']*100:.2f}%")
    print(f"{'mAcc':<28} {results['macc']*100:.2f}%")
    print(f"{'freq-mIoU':<28} {results['fiou']*100:.2f}%")
    print(f"{'freq-mAcc':<28} {results['facc']*100:.2f}%")
    print(f"{'Overall Acc':<28} {results['overall_acc']*100:.2f}%")
    print(f"{'Valid points':<28} {n_valid:,} / {n_total:,}")
    print("=" * 70 + "\n")


def save_csv(path, results, n_valid, n_total, class_names=None):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "iou(%)", "acc(%)"])
        for c in results["eval_classes"]:
            name = (class_names[c] if class_names and c < len(class_names) else str(c))
            w.writerow([c, name,
                        f"{results['per_class_iou'][c]*100:.4f}",
                        f"{results['per_class_acc'][c]*100:.4f}"])
        w.writerow(["", "mIoU",        f"{results['miou']*100:.4f}", ""])
        w.writerow(["", "mAcc",        f"{results['macc']*100:.4f}", ""])
        w.writerow(["", "freq-mIoU",   f"{results['fiou']*100:.4f}", ""])
        w.writerow(["", "freq-mAcc",   f"{results['facc']*100:.4f}", ""])
        w.writerow(["", "Overall Acc", f"{results['overall_acc']*100:.4f}", ""])
        w.writerow(["", "valid_pts",   f"{n_valid}", ""])
        w.writerow(["", "total_gt_pts",f"{n_total}", ""])
    print(f"CSV 已儲存: {path}")


# ── GT colored PCD export ─────────────────────────────────────────────────────

# Tab20 colormap（與 save_semantic_ply_pixel.py 的 class_colors 對應）
# index 對應 ScanNet20 1-indexed：index 0 = unlabeled（灰色），index 1~20 = 各類別
def build_scannet20_colormap() -> np.ndarray:
    """
    回傳 (21, 3) uint8 array，index = ScanNet20 ID（0~20）。
    index 1~20 使用 Tab20 colormap，與 save_semantic_ply_pixel.py 的 class_colors 相同。
    index 0（unlabeled）= 灰色 (128, 128, 128)。
    """
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab20', 20)
    colors = np.zeros((21, 3), dtype=np.uint8)
    colors[0] = [128, 128, 128]   # unlabeled → 灰色
    for i in range(20):
        r, g, b, _ = cmap(i)
        colors[i + 1] = [int(r * 255), int(g * 255), int(b * 255)]
    return colors


def save_colored_pcd(save_path: str, xyz: np.ndarray, sc20_labels: np.ndarray,
                     colormap: np.ndarray) -> None:
    """
    將 GT mesh 頂點存成帶顏色的 PCD ply。
    顏色依照 ScanNet20 colormap，與 pred pcd 的語意顏色對應相同類別同色。

    Args:
        save_path    : 輸出路徑
        xyz          : (N, 3) float，GT mesh 頂點座標
        sc20_labels  : (N,) int，已 remap 的 ScanNet20 ID（0~20）
        colormap     : (21, 3) uint8，build_scannet20_colormap() 的輸出
    """
    n = len(xyz)
    vertex = np.zeros(n, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertex['x'] = xyz[:, 0]
    vertex['y'] = xyz[:, 1]
    vertex['z'] = xyz[:, 2]

    # clip label 到合法範圍，以防萬一
    safe_labels = np.clip(sc20_labels, 0, 20).astype(np.int64)
    rgb = colormap[safe_labels]          # (N, 3) uint8
    vertex['red']   = rgb[:, 0]
    vertex['green'] = rgb[:, 1]
    vertex['blue']  = rgb[:, 2]

    el = plyfile.PlyElement.describe(vertex, 'vertex')
    plyfile.PlyData([el], text=False).write(save_path)
    print(f"GT colored PCD 已儲存: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # 1. 讀 GT
    print(f"\n[1/4] 讀取 GT ply: {args.gt_ply}")
    gt_xyz, gt_labels_raw = read_ply(args.gt_ply, args.gt_label_field)
    print(f"      頂點數: {len(gt_xyz):,}")
    print(f"      label 範圍: [{gt_labels_raw.min()}, {gt_labels_raw.max()}]，"
          f"unique: {sorted(np.unique(gt_labels_raw).tolist())}")

    print("      GT remap: NYU40 → ScanNet20 ...")
    gt_labels = nyu40_to_scannet20(gt_labels_raw)
    print(f"      remap 後 unique: {sorted(np.unique(gt_labels).tolist())}")

    # 2. 讀預測
    print(f"\n[2/4] 讀取預測 ply: {args.pred_ply}")
    pred_xyz, pred_labels_raw = read_ply(args.pred_ply, args.pred_label_field)
    print(f"      點數: {len(pred_xyz):,}")
    print(f"      label 範圍: [{pred_labels_raw.min()}, {pred_labels_raw.max()}]，"
          f"unique: {sorted(np.unique(pred_labels_raw).tolist())}")

    fmt = args.pred_label_format
    if fmt == "nyu40":
        print("      pred remap: NYU40 → ScanNet20 ...")
        pred_labels = nyu40_to_scannet20(pred_labels_raw)
    elif fmt == "zero_indexed":
        print("      pred remap: 0-indexed → 1-indexed ...")
        pred_labels = pred_labels_raw + 1
    else:
        pred_labels = pred_labels_raw.copy()
    print(f"      remap 後 unique: {sorted(np.unique(pred_labels).tolist())}")

    # 3. KD-Tree k=5 majority vote（對齊 OVO）
    dist_str = f"{args.max_dist}m" if args.max_dist is not None else "不限制"
    print(f"\n[3/4] KD-Tree 最近鄰匹配（k={args.k}, max_dist={dist_str}）...")
    matched_pred = match_pred_to_gt(
        gt_xyz, pred_xyz, pred_labels,
        k=args.k, max_dist=args.max_dist
    )

    # 4. （可選）儲存 GT colored PCD，顏色與 pred pcd 相同 colormap
    if args.save_gt_pcd:
        import os
        colormap = build_scannet20_colormap()
        gt_pcd_path = args.save_gt_pcd
        # 若只給目錄，自動命名
        if os.path.isdir(gt_pcd_path) or not gt_pcd_path.endswith(".ply"):
            os.makedirs(gt_pcd_path, exist_ok=True)
            from pathlib import Path as _Path
            stem = _Path(args.gt_ply).stem
            gt_pcd_path = os.path.join(gt_pcd_path, f"gt_colored_{stem}.ply")
        print(f"\n[opt] 儲存 GT colored PCD → {gt_pcd_path}")
        save_colored_pcd(gt_pcd_path, gt_xyz, gt_labels, colormap)

    # 5. 計算指標
    print(f"\n[4/4] 計算指標（ignore: {args.ignore_ids}, num_classes: {args.num_classes}）...")
    conf, n_valid = build_confmat(gt_labels, matched_pred, args.num_classes, args.ignore_ids)
    results = compute_metrics(conf, args.num_classes, args.ignore_ids)

    class_names = SCANNET20_NAMES if args.num_classes == 20 else None
    print_results(results, n_valid, len(gt_labels), class_names)

    if args.save_csv:
        save_csv(args.save_csv, results, n_valid, len(gt_labels), class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Semantic Evaluation — mIoU + mAcc (aligned with OVO eval_utils)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gt_ply",            required=True,  help="GT mesh ply（NYU40 label）")
    parser.add_argument("--pred_ply",          required=True,  help="預測 pcd ply")
    parser.add_argument("--gt_label_field",    default="label",help="GT ply label 欄位名")
    parser.add_argument("--pred_label_field",  default="label",help="預測 ply label 欄位名")
    parser.add_argument("--pred_label_format", default="nyu40",
                        choices=["scannet20", "nyu40", "zero_indexed"],
                        help="預測 label 格式：nyu40 / scannet20 / zero_indexed")
    parser.add_argument("--num_classes",  default=20,   type=int,   help="類別數（ScanNet20=20）")
    parser.add_argument("--ignore_ids",   default=[0],  nargs="+",  type=int,
                        help="忽略的 label ID（remap 後，0=unlabeled）")
    parser.add_argument("--k",            default=5,    type=int,   help="KD-Tree 最近鄰數（對齊 OVO 預設 k=5）")
    parser.add_argument("--max_dist",     default=None, type=float, help="最大接受距離(m)，不指定則不限制")
    parser.add_argument("--save_csv",     default=None,             help="結果存成 CSV（可選）")
    parser.add_argument("--save_gt_pcd",  default=None,
                        help="（可選）儲存 GT colored PCD ply 的路徑或目錄。"
                             "顏色與 pred pcd 的 Tab20 colormap 對應相同類別同色，方便視覺化比對。"
                             "可傳入完整檔名（.ply）或目錄（自動命名為 gt_colored_<stem>.ply）。")
    args = parser.parse_args()
    main(args)