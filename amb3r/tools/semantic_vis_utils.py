"""
semantic_vis_utils.py
=====================
PLY export utilities for semantic and instance features.

Three export strategies:
    1. Voxel-centre PLY  — one point per occupied voxel (sparse, minimal storage)
    2. Per-point PLY     — query voxel map for every geometry point (dense, best viz)
    3. Raw-feature PLY   — save raw float features as vertex attributes (for analysis)

The "save raw feature" functions (export_semantic_feat_ply,
export_instance_feat_ply) are defined but NOT called from run_semantic.py
by default—they are there for offline analysis if needed.
"""

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import math
import os
import struct
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
import plyfile

from .scannet200_constants import (
    VALID_CLASS_IDS_20, CLASS_LABELS_20, SCANNET_COLOR_MAP_20,
    VALID_CLASS_IDS_200, CLASS_LABELS_200, SCANNET_COLOR_MAP_200,
)


# ============================================================
# Label / colour map helpers
# ============================================================

def get_scannet_label_and_color_map(label_set="scannet20"):
    """
    Returns:
        labels:      list[str]
        class_ids:   list[int]
        color_table: np.ndarray [K, 3] uint8
    """
    if label_set.lower() in ["scannet20", "scannet-20", "20"]:
        labels    = list(CLASS_LABELS_20)
        class_ids = list(VALID_CLASS_IDS_20)
        color_map = SCANNET_COLOR_MAP_20
    elif label_set.lower() in ["scannet200", "scannet-200", "200"]:
        labels    = list(CLASS_LABELS_200)
        class_ids = list(VALID_CLASS_IDS_200)
        color_map = SCANNET_COLOR_MAP_200
    else:
        raise ValueError(f"Unknown label_set: {label_set}")

    color_table = np.asarray(
        [[int(color_map[cid][0]), int(color_map[cid][1]), int(color_map[cid][2])]
         for cid in class_ids],
        dtype=np.uint8,
    )
    return labels, class_ids, color_table


# ============================================================
# Text embedding helpers
# ============================================================

def build_text_embeddings(clip_model, tokenizer, labels, device,
                           template="a photo of a {}"):
    """
    Build L2-normalised CLIP text embeddings for a list of class labels.

    Returns:
        text_feat: [K, C] float32 (on `device`, L2-normalised)
    """
    prompts = [template.format(x) for x in labels]
    tokens  = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_feat = clip_model.encode_text(tokens)
    return F.normalize(text_feat.float(), dim=-1)


# ============================================================
# Cosine matching helper
# ============================================================

def semantic_feat_to_label_color(
    semantic_feat: torch.Tensor,   # [N, C]
    text_feat:     torch.Tensor,   # [K, C]
    color_table:   np.ndarray,     # [K, 3] uint8
):
    """
    For each point assign the label with the highest cosine similarity.

    Returns:
        pred_idx: [N] np.int64
        colors:   [N, 3] np.uint8
    """
    sem      = F.normalize(semantic_feat.detach().float(), dim=-1)
    txt      = F.normalize(text_feat.detach().float(),     dim=-1)
    sim      = sem @ txt.T                        # [N, K]
    pred_idx = sim.argmax(dim=-1).cpu().numpy()
    colors   = color_table[pred_idx]
    return pred_idx, colors


# ============================================================
# Low-level PLY writer  (binary, arbitrary float vertex props)
# ============================================================

def _write_ply_with_props(points_np: np.ndarray,
                          prop_dict:  dict,
                          save_file:  str,
                          colors_np:  np.ndarray = None):
    """
    Write a binary-little-endian PLY file with XYZ + optional RGB +
    arbitrary per-vertex float properties.

    Params:
        points_np: (N, 3) float32
        prop_dict: {prefix: feat_np (N, C)} — stored as "{prefix}_0", …
        save_file: output path
        colors_np: (N, 3) uint8 optional RGB colours
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
    N = points_np.shape[0]

    header_lines = ["ply", "format binary_little_endian 1.0", f"element vertex {N}"]
    header_lines += ["property float x", "property float y", "property float z"]

    if colors_np is not None:
        header_lines += ["property uchar red", "property uchar green", "property uchar blue"]

    col_list = [points_np[:, 0], points_np[:, 1], points_np[:, 2]]
    fmt      = "fff"

    if colors_np is not None:
        col_list += [colors_np[:, 0], colors_np[:, 1], colors_np[:, 2]]
        fmt += "BBB"
        for name in ["red", "green", "blue"]:
            pass  # already added to header above

    for prefix, feat in prop_dict.items():
        for i in range(feat.shape[1]):
            header_lines.append(f"property float {prefix}_{i}")
        for i in range(feat.shape[1]):
            col_list.append(feat[:, i].astype(np.float32))
        fmt += "f" * feat.shape[1]

    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    col_matrix = np.column_stack(col_list)
    if colors_np is not None:
        # Mixed dtype: float + uint8 — use struct packing
        row_fmt = "<" + fmt
        row_size = struct.calcsize(row_fmt)
        data = bytearray(N * row_size)
        for i in range(N):
            row = tuple(col_matrix[i, :3].astype(np.float32).tolist()
                        + [int(col_matrix[i, 3]), int(col_matrix[i, 4]), int(col_matrix[i, 5])]
                        + col_matrix[i, 6:].astype(np.float32).tolist())
            struct.pack_into(row_fmt, data, i * row_size, *row)
        with open(save_file, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(data)
    else:
        col_matrix = col_matrix.astype(np.float32)
        with open(save_file, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(col_matrix.tobytes())



# ============================================================
# Helper: Coordinate transformation

def apply_similarity_transform(points_xyz, transformation=None, scale=None):
    """
    points_xyz: (..., 3) torch.Tensor or np.ndarray
    transformation: (4, 4), maps local -> final
    scale: optional scalar, applied before rigid transform

    return: same type as input, transformed points in final frame
    """
    if transformation is None and scale is None:
        return points_xyz

    is_numpy = isinstance(points_xyz, np.ndarray)
    pts = torch.from_numpy(points_xyz).float() if is_numpy else points_xyz.detach().float().cpu()

    if scale is not None:
        pts = pts * float(scale)

    if transformation is not None:
        T = transformation
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=torch.float32)
        else:
            T = T.detach().float().cpu()

        ones = torch.ones((pts.shape[0], 1), dtype=torch.float32)
        pts_h = torch.cat([pts, ones], dim=1)   # (N, 4)
        pts = (T @ pts_h.T).T[:, :3]

    return pts.numpy() if is_numpy else pts

def invert_similarity_transform(points_xyz, transformation=None, scale=None):
    """
    points_xyz: (..., 3) torch.Tensor or np.ndarray in final frame
    transformation: (4, 4), maps local -> final
    scale: optional scalar used in forward transform

    return: points mapped back to local voxel-map frame
    """
    if transformation is None and scale is None:
        return points_xyz

    is_numpy = isinstance(points_xyz, np.ndarray)
    pts = torch.from_numpy(points_xyz).float() if is_numpy else points_xyz.detach().float().cpu()

    # inverse rigid transform first
    if transformation is not None:
        T = transformation
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=torch.float32)
        else:
            T = T.detach().float().cpu()

        T_inv = torch.linalg.inv(T)
        ones = torch.ones((pts.shape[0], 1), dtype=torch.float32)
        pts_h = torch.cat([pts, ones], dim=1)
        pts = (T_inv @ pts_h.T).T[:, :3]

    # then inverse scale
    if scale is not None:
        pts = pts / float(scale)

    return pts.numpy() if is_numpy else pts

# Helper: Save label
def _save_ply_with_label(xyz: np.ndarray, colors: np.ndarray, labels: np.ndarray, save_file: str):
    """
    Write binary PLY with x,y,z,r,g,b,label.
    labels: (N,) uint8 — scannet20 sequential 1-indexed (1=wall…20=otherfurniture, 0=unlabeled).
    """
    n = len(xyz)
    vertex = np.zeros(n, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('label', 'u1'),
    ])
    vertex['x'] = xyz[:, 0].astype(np.float32)
    vertex['y'] = xyz[:, 1].astype(np.float32)
    vertex['z'] = xyz[:, 2].astype(np.float32)
    vertex['red']   = colors[:, 0].astype(np.uint8)
    vertex['green'] = colors[:, 1].astype(np.uint8)
    vertex['blue']  = colors[:, 2].astype(np.uint8)
    vertex['label'] = labels.astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
    el = plyfile.PlyElement.describe(vertex, 'vertex')
    plyfile.PlyData([el], text=False).write(save_file)

# ============================================================
# ── NEW ──  Voxel-based semantic export  (main export path)
# ============================================================

# ============================================================

def export_semantic_voxels_ply(
    voxel_map,                  # VoxelFeatureMap
    text_feat: torch.Tensor,    # [K, C]
    color_table: np.ndarray,    # [K, 3]
    save_file: str,
    transformation=None,        # optional (4, 4), local -> final
    scale=None,                 # optional scalar
):
    """
    Export all occupied voxel centres with semantic colour assignment.

    If transformation / scale is provided, voxel centres are transformed
    from voxel-map local frame to final output frame before saving.
    """
    centers, features = voxel_map.get_all()    # (V, 3), (V, C)
    if centers.shape[0] == 0:
        print("[WARN] export_semantic_voxels_ply: voxel map is empty, skipping.")
        return

    centers_out = apply_similarity_transform(
        centers, transformation=transformation, scale=scale
    )

    pred_idx, colors_np = semantic_feat_to_label_color(features, text_feat, color_table)
    labels = (pred_idx + 1).astype(np.uint8)   # scannet20 sequential 1-indexed
    _save_ply_with_label(centers_out.numpy(), colors_np, labels, save_file)


def export_semantic_per_point_ply(
    points_xyz: torch.Tensor,    # [N, 3] in FINAL output frame
    voxel_map,
    text_feat: torch.Tensor,
    color_table: np.ndarray,
    save_file: str,
    transformation=None,        # optional (4,4), local -> final
    scale=None,                 # optional scalar
):
    """
    Query voxel map for every geometry point and export semantic PLY.

    If transformation / scale is provided:
      - points_xyz is assumed to be in FINAL frame
      - before querying voxel_map, points are mapped back to LOCAL voxel frame
      - saved coordinates remain in FINAL frame
    """
    points_out = points_xyz.detach().float().cpu()
    points_query = invert_similarity_transform(
        points_out, transformation=transformation, scale=scale
    )

    feat_pp = voxel_map.query(points_query)    # query in local voxel frame

    pred_idx, colors_np = semantic_feat_to_label_color(feat_pp, text_feat, color_table)
    labels = (pred_idx + 1).astype(np.uint8)
    _save_ply_with_label(points_out.numpy(), colors_np, labels, save_file)


def export_instance_voxels_ply(
    voxel_map,
    save_file: str,
    transformation=None,        # optional (4, 4), local -> final
    scale=None,                 # optional scalar
):
    """
    Export all occupied voxel centres with PCA-coloured instance feature.

    If transformation / scale is provided, voxel centres are transformed
    from local voxel frame to final output frame before saving.
    """
    centers, features = voxel_map.get_all()
    if centers.shape[0] == 0:
        print("[WARN] export_instance_voxels_ply: voxel map is empty, skipping.")
        return

    centers_out = apply_similarity_transform(
        centers, transformation=transformation, scale=scale
    )

    colors_np = feature_to_pca_color(features)
    pc = trimesh.points.PointCloud(
        centers_out.numpy(), colors=colors_np.astype(np.uint8)
    )
    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
    pc.export(save_file)



def export_instance_per_point_ply(
    points_xyz: torch.Tensor,    # [N, 3] in FINAL output frame
    voxel_map,
    save_file: str,
    transformation=None,        # optional (4, 4), local -> final
    scale=None,                 # optional scalar
):
    """
    Query the instance voxel map for every geometry point, colour by PCA.

    If transformation / scale is provided:
      - points_xyz is assumed to be in FINAL frame
      - query is done in LOCAL voxel-map frame
      - exported coordinates remain in FINAL frame
    """
    points_out = points_xyz.detach().float().cpu()
    points_query = invert_similarity_transform(
        points_out, transformation=transformation, scale=scale
    )

    feat_pp = voxel_map.query(points_query)

    colors_np = feature_to_pca_color(feat_pp)
    pc = trimesh.points.PointCloud(
        points_out.numpy(), colors=colors_np.astype(np.uint8)
    )
    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
    pc.export(save_file)



# ============================================================
# ── Raw feature export (NOT used in run_semantic.py by default)
#    Useful for offline analysis / re-matching with new labels.
# ============================================================

def export_semantic_feat_ply(
    voxel_map,
    save_file: str,
    transformation=None,        # optional (4, 4), local -> final
    scale=None,                 # optional scalar
):
    """
    Save all voxel centres with raw semantic feature vectors as PLY vertex attributes.
    If transformation / scale is provided, voxel centres are exported in final frame.
    """
    centers, features = voxel_map.get_all()
    if centers.shape[0] == 0:
        print("[WARN] export_semantic_feat_ply: voxel map is empty, skipping.")
        return

    centers_out = apply_similarity_transform(
        centers, transformation=transformation, scale=scale
    )

    _write_ply_with_props(
        centers_out.numpy().astype(np.float32),
        {'sem': features.numpy().astype(np.float32)},
        save_file,
    )


def export_instance_feat_ply(
    voxel_map,
    save_file: str,
    transformation=None,        # optional (4, 4), local -> final
    scale=None,                 # optional scalar
):
    """
    Save all voxel centres with raw instance feature vectors as PLY vertex attributes.
    If transformation / scale is provided, voxel centres are exported in final frame.
    """
    centers, features = voxel_map.get_all()
    if centers.shape[0] == 0:
        print("[WARN] export_instance_feat_ply: voxel map is empty, skipping.")
        return

    centers_out = apply_similarity_transform(
        centers, transformation=transformation, scale=scale
    )

    _write_ply_with_props(
        centers_out.numpy().astype(np.float32),
        {'ins': features.numpy().astype(np.float32)},
        save_file,
    )


# ============================================================
# Legend image
# ============================================================

def save_semantic_color_legend(
    labels, color_table, save_file,
    title="Semantic Color Legend", ncols=2,
    figsize_per_item=(4.0, 0.5), font_size=12,
):
    assert len(labels) == len(color_table)
    num_items = len(labels)
    nrows     = math.ceil(num_items / ncols)
    fig_w     = max(6, figsize_per_item[0] * ncols)
    fig_h     = max(2, figsize_per_item[1] * nrows + 1.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, ncols); ax.set_ylim(0, nrows); ax.axis("off")

    for idx, (label, color) in enumerate(zip(labels, color_table)):
        row = idx % nrows; col = idx // nrows
        y   = nrows - row - 1; x = col
        rgb = np.array(color, dtype=np.float32) / 255.0
        ax.add_patch(Rectangle((x+0.05, y+0.2), 0.18, 0.6,
                                facecolor=rgb, edgecolor='black', linewidth=0.8))
        ax.text(x+0.28, y+0.5, label, va='center', ha='left', fontsize=font_size)

    fig.suptitle(title, fontsize=font_size+2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
    plt.savefig(save_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PCA colour helper  (unchanged)
# ============================================================

def feature_to_pca_color(feat: torch.Tensor) -> np.ndarray:
    """[N, C] → [N, 3] uint8 via PCA + min-max normalisation."""
    feat_np = feat.detach().float().cpu().numpy()
    if feat_np.shape[0] < 3:
        return np.zeros((feat_np.shape[0], 3), dtype=np.uint8)
    pca      = PCA(n_components=3)
    feat_3d  = pca.fit_transform(feat_np)
    f_min    = feat_3d.min(axis=0, keepdims=True)
    f_max    = feat_3d.max(axis=0, keepdims=True)
    denom    = np.maximum(f_max - f_min, 1e-8)
    return ((feat_3d - f_min) / denom * 255.0).clip(0, 255).astype(np.uint8)


# ============================================================
# Legacy wrappers (backward compat with training code)
# ============================================================

def export_back_semantic_pca_ply(points_xyz, semantic_feat, save_file):
    """Training-era function: export per-point PCA PLY from raw feat tensor."""
    points_np = points_xyz.detach().float().cpu().numpy()
    colors_np = feature_to_pca_color(semantic_feat)
    pc = trimesh.points.PointCloud(points_np, colors=colors_np)
    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
    pc.export(save_file)


def export_back_semantic_textmatch_ply(points_xyz, semantic_feat,
                                        text_feat, color_table, save_file):
    """Training-era function: text-match PLY from raw feat tensor."""
    points_np = points_xyz.detach().float().cpu().numpy()
    _, colors_np = semantic_feat_to_label_color(semantic_feat, text_feat, color_table)
    pc = trimesh.points.PointCloud(points_np, colors=colors_np.astype(np.uint8))
    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
    pc.export(save_file)