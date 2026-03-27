import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import math
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA

from .scannet200_constants import (
    VALID_CLASS_IDS_20, CLASS_LABELS_20, SCANNET_COLOR_MAP_20,
    VALID_CLASS_IDS_200, CLASS_LABELS_200, SCANNET_COLOR_MAP_200,
)


def get_scannet_label_and_color_map(label_set="scannet20"):
    """
    Returns:
        labels: list[str]
        class_ids: list[int]
        color_table: np.ndarray of shape [num_classes, 3], uint8
    """
    if label_set.lower() in ["scannet20", "scannet-20", "20"]:
        labels = list(CLASS_LABELS_20)
        class_ids = list(VALID_CLASS_IDS_20)
        color_map = SCANNET_COLOR_MAP_20
    elif label_set.lower() in ["scannet200", "scannet-200", "200"]:
        labels = list(CLASS_LABELS_200)
        class_ids = list(VALID_CLASS_IDS_200)
        color_map = SCANNET_COLOR_MAP_200
    else:
        raise ValueError(f"Unknown label_set: {label_set}")

    color_table = []
    for cid in class_ids:
        color = color_map[cid]
        color_table.append([int(color[0]), int(color[1]), int(color[2])])

    color_table = np.asarray(color_table, dtype=np.uint8)
    return labels, class_ids, color_table


# ===================================== SAVE Semantic =====================================
def build_text_embeddings(clip_model, tokenizer, labels, device, template="a photo of a {}"):
    """
    labels: list[str]
    return: [K, C]
    """
    prompts = [template.format(x) for x in labels]

    tokens = tokenizer(prompts).to(device) #
    # tokens = clip_model.tokenize(prompts).to(device)
    with torch.no_grad():
        text_feat = clip_model.encode_text(tokens)

    text_feat = F.normalize(text_feat, dim=-1)
    return text_feat

def semantic_feat_to_label_color(
    semantic_feat: torch.Tensor,     # [N, C]
    text_feat: torch.Tensor,         # [K, C]
    color_table: np.ndarray,         # [K, 3]
):
    """
    return:
        pred_idx: [N] np.int64
        colors: [N, 3] np.uint8
    """
    sem = F.normalize(semantic_feat.detach().float(), dim=-1)
    txt = F.normalize(text_feat.detach().float(), dim=-1)

    sim = sem @ txt.T   # [N, K]
    pred_idx = sim.argmax(dim=-1).cpu().numpy()

    colors = color_table[pred_idx]
    return pred_idx, colors

def export_back_semantic_textmatch_ply(
    points_xyz: torch.Tensor,
    semantic_feat: torch.Tensor,
    text_feat: torch.Tensor,
    color_table: np.ndarray,
    save_file: str,
):
    """
    points_xyz: [N, 3]
    semantic_feat: [N, C]
    text_feat: [K, C]
    color_table: [K, 3]
    """
    # print(f'semantic feat:{semantic_feat.shape}, text feat:{text_feat.shape}')
    points_np = points_xyz.detach().float().cpu().numpy()
    _, colors_np = semantic_feat_to_label_color(
        semantic_feat=semantic_feat,
        text_feat=text_feat,
        color_table=color_table,
    )

    pc = trimesh.points.PointCloud(points_np, colors=colors_np.astype(np.uint8))
    pc.export(save_file)

def save_semantic_color_legend(
    labels,
    color_table,
    save_file,
    title="Semantic Color Legend",
    ncols=2,
    figsize_per_item=(4.0, 0.5),
    font_size=12,
):
    """
    Save a semantic color legend image.

    Args:
        labels: list[str], class names
        color_table: np.ndarray [K, 3], uint8 or int, RGB colors in [0,255]
        save_file: str, output image path (.png)
        title: str
        ncols: int, number of columns in legend
        figsize_per_item: tuple(float, float), controls image size
        font_size: int
    """
    assert len(labels) == len(color_table), \
        f"labels ({len(labels)}) and color_table ({len(color_table)}) must have same length"

    num_items = len(labels)
    nrows = math.ceil(num_items / ncols)

    fig_w = max(6, figsize_per_item[0] * ncols)
    fig_h = max(2, figsize_per_item[1] * nrows + 1.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.axis("off")

    for idx, (label, color) in enumerate(zip(labels, color_table)):
        row = idx % nrows
        col = idx // nrows

        # 讓第一個類別在左上角開始往下排
        y = nrows - row - 1
        x = col

        rgb = np.array(color, dtype=np.float32) / 255.0

        # color box
        rect = Rectangle((x + 0.05, y + 0.2), 0.18, 0.6,
                         facecolor=rgb, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)

        # text
        ax.text(
            x + 0.28, y + 0.5, label,
            va='center', ha='left',
            fontsize=font_size
        )

    fig.suptitle(title, fontsize=font_size + 2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    plt.savefig(save_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ===================================== SAVE PCA Feature =====================================
def feature_to_pca_color(feat: torch.Tensor):
    """
    feat: [N, C] torch tensor
    return: [N, 3] np.uint8
    """
    feat_np = feat.detach().float().cpu().numpy()

    # 防止全部一樣造成 PCA / normalize 問題
    if feat_np.shape[0] < 3:
        colors = np.zeros((feat_np.shape[0], 3), dtype=np.uint8)
        return colors

    pca = PCA(n_components=3)
    feat_3d = pca.fit_transform(feat_np)

    feat_3d_min = feat_3d.min(axis=0, keepdims=True)
    feat_3d_max = feat_3d.max(axis=0, keepdims=True)
    denom = np.maximum(feat_3d_max - feat_3d_min, 1e-8)

    feat_3d_norm = (feat_3d - feat_3d_min) / denom
    colors = (feat_3d_norm * 255.0).clip(0, 255).astype(np.uint8)
    return colors

def export_back_semantic_pca_ply(
    points_xyz: torch.Tensor,
    semantic_feat: torch.Tensor,
    save_file: str,
):
    """
    points_xyz: [N, 3]
    semantic_feat: [N, C]
    """
    points_np = points_xyz.detach().float().cpu().numpy()
    colors_np = feature_to_pca_color(semantic_feat)
    # print("CPAcolors min/max:", colors_np.min(), colors_np.max())
    # print("PCA unique first 10:", np.unique(colors_np.reshape(-1, 3), axis=0)[:10])
    # print("PCA mean color:", colors_np.mean(axis=0))
    pc = trimesh.points.PointCloud(points_np, colors=colors_np)
    pc.export(save_file)