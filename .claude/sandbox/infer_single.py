"""
infer_single.py
===============
Single-pass inference with AMB3RStage1FullFT on a user-supplied list of images.

Edit the two sections marked  ── USER CONFIG ──  then run:
    python infer_single.py

Outputs (all in OUTPUT_DIR):
    scene_results.npz               — pts, conf, pose, kf_idx, sky_mask, images
    scene_semantic_voxels.ply       — semantic colour PLY
    scene_semantic_per_point.ply
    scene_semantic_legend.png
    scene_instance_voxels.ply       — instance PCA-colour PLY
    scene_instance_per_point.ply
    scene_instance_voxel_feats.npz  — for instance_hdbscan_cluster.py
    scene_semantic_voxel_feats.npz  — (optional, see SAVE_SEMANTIC_NPZ)
    scene_geo_kf.ply                — geometry point cloud

To cluster instances afterwards:
    python .claude/sandbox/instance_hdbscan_cluster.py \
        --feats_npz   <OUTPUT_DIR>/scene_instance_voxel_feats.npz \
        --results_npz <OUTPUT_DIR>/scene_results.npz \
        --output      <OUTPUT_DIR>/clusters
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d

import PIL.Image
try:
    _LANCZOS = PIL.Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = PIL.Image.LANCZOS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── USER CONFIG ───────────────────────────────────────────────────────────────

CKPT_PATH = "./checkpoints/amb3r_semantic.pt"

IMAGE_PATHS = [
    # Add your image paths here, e.g.:
    # "/data/scene/color/000000.jpg",
    # "/data/scene/color/000005.jpg",
]

RESOLUTION       = (518, 336)    # (W, H) fed to the model
OUTPUT_DIR       = "./outputs/infer_single/scene"
TARGET_PTS_COUNT = 3_000_000     # random downsample geometry PLY beyond this
CONF_THRESHOLD   = 0.0           # keep pts with conf >= this

SAVE_SEMANTIC      = True
SAVE_INSTANCE      = True
SAVE_SEMANTIC_NPZ  = True
SAVE_INSTANCE_NPZ  = True
LABEL_SET          = "scannet20"   # "scannet20" or "scannet200"

# ── END USER CONFIG ───────────────────────────────────────────────────────────

from amb3r.model_stage1_wo_lora import AMB3RStage1FullFT as AMB3R
from amb3r.tools.pts_vis import get_pts_mask
from amb3r.tools.semantic_vis_utils import (
    get_scannet_label_and_color_map,
    build_text_embeddings,
    save_semantic_color_legend,
    export_semantic_voxels_ply,
    export_semantic_per_point_ply,
    export_instance_voxels_ply,
    export_instance_per_point_ply,
)
from slam_semantic.memory import SLAMemory
from slam_semantic.semantic_voxel_map import VoxelFeatureMap
from lang_seg.modules.models.lseg_net import clip


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(path: str, resolution) -> torch.Tensor:
    """
    Load one image, center-crop to target aspect ratio, resize, normalise to [-1,1].
    Returns (3, H, W) float32.
    """
    target_W, target_H = resolution
    img = PIL.Image.open(path).convert("RGB")
    W, H = img.size

    target_aspect = target_W / target_H
    image_aspect  = W / H

    if image_aspect > target_aspect:
        new_W = int(round(H * target_aspect))
        left  = (W - new_W) // 2
        img   = img.crop((left, 0, left + new_W, H))
    elif image_aspect < target_aspect:
        new_H = int(round(W / target_aspect))
        top   = (H - new_H) // 2
        img   = img.crop((0, top, W, top + new_H))

    img = img.resize((target_W, target_H), resample=_LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0 * 2.0 - 1.0  # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1)                  # (3, H, W)


def load_images(paths, resolution):
    """Returns (1, T, 3, H, W) float32 tensor in [-1, 1]."""
    frames = [load_image(p, resolution) for p in paths]
    return torch.stack(frames).unsqueeze(0)   # (1, T, 3, H, W)


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str) -> AMB3R:
    model = AMB3R(metric_scale=True, clip_dim=512, sem_dim=512, ins_dim=16)
    if os.path.isfile(ckpt_path):
        model.load_weights(ckpt_path, data_type='bf16', strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint not found: {ckpt_path} — using random weights.")
    return model.cuda().eval()


# ── Helpers ───────────────────────────────────────────────────────────────────

def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones  = np.ones((len(pts), 1), dtype=pts.dtype)
    pts_h = np.concatenate([pts, ones], axis=1)
    return (T @ pts_h.T).T[:, :3]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    assert IMAGE_PATHS, "Set IMAGE_PATHS in the USER CONFIG section."
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    T = len(IMAGE_PATHS)
    print(f"Loading {T} images at {RESOLUTION[0]}×{RESOLUTION[1]} …")
    images = load_images(IMAGE_PATHS, RESOLUTION)   # (1, T, 3, H, W)
    _, _, _, H, W = images.shape

    # ── Model forward ─────────────────────────────────────────────────────
    model = load_model(CKPT_PATH)
    print(f"Running inference on {T} frames …")
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        res = model.forward({'images': images.cuda()})[0]

    # Core outputs
    pts_pred  = res['world_points'][0].float().cpu()         # (T, H, W, 3)
    conf_pred = res['world_points_conf'][0].float().cpu()    # (T, H, W)
    pose_pred = res['pose'][0].float().cpu()                 # (T, 4, 4)

    conf_sig = (conf_pred - 1) / conf_pred
    conf_sig[conf_sig == 0] = 1e-6

    # All frames are keyframes (no SLAM loop)
    kf_idx = torch.arange(T)

    has_semantic = 'semantic_feat' in res
    has_instance = 'instance_feat' in res

    # ── Voxel maps ────────────────────────────────────────────────────────
    voxel_size = 0.05
    semantic_voxel_map = VoxelFeatureMap(voxel_size=voxel_size, feat_dim=512) if has_semantic else None
    instance_voxel_map = VoxelFeatureMap(voxel_size=voxel_size, feat_dim=16)  if has_instance else None

    if has_semantic:
        SLAMemory._push_to_voxel_map(
            semantic_voxel_map,
            pts_pred,
            res['semantic_feat'][0].float().cpu(),   # (T, C, H_f, W_f)
            res['semantic_conf'][0].float().cpu(),
        )
        print(f"Semantic voxels: {semantic_voxel_map.num_voxels:,}")

    if has_instance:
        SLAMemory._push_to_voxel_map(
            instance_voxel_map,
            pts_pred,
            res['instance_feat'][0].float().cpu(),
            res['instance_conf'][0].float().cpu(),
        )
        print(f"Instance voxels: {instance_voxel_map.num_voxels:,}")

    # ── Geometry PLY ──────────────────────────────────────────────────────
    pts_mask, sky_mask = get_pts_mask(
        pts_pred, images[0], conf_sig,
        conf_threshold=CONF_THRESHOLD,
    )

    pts_kf   = pts_pred[kf_idx].numpy().reshape(-1, 3)
    color_kf = (
        images[0, kf_idx].numpy().transpose(0, 2, 3, 1).reshape(-1, 3) + 1.0
    ) / 2.0
    mask_kf  = pts_mask[kf_idx].reshape(-1).numpy()
    pts_kf   = pts_kf[mask_kf]
    color_kf = color_kf[mask_kf]

    sampled_indices = None
    if len(pts_kf) > TARGET_PTS_COUNT:
        print(f"Downsampling: {len(pts_kf):,} → {TARGET_PTS_COUNT:,}")
        sampled_indices = np.random.choice(len(pts_kf), TARGET_PTS_COUNT, replace=False)
        pts_kf   = pts_kf[sampled_indices]
        color_kf = color_kf[sampled_indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_kf)
    pcd.colors = o3d.utility.Vector3dVector(color_kf)
    geo_ply = os.path.join(OUTPUT_DIR, "scene_geo_kf.ply")
    o3d.io.write_point_cloud(geo_ply, pcd)
    print(f"Saved geometry PLY         → {geo_ply}")

    # ── results.npz (needed by instance_hdbscan_cluster.py) ──────────────
    results_npz = os.path.join(OUTPUT_DIR, "scene_results.npz")
    np.savez_compressed(
        results_npz,
        pts      = pts_pred.numpy(),
        conf     = conf_sig.numpy(),
        pose     = pose_pred.numpy(),
        images   = images[0].numpy(),
        sky_mask = sky_mask,
        kf_idx   = kf_idx.numpy(),
    )
    print(f"Saved results.npz          → {results_npz}")

    # ── Semantic export ───────────────────────────────────────────────────
    T_voxel = None  # no coordinate transform for single-pass inference

    if SAVE_SEMANTIC and semantic_voxel_map is not None:
        labels, _, color_table = get_scannet_label_and_color_map(LABEL_SET)
        device    = next(model.parameters()).device
        text_feat = build_text_embeddings(
            clip_model=model.lseg.clip_pretrained,
            tokenizer=clip.tokenize,
            labels=labels,
            device=device,
        ).cpu()

        export_semantic_voxels_ply(
            semantic_voxel_map, text_feat, color_table,
            os.path.join(OUTPUT_DIR, "scene_semantic_voxels.ply"),
            transformation=T_voxel,
        )
        export_semantic_per_point_ply(
            torch.from_numpy(pts_kf), semantic_voxel_map,
            text_feat, color_table,
            os.path.join(OUTPUT_DIR, "scene_semantic_per_point.ply"),
            transformation=T_voxel,
        )
        save_semantic_color_legend(
            labels, color_table,
            os.path.join(OUTPUT_DIR, "scene_semantic_legend.png"),
        )
        print("Saved semantic PLYs + legend")

        if SAVE_SEMANTIC_NPZ:
            centers, features = semantic_voxel_map.get_all()
            np.savez_compressed(
                os.path.join(OUTPUT_DIR, "scene_semantic_voxel_feats.npz"),
                voxel_centers  = centers.numpy().astype(np.float32),
                voxel_features = features.numpy().astype(np.float32),
            )
            print("Saved scene_semantic_voxel_feats.npz")

    # ── Instance export ───────────────────────────────────────────────────
    if SAVE_INSTANCE and instance_voxel_map is not None:
        export_instance_voxels_ply(
            instance_voxel_map,
            os.path.join(OUTPUT_DIR, "scene_instance_voxels.ply"),
            transformation=T_voxel,
        )
        export_instance_per_point_ply(
            torch.from_numpy(pts_kf), instance_voxel_map,
            os.path.join(OUTPUT_DIR, "scene_instance_per_point.ply"),
            transformation=T_voxel,
        )
        print("Saved instance PLYs")

        if SAVE_INSTANCE_NPZ:
            centers, features = instance_voxel_map.get_all()
            np.savez_compressed(
                os.path.join(OUTPUT_DIR, "scene_instance_voxel_feats.npz"),
                voxel_centers  = centers.numpy().astype(np.float32),
                voxel_features = features.numpy().astype(np.float32),
            )
            print("Saved scene_instance_voxel_feats.npz")

    print("\nDone. To cluster instances:")
    print(f"  python .claude/sandbox/instance_hdbscan_cluster.py \\")
    print(f"      --feats_npz   {OUTPUT_DIR}/scene_instance_voxel_feats.npz \\")
    print(f"      --results_npz {OUTPUT_DIR}/scene_results.npz \\")
    print(f"      --output      {OUTPUT_DIR}/clusters")


if __name__ == '__main__':
    main()
