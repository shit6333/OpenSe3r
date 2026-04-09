"""
scannet_slam_gt_depth.py
========================
ScanNet demo dataset that additionally loads GT depth maps.

Expected directory layout (standard ScanNet v2 export):

    ROOT/
        color/             ← JPEG/PNG color images  (1296×968 or similar)
        depth/             ← 16-bit PNG depth maps   (640×480)
        poses.npy          ← (T, 4, 4) c2w poses
        intrinsic/
            intrinsic_depth.txt   ← 4×4 depth-camera intrinsic matrix

Depth maps are loaded in the DEPTH camera coordinate system (640×480).
Color images are first resized to the depth resolution so that both share the
same spatial grid, then both are center-cropped and resized to `resolution`.
Depth intrinsics are updated to track the crop/resize transformations.

Depth units: raw 16-bit values are divided by 1000 to obtain metres; pixels
with value 0 (invalid / no return) remain 0.0 m.
"""

import os
import os.path as osp
from glob import glob

import cv2
import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset

try:
    lanczos = PIL.Image.Resampling.LANCZOS
except AttributeError:
    lanczos = PIL.Image.LANCZOS


class ScannetDemoDatasetWithDepth(Dataset):
    """
    Drop-in replacement for ScannetDemoDataset that also returns depth maps.

    Args:
        ROOT:       path to a single ScanNet scene directory.
        resolution: (target_W, target_H) for color images and depth maps.

    __getitem__ returns (views, views_all) where views_all contains:
        images            (T, 3, H, W)  float32  in [-1, 1]
        camera_pose       (T, 4, 4)     float32  c2w
        camera_intrinsics (T, 3, 3)     float32  depth-cam intrinsics at target res
        depth             (T, H, W)     float32  metres; 0 = invalid
    """

    DEPTH_SCALE = 1000.0   # ScanNet raw value / DEPTH_SCALE = metres

    def __init__(self, ROOT: str, resolution=(518, 336), **kwargs):
        self.ROOT       = ROOT
        self.resolution = resolution   # (W, H)

        self.color_dir  = osp.join(ROOT, "color")
        self.depth_dir  = osp.join(ROOT, "depth")
        self.pose_path  = osp.join(ROOT, "poses.npy")
        self.intri_path = osp.join(ROOT, "intrinsic", "intrinsic_depth.txt")

        if not osp.isdir(self.color_dir):
            raise FileNotFoundError(f"Color directory not found: {self.color_dir}")
        if not osp.isdir(self.depth_dir):
            raise FileNotFoundError(
                f"Depth directory not found: {self.depth_dir}\n"
                "This dataset requires GT depth maps — ensure 'depth/' exists."
            )

        self.pose_all = np.load(self.pose_path).astype(np.float32)
        self.intri    = np.loadtxt(self.intri_path).astype(np.float32)[:3, :3]

        # Sort color and depth files together
        self._color_files = sorted(
            f for f in os.listdir(self.color_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        self._depth_files = sorted(
            f for f in os.listdir(self.depth_dir)
            if f.lower().endswith('.png')
        )

        # Verify counts match poses
        n_frames = len(self._color_files)
        assert n_frames == len(self._depth_files), (
            f"Color ({n_frames}) and depth ({len(self._depth_files)}) frame counts differ"
        )
        assert n_frames == len(self.pose_all), (
            f"Color ({n_frames}) and pose ({len(self.pose_all)}) counts differ"
        )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        views = self._get_views()

        images      = [
            torch.from_numpy(np.asarray(v['img'])).permute(2, 0, 1).float()
            / 255.0 * 2.0 - 1.0
            for v in views
        ]
        poses       = [torch.from_numpy(v['camera_pose'])       for v in views]
        intrinsics  = [torch.from_numpy(v['camera_intrinsics']) for v in views]
        depths      = [torch.from_numpy(v['depth'])             for v in views]

        views_all = {
            'images':            torch.stack(images),       # (T, 3, H, W)
            'camera_pose':       torch.stack(poses),        # (T, 4, 4)
            'camera_intrinsics': torch.stack(intrinsics),   # (T, 3, 3)
            'depth':             torch.stack(depths),       # (T, H, W)
        }
        return views, views_all

    # ──────────────────────────────────────────────────────────────────────
    def _get_views(self):
        views = []
        for i, color_name in enumerate(self._color_files):
            # ── Color image ───────────────────────────────────────────────
            color_path = osp.join(self.color_dir, color_name)
            color_img  = PIL.Image.open(color_path).convert('RGB')

            # ── Depth map ────────────────────────────────────────────────
            # Use the matching depth file (same index)
            depth_name = self._depth_files[i]
            depth_path = osp.join(self.depth_dir, depth_name)
            depth_raw  = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise FileNotFoundError(f"Cannot read depth: {depth_path}")
            depth_m = depth_raw.astype(np.float32) / self.DEPTH_SCALE  # metres

            # ── Align color to depth resolution ──────────────────────────
            depth_H, depth_W = depth_m.shape
            color_img = color_img.resize((depth_W, depth_H), resample=lanczos)

            # ── Shared center-crop + resize to target resolution ─────────
            intri = self.intri.copy()
            color_img, depth_m, intri = self._crop_resize(
                color_img, depth_m, intri, self.resolution
            )

            views.append(dict(
                img               = np.asarray(color_img),   # (H, W, 3) uint8
                depth             = depth_m,                  # (H, W) float32 metres
                camera_pose       = self.pose_all[i],         # (4, 4)
                camera_intrinsics = intri,                    # (3, 3)
                dataset           = 'scannet_demo_gt_depth',
                label             = osp.join(osp.basename(self.ROOT), f'{i:06d}'),
                instance          = color_name,
                is_metric         = True,
            ))
        return views

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _crop_resize(
        color_img: PIL.Image.Image,
        depth_m:   np.ndarray,
        intrinsics: np.ndarray,
        resolution,          # (target_W, target_H)
    ):
        """
        Center-crop both color and depth to the target aspect ratio,
        then resize both to `resolution`.  Intrinsics are updated in-place.

        Depth is resized with nearest-neighbour to avoid blending valid/invalid
        depth across discontinuities.
        """
        W, H = color_img.size          # PIL gives (W, H)
        target_W, target_H = resolution
        intri = intrinsics.copy().astype(np.float32)

        # Handle portrait orientation
        if target_W >= target_H and H > 1.1 * W:
            target_W, target_H = target_H, target_W

        target_aspect = target_W / target_H
        image_aspect  = W / H

        if image_aspect > target_aspect:
            # Crop width
            new_W = int(round(H * target_aspect))
            left  = (W - new_W) // 2
            color_img = color_img.crop((left, 0, left + new_W, H))
            depth_m   = depth_m[:, left: left + new_W]
            intri[0, 2] -= left
            crop_W, crop_H = new_W, H

        elif image_aspect < target_aspect:
            # Crop height
            new_H = int(round(W / target_aspect))
            top   = (H - new_H) // 2
            color_img = color_img.crop((0, top, W, top + new_H))
            depth_m   = depth_m[top: top + new_H, :]
            intri[1, 2] -= top
            crop_W, crop_H = W, new_H

        else:
            crop_W, crop_H = W, H

        # Resize
        color_img = color_img.resize((target_W, target_H), resample=lanczos)
        depth_m   = cv2.resize(
            depth_m, (target_W, target_H), interpolation=cv2.INTER_NEAREST
        )

        scale_x = target_W / float(crop_W)
        scale_y = target_H / float(crop_H)

        intri[0, 0] *= scale_x
        intri[1, 1] *= scale_y
        intri[0, 2] *= scale_x
        intri[1, 2] *= scale_y

        return color_img, depth_m.astype(np.float32), intri.astype(np.float32)
