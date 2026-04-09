"""
ScannetppSequence
==================
Extends Scannetpp_Arrow to return temporally-consecutive frame sequences
suitable for Stage-2 recurrent training.

Key differences from Scannetpp_Arrow:
  - _get_views samples `num_frames` consecutive frames (by frame_index order)
    instead of ranking-based random selection.
  - A `stride` parameter controls subsampling (default=1 → every frame).
  - Frames are always returned in temporal order.

The output format is identical to Scannetpp_Arrow so it plugs straight into
the existing BaseStereoViewDataset.__getitem__ collation pipeline.

Example (training script):
    trainset = (
        "ScannetppSequence("
        "  split='train',"
        "  ROOT='/mnt/HDD4/ricky/data/scannetpp',"
        "  resolution=[(518,336)],"
        "  num_frames=24,"   # seq_len = n_chunks * chunk_size
        "  num_seq=1,"
        "  stride=2,"
        ")"
    )
"""

import os
import cv2
import numpy as np
import os.path as osp
from collections import deque

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from amb3r.datasets.scannetpp_arrow import Scannetpp_Arrow


class ScannetppSequence(Scannetpp_Arrow):
    """
    Consecutive-frame sequence dataset for Stage-2 training.

    Extra parameters (on top of Scannetpp_Arrow):
        stride : int   — temporal subsampling interval (default 1 = every frame)
    """

    def __init__(self, stride: int = 1, *args, **kwargs):
        self.stride = int(stride)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    def _get_views(self, idx, resolution, num_frames, rng, attempts=0):
        """
        Returns `num_frames` consecutive frames from a random start position.

        Consecutive here means ordered by original frame_index with step=stride.
        If the scene has fewer than num_frames * stride frames we wrap around.
        """
        scene_id  = self.scene_list[idx // self.num_seq]
        scene_dir = osp.join(self.ROOT, self.folder, scene_id)

        frame_records, pose_all, intri_all, _ = self._load_scene_data(scene_dir)
        total = len(frame_records)

        # ── Pick a random start and build the index list ──────────────────
        needed  = num_frames * self.stride
        if total >= needed:
            max_start = total - needed
            start = int(rng.integers(0, max_start + 1))
            raw_indices = [start + i * self.stride for i in range(num_frames)]
        else:
            # scene is shorter than the sequence: tile and truncate
            start = int(rng.integers(0, total))
            raw_indices = [(start + i * self.stride) % total for i in range(num_frames)]

        views = []
        for im_idx in raw_indices:
            record      = frame_records[im_idx]
            frame_name  = record['frame_name']
            frame_stem  = osp.splitext(frame_name)[0]
            frame_info  = osp.join(scene_id, frame_name)

            rgb_image = self._decode_image_bytes(record['image_bytes'])
            depthmap  = self._decode_depth_bytes(record['depth_bytes'])

            if rgb_image.shape[:2] != depthmap.shape[:2]:
                rgb_image = cv2.resize(
                    rgb_image, (depthmap.shape[1], depthmap.shape[0])
                )

            from amb3r.tools.utils import threshold_depth_map
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap = threshold_depth_map(
                depthmap, min_percentile=-1, max_percentile=98
            ).astype(np.float32)

            camera_pose = pose_all[im_idx].astype(np.float32)
            intri       = intri_all[im_idx].astype(np.float32)

            rgb_ori   = rgb_image.copy()
            depth_ori = depthmap.copy()
            intri_ori = intri.copy()

            # ── Crop/resize RGB + depth ───────────────────────────────────
            rng_state_before = rng.bit_generator.state
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intri, resolution, rng=rng, info=frame_info
            )
            rng_state_after = rng.bit_generator.state

            # ── Crop/resize instance mask (nearest, same RNG state) ───────
            instance_mask = self._decode_mask_bytes(record['mask_bytes'])
            if instance_mask.shape[:2] != rgb_ori.shape[:2]:
                instance_mask = cv2.resize(
                    instance_mask,
                    (rgb_ori.shape[1], rgb_ori.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            rng.bit_generator.state = rng_state_before
            _, instance_mask, _ = self._crop_resize_if_necessary(
                rgb_ori, instance_mask, intri_ori, resolution, rng=rng, info=frame_info
            )
            rng.bit_generator.state = rng_state_after
            instance_mask = np.asarray(instance_mask).astype(np.int32)

            # ── Validity check ────────────────────────────────────────────
            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0 or not np.isfinite(camera_pose).all():
                # replace bad frame with the previous valid one (or retry)
                if len(views) > 0:
                    views.append(views[-1])   # duplicate last good frame
                    continue
                elif attempts < 5:
                    return self._get_views(idx, resolution, num_frames, rng,
                                           attempts + 1)
                else:
                    new_idx = int(rng.integers(0, len(self)))
                    return self._get_views(new_idx, resolution, num_frames, rng)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                instance_mask=instance_mask,
                dataset='scannetpp_seq',
                label=osp.join(scene_id, frame_stem),
                instance=frame_name,
                is_metric=True,
                orig_img=rgb_ori,
                orig_depthmap=depth_ori,
                orig_camera_intrinsics=intri_ori,
            ))

        return views
