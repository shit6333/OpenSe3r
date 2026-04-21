import io
import os
import cv2
import numpy as np
import os.path as osp
from collections import deque

import pyarrow as pa
import pyarrow.ipc as ipc

from .base_many_view_dataset import BaseManyViewDataset
from amb3r.tools.utils import threshold_depth_map
from amb3r.tools.pose_dist import compute_ranking


class Scannetpp_Arrow(BaseManyViewDataset):
    """
    ScanNet++ loader for per-scene Arrow export.

    Expected Arrow layout:
        ROOT/
          train/
            <scene_name>/
              sensor.arrow   # one row per frame: image/depth/mask bytes
              meta.arrow     # one row per frame: pose/intrinsics/frame info
              scene.arrow    # one row per scene: ranking + scene-level info
          val/
            <scene_name>/
              sensor.arrow
              meta.arrow
              scene.arrow

    Notes:
    - Keeps the same external behavior / output-view format as the original
      raw-file Scannetpp dataset class.
    - RGB / depth / mask are assumed to have already been resized during export
      (for example with scale=0.5), and intrinsics are assumed to already match
      those stored arrays.
    """

    def __init__(
        self,
        num_seq=1,
        test_id=None,
        full_video=False,
        kf_every=1,
        cache_scenes=False,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.cache_scenes = cache_scenes
        self._scene_cache = {}

        self.load_all_scenes(ROOT)

    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):
        self.folder = {
            'train': 'train',
            'val': 'val',
            'test': 'val',
        }[self.split]

        split_root = osp.join(base_dir, self.folder)
        if not osp.isdir(split_root):
            raise FileNotFoundError(f"Split root not found: {split_root}")

        if self.test_id is None:
            self.scene_list = [
                d for d in os.listdir(split_root)
                if os.path.isdir(osp.join(split_root, d))
            ]
        else:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]
            print(f"Test_id: {self.test_id}")

    def _open_arrow_table(self, arrow_path):
        if not osp.exists(arrow_path):
            raise FileNotFoundError(f"Missing arrow file: {arrow_path}")
        with pa.memory_map(arrow_path, 'r') as source:
            return ipc.open_file(source).read_all()

    def _decode_image_bytes(self, image_bytes):
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            raise RuntimeError("Failed to decode RGB image bytes")
        return img

    def _decode_depth_bytes(self, depth_bytes):
        arr = np.frombuffer(depth_bytes, dtype=np.uint8)
        depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError("Failed to decode depth bytes")
        return depth

    def _decode_mask_bytes(self, mask_bytes):
        with np.load(io.BytesIO(mask_bytes), allow_pickle=False) as data:
            if 'mask' in data:
                mask = data['mask']
            else:
                # Fallback for older export variants.
                keys = list(data.keys())
                if len(keys) == 0:
                    raise KeyError("Mask NPZ contains no arrays")
                mask = data[keys[0]]

        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        return np.asarray(mask).astype(np.int32)

    def _load_scene_data(self, scene_dir):
        if self.cache_scenes and scene_dir in self._scene_cache:
            return self._scene_cache[scene_dir]

        sensor_path = osp.join(scene_dir, 'sensor.arrow')
        meta_path = osp.join(scene_dir, 'meta.arrow')
        scene_path = osp.join(scene_dir, 'scene.arrow')

        sensor_table = self._open_arrow_table(sensor_path).to_pydict()
        meta_table = self._open_arrow_table(meta_path).to_pydict()
        scene_table = self._open_arrow_table(scene_path).to_pydict()

        num_sensor = len(sensor_table['frame_index'])
        num_meta = len(meta_table['frame_index'])
        if num_sensor != num_meta:
            raise ValueError(
                f"sensor/meta row mismatch in {scene_dir}: "
                f"sensor={num_sensor}, meta={num_meta}"
            )
        if len(scene_table['scene_id']) != 1:
            raise ValueError(f"scene.arrow in {scene_dir} must contain exactly one row")

        sensor_order = np.argsort(np.asarray(sensor_table['frame_index'], dtype=np.int32))
        meta_order = np.argsort(np.asarray(meta_table['frame_index'], dtype=np.int32))

        frame_records = []
        pose_all = []
        intri_all = []

        for s_idx, m_idx in zip(sensor_order.tolist(), meta_order.tolist()):
            s_frame_name = sensor_table['frame_name'][s_idx]
            m_frame_name = meta_table['frame_name'][m_idx]
            if s_frame_name != m_frame_name:
                raise ValueError(
                    f"frame_name mismatch in {scene_dir}: sensor={s_frame_name}, meta={m_frame_name}"
                )

            camera_pose = np.asarray(meta_table['camera_pose'][m_idx], dtype=np.float32).reshape(4, 4)
            camera_intrinsics = np.asarray(meta_table['camera_intrinsics'][m_idx], dtype=np.float32).reshape(3, 3)

            record = {
                'frame_index': int(sensor_table['frame_index'][s_idx]),
                'frame_name': s_frame_name,
                'height': int(sensor_table['height'][s_idx]),
                'width': int(sensor_table['width'][s_idx]),
                'image_bytes': sensor_table['image_bytes'][s_idx],
                'depth_bytes': sensor_table['depth_bytes'][s_idx],
                'mask_bytes': sensor_table['mask_bytes'][s_idx],
                'camera_pose': camera_pose,
                'camera_intrinsics': camera_intrinsics,
            }
            frame_records.append(record)
            pose_all.append(camera_pose)
            intri_all.append(camera_intrinsics)

        pose_all = np.stack(pose_all, axis=0)
        intri_all = np.stack(intri_all, axis=0)

        ranking = None
        if 'ranking_bytes' in scene_table and scene_table['ranking_bytes'][0] is not None:
            ranking_bytes = scene_table['ranking_bytes'][0]
            ranking = np.load(io.BytesIO(ranking_bytes), allow_pickle=False)
            ranking = np.asarray(ranking)

        if ranking is None:
            ranking, _ = compute_ranking(pose_all, lambda_t=1.0, normalize=True, batched=True)
            ranking = ranking[:, :512]
            print(f"Ranking computed on-the-fly for {scene_dir}")

        if ranking.ndim != 2 or ranking.shape[0] != len(frame_records):
            raise ValueError(
                f"Invalid ranking shape in {scene_dir}: {ranking.shape}, "
                f"expected ({len(frame_records)}, K)"
            )

        result = (frame_records, pose_all, intri_all, ranking)
        if self.cache_scenes:
            self._scene_cache[scene_dir] = result
        return result

    def _get_views(self, idx, resolution, num_frames, rng, attempts=0):
        scene_id = self.scene_list[idx // self.num_seq]
        scene_dir = osp.join(self.ROOT, self.folder, scene_id)

        frame_records, pose_all, intri_all, ranking = self._load_scene_data(scene_dir)
        num_files = len(frame_records)

        while True:
            ref_idx = np.random.randint(0, num_files)
            k = 26
            topk = ranking[ref_idx][1:k + 1]
            replace = False if len(topk) >= num_frames - 1 else True
            support_ids = np.random.choice(topk, size=num_frames - 1, replace=replace)

            imgs_idxs = [ref_idx] + support_ids.tolist()
            poses = pose_all[imgs_idxs]

            if np.isfinite(poses).all():
                break

        imgs_idxs = deque(imgs_idxs)
        views = []

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()
            record = frame_records[im_idx]
            frame_name = record['frame_name']
            frame_stem = osp.splitext(frame_name)[0]
            frame_info = osp.join(scene_id, frame_name)

            rgb_image = self._decode_image_bytes(record['image_bytes'])
            depthmap = self._decode_depth_bytes(record['depth_bytes'])

            # Keep the same behavior as the original loader: RGB is aligned to
            # depth resolution before later crop / resize.
            if rgb_image.shape[:2] != depthmap.shape[:2]:
                rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap = threshold_depth_map(
                depthmap, min_percentile=-1, max_percentile=98
            ).astype(np.float32)

            camera_pose = pose_all[im_idx].astype(np.float32)
            intri = intri_all[im_idx].astype(np.float32)

            rgb_ori = rgb_image.copy()
            depth_ori = depthmap.copy()
            intri_ori = intri.copy()

            rng_state_before = rng.bit_generator.state
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intri, resolution, rng=rng, info=frame_info
            )
            rng_state_after = rng.bit_generator.state

            instance_mask = self._decode_mask_bytes(record['mask_bytes'])
            if instance_mask.shape[:2] != rgb_ori.shape[:2]:
                instance_mask = cv2.resize(
                    instance_mask,
                    (rgb_ori.shape[1], rgb_ori.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            rng.bit_generator.state = rng_state_before
            _, instance_mask, _ = self._crop_resize_if_necessary(
                rgb_ori, instance_mask, intri_ori, resolution, rng=rng, info=frame_info
            )
            rng.bit_generator.state = rng_state_after
            instance_mask = np.asarray(instance_mask).astype(np.int32)

            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0 or (not np.isfinite(camera_pose).all()):
                if self.full_video:
                    print(f"Warning: No valid depthmap found for {frame_info}")
                    continue
                else:
                    if attempts >= 5:
                        new_idx = rng.integers(0, self.__len__() - 1)
                        return self._get_views(new_idx, resolution, num_frames, rng)
                    return self._get_views(idx, resolution, num_frames, rng, attempts + 1)

            dict_info = dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                instance_mask=instance_mask,
                dataset='scannetpp',
                label=osp.join(scene_id, frame_stem),
                instance=frame_name,
                is_metric=True,
                orig_img=rgb_ori,
                orig_depthmap=depth_ori,
                orig_camera_intrinsics=intri_ori,
            )

            views.append(dict_info)

        return views


if __name__ == "__main__":
    print('loading arrow dataset')

    dataset = Scannetpp_Arrow(
        split='train',
        ROOT='./data/scannetpp_arrow',
        resolution=224,
        num_seq=1,
        num_frames=5,
    )
    print(f"Loaded {len(dataset)} samples from {len(dataset.scene_list)} scenes")
