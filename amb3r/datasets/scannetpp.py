import os
import cv2
import glob
import numpy as np
import os.path as osp
from collections import deque

from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset
from amb3r.tools.utils import threshold_depth_map
from amb3r.tools.pose_dist import compute_ranking


class Scannetpp(BaseManyViewDataset):
    """
    ScanNet++ loader that keeps the same external behavior / output-view format
    as scannet.py, while only changing the raw data reading logic.

    Expected raw layout:
        ROOT/
          processed_scannetpp_v2/
            <scene_name>/
              images/frame_*.jpg
              depth/frame_*.png
              refined_ins_ids/frame_*.jpg.npy
              new_scene_metadata.npz
                or
              scene_iphone_metadata.npz
    """

    def __init__(self, num_seq=1,
                 test_id=None, full_video=False,
                 kf_every=1, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every

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

    def _get_metadata_path(self, scene_dir):
        default_path = osp.join(scene_dir, 'new_scene_metadata.npz')
        iphone_path = osp.join(scene_dir, 'scene_iphone_metadata.npz')

        if osp.exists(default_path):
            return default_path
        if osp.exists(iphone_path):
            return iphone_path

        raise FileNotFoundError(
            f'Cannot find metadata file in {scene_dir}. '
            f'Expected new_scene_metadata.npz or scene_iphone_metadata.npz.'
        )

    def _load_scene_data(self, scene_dir):
        # only use the real frame images, ignore any other images in the folder
        rgb_paths = sorted(glob.glob(osp.join(scene_dir, 'images', 'frame_*.jpg')))
        depth_paths = sorted(glob.glob(osp.join(scene_dir, 'depth', 'frame_*.png')))
        instance_mask_paths = [
            p.replace(f'{os.sep}images{os.sep}', f'{os.sep}refined_ins_ids{os.sep}') + '.npy'
            for p in rgb_paths
        ]

        metadata_path = self._get_metadata_path(scene_dir)
        annotations = np.load(metadata_path, allow_pickle=True)

        image_list = annotations['images']
        dsc_count = len([s for s in image_list if str(s).startswith('DSC')])

        pose_list = []
        for index, anno in enumerate(annotations['trajectories']):
            if index >= dsc_count:
                pose = np.array(anno, dtype=np.float32)
                assert pose.shape == (4, 4), f"Pose shape mismatch in {metadata_path}: {pose.shape}"
                pose_list.append(pose)

        intri_list = []
        for index, anno in enumerate(annotations['intrinsics']):
            if index >= dsc_count:
                intri = np.array(anno, dtype=np.float32)
                assert intri.shape == (3, 3), f"Intrinsic shape mismatch in {metadata_path}: {intri.shape}"
                intri_list.append(intri)

        pose_all = np.stack(pose_list, axis=0)
        intri_all = np.stack(intri_list, axis=0)

        n = len(rgb_paths)
        assert len(depth_paths) == n and \
               len(instance_mask_paths) == n and \
               len(pose_all) == n and \
               len(intri_all) == n, \
               (f"Number mismatch in {scene_dir}: rgb={len(rgb_paths)}, depth={len(depth_paths)}, "
                f"mask={len(instance_mask_paths)}, pose={len(pose_all)}, intri={len(intri_all)}")

        # make sure all expected mask files exist
        missing_masks = [p for p in instance_mask_paths if not osp.exists(p)]
        if len(missing_masks) > 0:
            raise FileNotFoundError(
                f"Missing {len(missing_masks)} refined_ins_ids mask files in {scene_dir}. "
                f"First missing file: {missing_masks[0]}"
            )

        return rgb_paths, depth_paths, instance_mask_paths, pose_all, intri_all

    def _load_instance_mask(self, mask_path):
        mask = np.load(mask_path)

        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        return mask.astype(np.int32)

    def _get_views(self, idx, resolution, num_frames, rng, attempts=0):
        scene_id = self.scene_list[idx // self.num_seq]
        scene_dir = osp.join(self.ROOT, self.folder, scene_id)

        rgb_paths, depth_paths, instance_mask_paths, pose_all, intri_all = self._load_scene_data(scene_dir)
        num_files = len(rgb_paths)

        rank_path = osp.join(scene_dir, 'ranking.npy')
        if os.path.exists(rank_path):
            ranking = np.load(rank_path)
        else:
            ranking, dist = compute_ranking(pose_all, lambda_t=1.0, normalize=True, batched=True)
            np.save(rank_path, ranking[:, :512])
            ranking = ranking[:, :512]
            print(f"Ranking saved to {rank_path}")

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

            impath = rgb_paths[im_idx]
            depthpath = depth_paths[im_idx]
            instance_mask_path = instance_mask_paths[im_idx]

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)

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
                rgb_image, depthmap, intri, resolution, rng=rng, info=impath
            )
            rng_state_after = rng.bit_generator.state

            instance_mask = self._load_instance_mask(instance_mask_path)
            if instance_mask.shape[:2] != rgb_ori.shape[:2]:
                instance_mask = cv2.resize(
                    instance_mask,
                    (rgb_ori.shape[1], rgb_ori.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            rng.bit_generator.state = rng_state_before
            _, instance_mask, _ = self._crop_resize_if_necessary(
                rgb_ori, instance_mask, intri_ori, resolution, rng=rng, info=impath
            )
            rng.bit_generator.state = rng_state_after
            instance_mask = np.asarray(instance_mask).astype(np.int32)

            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0 or (not np.isfinite(camera_pose).all()):
                if self.full_video:
                    print(f"Warning: No valid depthmap found for {impath}")
                    continue
                else:
                    if attempts >= 5:
                        new_idx = rng.integers(0, self.__len__() - 1)
                        return self._get_views(new_idx, resolution, num_frames, rng)
                    return self._get_views(idx, resolution, num_frames, rng, attempts + 1)

            frame_stem = osp.splitext(osp.basename(impath))[0]

            dict_info = dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                instance_mask=instance_mask,
                dataset='scannetpp',
                label=osp.join(scene_id, frame_stem),
                instance=osp.split(impath)[1],
                is_metric=True,
                orig_img=rgb_ori,
                orig_depthmap=depth_ori,
                orig_camera_intrinsics=intri_ori
            )

            views.append(dict_info)

        return views


if __name__ == "__main__":
    print('loading dataset')

    dataset = Scannetpp(
        split='train',
        ROOT="./data/scannetpp",
        resolution=224,
        num_seq=1,
        num_frames=5,
    )
