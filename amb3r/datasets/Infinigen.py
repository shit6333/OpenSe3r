import os
import cv2
import glob
import numpy as np
import os.path as osp
from collections import deque
from PIL import Image

from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset
from amb3r.tools.utils import threshold_depth_map
from amb3r.tools.pose_dist import compute_ranking


class InfinigenManyView(BaseManyViewDataset):
    """
    Infinigen loader with Scannetpp-like output format.

    Expected raw layout:
        ROOT/
          train/ or val/ or test/   # mapped by self.split
            scene_xxx/
              subscene_xxx/
                frames/
                  Image/camera_0/Image_*.png
                  Depth/camera_0/Depth_*.npy
                  ObjectSegmentation/camera_0/ObjectSegmentation_*.npy
                  camview/camera_0/camview_*.npz   # contains K, T

    Output view dict matches scannetpp.py style:
        {
            img,
            depthmap,
            camera_pose,
            camera_intrinsics,
            instance_mask,
            dataset,
            label,
            instance,
            is_metric,
            orig_img,
            orig_depthmap,
            orig_camera_intrinsics,
        }
    """

    def __init__(
        self,
        num_seq=1,
        test_id=None,
        full_video=False,
        kf_every=1,
        min_frames=24,
        topk_save=512,
        support_pool=26,
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

        self.min_frames = min_frames
        self.topk_save = topk_save
        self.support_pool = support_pool

        self.load_all_scenes(ROOT)

    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):
        # keep same convention as your scannetpp
        self.folder = {
            "train": "train",
            "val": "val",
            "test": "test",
        }[self.split]

        split_root = osp.join(base_dir, self.folder)
        if not osp.isdir(split_root):
            raise FileNotFoundError(f"Split root not found: {split_root}")

        all_items = []
        for scene_name in sorted(os.listdir(split_root)):
            scene_dir = osp.join(split_root, scene_name)
            if not osp.isdir(scene_dir):
                continue
            if not scene_name.startswith("scene"):
                continue

            for sub_name in sorted(os.listdir(scene_dir)):
                sub_dir = osp.join(scene_dir, sub_name)
                if not osp.isdir(sub_dir):
                    continue

                frames_dir = osp.join(sub_dir, "frames")
                if not osp.isdir(frames_dir):
                    continue

                rgb_dir = osp.join(frames_dir, "Image", "camera_0")
                n_rgb = len(glob.glob(osp.join(rgb_dir, "Image_*.png")))
                if n_rgb < self.min_frames:
                    continue

                all_items.append(
                    {
                        "scene_id": scene_name,
                        "subscene_id": sub_name,
                        "subscene_dir": sub_dir,
                    }
                )

        if self.test_id is None:
            self.scene_list = all_items
        else:
            if isinstance(self.test_id, list):
                wanted = set(self.test_id)
            else:
                wanted = {self.test_id}

            # allow matching by scene_id or "scene_id/subscene_id"
            self.scene_list = []
            for item in all_items:
                key1 = item["scene_id"]
                key2 = f'{item["scene_id"]}/{item["subscene_id"]}'
                if key1 in wanted or key2 in wanted:
                    self.scene_list.append(item)

            print(f"Test_id: {self.test_id}")

        print(f"[InfinigenManyView] found {len(self.scene_list)} valid subscenes in {split_root}")

    def _load_scene_data(self, subscene_dir):
        frames_dir = osp.join(subscene_dir, "frames")

        rgb_paths = sorted(
            glob.glob(osp.join(frames_dir, "Image", "camera_0", "Image_*.png"))
        )
        depth_paths = sorted(
            glob.glob(osp.join(frames_dir, "Depth", "camera_0", "Depth_*.npy"))
        )
        instance_mask_paths = sorted(
            glob.glob(
                osp.join(
                    frames_dir,
                    "ObjectSegmentation",
                    "camera_0",
                    "ObjectSegmentation_*.npy",
                )
            )
        )
        camview_paths = sorted(
            glob.glob(
                osp.join(
                    frames_dir,
                    "camview",
                    "camera_0",
                    "camview_*.npz",
                )
            )
        )

        n = len(rgb_paths)
        assert n > 0, f"No rgb files found in {subscene_dir}"
        assert len(depth_paths) == n, f"Depth count mismatch in {subscene_dir}"
        assert len(instance_mask_paths) == n, f"Mask count mismatch in {subscene_dir}"
        assert len(camview_paths) == n, f"Camview count mismatch in {subscene_dir}"

        pose_list = []
        intri_list = []

        for anno_path in camview_paths:
            anno = np.load(anno_path)
            pose = np.array(anno["T"], dtype=np.float32)
            intri = np.array(anno["K"], dtype=np.float32)

            assert pose.shape == (4, 4), f"Pose shape mismatch in {anno_path}: {pose.shape}"
            assert intri.shape == (3, 3), f"Intrinsic shape mismatch in {anno_path}: {intri.shape}"

            pose_list.append(pose)
            intri_list.append(intri)

        pose_all = np.stack(pose_list, axis=0)
        intri_all = np.stack(intri_list, axis=0)

        return rgb_paths, depth_paths, instance_mask_paths, pose_all, intri_all

    def _load_instance_mask(self, mask_path):
        mask = np.load(mask_path)

        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        return mask.astype(np.int32)

    def _load_depth(self, depth_path):
        depth = np.load(depth_path)

        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth > 1000.0] = 0.0
        # Assumption: Infinigen depth npy is already in meters.
        # If your actual files are in millimeters, replace with:
        # depth = depth / 1000.0
        depth = threshold_depth_map(depth, min_percentile=-1, max_percentile=98).astype(np.float32)
        return depth

    def _get_ranking(self, subscene_dir, pose_all):
        rank_path = osp.join(subscene_dir, "ranking.npy")

        if osp.exists(rank_path):
            ranking = np.load(rank_path)
        else:
            ranking, _ = compute_ranking(
                pose_all,
                lambda_t=1.0,
                normalize=True,
                batched=True,
            )
            ranking = ranking[:, : self.topk_save]
            np.save(rank_path, ranking)
            print(f"Ranking saved to {rank_path}")

        return ranking

    def _get_views(self, idx, resolution, num_frames, rng, attempts=0):
        item = self.scene_list[idx // self.num_seq]
        scene_id = item["scene_id"]
        subscene_id = item["subscene_id"]
        subscene_dir = item["subscene_dir"]

        rgb_paths, depth_paths, instance_mask_paths, pose_all, intri_all = self._load_scene_data(subscene_dir)
        ranking = self._get_ranking(subscene_dir, pose_all)

        num_files = len(rgb_paths)
        if num_files == 0:
            raise RuntimeError(f"No files found in {subscene_dir}")

        while True:
            ref_idx = np.random.randint(0, num_files)

            # skip self at ranking[ref_idx][0], follow your current scannetpp style
            topk = ranking[ref_idx][1 : self.support_pool + 1]
            if len(topk) == 0:
                topk = np.array([i for i in range(num_files) if i != ref_idx], dtype=np.int64)

            replace = False if len(topk) >= (num_frames - 1) else True
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
            depthmap = self._load_depth(depthpath)

            # keep same behavior as scannetpp: align rgb size to depth size before crop/resize
            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

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
                    interpolation=cv2.INTER_NEAREST,
                )

            # replay the exact same crop/resize randomness onto instance mask
            rng.bit_generator.state = rng_state_before
            _, instance_mask, _ = self._crop_resize_if_necessary(
                rgb_ori, instance_mask, intri_ori, resolution, rng=rng, info=impath
            )
            rng.bit_generator.state = rng_state_after
            instance_mask = np.asarray(instance_mask).astype(np.int32)

            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0 or (not np.isfinite(camera_pose).all()):
                if self.full_video:
                    print(f"Warning: invalid sample for {impath}")
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
                dataset="infinigen",
                label=osp.join(scene_id, subscene_id, frame_stem),
                instance=osp.split(impath)[1],
                is_metric=True,
                orig_img=rgb_ori,
                orig_depthmap=depth_ori,
                orig_camera_intrinsics=intri_ori,
            )

            views.append(dict_info)

        return views


if __name__ == "__main__":
    print("loading dataset")

    dataset = InfinigenManyView(
        split="train",
        ROOT="./data/infinigen",
        resolution=224,
        num_seq=1,
        num_frames=5,
    )