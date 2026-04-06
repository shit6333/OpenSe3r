import os
import os.path as osp
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from dust3r.utils.image import imread_cv2

try:
    import PIL.Image
    lanczos = PIL.Image.Resampling.LANCZOS
except AttributeError:
    import PIL.Image
    lanczos = PIL.Image.LANCZOS
    
class ScannetDemoDataset(Dataset):
    def __init__(self, ROOT, resolution=(518, 392), **kwargs):
        self.ROOT = ROOT
        self.resolution = resolution
        self.color_dir = osp.join(ROOT, "color")
        self.pose_path = osp.join(ROOT, "poses.npy")
        self.intri_path = osp.join(ROOT, "intrinsic", "intrinsic_depth.txt")

        self.pose_all = np.load(self.pose_path).astype(np.float32)
        self.intri = np.loadtxt(self.intri_path).astype(np.float32)[:3, :3]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        views = self._get_views()

        images = [torch.from_numpy(np.asarray(v['img'])).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0 for v in views]
        camera_poses = [torch.from_numpy(v['camera_pose']) for v in views]
        camera_intrinsics = [torch.from_numpy(v['camera_intrinsics']) for v in views]

        views_all = {
            'images': torch.stack(images),
            'camera_pose': torch.stack(camera_poses),
            'camera_intrinsics': torch.stack(camera_intrinsics),
        }
        return views, views_all

    def _get_views(self):
        image_files = sorted([f for f in os.listdir(self.color_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        views = []
        for i, image_name in enumerate(image_files):
            image_path = osp.join(self.color_dir, image_name)

            image = PIL.Image.open(image_path).convert('RGB')
            image, intrinsics = self._center_crop_and_resize_with_intrinsics(
                image, self.intri.copy(), self.resolution
            )

            views.append(dict(
                img=np.asarray(image),
                camera_pose=self.pose_all[i],
                camera_intrinsics=intrinsics,
                dataset='scannet_demo',
                label=osp.join(osp.basename(self.ROOT), f'{i:06d}'),
                instance=image_name,
                is_metric=True,
            ))
        return views
    
    @staticmethod
    def _center_crop_and_resize_with_intrinsics(image, intrinsics, resolution):
        """
        Center-crop the image to match the target aspect ratio, then resize
        to the exact target resolution, while updating the REAL intrinsics.

        Args:
            image: PIL Image
            intrinsics: (3, 3) numpy array
            resolution: (target_W, target_H)

        Returns:
            image, intrinsics
        """
        W, H = image.size
        target_W, target_H = resolution

        intrinsics = intrinsics.copy().astype(np.float32)

        # Handle portrait images: swap target if image is portrait
        if target_W >= target_H:
            if H > 1.1 * W:
                target_W, target_H = target_H, target_W

        target_aspect = target_W / target_H
        image_aspect = W / H

        # ----- center crop -----
        if image_aspect > target_aspect:
            # crop width
            new_W = int(round(H * target_aspect))
            left = (W - new_W) // 2
            image = image.crop((left, 0, left + new_W, H))

            # principal point shifts after crop
            intrinsics[0, 2] -= left

            crop_W, crop_H = new_W, H

        elif image_aspect < target_aspect:
            # crop height
            new_H = int(round(W / target_aspect))
            top = (H - new_H) // 2
            image = image.crop((0, top, W, top + new_H))

            # principal point shifts after crop
            intrinsics[1, 2] -= top

            crop_W, crop_H = W, new_H

        else:
            crop_W, crop_H = W, H

        # ----- resize -----
        image = image.resize((target_W, target_H), resample=lanczos)

        scale_x = target_W / float(crop_W)
        scale_y = target_H / float(crop_H)

        intrinsics[0, 0] *= scale_x
        intrinsics[1, 1] *= scale_y
        intrinsics[0, 2] *= scale_x
        intrinsics[1, 2] *= scale_y

        return image, intrinsics.astype(np.float32)