import os
import glob
import numpy as np
import os.path as osp
from amb3r.tools.pose_dist import compute_ranking


def build_ranking_for_subscene(subscene_dir, topk_save=512):
    camview_dir = osp.join(subscene_dir, "frames", "camview", "camera_0")
    camview_paths = sorted(glob.glob(osp.join(camview_dir, "camview_*.npz")))

    if len(camview_paths) == 0:
        print(f"[skip] no camview files: {subscene_dir}")
        return

    rank_path = osp.join(subscene_dir, "ranking.npy")
    if osp.exists(rank_path):
        print(f"[exists] {rank_path}")
        return

    pose_list = []
    for p in camview_paths:
        anno = np.load(p)
        if "T" not in anno:
            raise KeyError(f"Missing key 'T' in {p}")
        pose = np.array(anno["T"], dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError(f"Pose shape mismatch in {p}: {pose.shape}")
        pose_list.append(pose)

    pose_all = np.stack(pose_list, axis=0)

    ranking, dist = compute_ranking(
        pose_all,
        lambda_t=1.0,
        normalize=True,
        batched=True
    )
    ranking = ranking[:, :topk_save]
    np.save(rank_path, ranking)
    print(f"[saved] {rank_path}  shape={ranking.shape}")


def find_all_subscenes(root, splits=("train", "val", "test"), min_frames=1):
    subscene_dirs = []

    for split in splits:
        split_root = osp.join(root, split)
        if not osp.isdir(split_root):
            print(f"[skip split] not found: {split_root}")
            continue

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

                rgb_dir = osp.join(sub_dir, "frames", "Image", "camera_0")
                n_rgb = len(glob.glob(osp.join(rgb_dir, "Image_*.png")))
                if n_rgb < min_frames:
                    continue

                subscene_dirs.append(sub_dir)

    return subscene_dirs


if __name__ == "__main__":
    ROOT = "/mnt/HDD4/ricky/data/data/InsScene-15K/processed_infinigen_data"   # 改成你的 root
    SPLITS = ("train", "")
    TOPK_SAVE = 512
    MIN_FRAMES = 1

    subscene_dirs = find_all_subscenes(ROOT, splits=SPLITS, min_frames=MIN_FRAMES)
    print(f"Found {len(subscene_dirs)} subscenes.")

    for i, subscene_dir in enumerate(subscene_dirs):
        print(f"\n[{i+1}/{len(subscene_dirs)}] processing {subscene_dir}")
        try:
            build_ranking_for_subscene(subscene_dir, topk_save=TOPK_SAVE)
        except Exception as e:
            print(f"[error] {subscene_dir}: {e}")