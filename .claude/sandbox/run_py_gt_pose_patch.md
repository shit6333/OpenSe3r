# run.py — GT Pose 模式的修改說明

## 需要加的 argument（在 get_args_parser 裡）

```python
parser.add_argument('--use_gt_pose', action='store_true', default=False,
                    help='Use GT camera poses for semantic mapping (bypasses SLAM pose estimation)')
```

## 需要修改的 pipeline.run() 呼叫（main() 裡，第 123 行附近）

將：
```python
memory = pipeline.run(images)
```

改成：
```python
poses_gt = None
if args.use_gt_pose and args.demo_type == 'scannet':
    # views_all['camera_pose'] shape: (1, T, 4, 4)
    poses_gt = views_all['camera_pose'][0].cpu()  # (T, 4, 4)

memory = pipeline.run(images, poses_gt=poses_gt, use_gt_pose=args.use_gt_pose)
```

## 重點

- 只有 `demo_type == 'scannet'` 才有 GT poses（ScannetDemoDataset 提供）
- `use_gt_pose=True` 時 pipeline 會走新的 `_run_gt_pose_mode()` 而非 SLAM loop
- 幾何 pts/poses 都用 GT 對齊，semantic/instance voxel map 結果比較乾淨

===

● 你說到了關鍵問題。讓我先梳理清楚整個 coordinate 對應關係：

  Read 1 file (ctrl+o to expand)

● 現在有完整的 picture。關鍵問題梳理：

  world_points 的坐標系：model 每次 call 的 output 是在 第一個 input frame 的 camera 坐標系（即 cur_kf_idx[0] 那個 keyframe）。所以 c2w_local[0] ≈ identity。

  Map idx 對應：
  model input:  [kf0,      kf1,      ...,  new_f0,    new_f1  ]
  map_idx:      [kf_idx[0], kf_idx[1], ..., start_idx, start+1 ]
  poses_gt:      poses_gt[kf_idx[0]], poses_gt[kf_idx[1]], ...
  這部分目前是對的。

  根本問題：initialization 時 pts 存成 model scale，poses 存成 GT metric scale → 兩者 inconsistent → 後續 _estimate_chunk_scale 拿 model-scale pts 比 GT-metric poses 就爛掉了。

  Fix: initialization 就要把 pts 轉成 metric scale（用 GT pose translations vs predicted translations 估 scale）。

  Read 1 file (ctrl+o to expand)

● 完成。以下是這版修了哪些根本問題：

  ---
  修正的根本問題

  問題 1：Init pts 和 poses 的 scale 不一致

  之前：init 存 model-scale pts + GT-metric poses → 後面的 scale 估計全錯

  現在：init 時就用 GT translations vs predicted translations 估出 metric_scale，把 pts 也轉成 metric → memory 裡 pts 和 poses 都在同一個 metric 坐標系

  問題 2：後續 chunk 的 scale 估計

  之前：用全局估的一個 scale，在不同 chunk 可能偏掉

  現在：每個 chunk 都重新用 _scale_from_pts 估（同 coordinate_alignment 內部邏輯）：拿 local model-scale pts 跟 stored metric pts 比 → 正確估出 model→metric scale

  問題 3：GT pose 對應確認

  - model input: [kf0, kf1, ..., new_frame0, new_frame1]
  - map_idx = [cur_kf_idx[0], cur_kf_idx[1], ..., start_idx, ...]
  - 對每個 local_i：poses_gt[map_idx[local_i]] = 正確的 GT pose ✓
  - c2w_local[0] ≈ identity（kf0 相對自己）= model world frame origin ✓

  ---
  run.py 的使用方式

  if args.use_gt_pose and args.demo_type == 'scannet':
      from slam_semantic.pipeline_gt import AMB3R_VO_GT
      poses_gt = views_all['camera_pose'][0].cpu().float()  # (T, 4, 4)
      pipeline = AMB3R_VO_GT(model)
      memory   = pipeline.run(images, poses_gt)
  else:
      from slam_semantic.pipeline import AMB3R_VO
      pipeline = AMB3R_VO(model)
      memory   = pipeline.run(images)
