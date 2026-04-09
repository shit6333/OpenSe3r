#!/usr/bin/env bash
# Example invocation for the GT-depth pipeline.
# Requires: ROOT/color/, ROOT/depth/, ROOT/poses.npy, ROOT/intrinsic/intrinsic_depth.txt

python slam_semantic/run_gt_depth.py \
    --data_path  /path/to/scene \
    --demo_name  scene0000_00 \
    --results_path ./outputs/slam/scene0000_00 \
    --ckpt_path  ./checkpoints/amb3r_semantic.pt \
    --resolution 518 336 \
    --save_semantic \
    --save_instance \
    --label_set scannet20
