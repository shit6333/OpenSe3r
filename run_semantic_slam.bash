
# python slam_semantic/run.py \
#     --data_path /mnt/HDD4/ricky/data/data/InsScene-15K/processed_scannetpp_v2/data/test/1ada7a0617/images \
#     --ckpt_path /mnt/HDD4/ricky/feedforward/amb3r/outputs/exp_stage1_wo_lora/checkpoint-last.pth \
#     --results_path /mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/exp_stage1_wo_lora/1ada7a0617 \
#     --resolution 518 336 \
#     --save_semantic \
#     --save_instance \
#     --demo_name 1ada7a0617 \
#     --save_semantic \
#     --save_instance \
#     --save_instance_npz

python slam_semantic/run_gt_depth.py \
    --data_path /mnt/HDD4/ricky/data/scannet_processed_test/scene0050_00 \
    --ckpt_path /mnt/HDD4/ricky/feedforward/amb3r/outputs/exp_stage1_wo_lora/checkpoint-last.pth \
    --results_path /mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/exp_stage1_wo_lora/gt_depth/scene0050_00 \
    --resolution 518 336 \
    --save_semantic \
    --save_instance \
    --demo_name scene0050_00 \
    --save_semantic \
    --save_instance \
    --save_instance_npz

# Original model (frontend + backend + semantic/instance head)
# python slam_semantic/run.py \
#     --data_path /mnt/HDD4/ricky/data/scannet_processed_test/scene0050_00 \
#     --ckpt_path /mnt/HDD4/ricky/feedforward/amb3r/outputs/exp_amb3r/checkpoint-best.pth \
#     --results_path /mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/scene0050_00 \
#     --resolution 518 336 \
#     --save_semantic \
#     --save_instance \
#     --demo_name scene0050_00 \
#     --demo_type scannet \
#     --save_semantic \
#     --save_instance \
#     --save_instance_npz

    
python instance_hdbscan_cluster.py \
    --feats_npz /mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/exp_stage1_wo_lora/scene0050_00/scene_scene0050_instance_voxel_feats.npz \
    --results_npz /mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/exp_stage1_wo_lora/scene0050_00/scene_scene0050_00_results.npz \
    --output /mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/exp_stage1_wo_lora/scene0050_00 \
    --min_cluster_size 50

#  python instance_hdbscan_cluster.py \
#     --feats_npz /mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/scene0050_00/scene_scene0050_00_instance_voxel_feats.npz \
#     --results_npz /mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/scene0050_00/scene_scene0050_00_results.npz \
#     --output /mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/scene0050_00 \
#     --min_cluster_size 50

# Noraml Data Demo (In the Wild)
# python slam_semantic/run.py \
#     --data_path /mnt/HDD4/ricky/data/scannet_processed_test/scene0050_00/color \
#     --ckpt_path /mnt/HDD4/ricky/feedforward/amb3r/outputs/exp_amb3r/checkpoint-best.pth \
#     --results_path /mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/scene0000_00 \
#     --resolution 518 336 \
#     --save_semantic
    

# python slam_semantic/run.py \
#     --data_path /mnt/HDD4/ricky/data/lab/lab12/color \
#     --ckpt_path /mnt/HDD4/ricky/feedforward/amb3r/outputs/exp_amb3r/checkpoint-best.pth \
#     --results_path /mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/lab12 \
#     --resolution 518 336 \
#     --save_semantic
    

# SAVE Instance Cluster
# python instance_hdbscan_cluster.py \\
#         --feats_npz   /path/to/scene_X_instance_voxel_feats.npz \\
#         --results_npz /path/to/scene_X_results.npz \\
#         --output      /path/to/output_dir \\
#         [--conf_threshold 0.0] \\
#         [--min_cluster_size 50] \\
#         [--min_samples 10] \\
#         [--use_pca 8]
#         [--normalize]
#         [--max_points 6000000]