# python eval_semantic_slam.py \
#     --results_path  /mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/scene0050_00 \
#     --scene_name    scene0050_00 \
#     --data_path     /mnt/HDD4/ricky/data/scannet_processed_test/scene0050_00 \
#     --gt_labels_ply /mnt/HDD4/ricky/online_slam/LoopSplat/data/ScanNet/scene0050_00/scene0050_00_vh_clean_2.labels.ply \
#     --save_csv      /mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/scene0050_00/eval_results.csv


python eval_semantic_slam.py \
    --results_path  /mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/exp_stage1_wo_lora_long/slam_w_gt_depth/scene0518_00 \
    --scene_name    scene0518_00 \
    --data_path     /mnt/HDD4/ricky/data/scannet_processed_test/scene0518_00 \
    --gt_labels_ply /mnt/HDD4/ricky/online_slam/LoopSplat/data/ScanNet/scene0518_00/scene0518_00_vh_clean_2.labels.ply \
    --save_csv      /mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/exp_stage1_wo_lora_long/slam_w_gt_depth/scene0518_00/eval_results_table_desk_merge.csv \
    --merge_table_desk