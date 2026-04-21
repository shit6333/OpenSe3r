# 設定場景變數
SCENE=scene0518_00
BASE_OUT=/mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slamv2/exp_stage2v4_mode3/$SCENE

GT_LABELS_ROOT="/mnt/HDD4/ricky/online_slam/LoopSplat/data/ScanNet"
GT_LABELS_PLY="${GT_LABELS_ROOT}/${SCENE}/${SCENE}_vh_clean_2.labels.ply"

# 第一階段：SLAM 執行
python slam_stage2v4/run.py \
    --stage1_ckpt /mnt/HDD4/ricky/feedforward/amb3r/outputs/exp_stage1_wo_lora_long/checkpoint-last.pth \
    --stage1_5_ckpt /mnt/HDD4/ricky/feedforward/amb3r/outputs/exp_stage1_5/checkpoint-last.pth \
    --stage2_ckpt /mnt/HDD4/ricky/feedforward/amb3r/outputs/exp_stage2v4_mode3/checkpoint-last.pth \
    --data_path /mnt/HDD4/ricky/data/scannet_processed_test/$SCENE \
    --results_path $BASE_OUT \
    --demo_type scannet \
    --demo_name $SCENE \
    --resolution 518 336 \
    --save_semantic --save_instance \
    --save_instance_npz \
    --use_gt_depth \
    --save_voxel_store

# 第二階段：HDBSCAN 分群
# python instance_hdbscan_cluster.py \
#     --feats_npz $BASE_OUT/scene_${SCENE}_instance_voxel_feats.npz \
#     --results_npz $BASE_OUT/scene_${SCENE}_results.npz \
#     --output $BASE_OUT \
#     --min_cluster_size 50

# New Cluster Params
# python instance_hdbscan_cluster.py \
#     --feats_npz $BASE_OUT/scene_${SCENE}_instance_voxel_feats.npz \
#     --results_npz $BASE_OUT/scene_${SCENE}_results.npz \
#     --output $BASE_OUT \
#     --min_cluster_size 20 \
#     --cluster_selection_epsilon 0.0 --min_samples 10

# Semantic Eval
python eval_semantic_slam.py \
    --results_path $BASE_OUT \
    --scene_name $SCENE \
    --data_path /mnt/HDD4/ricky/data/scannet_processed_test/$SCENE \
    --gt_labels_ply "${GT_LABELS_PLY}" \
    --save_csv "${BASE_OUT}/eval_results.csv"