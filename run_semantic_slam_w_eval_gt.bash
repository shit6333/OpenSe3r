#!/usr/bin/env bash
set -euo pipefail

# =========================
# User-configurable params
# =========================

# SCENE_NAME="scene0518_00"
# SCENE_NAME="scene0378_00"
# SCENE_NAME="scene0231_00"
SCENE_NAME="scene0050_00"
# SCENE_NAME="scene0011_00"

DATA_ROOT="/mnt/HDD4/ricky/data/scannet_processed_test"
CKPT_PATH="/mnt/HDD4/ricky/feedforward/amb3r/outputs/exp_amb3r_nolora/checkpoint-best.pth"
RESULTS_ROOT="/mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/exp_amb3r_nolora/slam_w_gt_depth_long"
# CKPT_PATH="/mnt/HDD4/ricky/feedforward/amb3r/outputs/exp_stage1_wo_lora/checkpoint-last.pth"
# CKPT_PATH="/mnt/HDD4/ricky/feedforward/amb3r/outputs/exp_stage1_wo_lora_long/checkpoint-last.pth"
# RESULTS_ROOT="/mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam/exp_stage1_wo_lora_long/slam_w_gt_depth"

GT_LABELS_ROOT="/mnt/HDD4/ricky/online_slam/LoopSplat/data/ScanNet"

RES_W=518
RES_H=336

SAVE_SEMANTIC=true
SAVE_INSTANCE=true
SAVE_INSTANCE_NPZ=true

MIN_CLUSTER_SIZE=20

# =========================
# Auto-generated paths
# =========================

DATA_PATH="${DATA_ROOT}/${SCENE_NAME}"
RESULTS_PATH="${RESULTS_ROOT}/${SCENE_NAME}"
GT_LABELS_PLY="${GT_LABELS_ROOT}/${SCENE_NAME}/${SCENE_NAME}_vh_clean_2.labels.ply"
SAVE_CSV="${RESULTS_PATH}/eval_results.csv"

# 注意：這裡的檔名格式請依照你 run_gt_depth.py 實際產生的檔名為主
FEATS_NPZ="${RESULTS_PATH}/scene_${SCENE_NAME}_instance_voxel_feats.npz"
RESULTS_NPZ="${RESULTS_PATH}/scene_${SCENE_NAME}_results.npz"

# =========================
# Print config
# =========================

echo "=================================="
echo "SCENE_NAME        : ${SCENE_NAME}"
echo "DATA_PATH         : ${DATA_PATH}"
echo "CKPT_PATH         : ${CKPT_PATH}"
echo "RESULTS_PATH      : ${RESULTS_PATH}"
echo "RESOLUTION        : ${RES_W} ${RES_H}"
echo "MIN_CLUSTER_SIZE  : ${MIN_CLUSTER_SIZE}"
echo "FEATS_NPZ         : ${FEATS_NPZ}"
echo "RESULTS_NPZ       : ${RESULTS_NPZ}"
echo "GT_LABELS_PLY     : ${GT_LABELS_PLY}"
echo "SAVE_CSV          : ${SAVE_CSV}"
echo "=================================="

mkdir -p "${RESULTS_PATH}"

# =========================
# Build optional flags
# =========================

RUN_ARGS=(
    --data_path "${DATA_PATH}"
    --ckpt_path "${CKPT_PATH}"
    --results_path "${RESULTS_PATH}"
    --resolution "${RES_W}" "${RES_H}"
    --demo_name "${SCENE_NAME}"
)

if [ "${SAVE_SEMANTIC}" = true ]; then
    RUN_ARGS+=(--save_semantic)
fi

if [ "${SAVE_INSTANCE}" = true ]; then
    RUN_ARGS+=(--save_instance)
fi

if [ "${SAVE_INSTANCE_NPZ}" = true ]; then
    RUN_ARGS+=(--save_instance_npz)
fi

# =========================
# [1/3] Run semantic SLAM (GT Depth)
# =========================

echo
echo "[1/3] Running semantic SLAM with GT Depth..."
python slam_semantic/run_gt_depth.py "${RUN_ARGS[@]}"

# =========================
# [2/3] Run HDBSCAN Clustering
# =========================

echo
echo "[2/3] Running HDBSCAN Instance Clustering..."
python instance_hdbscan_cluster.py \
    --feats_npz "${FEATS_NPZ}" \
    --results_npz "${RESULTS_NPZ}" \
    --output "${RESULTS_PATH}" \
    --min_cluster_size "${MIN_CLUSTER_SIZE}"

# =========================
# [3/3] Run evaluation
# =========================

echo
echo "[3/3] Running evaluation..."
python eval_semantic_slam.py \
    --results_path "${RESULTS_PATH}" \
    --scene_name "${SCENE_NAME}" \
    --data_path "${DATA_PATH}" \
    --gt_labels_ply "${GT_LABELS_PLY}" \
    --save_csv "${SAVE_CSV}"

echo
echo "Done."