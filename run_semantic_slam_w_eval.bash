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
CKPT_PATH="/mnt/HDD4/ricky/feedforward/amb3r/outputs/exp_amb3r/checkpoint-best.pth"
RESULTS_ROOT="/mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slam_gtpose"
GT_LABELS_ROOT="/mnt/HDD4/ricky/online_slam/LoopSplat/data/ScanNet"

RES_W=518
RES_H=336

SAVE_SEMANTIC=true
SAVE_INSTANCE=true
GT_POSE=true


DEMO_TYPE="scannet"

# =========================
# Auto-generated paths
# =========================

DATA_PATH="${DATA_ROOT}/${SCENE_NAME}"
RESULTS_PATH="${RESULTS_ROOT}/${SCENE_NAME}"
GT_LABELS_PLY="${GT_LABELS_ROOT}/${SCENE_NAME}/${SCENE_NAME}_vh_clean_2.labels.ply"
SAVE_CSV="${RESULTS_PATH}/eval_results.csv"

# =========================
# Print config
# =========================

echo "=================================="
echo "SCENE_NAME   : ${SCENE_NAME}"
echo "DATA_PATH    : ${DATA_PATH}"
echo "CKPT_PATH    : ${CKPT_PATH}"
echo "RESULTS_PATH : ${RESULTS_PATH}"
echo "RESOLUTION   : ${RES_W} ${RES_H}"
echo "DEMO_TYPE    : ${DEMO_TYPE}"
echo "GT_LABELS_PLY: ${GT_LABELS_PLY}"
echo "SAVE_CSV     : ${SAVE_CSV}"
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
    --demo_type "${DEMO_TYPE}"
)

if [ "${SAVE_SEMANTIC}" = true ]; then
    RUN_ARGS+=(--save_semantic)
fi

if [ "${SAVE_INSTANCE}" = true ]; then
    RUN_ARGS+=(--save_instance)
fi

if [ "${GT_POSE}" = true ]; then
    RUN_ARGS+=(--use_gt_pose)
fi

# =========================
# Run semantic SLAM
# =========================

echo
echo "[1/2] Running semantic SLAM..."
python slam_semantic/run.py "${RUN_ARGS[@]}"

# =========================
# Run evaluation
# =========================

echo
echo "[2/2] Running evaluation..."
python eval_semantic_slam.py \
    --results_path "${RESULTS_PATH}" \
    --scene_name "${SCENE_NAME}" \
    --data_path "${DATA_PATH}" \
    --gt_labels_ply "${GT_LABELS_PLY}" \
    --save_csv "${SAVE_CSV}"

echo
echo "Done."