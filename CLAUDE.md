# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## File Access Policy

> **IMPORTANT — Read before any file operation.**

- **`.claude/sandbox/`** is the **only directory where you may create or edit files**.
- **All other files and directories in this repository are read-only.** Do not attempt to edit, create, or delete them, even if the user asks you to modify existing code directly.
- If a task requires changing code outside `.claude/sandbox/`, write the proposed changes as new files inside `.claude/sandbox/` and explain what would need to be changed and where.

## Project Overview

**AMB3R** (Accurate Feed-forward Metric-scale 3D Reconstruction with Backend) is a research codebase for multi-view 3D reconstruction, monocular depth estimation, and semantic SLAM. It extends VGGT (a vision transformer foundation model) with a voxel-based backend and semantic/instance segmentation heads using LSeg/CLIP features.

Paper: https://arxiv.org/abs/2511.20343

## Environment Setup

```bash
conda create -n amb3r python=3.9 cmake=3.14.0 -y
conda activate amb3r
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8" --no-build-isolation
pip install flash-attn==2.7.3 --no-build-isolation
pip install -r requirements.txt
```

Weights go in `./checkpoints/`. The semantic model checkpoint is referenced in scripts as `outputs/exp_amb3r/checkpoint-best.pth`.

## Key Commands

### Interactive Demo
```bash
python demo.py
```

### SLAM
```bash
# Non-semantic SLAM
python slam/run.py --data_path <path-to-video-folder>

# Semantic SLAM (ScanNet format)
python slam_semantic/run.py \
    --data_path <scene_path> \
    --ckpt_path <checkpoint.pth> \
    --results_path <output_dir> \
    --resolution 518 336 \
    --save_semantic --save_instance \
    --demo_type scannet
```

### SfM
```bash
python sfm/run.py --data_path <path-to-video-folder>
```

### Training
```bash
# Base AMB3R
torchrun --nproc_per_node $num_gpus train.py --batch_size $batch_size

# With fixed voxel/point coordinate alignment (use for new tasks)
torchrun --nproc_per_node $num_gpus train.py --batch_size $batch_size --interp_v2

# Semantic AMB3R (single GPU)
torchrun --nproc_per_node=1 train_semantic.py \
    --batch_size 1 --interp_v2 \
    --model "AMB3R(metric_scale=True)" \
    --accum_iter 2 --epochs 30 --lr 0.00005
```

## Architecture

The model has two main stages:

**Frontend** (`amb3r/frontend_semantic.py`, `amb3r/model_semantic.py`):
- VGGT encodes multi-view images into patch tokens
- Separate heads decode depth maps, camera poses (SE3), dense 3D point maps, and semantic/instance features
- LSeg/CLIP extracts frozen 512-dim dense semantic features per image
- `PatchConditionedDPTHead` (`amb3r/dpt_head_patch_cond.py`) fuses CLIP features into the decoder via token concatenation at each DPT layer

**Backend** (`amb3r/backend_semantic.py`):
- PointTransformerV3 processes voxelized 3D points from the frontend
- Semantic (512→256 dim) and instance (16→128 dim) features are projected and injected via zero-conv gating
- Voxel features are KNN-interpolated back to image-space patch tokens for a second frontend pass

**Data flow** (one or two iterations):
```
Images → VGGT patch tokens → decode depth/pose/points/semantic
       → LSeg features (frozen)
       → Backend voxel grid (conditioned on semantic/instance)
       → Refined depth/pose/points + semantic/instance features
```

**SLAM pipeline** (`slam_semantic/pipeline.py`): `AMB3R_VO` runs incremental mapping — keyframes are stored in `SLAMemory` (`slam_semantic/memory.py`), which tracks 3D points, poses, and semantic/instance features across frames. `SemanticVoxelMap` (`slam_semantic/semantic_voxel_map.py`) provides a persistent voxel → feature store.

## Key Files

| File | Purpose |
|------|---------|
| `amb3r/model_semantic.py` | Main model class (`AMB3R`) — forward pass, SLAM/benchmark interfaces |
| `amb3r/frontend_semantic.py` | `FrontEnd`: image encoding + head decoding |
| `amb3r/backend_semantic.py` | Voxel backend with semantic conditioning |
| `amb3r/loss_semantic.py` | `MultitaskLoss` combining geometry + semantic + instance losses |
| `amb3r/training_semantic.py` | Distributed training loop, GPU-adaptive resolution |
| `amb3r/dpt_head_patch_cond.py` | CLIP-conditioned DPT decoder head |
| `amb3r/lseg.py` | Frozen LSeg/CLIP feature extractor |
| `amb3r/blocks.py` | `ScaleProjector`, `FeatureExpander`, `ZeroConvBlock` |
| `amb3r/tools/semantic_vis_utils.py` | PLY export (voxel/per-point/raw-feat), CLIP text matching |
| `amb3r/tools/scannet200_constants.py` | ScanNet20/200 label maps and color palettes |
| `amb3r/datasets/scannet.py` | Primary training dataset loader |
| `slam_semantic/pipeline.py` | Semantic SLAM orchestration |

## Important Notes

- **Coordinate misalignment**: Pre-trained checkpoints were trained with a voxel/point coordinate offset that the network compensates for internally. Use `--interp_v2` only when training from scratch for new tasks — don't mix with the released checkpoint.
- **Resolution is GPU-dependent**: `training_semantic.py` auto-selects resolution and frame count based on GPU type (RTX 4090 uses 518×336, others use lower resolutions).
- **Two inference modes**: `forward(frames, iters=1)` for single-pass, `iters=2` for backend refinement loop.
- **Semantic features are always frozen**: LSeg/CLIP parameters are never updated during training.
- **Dataset format**: ScanNet is the primary dataset for semantic training. The `scannet.py` loader expects preprocessed data with depth, pose, and semantic label files.
