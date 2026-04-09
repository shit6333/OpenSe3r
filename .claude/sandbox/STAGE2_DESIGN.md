# AMB3R Stage 2 Design Document

## Overview

Stage 2 extends Stage 1 with **online temporal memory** to produce *consistent* semantic and instance features across frames. Stage 1 processes each chunk of K frames independently; Stage 2 adds a voxel-based memory that accumulates observations over time and conditions a powerful DPT-based head to refine the per-chunk outputs into globally consistent features.

---

## Problem Statement

- **Stage 1** (fixed): Input K frames → output semantic feat + instance feat + camera pose per chunk. Each chunk is processed independently → same physical object may get different embeddings across chunks.
- **Stage 2** (trainable): Input K frames + past memory → output *consistent* semantic/instance features. If a voxel was already observed, the new output should match the accumulated memory. If a voxel is new, it gets a unique instance embedding distinct from existing ones.

---

## Architecture

### Overall Data Flow

```
Input: K frames (current chunk) + VoxelMemory M_{t-1}

┌─────────────────────────────────────────────── STAGE 1 (frozen) ─────┐
│  encode_patch_tokens → DINOv2 patch embed                             │
│  _extract_lseg       → LSeg/CLIP features (frozen)                   │
│  clip_patch_fusion   → CLIP injected into patch tokens                │
│  decode_patch_tokens → aggregated_tokens_list  ◄── captured here      │
│  decode_heads        → world_points, sem_s1, ins_s1, lseg_feat        │
└───────────────────────────────────────────────────────────────────────┘
                    │                        │
         pts_for_query (globally-aligned)   │  (all detached, no grad)
         resize to (Hp, Wp)                 │
                    │                        │
           VoxelMemory M_{t-1}.query()      │
                    │                        │
         mem_sem, mem_ins, mem_mask          │
                    │                        │
┌─────────────────────────────────────── STAGE 2 (trainable) ──────────┐
│                                                                        │
│  mem_sem/ins/mask queried @ patch res → upsampled to (H, W)          │
│  sem_cond = cat[ mem_sem(512), sem_s1(512), mem_mask(1) ]  = 1025ch  │
│  ins_cond = cat[ mem_ins(16), sem_s1(512), ins_s1(16), mem_mask(1) ] │
│           = 545ch                                                      │
│  (all conditioning maps at full image resolution H × W)               │
│                                                                        │
│  SemanticMemoryDPTHead( aggregated_tokens_list, sem_cond )            │
│    → refined_sem (B, T, 512, H, W) + sem_conf  ← pixel-aligned       │
│                                                                        │
│  InstanceMemoryDPTHead( aggregated_tokens_list, ins_cond )            │
│    → refined_ins (B, T, 16, H, W)  + ins_conf  ← pixel-aligned       │
│                                                                        │
└───────────────────────────────────────────────────────────────────────┘
                    │
     update M_t ← downsample to patch res → update voxel map (stop-grad)
```

### Fixed vs. Trainable Parameters

| Module | Status | Notes |
|--------|--------|-------|
| DINOv2 patch_embed | **Fixed** | Stage 1 pretrained |
| VGGT frame/global blocks | **Fixed** | Stage 1 pretrained |
| LSeg / CLIP | **Fixed** | Always frozen |
| CLIPPatchFusion | **Fixed** | Stage 1 pretrained |
| PatchConditionedDPTHead (sem/ins) | **Fixed** | Stage 1 pretrained |
| `semantic_memory_head` | **TRAIN** | Stage 2 new |
| `instance_memory_head` | **TRAIN** | Stage 2 new |
| VoxelMemory (runtime state) | No params | Accumulates at inference |

---

## Head Architecture: `MemoryConditionedDPTHead`

Both Stage-2 heads are instances of `MemoryConditionedDPTHead`, which **directly inherits `PatchConditionedDPTHead`** (same as the Stage-1 semantic/instance heads). No new forward logic is needed — only the conditioning input dimension differs.

### `PatchConditionedDPTHead` recap

```
aggregated_tokens_list  (multi-scale VGGT tokens from 4 intermediate layers)
         │
         ▼  for each layer:
    reshape to (B*T, C, Hp, Wp)
    cat with sem_patch_feat  ──►  fusion_proj (1×1 conv)  ──►  DPT spatial ops
         │
         ▼
    scratch_forward (DPT decoder: merge 4 scales → 256ch feature map)
         │
         ▼
    output_proj → (B, T, output_channels, Hp, Wp)   ← feature
    conf_head   → (B, T, 1, Hp, Wp)                ← confidence
```

### Conditioning for Stage 2

| Head | `semantic_dim` | Conditioning content |
|------|----------------|----------------------|
| `semantic_memory_head` | 1025 | `cat[mem_sem(512), sem_s1(512), mem_mask(1)]` |
| `instance_memory_head` | 545 | `cat[mem_ins(16), sem_s1(512), ins_s1(16), mem_mask(1)]` |

**Why include `sem_s1` in instance conditioning?**  
The instance head needs to know the semantic category of each patch to produce embeddings that are discriminative between instances of the same class. Providing `sem_s1` as cross-task context helps the head avoid assigning similar embeddings to different objects of the same category.

**Why `mem_mask` in conditioning?**  
The head must learn different behaviors for seen vs. unseen voxels:
- `mem_mask ≈ 1` (seen) → pull refined feature toward memory
- `mem_mask ≈ 0` (unseen) → rely on `stage1_feat` only, produce a new discriminative embedding

### Conditioning projection (inside `PatchConditionedDPTHead`)

```python
# semantic_proj (applied to sem_cond at patch resolution):
Conv2d(semantic_dim → semantic_proj_dim, k=1) → BN → GELU
Conv2d(semantic_proj_dim → semantic_proj_dim, k=3, pad=1) → BN → GELU

# per-layer fusion (inside DPT):
cat[vggt_token_2D, sem_patch_feat]           # (C_token + semantic_proj_dim)
fusion_proj: Conv2d → BN → GELU → C_token   # project back to token dim
```

\`semantic_proj_dim = 128\` for both heads (same as Stage-1 heads).

### Output Resolution

Stage-1 heads use `PatchConditionedDPTHead` with `down_ratio=1`, so they output at **full image resolution (H, W)** — pixel-aligned with the input depth maps and point clouds.  Stage-2 heads inherit the same architecture and also output at **(H, W)**.

Voxel memory operations (query and update) are done at **patch resolution (H//14, W//14)** for efficiency (200× fewer voxel lookups than full resolution).  Memory features are upsampled back to (H, W) before conditioning the Stage-2 DPT heads.

---

## Coordinate Alignment

The voxel memory must be indexed in a **globally-consistent coordinate frame** across all chunks. Stage-1 `predictions['world_points']` are in VGGT's internal normalized frame (relative to the first frame of each chunk) — **not** globally consistent, so they cannot be used to index a persistent voxel map.

### Training: GT world points

The dataset (`ScannetppSequence`) provides `views_all['pts3d']` with shape `(B, T, H, W, 3)`, which contains GT world points computed from absolute ScanNet++ camera poses:

```
pts3d = depth × K^{-1} @ pixel_coords  transformed by abs. camera_pose
```

These are in the same global coordinate frame for all frames in the scene, so they are consistent across all chunks within one sequence.

**Usage in training:**
```python
gt_pts_chunk = views_all['pts3d'][:, s:e]   # (B, chunk_size, H, W, 3)

predictions = model(
    chunk,
    sem_voxel_map=sem_map,
    ins_voxel_map=ins_map,
    pts_for_query=gt_pts_chunk,             # ← globally-aligned
)

update_voxel_maps(sem_map, ins_map,
                  gt_pts=gt_pts_chunk, ...)  # ← same frame for update
```

### Inference: coordinate-aligned predicted points

At inference time there are no GT pts. The SLAM pipeline uses `coordinate_alignment()` (in `amb3r/tools/pts_align.py`) to align the predicted `world_points` to the global voxel map frame and stores them in `SLAMemory.pts`. These aligned pts are passed as `pts_for_query`.

```python
# In AMB3R_VO.local_mapping():
predictions = stage2_model(
    views_all,
    sem_voxel_map=self.keyframe_memory.semantic_voxel_map,
    ins_voxel_map=self.keyframe_memory.instance_voxel_map,
    pts_for_query=aligned_pts,   # from SLAMemory after coordinate_alignment()
)
```

### Train / Inference Gap

Training uses GT pts; inference uses aligned predicted pts. The gap is small because:
1. Stage-1 geometry is accurate (VGGT pretrained on large-scale data).
2. `coordinate_alignment()` is a scale-invariant SE3 alignment that minimises residual.
3. Both point sets index the same voxel grid; any small misalignment only causes mild soft-assignment error.

---

## Memory System

### `VoxelFeatureMapV2` (extends `VoxelFeatureMap`)

Sparse voxel accumulator with confidence-weighted running average.

**New method: `query_conf(pts) → (N, 1) in [0, 1]`**
```python
mask = clamp(conf_sum[voxel(pt)] / conf_scale, 0, 1)
# conf_scale default = 1.0  →  saturates to 1 after ~1 observation unit
```

Two maps are maintained per sequence:
- `sem_voxel_map`: `VoxelFeatureMapV2(voxel_size=0.05, feat_dim=512)`
- `ins_voxel_map`: `VoxelFeatureMapV2(voxel_size=0.05, feat_dim=16)`

### Memory Query

World points are resized from image resolution to patch resolution `(Hp, Wp)`:
```python
# 1. Query at patch resolution (Hp = H//14, Wp = W//14)
pts_patch = bilinear_resize(pts_for_query, (Hp, Wp))   # (B*T, Hp, Wp, 3)
pts_flat  = pts_patch.reshape(-1, 3)                    # (B*T*Hp*Wp, 3)
mem_sem_p = sem_voxel_map.query(pts_flat)               # (N, 512) patch-res
mem_ins_p = ins_voxel_map.query(pts_flat)               # (N, 16)  patch-res
mem_mask_p= sem_voxel_map.query_conf(pts_flat)          # (N, 1)   patch-res

# 2. Upsample to full image resolution before conditioning
mem_sem   = bilinear_upsample(mem_sem_p,  (H, W))      # (B*T, 512, H, W)
mem_ins   = bilinear_upsample(mem_ins_p,  (H, W))      # (B*T, 16,  H, W)
mem_mask  = nearest_upsample(mem_mask_p,  (H, W))      # (B*T, 1,   H, W)
```

### Memory Update (after each chunk, stop-grad)

```python
with torch.no_grad():
    # Downsample pixel-aligned features to patch resolution before storing
    pts_patch   = bilinear_downsample(gt_pts, (Hp, Wp))
    sem_patch   = bilinear_downsample(refined_sem, (Hp, Wp))
    ins_patch   = bilinear_downsample(refined_ins, (Hp, Wp))
    conf_patch  = bilinear_downsample(sem_conf,    (Hp, Wp))
    sem_voxel_map.update(pts_patch, sem_patch.detach(), conf_patch.detach())
    ins_voxel_map.update(pts_patch, ins_patch.detach(), conf_patch.detach())
```

---

## Recurrent Training (Truncated BPTT)

### Dataset: `ScannetppSequence`

Extends `Scannetpp_Arrow` with consecutive frame sampling.

```
__init__ params (on top of Scannetpp_Arrow):
    stride  : int  — temporal subsampling step (default 2)
    num_frames passed at instantiation = seq_len (e.g. 24)

_get_views:
    1. random start_idx in [0, total_frames - seq_len * stride]
    2. frame_indices = [start + i*stride for i in range(seq_len)]
    3. return seq_len consecutive frames (duplicates last valid if bad frame)
```

Output format is identical to `Scannetpp_Arrow` (compatible with existing collation pipeline).

### Training Loop Structure

```python
for batch in dataloader:                     # (B=1, N=seq_len, ...)
    views, views_all = batch
    views_all = move_to_device(views_all)

    # Reset memory for each new sequence
    sem_map, ins_map = make_voxel_maps()

    total_loss = 0
    for c in range(n_chunks):               # n_chunks = seq_len // chunk_size
        s, e = c * chunk_size, (c+1) * chunk_size
        chunk = slice_chunk(views_all, s, e)   # (B=1, K, ...)

        gt_pts_chunk = views_all['pts3d'][:, s:e]  # (B, K, H, W, 3)

        predictions = model(
            chunk,
            sem_voxel_map = sem_map if c > 0 else None,
            ins_voxel_map = ins_map if c > 0 else None,
            pts_for_query = gt_pts_chunk,            # globally-aligned GT pts
        )

        loss_c = criterion(predictions, chunk)
        total_loss += loss_c               # accumulate over chunks

        # Update memory (stop-grad, before next chunk)
        update_voxel_maps(sem_map, ins_map,
                          gt_pts=gt_pts_chunk,
                          refined_sem=predictions['semantic_feat'].detach(),
                          refined_ins=predictions['instance_feat'].detach(),
                          sem_conf=predictions['semantic_conf'])

    (total_loss / n_chunks).backward()
    optimizer.step()
```

**Why Truncated BPTT?**  
Memory is updated with `.detach()`, so gradients never flow through historical chunks. Each chunk's gradient only passes through `semantic_memory_head` and `instance_memory_head` parameters. This matches inference behavior (SLAM pipeline processes one chunk at a time with persistent memory) and avoids exploding gradients.

---

## Loss Functions (`Stage2Loss`)

| Loss | Formula | When active |
|------|---------|-------------|
| `loss_sem_align` | `1 - cosine_sim(refined_sem, lseg_gt_resized)` per instance | Always |
| `loss_sem_memory` | `(1 - cosine_sim(refined_sem, mem_sem)) * mem_mask` | chunk > 0, seen voxels |
| `loss_ins_contrast` | `ContrastiveLoss(intra + inter)` w/ GT instance masks | Always |
| `loss_ins_memory` | `(1 - cosine_sim(refined_ins, mem_ins)) * mem_mask` | chunk > 0, seen voxels |

Default weights: `w_sem_align=0.5, w_sem_memory=1.0, w_ins_contrast=1.0, w_ins_memory=1.0`

**`loss_sem_align`**: Keeps refined_sem aligned to CLIP space (inherits Stage-1 supervision signal so semantic meaning is not lost).

**`loss_sem_memory` / `loss_ins_memory`**: Memory consistency — where a voxel has been observed (`mem_mask > 0`), the new prediction is pulled toward the accumulated memory feature. `mem_mask` acts as a soft weight.

**`loss_ins_contrast`**: `ContrastiveLoss` with `inter_mode='hinge', inter_margin=0.2` — within a frame, same-instance patches cluster together, different-instance patches are pushed apart.

---

## Files

```
amb3r/
  stage2_heads.py              ← MemoryConditionedDPTHead (extends PatchConditionedDPTHead)
                                  build_semantic_memory_head()
                                  build_instance_memory_head()
  model_stage2.py              ← AMB3RStage2
                                  _run_stage1()       captures aggregated_tokens_list
                                  forward(pts_for_query)  memory query + DPT heads
                                  load_stage1_weights()
                                  load_stage2_weights()
  loss_stage2.py               ← Stage2Loss, compute_sem_align_loss,
                                  compute_sem_memory_loss, compute_ins_contrastive_loss,
                                  compute_ins_memory_loss
  datasets/
    scannetpp_sequence.py      ← ScannetppSequence (consecutive frames, extends Scannetpp_Arrow)

slam_semantic/
  semantic_voxel_map_v2.py    ← VoxelFeatureMapV2 (adds query_conf())

train_stage2.py                ← training script
```

**Checkpoint saves only Stage-2 weights:**
```python
state = {k: v for k, v in model.state_dict().items()
         if k.startswith('semantic_memory_head.') or
            k.startswith('instance_memory_head.')}
```

---

## Training Command

```bash
torchrun --nproc_per_node=1 .claude/sandbox/train_stage2.py \
    --stage1_ckpt outputs/exp_stage1_wo_lora/checkpoint-best.pth \
    --data_root /mnt/HDD4/ricky/data/scannetpp_arrow \
    --seq_len 24 --chunk_size 6 --stride 2 \
    --batch_size 1 --accum_iter 4 \
    --epochs 30 --lr 5e-4 \
    --w_sem_align 0.5 --w_sem_memory 1.0 \
    --w_ins_contrast 1.0 --w_ins_memory 1.0 \
    --output_dir outputs/exp_stage2
```

**GPU memory reference:**
- A6000 (48 GB): `chunk_size=4~5`, `stride=2`
- Pro6000 (96 GB): `chunk_size=6~8`, `stride=2`

---

## Inference Integration (SLAM)

Stage 2 hooks into the existing `slam_semantic/pipeline.py` with minimal changes.

```python
# In AMB3R_VO.local_mapping(), after Stage-1 produces res:

# 1. Get coordinate-aligned predicted pts from SLAM memory
#    (SLAMemory.update() already runs coordinate_alignment() and stores
#     globally-aligned pts in self.pts[map_idx])
aligned_pts = get_aligned_pts(self.keyframe_memory, views_all)  # (B, T, H, W, 3)

# 2. Stage-2 refinement
predictions = stage2_model(
    views_all,
    sem_voxel_map=self.keyframe_memory.semantic_voxel_map,
    ins_voxel_map=self.keyframe_memory.instance_voxel_map,
    pts_for_query=aligned_pts,
)
res['semantic_feat'] = predictions['semantic_feat']
res['instance_feat'] = predictions['instance_feat']

# 3. keyframe_memory.update() proceeds as before using res
```

The training recurrent loop (chunk-by-chunk with persistent memory + `pts_for_query`) directly mirrors this SLAM inference pattern — only the source of globally-aligned pts differs (GT from dataset vs. coordinate-aligned predicted pts).

---

## Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| DPT head (not Conv head) for Stage 2 | Needs multi-scale spatial reasoning over patch tokens to produce spatially coherent consistent features |
| `aggregated_tokens_list` captured step-by-step | Stage-1 `forward()` with `has_backend=True` doesn't store it; step-by-step call to `decode_patch_tokens` gives access |
| No LSeg in Stage-2 conditioning | `sem_s1` (Stage-1 output) already encodes CLIP-aligned semantics; avoids redundancy and removes LSeg inference overhead |
| `sem_s1` in instance head conditioning | Cross-task context: instance head should know object category when assigning embeddings |
| `mem_mask` always in conditioning (not just gate) | Head learns when to trust/ignore memory; cleaner than hard gating |
| GT pts for voxel indexing (training) | Simpler and cleaner than aligning predicted pts during training; avoids needing `coordinate_alignment()` in the training loop |
| `pts_for_query` parameter in `forward()` | Decouples the model from the coordinate alignment strategy; caller provides globally-aligned pts, making inference and training interchangeable |
| Stop-grad on memory update | Prevents gradient explosion across chunks; matches inference behavior |
| Separate sem/ins voxel maps | Different feature dims (512 vs 16); independent confidence accumulation |
| Checkpoint stores only Stage-2 weights | Stage-1 checkpoint already managed separately; keeps Stage-2 checkpoints small |

---

## Known Limitations

1. **Train/inference coordinate gap**: Training uses GT abs. world pts; inference uses coordinate-aligned predicted pts. In practice the gap is small (~cm scale) but could cause subtle voxel mismatches at boundaries.

2. **No cross-sequence memory**: Memory resets per sequence during training (by design — avoids GPU OOM). SLAM inference accumulates across the full scene.

3. **Single-resolution training**: Both heads process one resolution at a time. Mixed-resolution training (like Stage-1) would require adapting `ScannetppSequence`.
