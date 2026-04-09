# AMB3R Stage-2 V2 Design Document

## Overview

Stage-2 V2 replaces the V1 DPT memory-head approach with a **PTv3 backend** that mirrors the original AMB3R geometry backend.  After Stage-1 processes a chunk of K frames independently, the PTv3 backend receives geo + memory + Stage-1 semantic/instance features, fuses them in 3D, and injects the result back into the frozen Stage-1 via zero-conv — causing Stage-1 to **re-predict everything** (depth, pose, pts, semantic, instance) in a memory-consistent way.

---

## V1 vs V2 Comparison

| Aspect | Stage-2 V1 | Stage-2 V2 |
|--------|-----------|-----------|
| Trainable component | 2 DPT memory heads | 1 PTv3 BackEnd |
| What gets re-predicted | semantic + instance only | **depth + pose + pts + semantic + instance** |
| Geometry supervised | ✗ | ✓ |
| Injection mechanism | DPT heads read frozen `aggregated_tokens_list` | Zero-conv injected into patch tokens → VGGT re-decode |
| Memory conditioning | concatenated into DPT head conditioning | Concatenated into PTv3 input as `semantic_feats` / `instance_feats` |
| VRAM | Lower (no second VGGT pass) | Higher (second VGGT decode per chunk) |
| Similarity to original AMB3R | Low | **High** (same PTv3 + zero-conv pattern) |

---

## Architecture

### Data Flow

```
Input: K frames + VoxelMemory M_{t-1}

╔══════════════════════════════════════════════════════════════════════╗
║  Stage-1  (frozen, no_grad)                                          ║
║                                                                       ║
║  encode_patch_tokens + clip_patch_fusion                             ║
║         │                                                            ║
║  decode_patch_tokens  (VGGT frame/global blocks — run ONCE)          ║
║         │                                                            ║
║  ├─ decode_heads(has_backend=False) → enc, dec, world_pts            ║
║  └─ semantic_head / instance_head  → sem_s1, ins_s1                  ║
╚══════════════════════════════════════════════════════════════════════╝
                │                    │
     world_pts (or gt_pts)       sem_s1, ins_s1
                │
        Memory query @ patch res (H/14)
                │
     mem_sem (512), mem_ins (16), mem_mask (1)
                │
  sem_cond = cat[mem_sem, sem_s1]        (1024-dim, patch res)
  ins_cond = cat[mem_ins, ins_s1, mem_mask]  (33-dim, patch res)
                │
╔══════════════════════════════════════════════════════════════════════╗
║  Stage-2 PTv3 BackEnd  (trainable, with grad)                        ║
║                                                                       ║
║  feat = cat[enc, dec].detach()  →  aligner  →  feat_geo (1024-dim)  ║
║  resize pts + feat_geo to H//7  (×2 vs patch res)                    ║
║  sem_cond, ins_cond → upsample to H//7 → flatten                     ║
║                                                                       ║
║  BackEnd.forward(pts, feat_geo,                                       ║
║                  semantic_feats=sem_cond_flat [1024-dim],            ║
║                  instance_feats=ins_cond_flat [33-dim])              ║
║         │                                                            ║
║  PTv3 voxelizes scene, refines features                               ║
║  KNN interpolation  → voxel_feat_aligned  (B*T, 1024, Hp, Wp)       ║
║  downsample + zero_conv + gate → voxel_feat_aligned_vis              ║
║  voxel_layer_list  (48 ZeroConvBlocks for per-VGGT-layer cond)       ║
╚══════════════════════════════════════════════════════════════════════╝
                │
  patch_tokens["x_norm_patchtokens"] = original + voxel_feat_vis
                │           (creates gradient path to Stage-2 backend)
╔══════════════════════════════════════════════════════════════════════╗
║  Stage-1 second decode  (frozen weights, grad via injection)         ║
║                                                                       ║
║  decode_patch_tokens_and_heads(patched_tokens,                       ║
║      voxel_feat=voxel_feat_aligned,                                  ║
║      voxel_layer_list=voxel_layer_list,                              ║
║      semantic_feats=lseg_feat, has_backend=True)                     ║
║         │                                                            ║
║  → depth, pose, pts, semantic_feat, instance_feat                    ║
╚══════════════════════════════════════════════════════════════════════╝
                │
  Update VoxelMemory (stop-grad, features from re-decoded output)
```

### Fixed vs Trainable Parameters

| Module | Status | Notes |
|--------|--------|-------|
| DINOv2 patch_embed | **Fixed** | Stage-1 pretrained |
| VGGT frame/global blocks | **Fixed** | Stage-1 pretrained |
| LSeg / CLIP | **Fixed** | Always frozen |
| CLIPPatchFusion | **Fixed** | Stage-1 pretrained |
| PatchConditionedDPTHead (sem/ins) | **Fixed** | Stage-1 pretrained |
| DPT heads (depth, point, camera) | **Fixed** | Stage-1 pretrained |
| `backend.aligner` | **TRAIN** | Projects enc+dec → 1024-dim |
| `backend.point_transformer` (PTv3) | **TRAIN** | Voxelized 3D feature learning |
| `backend.downsample` | **TRAIN** | Hs,Ws → Hp,Wp spatial reduction |
| `backend.zero_conv` + `gate_scale` | **TRAIN** | Patch token injection gate |
| `backend.zero_conv_layers[48]` + `gate_scales[48]` | **TRAIN** | Per-VGGT-layer injection |
| `backend.sem_proj` | **TRAIN** | Projects sem_cond (1024) → 256 |
| `backend.ins_proj` | **TRAIN** | Projects ins_cond (33) → 128 |
| `backend.feat_merger` | **TRAIN** | Merges geo + sem + ins → 1024 |
| VoxelMemory (runtime state) | No params | Accumulates at inference |

---

## Memory Conditioning

### Semantic Conditioning to PTv3

```
sem_cond = cat[mem_sem (512), sem_s1 (512)]   →   1024-dim

mem_sem  : feature retrieved from voxel memory (0 if unseen)
sem_s1   : Stage-1 semantic output (CLIP-aligned, 512-dim)

→ BackEnd.sem_proj: Linear(1024, 256) → merged with geo feat
```

**Rationale:** The PTv3 receives both what the memory already knows about each 3D point (`mem_sem`) and what Stage-1 predicts for the current observation (`sem_s1`). This enables learning to:
- Reinforce predictions where memory and current observation agree
- Correct noisy Stage-1 predictions using persistent memory
- Produce novel features for newly-seen regions

### Instance Conditioning to PTv3

```
ins_cond = cat[mem_ins (16), ins_s1 (16), mem_mask (1)]   →   33-dim

mem_ins  : instance embedding from memory (0 if unseen)
ins_s1   : Stage-1 instance output (16-dim)
mem_mask : soft confidence in [0,1] — indicates how much memory is trusted
```

**Rationale:** `mem_mask` is critical — the PTv3 must learn different behaviors for seen vs. unseen voxels. For seen voxels (`mem_mask ≈ 1`), it should maintain temporal consistency. For unseen voxels (`mem_mask ≈ 0`), it generates fresh discriminative embeddings.

---

## Gradient Flow

```
Loss
 │
 ▼
res_s2 (Stage-1 second decode outputs: depth, sem, ins, pose, pts)
 │
 ▼  (through frozen VGGT + DPT heads as constant operators)
patch_tokens["x_norm_patchtokens"] = original_pt + vf_flat
                                                    │
                                                    ▼
                               voxel_feat_aligned_vis → zero_conv (trainable)
                                                               │
AND voxel_layer_list[i](voxel_feat_aligned) at each VGGT block │
                                                               │
                                                               ▼
                                                    voxel_feat_aligned → downsample (trainable)
                                                               │
                                                               ▼
                                                    PTv3 output → knn interp
                                                               │
                                                               ▼
                                                    BackEnd.forward (PTv3, sem_proj, ins_proj, feat_merger, aligner)
                                                               │
                                                               ▼
                                                    Stage-2 BackEnd parameters ← gradients arrive here
```

Stage-1 weights receive **zero gradient** (frozen, `requires_grad=False`). Only Stage-2 BackEnd parameters are updated.

---

## Backend Dimensions

```python
BackEnd(
    in_dim  = 2048 + 1024,   # enc (1024) + dec (1024) = 2048 → aligner input
    out_dim = 1024,
    sem_dim = 512 * 2,       # = 1024  (cat[mem_sem, sem_s1])
    ins_dim = 16  * 2 + 1,   # = 33    (cat[mem_ins, ins_s1, mem_mask])
)

# aligner: Linear(3072, 1536) → Linear(1536, 1024)
# sem_proj: Linear(1024, 256)
# ins_proj: Linear(33,   128)
# feat_merger: Linear(1024+256+128, 1024) = Linear(1408, 1024)
# PTv3: PointTransformerV3(in_channels=1024, ...)
```

---

## Recurrent Training (Truncated BPTT)

Same structure as Stage-2 V1 — identical chunk-by-chunk loop, same `ScannetppSequence` dataset.

```python
for batch in dataloader:             # B=1, T=seq_len
    sem_map, ins_map = make_voxel_maps()

    for c in range(n_chunks):
        s, e = c*chunk_size, (c+1)*chunk_size
        chunk = views_all[:, s:e]
        gt_pts = views_all['pts3d'][:, s:e]

        predictions = model(
            chunk,
            sem_voxel_map = sem_map if c > 0 else None,
            ins_voxel_map = ins_map if c > 0 else None,
            pts_for_query = gt_pts,
        )

        loss = criterion(predictions, chunk)
        total_loss += loss

        update_voxel_maps(sem_map, ins_map,
                          gt_pts, predictions['semantic_feat'],
                          predictions['instance_feat'],
                          sem_conf=predictions.get('semantic_conf', ...))

    (total_loss / n_chunks).backward()
    optimizer.step()
```

Memory is updated with `.detach()` — gradients never flow through historical chunks.

---

## Loss Functions (`Stage2LossV2`)

| Loss | Formula | Active |
|------|---------|--------|
| `loss_conf_depth` / `loss_reg_depth` | SILog + confidence regression | Always |
| `loss_conf_point` / `loss_reg_point` | L2 + confidence on world points | Always |
| `loss_camera` | T + R + FL on camera pose encoding | Always |
| `loss_sem_align` | `1 - cosine_sim(sem_s2, lseg_gt)` per instance | Always |
| `loss_sem_memory` | `(1 - cosine_sim(sem_s2, mem_sem)) * mem_mask` | chunk > 0 |
| `loss_ins_contrast` | SupCon / hinge contrastive | Always |
| `loss_ins_memory` | `(1 - cosine_sim(ins_s2, mem_ins)) * mem_mask` | chunk > 0 |

Default weights:
```
w_geo_depth=1.0, w_geo_pts=1.0, w_geo_camera=0.5
w_sem_align=0.5, w_sem_memory=1.0
w_ins_contrast=1.0, w_ins_memory=1.0
```

---

## Files

```
.claude/sandbox/
  amb3r/
    model_stage2v2.py    ← AMB3RStage2V2
                             _encode_stage1()
                             _get_voxel_feat()
                             forward()
    loss_stage2v2.py     ← Stage2LossV2
                             geo (depth + pts + camera) + sem + ins + memory
  train_stage2v2.py      ← training script (same BPTT structure as V1)
```

Reuses without modification:
```
amb3r/backend_semantic.py   — BackEnd (PTv3 + zero_conv)
slam_semantic/semantic_voxel_map_v2.py — VoxelFeatureMapV2
amb3r/datasets/scannetpp_sequence.py   — ScannetppSequence
```

---

## Training Command

```bash
torchrun --nproc_per_node=1 .claude/sandbox/train_stage2v2.py \
    --stage1_ckpt outputs/exp_stage1_wo_lora/checkpoint-best.pth \
    --data_root /mnt/HDD4/ricky/data/scannetpp_arrow \
    --seq_len 24 --chunk_size 6 --stride 2 \
    --batch_size 1 --accum_iter 4 \
    --epochs 30 --lr 1e-4 \
    --voxel_res 0.01 --voxel_size 0.05 \
    --w_geo_depth 1.0 --w_geo_pts 1.0 --w_geo_camera 0.5 \
    --w_sem_align 0.5 --w_sem_memory 1.0 \
    --w_ins_contrast 1.0 --w_ins_memory 1.0 \
    --output_dir outputs/exp_stage2v2
```

**GPU memory reference** (B=1):
- A6000 (48 GB): `chunk_size=3~4`, `stride=2`
- Pro6000 (96 GB): `chunk_size=4~6`, `stride=2`

Note: V2 requires more VRAM than V1 due to the second VGGT decode pass with gradient tracking.

---

## Checkpoint Format

Only BackEnd weights are saved:
```python
state = {k: v for k, v in model.state_dict().items() if k.startswith('backend.')}
```

Stage-1 is loaded separately via `--stage1_ckpt`.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Reuse `BackEnd` directly (changed `sem_dim`, `ins_dim`) | Zero new code for PTv3/zero_conv/voxel logic; all tested infrastructure |
| `sem_cond = cat[mem_sem, sem_s1]` | PTv3 sees both "what we know" and "what we predict now" — can reconcile or reinforce |
| `ins_cond = cat[mem_ins, ins_s1, mem_mask]` | `mem_mask` teaches PTv3 when to enforce consistency vs. generate novel embeddings |
| VGGT decode twice (no_grad + grad) | First pass: cheap encoder output for memory query. Second pass: full backprop through voxel injection |
| `patch_tokens_cond = dict(patch_tokens); patch_tokens_cond["x_norm"] = old + vf` | Non-in-place addition preserves autograd graph (original tensor has `requires_grad=False`, result has `True` from `vf`) |
| Geometry loss in Stage-2 | PTv3 sees world geometry context → can improve geo. Loss signal from geometry also drives backend to learn better 3D structure |
| GT pts for voxel indexing (training) | Same as V1 — simpler, avoids `coordinate_alignment()` in training loop |
| Same `update_voxel_maps` as V1 | Memory stores Stage-2 re-predicted sem/ins (better quality than Stage-1 over time) |
