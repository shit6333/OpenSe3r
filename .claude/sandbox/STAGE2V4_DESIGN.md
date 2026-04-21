# Stage-2 V4: Decoupled Spatial-Memory Attention — 設計文件

## 1. 核心概念

Stage-2 V4 的目標是在 **不引入重量級 3D backbone (如 PTv3)** 的前提下，透過 **隱式空間記憶 (Implicit Spatial Memory)** 讓多幀之間的 semantic / instance feature 保持 **跨視角一致性**。

### 與 Stage-2 V2 (PTv3 backend) 的關鍵差異

| | V2 (PTv3) | V4 (Memory Attention) |
|---|---|---|
| 3D backbone | PointTransformerV3 (~40M) | ❌ 無 |
| 可訓練參數 | ~40M (PTv3 + heads) | ~3M (Fusion + Tokenizer) |
| 跨幀資訊傳遞 | 顯式 voxelization + 3D attention | 隱式 voxel hash map + EMA |
| 訓練方式 | Per-chunk backward (Truncated BPTT) | **Single backward (Chunked Unrolled BPTT)** |
| Gradient flow | 截斷於 chunk 邊界 | **完整穿越所有 chunk** |

---

## 2. 架構總覽

```
Sequence: [F0 F1 F2 F3 | F4 F5 F6 F7 | F8 F9 F10 F11 | F12 F13 F14 F15]
           ── chunk 0 ──  ── chunk 1 ──  ── chunk 2 ───  ── chunk 3 ───

Per-chunk forward:

    ┌─────────────────────────────────────────────────────────────────┐
    │ Step 1: Frozen Stage-1 backbone                                 │
    │   images → DINOv2 + VGGT frame/global blocks                    │
    │   → aggregated_tokens_list (atl), depth, pose, world_points    │
    │   → LSeg (frozen CLIP features)                                 │
    │                                                                 │
    │ Step 2: Multi-scale VGGT feature extraction                     │
    │   X_vggt_early = atl[4][:,:,patch_start:]   (early DPT layer) │
    │   X_vggt_late  = atl[23][:,:,patch_start:]  (last DPT layer)  │
    │                                                                 │
    │ Step 3: Memory retrieval                                        │
    │   pts → DifferentiableVoxelMap.query()                          │
    │   → X_mem [B*T, N_patch, 128]  (grad_fn from prev chunks)     │
    │   → null_token for unseen voxels                                │
    │                                                                 │
    │ Step 4: Multi-scale Memory Fusion (TRAINABLE)                   │
    │   X_fuse_early = MemoryFusion(X_vggt_early, X_mem)             │
    │   X_fuse_late  = MemoryFusion(X_vggt_late,  X_mem)   [共享參數] │
    │                                                                 │
    │ Step 5: Task decoding (frozen params, grad ENABLED)             │
    │   modified_atl[4]  = X_fuse_early                               │
    │   modified_atl[23] = X_fuse_late                                │
    │   sem_feat = semantic_head(modified_atl)                        │
    │   ins_feat = instance_head(modified_atl)                        │
    │                                                                 │
    │ Step 6: Memory Tokenizer (TRAINABLE)                            │
    │   avg(X_fuse_early, X_fuse_late) → MLP → M_readout + W_conf   │
    │   scatter_sum(M*W) / scatter_sum(W) → voxel_new                │
    │   voxel_map.EMA_update(voxel_new)   [grad_fn preserved!]       │
    └─────────────────────────────────────────────────────────────────┘
```

---

## 3. 模組設計

### 3.1 Memory 模組 (`memory_stage2v4.py`)

#### LearnableNullToken
- 1 × 128 的可學習參數，初始化為 N(0, 0.02²)
- 用於從未被觀察過的 voxel，提供 cold-start fallback

#### DifferentiableVoxelMap
- Hash map: `(ix, iy, iz) → Tensor[128]`，tensor 帶有 grad_fn
- **EMA 更新**: `store[k] = (1-α)·old + α·new`，新的 tensor 連接了 old（前一 chunk）和 new（當前 chunk）的計算圖
- **不會 detach!** 這是 Chunked Unrolled BPTT 的核心 — 梯度可以穿越所有 chunk

#### CrossChunkFeatureBuffer
- Hash map: `(ix, iy, iz) → (sem[512], ins[16])`，**完全 detach**
- **First-visit anchoring**: 只存第一次看到某個 voxel 時的 feature
- 後續 chunk 必須 align 到此 anchor → 強迫 MemoryFusion 利用 memory 重現一致的 feature
- 提供 cross-view consistency loss 的 target

### 3.2 NN 模組 (`model_stage2v4.py`)

#### MemoryFusionModule (Cross-attn + Self-attn)
```
Step 1 — Cross-attention (VGGT queries memory):
    Q = W_q(X_vggt),  K = W_k(X_mem),  V = W_v(X_mem)
    X_cross = X_vggt + γ_cross · OutProj(CrossAttn(Q, K, V))

Step 2 — Self-attention (contextualise fused patches):
    X_norm  = LayerNorm(X_cross)
    q = k = v = proj_in(X_norm)
    X_fuse  = X_cross + γ_self · OutProj(SelfAttn(q, k, v))
```
- **Zero-init γ**: γ_cross = γ_self = 0 at init → identity mapping
- Training 初期 = frozen Stage-1 performance（穩定起步）
- **Multi-scale**: 共享參數分別作用於 atl[4] 和 atl[23]

#### MemoryTokenizer
```
Linear(2048, 512) → GELU → LayerNorm(512) → Linear(512, 129)
→ M_readout[128] + W_conf[1] (sigmoid)
```
- Input: `avg(X_fuse_early, X_fuse_late)`
- Confidence 用於 intra-frame 加權聚合

### 3.3 Loss 設計 (`loss_stage2v4.py`)

| Loss | 類別 | 公式 | 權重 |
|------|------|------|------|
| `sem_align` | Per-chunk | cosine(sem_feat, LSeg_GT) per instance | 0.5 |
| `ins_contrastive` | Per-chunk | Hinge contrastive (margin=0.2) | 1.0 |
| `mem_consistency` | Per-chunk | cosine(M_readout, M_queried.detach) | 0.1 |
| **`cross_sem`** | **Cross-chunk** | cosine(sem_patch, sem_anchor.detach) | 0.5 |
| **`cross_ins`** | **Cross-chunk** | cosine(ins_patch, ins_anchor.detach) | **2.0** |

#### Per-chunk losses（`Stage2V4Loss.forward()`）

1. **`sem_align`** — 重用 `loss_stage2.compute_sem_align_loss()`
   - 對每個 instance mask 區域，計算 `sem_feat` 與 frozen LSeg GT 的 cosine distance
   - 驅動 semantic head 學到對齊 CLIP 空間的 feature
   
2. **`ins_contrastive`** — 重用 `loss_stage2.compute_ins_contrastive_loss()`
   - Hinge-based contrastive loss (margin=0.2)，同 instance 的 pixel 拉近，不同 instance 推遠
   - 只作用於單幀內部，不保證跨視角一致性
   
3. **`mem_consistency`** (optional) — M_readout ↔ M_queried
   - 當前 chunk 的 memory token 應該和從 voxel map 取回的舊 token 一致
   - M_queried 做 `.detach()`，避免雙重梯度不穩定
   - 主要作為 auxiliary stabiliser，權重較低 (0.1)

#### `compute_cross_view_consistency_loss()` — 跨 chunk 一致性

這是 Stage-2 V4 最關鍵的新設計，獨立於 `Stage2V4Loss` 之外，在 training loop 中直接呼叫。

**動機**: Per-chunk losses 只監督單個 chunk 內部的 feature 品質，但無法保證同一個 3D 位置在不同 chunk（不同視角）產生一致的 sem/ins feature。如果 memory fusion 沒有真正利用 spatial memory，跨視角 feature 就會不一致，SLAM pipeline 下游的 semantic voxel map 會出現「同一物體在不同角度看到不同 label / instance ID」的問題。

**設計**:
```
函數簽名:
    compute_cross_view_consistency_loss(
        sem_curr,    # [N, 512]  當前 chunk 的 patch-level sem features (有 grad)
        ins_curr,    # [N, 16]   當前 chunk 的 patch-level ins features (有 grad)
        sem_stored,  # [N, 512]  first-visit anchor (detached, 來自 CrossChunkFeatureBuffer)
        ins_stored,  # [N, 16]   first-visit anchor (detached)
        mask,        # [N, 1]    1.0 = 此 voxel 在之前的 chunk 已被觀察過
        w_sem=0.5,   # semantic 一致性權重
        w_ins=2.0,   # instance 一致性權重 (4× semantic)
    ) → dict
```

**計算流程**:
1. 從 `mask` 篩選出已被觀察過的 patches（`matched = mask.bool()`）
2. 對 matched patches:
   - L2 normalize `sem_curr[matched]` 和 `sem_stored[matched]`
   - 計算 cosine distance: `loss = mean(1 − Σ(s_curr · s_anchor))`
   - 同理計算 `ins_curr` 和 `ins_stored`
3. 加權求和: `cross_objective = w_sem * sem_loss + w_ins * ins_loss`

**返回值**:
```python
{
    'loss_cross_sem':     w_sem * sem_loss,     # 加權後的 loss (有 grad)
    'loss_cross_sem_det': sem_loss.detach(),     # 未加權的 raw loss (for logging)
    'loss_cross_ins':     w_ins * ins_loss,
    'loss_cross_ins_det': ins_loss.detach(),
    'cross_n_matched':    int,                   # 匹配的 patch 數量
    'cross_objective':    loss_cross_sem + loss_cross_ins,
}
```

**為什麼 instance 權重是 semantic 的 4 倍?**
- Semantic feature 已有 `sem_align` loss 直接對齊 frozen LSeg GT，每個 frame 獨立就能學到不錯的 semantic feature
- Instance feature 只有 `ins_contrastive` 做 within-frame 同/異 instance 拉近/推遠，**沒有任何 per-frame 的跨視角監督**
- Cross-view instance loss 是唯一驅動 memory fusion 學到跨幀 instance 對應的訊號

**為什麼不放在 `Stage2V4Loss` 裡?**
- 此 loss 需要存取 `CrossChunkFeatureBuffer`（training loop 層級的狀態）
- `Stage2V4Loss.forward()` 只接收 `(predictions, batch)` 兩個參數，不應耦合 memory buffer
- 分離設計讓 loss 模組保持乾淨：`Stage2V4Loss` = per-chunk losses，`compute_cross_view_consistency_loss` / `compute_instance_id_cross_chunk_loss` = cross-chunk losses

#### `compute_instance_id_cross_chunk_loss()` — 用 Instance ID 做跨 chunk 一致性

**動機**: `compute_cross_view_consistency_loss` 用 3D 空間位置（voxel）匹配跨 chunk 的 patches，但在視角差異大時，兩個 chunk 觀察到同一物體的 patches 可能 voxel 不完全重疊，造成 anchor 缺失。ScanNet 的 instance_mask 是**整個 scene 全域一致的整數 ID**（所有 frame 共用同一個 ID 集合），因此可以用 instance ID 直接作為 matching key，matching 更穩定，不依賴 3D 座標對齊。

**設計**:
```
函數簽名:
    compute_instance_id_cross_chunk_loss(
        sem_curr,           # [N, sem_dim]  當前 chunk 的 patch-level sem features (有 grad)
        ins_curr,           # [N, ins_dim]  當前 chunk 的 patch-level ins features (有 grad)
        instance_mask_flat, # [N]           per-patch integer instance IDs (0 = background)
        feat_buffer,        # CrossChunkFeatureBuffer — query_by_id / update_by_id
        w_sem=0.5,          # semantic 一致性權重
        w_ins=2.0,          # instance 一致性權重
    ) → dict
```

**計算流程**:
1. 呼叫 `feat_buffer.query_by_id(instance_mask_flat, device)` → 取得 `sem_stored, ins_stored, matched`
   - `matched=True`：此 patch 的 instance ID 在前一個 chunk 已被記錄過
2. 對 matched patches:
   - L2 normalize `ins_curr[matched]` 和 `ins_stored[matched]`
   - 計算 cosine distance: `loss = mean(1 − Σ(ic · it))`
3. 返回加權 loss

**返回值**:
```python
{
    'loss_instid_cross_sem':     w_sem * sem_loss,   # 加權後 (有 grad)
    'loss_instid_cross_sem_det': sem_loss.detach(),  # raw loss
    'loss_instid_cross_ins':     w_ins * ins_loss,
    'loss_instid_cross_ins_det': ins_loss.detach(),
    'instid_n_matched':          int,
    'instid_objective':          sem + ins,          # 加進 total loss
}
```

**與 voxel-based cross-view loss 的比較**:
| 維度 | `compute_cross_view_consistency_loss` | `compute_instance_id_cross_chunk_loss` |
|------|--------------------------------------|----------------------------------------|
| Matching key | 3D voxel 座標 | Integer instance ID |
| 適用場景 | 局部 overlap 大時準確 | Instance ID 全域一致即可用 |
| 覆蓋範圍 | Voxel 精確重疊的 patches | 所有屬於同一 instance 的 patches |
| Anchor 穩定性 | 受座標對齊精度影響 | 只依賴 dataset 的 instance annotation |
| 監督 semantic | 是（sem + ins 都監督） | 否（只監督 ins） |

兩者可同時使用，互補。

**梯度流**:
```
cross_view_loss
  ↓ grad
sem_feat_patch / ins_feat_patch          ← adaptive_avg_pool2d(DPT head output)
  ↓ grad
frozen DPT heads (params frozen, ops have grad)
  ↓ grad
modified_atl[4] / modified_atl[23]       ← X_fuse_{early,late}
  ↓ grad
MemoryFusionModule (γ_cross, γ_self, W_q/k/v, proj)
  ↓ grad
X_mem                                     ← DifferentiableVoxelMap.query()
  ↓ grad (through stored tensors with grad_fn)
DifferentiableVoxelMap._store[key]        ← EMA: (1-α)·old + α·new
  ↓ grad (crosses chunk boundary!)
MemoryTokenizer (previous chunk)
  ↓ ...
MemoryFusionModule (previous chunk)       ← BPTT 繼續向前傳播
```

---

## 4. 坐標系統 (Coordinate Frames)

> **重要**: Stage-2 V4 的 model 不在 metric scale 下運作。必須正確區分各個坐標系。

### 4.1 坐標系一覽

| 資料 | 坐標系 | 說明 |
|------|--------|------|
| `views_all['pts3d']` (raw) | **Dataset Global** | 原始 GT，metric scale，跨所有 frame 一致 |
| `views_all['pts3d']` (after normalize_gt) | **Local Normalized** | 以第一個 camera 為中心，unit scale |
| `res_geo['world_points']` (pred) | **VGGT Local** | 模型預測，非 metric，scale 未知 |
| Memory query coords (`pts_flat`) | **Dataset Global** (training) | 必須跨 chunk 一致 |
| Memory query coords (`pts_flat`) | **VGGT Local** (inference) | 無 GT 時用 predicted，同一 sequence 內一致 |
| PLY export coords | **Normalized + Scale-aligned** | align_pred_to_gt 後的結果 |

### 4.2 Training 流程中的坐標處理

```python
# ① 捕獲 RAW GT 在 global frame (跨 chunk 一致)
gt_pts_global = views_all['pts3d'].clone()   # [B, T, H, W, 3] dataset global

# ② normalize_gt: in-place 轉換 GT 到 local normalized frame
normalize_gt(views_all)
#   views_all['pts3d']      → local normalized (centered at cam0, unit scale)
#   views_all['extrinsics'] → local normalized
#   views_all['depthmap']   → local normalized

# ③ 每個 chunk:
#   Memory queries: 使用 gt_pts_global[:, s:e]   → global frame (一致性保證!)
#   Loss 計算:      sem/ins losses 不依賴坐標     → 不受影響
#   Feature buffer: 使用 gt_pts_global            → global frame (與 voxel_map 同一坐標系)
```

### 4.3 為什麼 Memory 必須在 Global Frame?

```
Chunk 0: frames [F0, F1, F2, F3]
   voxel_map.update() → 把 F0~F3 看到的 voxel 存入 memory
   key = quantize(gt_pts_global)  ← global frame

Chunk 1: frames [F4, F5, F6, F7]
   voxel_map.query()  → 用 F4~F7 的 gt_pts_global 查詢
   key = quantize(gt_pts_global)  ← 同一個 global frame → ✓ 命中 chunk 0 存的 voxel

如果改用 predicted world_points (VGGT local frame):
   - 不同 chunk 的 VGGT forward 是獨立的
   - 預測坐標可能有 drift / scale 差異
   - → 查詢命中率低，memory 形同虛設
```

### 4.4 Eval 時的坐標處理

```python
# ① 同樣先捕獲 raw GT → global frame for memory
gt_pts_global = views_all['pts3d'].clone()
normalize_gt(views_all)  # GT → local normalized

# ② 每個 chunk forward 後:
align_pred_to_gt(preds, chunk_gt)
#   pred['world_points'] → scale-aligned to normalized GT
#   pred['depth']        → scale-aligned

# ③ PLY export 使用 aligned pred['world_points']
#   → 幾何品質可視化正確
```

### 4.5 Inference (SLAM) 時的注意事項

推理時沒有 GT，memory query 使用 `res_geo['world_points']`（VGGT local frame）。
- 同一 sequence 內 VGGT 會輸出一致的 local frame（因為 frame/global attention 共享資訊）
- `voxel_size` 的有效值取決於 predicted scale — 若 predicted 場景尺度與 metric 差很多，可能需要調整 `voxel_size`
- 建議在 SLAM pipeline 中加入 scale estimation（例如從 known object size 或 IMU）來校正 voxel_size

---

## 5. 梯度流分析

### BPTT 梯度路徑 (跨 chunk)

```
   chunk c                          chunk c-1
   ──────                           ─────────
   cross_view_loss                  
     ↓                              
   sem/ins_feat_patch               
     ↓                              
   frozen DPT heads (grad enabled)  
     ↓                              
   X_fuse_{early,late}              
     ↓                              
   MemoryFusionModule               
   (γ_cross, γ_self, W_q/k/v)      
     ↓                              
   X_mem (from voxel_map.query)     
     ↓                              
   DifferentiableVoxelMap._store    
     ↓  EMA: (1-α)·old + α·new     
                    ↓               
              voxel_new (chunk c-1)  
                    ↓               
              MemoryTokenizer        
              (chunk c-1 的 MLP)     
                    ↓               
              X_fuse (chunk c-1)     
                    ↓               
              MemoryFusionModule     
              (chunk c-1)            
                    ↓               
              X_mem (chunk c-2)...   
```

**關鍵**: 整個 sequence 只做 **一次 backward**，梯度從最後一個 chunk 的 loss 一路回傳到第一個 chunk 的 MemoryFusionModule。

---

## 6. 訓練策略 (`train_stage2v4.py`)

### Chunked Unrolled BPTT 流程
```python
voxel_map   = model.make_voxel_map()      # 每個 sequence 清空
feat_buffer = model.make_feat_buffer()

total_loss = 0
for c in range(n_chunks):
    preds, M_r, W_c, pts = model.forward_chunk(chunk, voxel_map)
    
    # Per-chunk losses
    losses = criterion(preds, chunk)
    chunk_loss = losses['objective']
    
    # Cross-view loss (chunk 1 onwards)
    if c > 0:
        sem_s, ins_s, mask = feat_buffer.query(pts)
        cross = cross_view_loss(preds['sem_feat_patch'], preds['ins_feat_patch'],
                                sem_s, ins_s, mask)
        chunk_loss += cross['cross_objective']
    
    total_loss += chunk_loss
    
    # Memory updates (NO detach for voxel_map, detach for feat_buffer)
    AMB3RStage2V4.update_voxel_map(voxel_map, M_r, W_c, pts)
    feat_buffer.update(pts, sem_p.detach(), ins_p.detach())

(total_loss / n_chunks).backward()   # 單次 backward 穿越所有 chunks
optimizer.step()
```

### Evaluation + PLY Export
- 每 `--eval_freq` 個 epoch 執行一次
- 對 eval set 的每個 sequence:
  1. Chunk-by-chunk forward（填充 voxel_map）
  2. 累積所有 chunk 的 world_points + sem/ins features
  3. 匯出 PLY:
     - `{name}_geo_{i}.ply` — RGB 著色幾何
     - `{name}_sem_pca_{i}.ply` — Semantic PCA 著色
     - `{name}_sem_text_{i}.ply` — ScanNet20 text-match 著色
     - `{name}_sem_text_gt_{i}.ply` — GT CLIP text-match
     - `{name}_ins_pca_{i}.ply` — Instance PCA 著色

---

## 7. 檔案結構

```
.claude/sandbox/
├── amb3r/
│   ├── memory_stage2v4.py    ← 空間記憶結構 (318 行)
│   │   ├── LearnableNullToken
│   │   ├── DifferentiableVoxelMap
│   │   └── CrossChunkFeatureBuffer
│   │
│   ├── model_stage2v4.py     ← NN 模組 + 主模型 (446 行)
│   │   ├── MemoryFusionModule
│   │   ├── MemoryTokenizer
│   │   └── AMB3RStage2V4
│   │
│   └── loss_stage2v4.py      ← Loss 函數 (178 行)
│       ├── compute_cross_view_consistency_loss
│       └── Stage2V4Loss
│
├── train_stage2v4.py         ← 訓練 + 評估 (685 行)
│   ├── train_one_sequence()  — Chunked Unrolled BPTT
│   ├── train_one_epoch()     — Gradient accumulation
│   └── eval_one_epoch()      — Eval + PLY export
│
└── STAGE2V4_DESIGN.md        ← 本文件
```

### 相依關係

```
loss_stage2v4.py
  └── imports: amb3r.loss_stage2.{compute_sem_align_loss, compute_ins_contrastive_loss}

model_stage2v4.py
  └── imports: amb3r.memory_stage2v4.{LearnableNullToken, DifferentiableVoxelMap, ...}

train_stage2v4.py
  └── imports: model_stage2v4.AMB3RStage2V4
  └── imports: loss_stage2v4.{Stage2V4Loss, compute_cross_view_consistency_loss,
                               compute_instance_id_cross_chunk_loss}
             memory_stage2v4.CrossChunkFeatureBuffer
  └── imports: memory_stage2v4.CrossChunkFeatureBuffer
  └── imports: amb3r.model_stage1_wo_lora.AMB3RStage1FullFT
  └── imports: amb3r.tools.semantic_vis_utils.*  (eval PLY export)
```

---

## 8. 使用方式

### 訓練
```bash
torchrun --nproc_per_node=1 .claude/sandbox/train_stage2v4.py \
    --stage1_ckpt outputs/exp_stage1_wo_lora/checkpoint-best.pth \
    --seq_len 16 --chunk_size 4 --stride 2 \
    --batch_size 1 --accum_iter 4 --epochs 30 --lr 3e-4 \
    --w_cross_sem 0.5 --w_cross_ins 2.0 \
    --eval_freq 5 --eval_ply_n 5 \
    --use_checkpoint
```

### 關鍵參數
| 參數 | 預設 | 說明 |
|------|------|------|
| `--seq_len` | 16 | 每個訓練序列的總幀數 |
| `--chunk_size` | 4 | 每個 BPTT chunk 的幀數 |
| `--ema_alpha` | 0.5 | Voxel memory EMA 權重 |
| `--voxel_size` | 0.05 | Voxel 邊長（公尺） |
| `--mem_dim` | 128 | 隱式記憶 token 維度 |
| `--hidden_dim` | 256 | Fusion 注意力隱藏層維度 |
| `--w_cross_ins` | 2.0 | Cross-view instance 一致性權重 (voxel-based) |
| `--w_instid_cross_sem` | 0.5 | Instance-ID-based cross-view semantic 一致性權重 |
| `--w_instid_cross_ins` | 2.0 | Instance-ID-based cross-view instance 一致性權重 |
| `--eval_freq` | 5 | 每 N 個 epoch 做一次 eval + PLY |

---

## 9. FAQ

**Q: 為什麼 instance cross-view 權重要比 semantic 高 4 倍?**

A: Semantic feature 已經有 LSeg alignment loss (直接對齊 frozen CLIP GT)，每個 frame 獨立就能學到不錯的 semantic feature。但 instance feature 只有 contrastive loss 做 within-frame 監督，跨視角一致性完全靠 cross-view loss。所以 instance 需要更高權重來驅動 memory fusion 學到有用的跨幀 instance 對應。

**Q: 為什麼用 first-visit anchoring 而不是 EMA 更新 feature buffer?**

A: EMA 會造成 target drift — 後續 chunk 的 target 會被前一個 chunk（已經受到 memory conditioning 影響）的 feature 污染。First-visit anchoring 保證 target 永遠是「第一次看到某個 3D 位置時的 raw feature」，這是最乾淨的一致性目標。

**Q: 為什麼 MemoryFusionModule 要同時用 cross-attn 和 self-attn?**

A: Cross-attn 負責「從 memory 讀取資訊注入到 VGGT feature」，但注入後各 patch 之間缺乏互動。Self-attn 讓 fused patch 之間可以互相 contextualise（例如：鄰近 patch 可以 smooth 出不一致的 memory 讀取）。兩個 γ 都 zero-init，所以訓練初期不會有影響。

**Q: 為什麼在 multi-scale fusion 中共享 MemoryFusion 的參數?**

A: 兩個 scale (atl[4] 和 atl[23]) 的維度相同 (C_VGGT=2048)，共享參數可以：
1. 減少可訓練參數量（不需要翻倍）
2. 強迫 fusion module 學到 scale-agnostic 的 memory 整合策略
3. Tokenizer 輸入兩者的平均，也鼓勵兩個 scale 的 fused representation 在相似的空間中
