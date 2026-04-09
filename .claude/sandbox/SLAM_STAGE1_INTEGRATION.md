# Semantic SLAM — Stage 1 Model Integration Guide

## Overview

要讓 `slam_semantic` 能執行 Stage 1 model (`AMB3RStage1`)，需要：

1. **新增** `run_amb3r_sem_vo` 到 `AMB3RStage1`（主要改動）
2. **`pipeline.py` 的屬性引用**：大部分已相容，僅一個需要確認
3. **`slam_semantic/run.py`**：更換 import 和 model 載入函數

---

## 1. `amb3r/model_stage1.py` — 新增 `run_amb3r_sem_vo`

**位置**：在 `AMB3RStage1` class 中新增一個方法（`forward()` 之後）

Stage 1 沒有 backend，因此這個函數比 `model_semantic.py` 的版本簡單很多：
- 沒有 frontend/backend blending 邏輯
- 沒有多結果 keyframe consistency 比較
- 直接呼叫 `self.forward(frames)[0]` 即可（內部已做 CLIP fusion + semantic/instance heads）

```python
@torch.inference_mode()
def run_amb3r_sem_vo(self, frames, cfg, keyframe_memory):
    """
    Stage-1 inference interface for semantic SLAM.
    No backend → single forward pass; no blending.
    """
    predictions = self.forward(frames)[0]
    return predictions
```

> **注意**：`cfg` 和 `keyframe_memory` 參數保留（與 `AMB3R.run_amb3r_sem_vo` 簽名相同），
> 但 Stage 1 不需要用到它們。這樣 `pipeline.py` 可以不改就直接呼叫。

---

## 2. `slam_semantic/pipeline.py` — 屬性引用確認

`pipeline.py` 透過 `self.model.*` 存取以下屬性，逐一確認 `AMB3RStage1` 的相容性：

| `pipeline.py` 引用 | `AMB3R` 路徑 | `AMB3RStage1` 路徑 | 狀態 |
|---|---|---|---|
| `self.model.run_amb3r_sem_vo` (line 28) | `AMB3R.run_amb3r_sem_vo` | **不存在** → 第 1 節新增後解決 | ⚠️ 需新增 |
| `self.model.lseg` (line 71) | `AMB3R.lseg` | `AMB3RStage1.lseg` ✓ | ✅ |
| `self.model.front_end.model.semantic_head` (line 72) | `FrontEnd.model.semantic_head` | `FrontEndLoRA.model.semantic_head`（在 `__init__` 設定）✓ | ✅ |
| `self.model.front_end.model.instance_head` (line 75) | 同上 | 同上 ✓ | ✅ |
| `self.model.clip_dim` (line 82) | `AMB3R.clip_dim` | `AMB3RStage1.clip_dim` ✓ | ✅ |
| `self.model.ins_dim` (line 83) | `AMB3R.ins_dim` | `AMB3RStage1.ins_dim` ✓ | ✅ |

**結論**：`pipeline.py` 本身不需要改，只要 `run_amb3r_sem_vo` 存在即可。

---

## 3. `slam_semantic/run.py` — 更換 model 載入

**位置**：檔案開頭 import 區段 + `load_semantic_model` 函數

### 3a. 更換 import（約 line 14）

```python
# 原本
from amb3r.model_semantic import AMB3R

# 改為
from amb3r.model_stage1 import AMB3RStage1
```

### 3b. 更換 `load_semantic_model` 函數（約 line 69-78）

```python
# 原本
def load_semantic_model(ckpt_path: str) -> AMB3R:
    model = AMB3R(metric_scale=True, clip_dim=512, sem_dim=512, ins_dim=16)
    if os.path.isfile(ckpt_path):
        model.load_weights(ckpt_path, data_type='bf16', strict=False)
        ...
    model.cuda()
    model.eval()
    return model

# 改為
def load_semantic_model(ckpt_path: str) -> AMB3RStage1:
    model = AMB3RStage1(metric_scale=True, clip_dim=512, sem_dim=512, ins_dim=16)
    if os.path.isfile(ckpt_path):
        model.load_weights(ckpt_path, data_type='bf16', strict=False)
        ...
    model.cuda()
    model.eval()
    return model
```

`AMB3RStage1.load_weights` 已存在（`model_stage1.py` line 471），簽名相同。

### 3c. `model.lseg.clip_pretrained`（line 214）

`run.py` 在 build text embeddings 時呼叫 `model.lseg.clip_pretrained`。
`AMB3RStage1` 有 `self.lseg`，而 `LSegFeatureExtractor.clip_pretrained` 屬性存在，✅ 不需要改。

---

## 架構差異補充說明

| 項目 | `AMB3R`（Stage 2） | `AMB3RStage1`（Stage 1） |
|---|---|---|
| Backend | 有（PointTransformerV3） | 無 |
| CLIP 注入方式 | 透過 `get_voxel_feat` → backend conditioning | `CLIPPatchFusion`（cross-attn 注入 patch tokens）|
| `run_amb3r_sem_vo` 複雜度 | frontend → backend → blend → keyframe select | 單次 `forward()` |
| semantic/instance heads | `has_backend=True` 時在 `decode_heads` 內觸發 | 同，但由 `forward()` 內部以 `has_backend=True` 觸發 |
| CLIP feature 來源 | 在 `run_amb3r_sem_vo` 外部抽取、傳入 backend | 在 `AMB3RStage1.forward()` 內部 `_extract_lseg` 處理 |

---

## 修改清單（Summary）

| 檔案 | 動作 | 說明 |
|---|---|---|
| `amb3r/model_stage1.py` | **新增方法** `AMB3RStage1.run_amb3r_sem_vo` | 單次 forward，保留 cfg/keyframe_memory 簽名 |
| `slam_semantic/run.py` | **修改 import** + **修改 `load_semantic_model`** | 改用 `AMB3RStage1` 取代 `AMB3R` |
| `slam_semantic/pipeline.py` | **不需要改** | 所有屬性引用對 Stage 1 均相容 |

