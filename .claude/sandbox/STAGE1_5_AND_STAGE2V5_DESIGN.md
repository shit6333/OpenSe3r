# Stage 1.5 + Stage 2 V5 設計

## 一、整體架構

```
Stage 1 (frozen) → VGGT features (atl, sem/ins feat, lseg_feat)
       ↓
Stage 1.5 (train autoencoders, then freeze)
  ├── SemAutoencoder: X_vggt → enc_sem(bottleneck) → dec → frozen sem_head → sem_feat → vs CLIP GT
  └── InsAutoencoder: X_vggt → enc_ins(bottleneck) → dec → frozen ins_head → ins_feat → vs inst_mask
       ↓
Stage 2 (train fusion + optional adapter)
  ├── Memory: 依 mem_mode 選擇存什麼
  ├── Fusion: 分開處理 sem/ins（各自有 early+late fusion module）
  └── Optional: feat_adapter / finetune_head
```

## 二、Stage 2 改動

### 2.1 Separate Fusion (sem/ins 分開)

**現在**: 共用 fusion → 同一個 modified_atl → 給 sem_head 和 ins_head
**改為**: 分開 fusion → 兩個 modified_atl

```python
# Semantic path
X_fuse_sem_early = fusion_sem_early(X_vggt_early, X_mem)
X_fuse_sem_late  = fusion_sem_late(X_vggt_late,  X_mem)
modified_atl_sem[early_idx] = X_fuse_sem_early
modified_atl_sem[last_idx]  = X_fuse_sem_late
sem_feat = sem_head(modified_atl_sem)

# Instance path
X_fuse_ins_early = fusion_ins_early(X_vggt_early, X_mem)
X_fuse_ins_late  = fusion_ins_late(X_vggt_late,  X_mem)
modified_atl_ins[early_idx] = X_fuse_ins_early
modified_atl_ins[last_idx]  = X_fuse_ins_late
ins_feat = ins_head(modified_atl_ins)
```

Trainable modules: 4 MemoryFusionModule (sem_early, sem_late, ins_early, ins_late)

MemoryTokenizer: 只在 mem_mode=0 時需要。
其他 mode 下 memory 直接存 detached features，不需要 tokenizer。

### 2.2 Memory Modes (`--mem_mode`)

| Mode | 存什麼 | mem_dim | 需要 Tokenizer | BPTT through memory |
|------|--------|---------|----------------|---------------------|
| 0 | MemoryTokenizer(X_fuse) | 128 | ✓ | ✓ (但 gradient 弱) |
| 1 | detached sem_feat ⊕ ins_feat | sem+ins (528) | ✗ | ✗ |
| 2 | detached X_vggt_late | 2048 | ✗ | ✗ |
| 3 | frozen enc_sem(X) ⊕ enc_ins(X) | enc_dim*2 | ✗ | ✗ |

Mode 1-3: memory 本身有意義（不是隨機初始化的 noise），fusion 只需學「怎麼讀」。
Memory update 用 first-visit anchor（不用 EMA），detached 存入簡單 voxel store。

Mode 3 的 X_mem = concat(enc_sem_feat, enc_ins_feat)
每個 fusion module 可以 attend 到自己需要的部分。

### 2.3 Feature Adapter (`--use_feat_adapter`)

在 frozen DPT head output 後加 trainable residual adapter：

```python
class FeatureAdapter(nn.Module):
    # Small residual MLP
    def __init__(self, feat_dim, hidden=256):
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, feat_dim))
        nn.init.zeros_(self.mlp[-1].weight)  # zero-init → identity at start
    
    def forward(self, x):
        return x + self.mlp(x)  # residual

sem_feat_raw = frozen_sem_head(modified_atl_sem)
sem_feat = sem_adapter(sem_feat_raw)  # trainable, direct gradient path
```

### 2.4 Finetune Head (`--finetune_head`)

簡單 unfreeze sem_head / ins_head 的 parameters：
```python
if finetune_head:
    for p in sem_head.parameters(): p.requires_grad_(True)
    for p in ins_head.parameters(): p.requires_grad_(True)
```
搭配 lower LR（用 param group 區分）。

## 三、Stage 1.5

### 3.1 模型

```python
class VGGTAutoencoder(nn.Module):
    def __init__(self, input_dim=2048, bottleneck_dim=128):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, bottleneck_dim))
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, input_dim))

class Stage1_5(nn.Module):
    def __init__(self, stage1_model, bottleneck_dim=128):
        self.stage1 = stage1_model  # frozen
        self.sem_ae = VGGTAutoencoder(2048, bottleneck_dim)
        self.ins_ae = VGGTAutoencoder(2048, bottleneck_dim)
```

### 3.2 Training flow

```
For each batch:
    1. Frozen Stage1: images → atl, lseg_feat
    2. X_vggt = atl[last_dpt_idx] patch tokens  [B*T, N, 2048]
    
    3a. Semantic path:
        X_enc_sem = sem_ae.encoder(X_vggt)           # [B*T, N, bottleneck]
        X_dec_sem = sem_ae.decoder(X_enc_sem)         # [B*T, N, 2048]
        → inject X_dec_sem into atl → frozen sem_head → sem_feat
        → loss vs CLIP GT (cosine)
    
    3b. Instance path:
        X_enc_ins = ins_ae.encoder(X_vggt)
        X_dec_ins = ins_ae.decoder(X_enc_ins)
        → inject X_dec_ins into atl → frozen ins_head → ins_feat
        → loss vs instance_mask (contrastive)
    
    4. Total loss = sem_loss + ins_loss
```

### 3.3 Eval

和 Stage 2 類似：PLY export（geometry + sem PCA + ins PCA + text-match）

### 3.4 接到 Stage 2

Train 完後，Stage 2 mode 3 載入 frozen encoder：
```python
# Stage 2 init
self.sem_encoder = stage1_5.sem_ae.encoder  # frozen
self.ins_encoder = stage1_5.ins_ae.encoder  # frozen

# forward_chunk
enc_sem = self.sem_encoder(X_vggt_late.detach())  # [B*T, N, bottleneck]
enc_ins = self.ins_encoder(X_vggt_late.detach())  # [B*T, N, bottleneck]
X_mem_content = torch.cat([enc_sem, enc_ins], dim=-1)  # [B*T, N, 2*bottleneck]
# → store into voxel_map (detached, first-visit)
# → query from voxel_map → feed to fusion modules
```

## 四、檔案結構

```
.claude/sandbox/
├── amb3r/
│   ├── model_stage1_5.py      # Stage 1.5 autoencoders
│   ├── model_stage2v5.py      # Stage 2 V5 (separate fusion, mem modes, adapter)
│   └── memory_stage2v4.py     # 不大改，加 SimpleVoxelStore for mode 1-3
├── train_stage1_5.py          # Stage 1.5 training + eval
└── train_stage2v5.py          # Stage 2 V5 training (基於 v4 修改)
```

## 五、Implementation 順序

1. model_stage2v5.py — separate fusion + mem_modes + adapter + finetune_head
2. model_stage1_5.py — autoencoders
3. train_stage1_5.py — Stage 1.5 training pipeline
4. train_stage2v5.py — 更新 training loop for new modes
5. memory_stage2v4.py — 加 SimpleVoxelStore (detached, first-visit, for mode 1-3)
