# Stage-1 Integration: Required Changes to Existing Files

This document lists the minimal changes needed in the existing (read-only) codebase
to integrate the Stage-1 training files from `.claude/sandbox/`.

---

## New files to copy into the project

| Sandbox path | Destination |
|---|---|
| `.claude/sandbox/amb3r/clip_patch_fusion.py` | `amb3r/clip_patch_fusion.py` |
| `.claude/sandbox/amb3r/model_stage1.py`      | `amb3r/model_stage1.py`      |
| `.claude/sandbox/train_stage1.py`            | `train_stage1.py`            |

---

## 1. `amb3r/__init__.py` (if it exists)

No change needed. The new modules are imported directly in `model_stage1.py`.

---

## 2. `amb3r/frontend_semantic.py`

**No change required for Stage-1.**

`FrontEndLoRA` in `model_stage1.py` replicates the interface of `FrontEnd` using
`VGGTwLoRA` as the backbone. All method implementations (encode/decode) are
self-contained in `FrontEndLoRA`.

If you later want to unify the two classes, the minimal change is to add a
`use_lora: bool = False` flag to `FrontEnd.__init__` and branch on it:

```python
# In FrontEnd.__init__
if use_lora:
    from amb3r.vggt_w_lora import VGGTwLoRA
    self.model = VGGTwLoRA(return_depth_feat=metric_scale, use_lora=True,
                           lora_r=lora_r, lora_last_n=lora_last_n, ...)
else:
    self.model = VGGT(return_depth_feat=metric_scale)
```

---

## 3. `amb3r/model_semantic.py` — `AMB3R`

**No change required for Stage-1.**

`AMB3RStage1` is a standalone model. If you want to use a Stage-1 checkpoint as
a starting point for Stage-2 (AMB3R), you will need to transfer weights.

### Recommended weight transfer (Stage-1 → Stage-2)

After Stage-1 training, load the relevant sub-module weights into `AMB3R`:

```python
stage1_ckpt = torch.load('outputs/exp_stage1/checkpoint-best.pth')
stage1_state = stage1_ckpt['model']

# Keys to transfer:
transfer_keys = [
    'front_end.model',       # VGGTwLoRA (includes LoRA weights)
    'clip_patch_fusion',     # new cross-attn module
    'sem_expander',
    'instance_token',
]

# For AMB3R, map 'front_end.model' keys and add clip_patch_fusion:
for key, val in stage1_state.items():
    if any(key.startswith(k) for k in transfer_keys):
        # direct copy if AMB3R has the same sub-module path
        pass  # implement key remapping as needed
```

> **Note**: AMB3R's `FrontEnd` uses plain `VGGT`, not `VGGTwLoRA`.
> To use Stage-1 LoRA weights in Stage-2, `AMB3R` must also use `VGGTwLoRA`.
> See Section 4 below.

---

## 4. `amb3r/model_semantic.py` — enabling LoRA in AMB3R for Stage-2

To carry LoRA weights from Stage-1 into Stage-2, replace the `FrontEnd`
instantiation in `AMB3R.__init__`:

```python
# Current (line ~40):
self.front_end = FrontEnd(metric_scale=metric_scale, ...)

# Replace with FrontEndLoRA (import from model_stage1 or unify in frontend_semantic.py):
from amb3r.model_stage1 import FrontEndLoRA
self.front_end = FrontEndLoRA(
    metric_scale=metric_scale,
    clip_dim=clip_dim,
    sem_dim=sem_dim,
    lora_r=4, lora_last_n=8,
)
```

Then `AMB3R.__init__` should also instantiate `clip_patch_fusion` and
call it in `forward()` after `encode_patch_tokens`, before
`decode_patch_tokens_and_heads` — exactly as in `AMB3RStage1.forward()`.

---

## 5. `amb3r/training_semantic.py` — Stage-2 with Stage-1 init

No structural changes required. To start Stage-2 from a Stage-1 checkpoint:

```bash
torchrun --nproc_per_node=1 train_semantic.py \
    --batch_size 1 --interp_v2 \
    --model "AMB3R(metric_scale=True)" \
    --pretrained outputs/exp_stage1/checkpoint-best.pth \
    --accum_iter 2 --epochs 30 --lr 0.00005
```

The `strict=False` load in `AMB3R` will absorb the Stage-1 weights for
matching keys and ignore backend keys (which are not in the Stage-1 checkpoint).

---

## 6. `amb3r/vggt_w_lora.py`

**No change required.** Used as-is.

---

## 7. `amb3r/loss_semantic.py` — `MultitaskLoss`

**No change required.** The geometry weights are set to 0.0 in `train_stage1.py`,
so the camera / depth / point branches are simply skipped at runtime.

---

## Summary checklist

- [ ] Copy `clip_patch_fusion.py` → `amb3r/`
- [ ] Copy `model_stage1.py`      → `amb3r/`
- [ ] Copy `train_stage1.py`      → project root
- [ ] (Stage-2) Update `AMB3R.__init__` to use `FrontEndLoRA` + `clip_patch_fusion`
- [ ] (Stage-2) Update `AMB3R.forward()` to call `clip_patch_fusion` after `encode_patch_tokens`
