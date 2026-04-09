"""
Stage-2 Memory-Conditioned DPT Heads
======================================
MemoryConditionedDPTHead is the same architecture as PatchConditionedDPTHead
(VGGT multi-scale DPT decoder with patch-level conditioning) but the
conditioning signal is:

    cat[ memory_feat, stage1_feat, mem_mask ]

instead of LSeg/CLIP features.

- memory_feat  : (B*T, C_mem, Hp, Wp)  — voxel-map query result (zeros if unseen)
- stage1_feat  : (B*T, C_feat, Hp, Wp) — Stage-1 output (sem or ins) for this chunk
- mem_mask     : (B*T, 1,      Hp, Wp) — soft confidence in [0, 1]

The MV-Transformer patch embeddings (aggregated_tokens_list) are fed unchanged
from Stage 1 (detached), providing the geometric / appearance backbone.

When no memory exists (first chunk), mem_feat = zeros and mem_mask = zeros.
The head automatically degrades to using only stage1_feat as conditioning,
which still improves over Stage-1 output because the DPT head is re-run
with a stronger decoder.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from amb3r.dpt_head_patch_cond import PatchConditionedDPTHead


# ---------------------------------------------------------------------------
# MemoryConditionedDPTHead
# ---------------------------------------------------------------------------

class MemoryConditionedDPTHead(PatchConditionedDPTHead):
    """
    DPT head conditioned on memory + Stage-1 output features.

    Inherits everything from PatchConditionedDPTHead; the only structural
    difference is the conditioning input dimension:

        semantic_dim = mem_feat_dim + stage1_feat_dim + 1   (the +1 is mem_mask)

    All other hyper-parameters (depth, channels, patch_size, etc.) are kept
    identical to the Stage-1 semantic / instance heads.

    Conditioning is always passed as a pre-concatenated 2D map at patch
    resolution (Hp × Wp), so the parent's adaptive_avg_pool is a no-op.

    Parameters
    ----------
    mem_feat_dim    : channels in the voxel-memory query result
    stage1_feat_dim : channels of the Stage-1 feat being refined
    (all other kwargs forwarded to PatchConditionedDPTHead / VGGTDPTHead)
    """

    def __init__(
        self,
        mem_feat_dim: int,
        stage1_feat_dim: int,
        **kwargs,
    ):
        # semantic_dim = concat dim of [mem_feat, stage1_feat, mem_mask]
        semantic_dim = mem_feat_dim + stage1_feat_dim + 1
        super().__init__(semantic_dim=semantic_dim, **kwargs)

    # forward() is unchanged — PatchConditionedDPTHead.forward() handles
    # any 4-D / 5-D semantic_cond input and projects it via semantic_proj.


# ---------------------------------------------------------------------------
# factory helpers used by model_stage2.py
# ---------------------------------------------------------------------------

def build_semantic_memory_head(
    sem_dim: int = 512,
    dim_in: int = 2 * 1024,
    semantic_proj_dim: int = 128,
) -> MemoryConditionedDPTHead:
    """
    Semantic memory head.
    Conditioning: cat[mem_sem(sem_dim), sem_s1(sem_dim), mem_mask(1)]
    """
    return MemoryConditionedDPTHead(
        mem_feat_dim=sem_dim,
        stage1_feat_dim=sem_dim,
        output_channels=sem_dim,
        semantic_proj_dim=semantic_proj_dim,
        dim_in=dim_in,
        feature_only=True,
        input_identity=True,
        patch_size=14,
    )


def build_instance_memory_head(
    ins_dim: int = 16,
    sem_dim: int = 512,          # semantic context for cross-task reasoning
    dim_in: int = 2 * 1024,
    semantic_proj_dim: int = 128,
) -> MemoryConditionedDPTHead:
    """
    Instance memory head.
    Conditioning: cat[mem_ins(ins_dim), ins_s1(ins_dim), mem_mask(1), sem_s1(sem_dim)]

    sem_s1 is appended so the instance head can reason about semantic context
    (which category this patch belongs to) when assigning instance embeddings.
    Total conditioning channels = ins_dim + ins_dim + 1 + sem_dim.
    """
    mem_feat_dim_total = ins_dim + sem_dim   # memory ins + semantic context
    stage1_feat_dim    = ins_dim             # ins_s1
    return MemoryConditionedDPTHead(
        mem_feat_dim=mem_feat_dim_total,
        stage1_feat_dim=stage1_feat_dim,
        output_channels=ins_dim,
        semantic_proj_dim=semantic_proj_dim,
        dim_in=dim_in,
        feature_only=True,
        input_identity=True,
        patch_size=14,
    )
