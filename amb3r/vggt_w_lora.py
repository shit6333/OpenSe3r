import math
import sys
import os
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))
from vggt.models.vggt import VGGT as _VGGT


class LoRALinear(nn.Linear):
    """
    A true nn.Linear-compatible LoRA layer.

    Important properties:
    - Keeps the original parameter names: `weight` and `bias`
    - Adds only incremental LoRA params: `lora_A`, `lora_B`
    - Can load plain pretrained nn.Linear checkpoints directly
    - Can be enabled / disabled at runtime via `lora_enabled`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if r <= 0:
            raise ValueError(f"LoRA rank r must be > 0, got {r}")

        self.r = int(r)
        self.alpha = int(alpha)
        self.scale = float(alpha) / float(r)
        self.lora_enabled = True

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear.from_linear expects nn.Linear, got {type(base_layer)}")

        new_layer = cls(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            bias=base_layer.bias is not None,
            r=r,
            alpha=alpha,
            dropout=dropout,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )

        with torch.no_grad():
            new_layer.weight.copy_(base_layer.weight)
            if base_layer.bias is not None and new_layer.bias is not None:
                new_layer.bias.copy_(base_layer.bias)

        # Freeze the original linear weights; only LoRA params train by default.
        new_layer.weight.requires_grad_(False)
        if new_layer.bias is not None:
            new_layer.bias.requires_grad_(False)

        return new_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if not self.lora_enabled:
            return out
        lora_hidden = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_hidden, self.lora_B)
        return out + self.scale * lora_out


class VGGTwLoRA(_VGGT):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        return_depth_feat: bool = False,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_last_n: int = 4,
        lora_on_qkv: bool = True,
        lora_on_proj: bool = True,
        lora_on_mlp: bool = False,
        lora_on_frame_blocks: bool = True,
        lora_on_global_blocks: bool = True,
        lora_verbose: bool = False,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            return_depth_feat=return_depth_feat,
        )

        self.use_lora = use_lora
        self.lora_config = {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "last_n": lora_last_n,
            "on_qkv": lora_on_qkv,
            "on_proj": lora_on_proj,
            "on_mlp": lora_on_mlp,
            "on_frame_blocks": lora_on_frame_blocks,
            "on_global_blocks": lora_on_global_blocks,
        }
        self._lora_injected = False
        self._lora_module_names: List[str] = []

        if use_lora:
            self.inject_lora(verbose=lora_verbose)

    def inject_lora(self, verbose: bool = False) -> None:
        if self._lora_injected:
            return

        cfg = self.lora_config
        names = self._build_target_module_names(
            aggregator=self.aggregator,
            last_n=cfg["last_n"],
            on_qkv=cfg["on_qkv"],
            on_proj=cfg["on_proj"],
            on_mlp=cfg["on_mlp"],
            on_frame_blocks=cfg["on_frame_blocks"],
            on_global_blocks=cfg["on_global_blocks"],
        )
        self._replace_named_linears(
            root=self.aggregator,
            target_names=names,
            r=cfg["r"],
            alpha=cfg["alpha"],
            dropout=cfg["dropout"],
            verbose=verbose,
        )
        self._lora_injected = True

    def set_lora_enabled(self, enabled: bool = True) -> None:
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.lora_enabled = enabled

    def enable_lora(self) -> None:
        self.set_lora_enabled(True)

    def disable_lora(self) -> None:
        self.set_lora_enabled(False)

    def get_lora_state_dict(self) -> dict:
        return {k: v for k, v in self.state_dict().items() if "lora_" in k}

    def lora_trainable_keywords(self) -> List[str]:
        return ["lora_"]

    def print_lora_summary(self) -> None:
        print("[VGGTwLoRA] injected modules:")
        for name in self._lora_module_names:
            print("  ", name)

    @staticmethod
    def freeze_non_lora_params(module: nn.Module) -> None:
        for name, param in module.named_parameters():
            param.requires_grad_("lora_" in name)

    def _build_target_module_names(
        self,
        aggregator: nn.Module,
        last_n: int,
        on_qkv: bool,
        on_proj: bool,
        on_mlp: bool,
        on_frame_blocks: bool,
        on_global_blocks: bool,
    ) -> List[str]:
        target_names: List[str] = []

        block_groups: List[Tuple[str, nn.ModuleList]] = []
        if on_frame_blocks:
            block_groups.append(("frame_blocks", aggregator.frame_blocks))
        if on_global_blocks:
            block_groups.append(("global_blocks", aggregator.global_blocks))

        for group_name, blocks in block_groups:
            n_blocks = len(blocks)
            start_idx = max(0, n_blocks - int(last_n))
            for block_idx in range(start_idx, n_blocks):
                block = blocks[block_idx]
                for sub_name, sub_module in block.named_modules():
                    if not isinstance(sub_module, nn.Linear):
                        continue

                    leaf_name = sub_name.split(".")[-1].lower()
                    full_name = f"{group_name}.{block_idx}.{sub_name}"

                    is_qkv = "qkv" in leaf_name
                    is_proj = (leaf_name == "proj") or leaf_name.endswith("_proj") or ("out_proj" in leaf_name)
                    is_mlp = leaf_name in {"fc1", "fc2"}

                    if (on_qkv and is_qkv) or (on_proj and is_proj) or (on_mlp and is_mlp):
                        target_names.append(full_name)

        return target_names

    def _replace_named_linears(
        self,
        root: nn.Module,
        target_names: Sequence[str],
        r: int,
        alpha: int,
        dropout: float,
        verbose: bool = False,
    ) -> None:
        module_dict = dict(root.named_modules())
        for full_name in target_names:
            if full_name not in module_dict:
                if verbose:
                    print(f"[LoRA] skip (not found): {full_name}")
                continue

            old_module = module_dict[full_name]
            if not isinstance(old_module, nn.Linear):
                if verbose:
                    print(f"[LoRA] skip (not nn.Linear): {full_name} -> {type(old_module)}")
                continue

            parent, child_name = self._get_parent_module(root, full_name)
            new_module = LoRALinear.from_linear(old_module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, child_name, new_module)
            self._lora_module_names.append(full_name)
            if verbose:
                print(f"[LoRA] injected: {full_name}")

    @staticmethod
    def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
        parts = module_name.split(".")
        parent = root
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        return parent, parts[-1]


VGGT = VGGTwLoRA

__all__ = ["LoRALinear", "VGGTwLoRA", "VGGT"]
