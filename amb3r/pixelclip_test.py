"""
inference.py
============
Run PixelCLIP open-vocabulary segmentation on a single image and save result.

Usage:
    python inference.py \
        --image      photo.jpg \
        --checkpoint pixelclip.pth \
        --classes    sky tree road building person car \
        --output     seg_result.png \
        --device     cuda
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (Pascal VOC-style, auto-extended if > 20 classes)
# ─────────────────────────────────────────────────────────────────────────────

_BASE_PALETTE = np.array([
    [128,   0,   0], [  0, 128,   0], [128, 128,   0], [  0,   0, 128],
    [128,   0, 128], [  0, 128, 128], [128, 128, 128], [ 64,   0,   0],
    [192,   0,   0], [ 64, 128,   0], [192, 128,   0], [ 64,   0, 128],
    [192,   0, 128], [ 64, 128, 128], [192, 128, 128], [  0,  64,   0],
    [128,  64,   0], [  0, 192,   0], [128, 192,   0], [  0,  64, 128],
], dtype=np.uint8)


def get_palette(n: int) -> np.ndarray:
    palette = list(_BASE_PALETTE)
    rng = np.random.default_rng(seed=42)
    while len(palette) < n:
        palette.append(rng.integers(0, 256, size=3, dtype=np.uint8))
    return np.array(palette[:n], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Returns:
        tensor : [1, 3, H, W] float32 in [0, 255]
        arr    : [H, W, 3]    uint8 numpy
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    tensor = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0)
    return tensor, arr


def colorize(seg_map: np.ndarray, palette: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        canvas[seg_map == i] = color
    return canvas


def save_visualization(
    original: np.ndarray,
    seg_map:  np.ndarray,
    class_names: List[str],
    output_path: str,
    blend_alpha: float = 0.55,
) -> None:
    """Save 3-panel figure (original | seg | blended) + raw seg PNG."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    palette   = get_palette(len(class_names))
    color_seg = colorize(seg_map, palette)
    blended   = (original * (1 - blend_alpha) + color_seg * blend_alpha).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)
    for ax, img, title in zip(axes, [original, color_seg, blended],
                               ["Original", "Segmentation", "Blended"]):
        ax.imshow(img)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    patches = [mpatches.Patch(color=palette[i] / 255.0, label=n)
               for i, n in enumerate(class_names)]
    fig.legend(handles=patches, loc="lower center",
               ncol=min(len(class_names), 10), fontsize=10, framealpha=0.8)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    # also save the raw colour segmentation
    raw_path = Path(output_path).with_name(Path(output_path).stem + "_raw_seg.png")
    Image.fromarray(color_seg).save(str(raw_path))

    print(f"Saved overlay  → {output_path}")
    print(f"Saved raw seg  → {raw_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main routine
# ─────────────────────────────────────────────────────────────────────────────

def run(
    image_path:      str,
    checkpoint_path: str,
    class_names:     List[str],
    output_path:     str,
    clip_model_name: str = "ViT-B/16",
    clip_resolution: Tuple[int, int] = (320, 320),
    num_classes: int  = 64,
    prompt_length: int = 4,
    prompt_template: str = "A photo of a {} in the scene",
    device: str = "cuda",
) -> None:

    # ── 1. load custom CLIP from PixelCLIP repo ──────────────────────────────
    # This is the ONLY part that still needs PixelCLIP's third_party CLIP
    # (for the dense=True support in encode_image).
    print("Loading CLIP model...")
    from pixelclip.third_party import clip as pixelclip_clip
    clip_model, _ = pixelclip_clip.load(
        clip_model_name, device="cpu", jit=False,
        prompt_depth=0, prompt_length=0,
    )
    clip_model = clip_model.float()
    tokenizer  = pixelclip_clip.tokenize

    # ── 2. build PixelCLIP inference model ───────────────────────────────────
    from pixelclip_clean import build_model
    print(f"Loading checkpoint: {checkpoint_path}")
    model = build_model(
        checkpoint_path = checkpoint_path,
        clip_model      = clip_model,
        tokenizer       = tokenizer,
        clip_resolution = clip_resolution,
        num_classes     = num_classes,
        prompt_length   = prompt_length,
        device          = device,
    )

    # ── 3. load & preprocess image ───────────────────────────────────────────
    img_tensor, img_np = load_image(image_path)
    img_tensor = img_tensor.to(device)
    H, W = img_np.shape[:2]
    print(f"Image: {image_path}  ({W}×{H})")
    print(f"Classes ({len(class_names)}): {class_names}")

    # ── 4. inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        # Option A — end-to-end in one call
        logits = model(img_tensor, class_names, prompt_template)    # [1, N, H, W]

        # Option B — call separately (useful if you want to reuse feats)
        # img_feat  = model.encode_image(img_tensor, original_hw=(H, W))  # [1,H,W,C]
        # text_feat = model.encode_text(class_names, prompt_template)      # [N,C]
        # logits    = (img_feat @ text_feat.T).permute(0,3,1,2)           # [1,N,H,W]

    seg_map   = logits[0].argmax(dim=0).cpu().numpy()               # [H, W]  int
    max_conf  = F.softmax(logits[0], dim=0).max(dim=0).values.mean().item()
    print(f"Mean max-class confidence: {max_conf:.3f}")

    # ── 5. save result ───────────────────────────────────────────────────────
    save_visualization(img_np, seg_map, class_names, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PixelCLIP open-vocabulary segmentation")
    p.add_argument("--image",       required=True,  help="Input image path.")
    p.add_argument("--checkpoint",  required=True,  help="PixelCLIP .pth checkpoint.")
    p.add_argument("--classes", nargs="+",
                   default=[
                       "wall", "floor", "cabinet", "bed", "chair",
                       "sofa", "table", "door", "window", "bookshelf",
                       "picture", "counter", "desk", "curtain", "refrigerator",
                       "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
                   ],
                   help="Class names for open-vocabulary segmentation. Default: ScanNet-20.")
    p.add_argument("--output",      default="seg_output.png", help="Output path.")
    p.add_argument("--clip-model",  default="ViT-B/16", dest="clip_model",
                   help='CLIP backbone name, e.g. "ViT-B/16", "ViT-L/14".')
    p.add_argument("--clip-resolution", nargs=2, type=int, default=[512, 320],
                   metavar=("H", "W"), dest="clip_resolution",
                   help="CLIP input resolution (must match training config, default 320 320).")
    p.add_argument("--num-classes", type=int, default=64, dest="num_classes",
                   help="NUM_CLASSES from training yaml (default 64).")
    p.add_argument("--prompt-length", type=int, default=4, dest="prompt_length",
                   help="PROMPT_LENGTH from training yaml (default 4).")
    p.add_argument("--prompt-template", default="A photo of a {} in the scene",
                   dest="prompt_template",
                   help='Prompt format string, e.g. "a photo of a {} in the scene".')
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        image_path      = args.image,
        checkpoint_path = args.checkpoint,
        class_names     = args.classes,
        output_path     = args.output,
        clip_model_name = args.clip_model,
        clip_resolution = tuple(args.clip_resolution),
        num_classes     = args.num_classes,
        prompt_length   = args.prompt_length,
        prompt_template = args.prompt_template,
        device          = args.device,
    )