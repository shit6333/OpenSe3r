import os, sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms


sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))
from vggt.models.vggt import VGGT
from ProxyCLIP.open_clip import create_model, tokenizer as open_clip_tokenizer

# =========================================================
# ScanNet-20 labels
# =========================================================
SCANNET20_LABELS = [
    "wall", "floor", "cabinet", "bed", "chair",
    "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator",
    "shower curtain", "toilet", "sink", "bathtub", "otherfurniture"
]

# 簡單 palette，可自行改
SCANNET20_PALETTE = np.array([
    [174, 199, 232], [152, 223, 138], [31, 119, 180], [255, 187, 120], [188, 189, 34],
    [140, 86, 75], [255, 152, 150], [214, 39, 40], [197, 176, 213], [148, 103, 189],
    [196, 156, 148], [23, 190, 207], [247, 182, 210], [219, 219, 141], [255, 127, 14],
    [158, 218, 229], [44, 160, 44], [112, 128, 144], [227, 119, 194], [82, 84, 163],
], dtype=np.uint8)


# =========================================================
# Utils
# =========================================================
def load_rgb(path, image_h=None, image_w=None):
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    if image_h is not None and image_w is not None:
        img = img.resize((image_w, image_h), Image.BILINEAR)  # PIL 是 (W, H)
    img_np = np.array(img)
    return img, img_np, (orig_h, orig_w)


def overlay_mask(image_rgb, mask, palette, alpha=0.55):
    color_mask = palette[mask]
    overlay = (image_rgb * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    return overlay


def save_concat(images, save_path):
    images = [img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8) for img in images]
    h = max(im.shape[0] for im in images)
    resized = []
    for im in images:
        if im.shape[0] != h:
            scale = h / im.shape[0]
            w = int(im.shape[1] * scale)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)
        resized.append(im)
    canvas = np.concatenate(resized, axis=1)
    cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def make_legend(labels, palette, patch_h=28, width=240):
    rows = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, name in enumerate(labels):
        row = np.ones((patch_h, width, 3), dtype=np.uint8) * 255
        row[:, :patch_h] = palette[i]
        cv2.putText(row, f"{i:02d} {name}", (patch_h + 8, patch_h - 8),
                    font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(row)
    return np.concatenate(rows, axis=0)


# =========================================================
# Text features
# =========================================================
@torch.no_grad()
def build_text_features(proxyclip_model, device, class_names):
    templates = [
        "a photo of a {}.",
        "a photo of the {}.",
        "this is a {}.",
        "there is a {} in the scene.",
        "a picture of a {}."
    ]

    text_feats = []
    for cname in class_names:
        prompts = [t.format(cname) for t in templates]
        tokens = open_clip_tokenizer.tokenize(prompts).to(device)
        feat = proxyclip_model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feat = feat.mean(dim=0)
        feat = feat / feat.norm()
        text_feats.append(feat)
    text_feats = torch.stack(text_feats, dim=0)  # [C, D]
    return text_feats


# =========================================================
# VGGT feature extraction
# =========================================================
@torch.no_grad()
def extract_vggt_features(vggt_model, image_01, source="patch", decoder_layer=-1):
    """
    image_01: [1, 3, H, W], range [0,1]
    return ex_feats: [1, C, Hf, Wf]
    """
    device = image_01.device
    B = 1
    S = 1
    images = image_01.unsqueeze(1)  # [1,1,3,H,W]

    # 直接走 aggregator，這樣你可以同時拿 patch_tokens / aggregated_tokens_list
    agg_res = vggt_model.aggregator(images)
    patch_tokens = agg_res["patch_tokens"]                 # [B*S, P, C] 或 dict 後已取 patch
    aggregated_tokens_list = agg_res["aggregated_tokens_list"]
    patch_start_idx = agg_res["patch_start_idx"]

    patch_size = vggt_model.aggregator.patch_size
    H, W = image_01.shape[-2:]
    Hf, Wf = H // patch_size, W // patch_size

    if source == "patch":
        # patch_tokens: [1, P, C]
        feat = patch_tokens.view(B, S, Hf * Wf, -1)[:, 0]           # [1, HW, C]
        feat = feat.view(B, Hf, Wf, -1).permute(0, 3, 1, 2)         # [1, C, Hf, Wf]
        return feat

    elif source == "decoder":
        # aggregated_tokens_list[layer]: [B, S, P_total, 2C]
        print(f"decoder_layer={decoder_layer}/{len(aggregated_tokens_list)-1}")
        feat = aggregated_tokens_list[decoder_layer][:, 0, patch_start_idx:, :]  # [1, HW, 2C]
        feat = feat.view(B, Hf, Wf, -1).permute(0, 3, 1, 2)                       # [1, 2C, Hf, Wf]
        return feat

    else:
        raise ValueError(f"Unknown source={source}")

@torch.no_grad()
def extract_vggt_dinov2_intermediate_features(vggt_model, image_01):
    """
    image_01: [1, 3, H, W], range [0,1]
    return ex_feats: [1, C, Hf, Wf]
    """
    assert image_01.dim() == 4 and image_01.shape[0] == 1

    dino = vggt_model.aggregator.patch_embed
    patch_size = dino.patch_embed.patch_size  # 通常是 (14, 14)

    # 跟 Aggregator 一樣的 normalize
    mean = vggt_model.aggregator._resnet_mean[:, 0]   # [1,3,1,1]
    std = vggt_model.aggregator._resnet_std[:, 0]     # [1,3,1,1]
    imgs_norm = (image_01 - mean) / std

    # 直接照官方 dinov2 分支取 feature
    ex_feats = dino.get_intermediate_layers(imgs_norm, reshape=True)[0]

    print("dinov2 ex_feats:", ex_feats.shape)
    return ex_feats

# =========================================================
# ProxyCLIP dense matching
# =========================================================
@torch.no_grad()
def proxyclip_match(proxyclip_model, image_clip, ex_feats, text_features,
                    beta=1.2, gamma=3.0,
                    clip_train_size=224, clip_patch_size=16):
    """
    image_clip: [1,3,H,W], already normalized for ProxyCLIP/open_clip image encoder
    ex_feats:   [1,C,Hf,Wf]
    text_features: [num_classes, D]
    """
    B = image_clip.shape[0]
    Hf_orig, Wf_orig = ex_feats.shape[-2:]

    # CLIP train resolution 對應的 token grid
    Hf_clip = clip_train_size // clip_patch_size
    Wf_clip = clip_train_size // clip_patch_size

    # external feats 先縮到 CLIP train-size grid
    ex_feats_clip = F.interpolate(
        ex_feats,
        size=(Hf_clip, Wf_clip),
        mode="bilinear",
        align_corners=False
    )

    # encode_image 輸出低解析 dense features: [B, Hf_clip*Wf_clip, D]
    image_features = proxyclip_model.encode_image(
        image_clip,
        external_feats=ex_feats_clip,
        beta=beta,
        gamma=gamma
    )

    # 先 reshape 成 feature map
    B2, HW_clip, D = image_features.shape
    assert B2 == B
    assert HW_clip == Hf_clip * Wf_clip, f"{HW_clip=} != {Hf_clip * Wf_clip=}"

    image_features_map = image_features.permute(0, 2, 1).reshape(B, D, Hf_clip, Wf_clip)

    # 把 dense image features 放大回原本 external feature 尺度
    image_features_map_up = F.interpolate(
        image_features_map,
        size=(Hf_orig, Wf_orig),
        mode="bilinear",
        align_corners=False
    )

    # 再 flatten 回 [B, HW_orig, D]
    image_features_up = image_features_map_up.flatten(2).permute(0, 2, 1)

    # 這個就是你之後要接自己方法時可直接用的 dense feature
    image_features_up = image_features_up / image_features_up.norm(dim=-1, keepdim=True)

    logits = image_features_up @ text_features.T  # [B, Hf_orig*Wf_orig, C]

    # logits 先回到原本 ex_feats 尺度
    logits = logits.permute(0, 2, 1).reshape(B, text_features.shape[0], Hf_orig, Wf_orig)

    # 最後再放到輸入 image 尺度做 visualization / prediction
    logits_up = F.interpolate(
        logits,
        size=image_clip.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    pred = logits_up.argmax(dim=1)
    return logits_up, pred, image_features_up


# =========================================================
# Feature visualization (optional)
# =========================================================
@torch.no_grad()
def pca_vis(feat_map):
    """
    feat_map: [1,C,H,W]
    return rgb uint8 [H,W,3]
    """
    x = feat_map[0].permute(1, 2, 0).reshape(-1, feat_map.shape[1]).float().cpu().numpy()
    x = x - x.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(x, full_matrices=False)
    y = x @ Vt[:3].T
    y = (y - y.min(0, keepdims=True)) / (y.max(0, keepdims=True) - y.min(0, keepdims=True) + 1e-8)
    y = (y * 255).astype(np.uint8)
    H, W = feat_map.shape[-2:]
    return y.reshape(H, W, 3)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="single RGB image")
    parser.add_argument("--save_dir", type=str, default="./proxyclip_test")

    # VGGT
    parser.add_argument("--vggt_ckpt", type=str, default="/mnt/HDD4/ricky/feedforward/amb3r/checkpoints/VGGT.pt")
    parser.add_argument("--clip_train_size", type=int, default=224)
    parser.add_argument("--clip_patch_size", type=int, default=16)
    parser.add_argument("--input_h", type=int, default=336)
    parser.add_argument("--input_w", type=int, default=518)
    parser.add_argument("--metric_scal", action="store_true")

    # ProxyCLIP / open_clip
    parser.add_argument("--clip_type", type=str, default="openai")
    parser.add_argument("--model_type", type=str, default="ViT-B-16")

    # experiment
    parser.add_argument("--decoder_layer", type=int, default=-1)
    parser.add_argument("--beta", type=float, default=1.2)
    parser.add_argument("--gamma", type=float, default=3.0)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------
    pil_img, img_np, _ = load_rgb(args.image, image_h=args.input_h, image_w=args.input_w)

    to_tensor_01 = transforms.ToTensor()
    image_01 = to_tensor_01(pil_img).unsqueeze(0).to(device)  # [1,3,H,W], [0,1]

    # 給 ProxyCLIP/open_clip 的 image normalize
    # 這裡沿用 CLIP normalization
    clip_norm = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    image_clip = clip_norm(image_01[0]).unsqueeze(0).to(device)

    # -----------------------------------------------------
    # Load VGGT
    # -----------------------------------------------------
    vggt_model = VGGT(return_depth_feat=args.metric_scal)
    if args.vggt_ckpt is not None and os.path.isfile(args.vggt_ckpt):
        ckpt = torch.load(args.vggt_ckpt, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        missing, unexpected = vggt_model.load_state_dict(ckpt, strict=False)
        print("[VGGT] missing keys:", len(missing))
        print("[VGGT] unexpected keys:", len(unexpected))
    vggt_model = vggt_model.to(device).eval()

    dino = vggt_model.aggregator.patch_embed
    print(type(dino))
    print(hasattr(dino, "get_intermediate_layers"))

    # -----------------------------------------------------
    # Load ProxyCLIP open_clip
    # -----------------------------------------------------
    proxyclip_model = create_model(args.model_type, pretrained=args.clip_type, precision="fp16")
    proxyclip_model = proxyclip_model.to(device).eval()

    # -----------------------------------------------------
    # Build text features
    # -----------------------------------------------------
    text_features = build_text_features(proxyclip_model, device, SCANNET20_LABELS)

    # -----------------------------------------------------
    # Extract two feature sources
    # -----------------------------------------------------
    # feat_patch = extract_vggt_features(vggt_model, image_01, source="patch", decoder_layer=args.decoder_layer)
    feat_patch = extract_vggt_dinov2_intermediate_features(vggt_model, image_01)
    print("feat_patch (dino intermediate):", tuple(feat_patch.shape))
    feat_decoder = extract_vggt_features(vggt_model, image_01, source="decoder", decoder_layer=args.decoder_layer)

    print("feat_patch  :", tuple(feat_patch.shape))
    print("feat_decoder:", tuple(feat_decoder.shape))

    # -----------------------------------------------------
    # ProxyCLIP matching
    # -----------------------------------------------------
    logits_patch, pred_patch, dense_feat_patch = proxyclip_match(
        proxyclip_model, image_clip.half(), feat_patch.half(),
        text_features.half(), 
        beta=args.beta, gamma=args.gamma,
        clip_train_size=args.clip_train_size,
        clip_patch_size=args.clip_patch_size
    )

    logits_decoder, pred_decoder, dense_feat_decoder = proxyclip_match(
        proxyclip_model, image_clip.half(), feat_decoder.half(), 
        text_features.half(),
        beta=args.beta, gamma=args.gamma,
        clip_train_size=args.clip_train_size,
        clip_patch_size=args.clip_patch_size
    )

    pred_patch = pred_patch[0].detach().cpu().numpy().astype(np.int64)
    pred_decoder = pred_decoder[0].detach().cpu().numpy().astype(np.int64)

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------
    image_rgb = np.array(pil_img)

    overlay_patch = overlay_mask(image_rgb, pred_patch, SCANNET20_PALETTE, alpha=1.0) # alpha=0.55
    overlay_decoder = overlay_mask(image_rgb, pred_decoder, SCANNET20_PALETTE, alpha=1.0) # alpha=0.55

    pca_patch = pca_vis(feat_patch)
    pca_decoder = pca_vis(feat_decoder)

    legend = make_legend(SCANNET20_LABELS, SCANNET20_PALETTE)

    cv2.imwrite(os.path.join(args.save_dir, "pred_patch.png"),
                cv2.cvtColor((SCANNET20_PALETTE[pred_patch]).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.save_dir, "pred_decoder.png"),
                cv2.cvtColor((SCANNET20_PALETTE[pred_decoder]).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.save_dir, "overlay_patch.png"),
                cv2.cvtColor(overlay_patch, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.save_dir, "overlay_decoder.png"),
                cv2.cvtColor(overlay_decoder, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.save_dir, "pca_patch.png"),
                cv2.cvtColor(pca_patch, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.save_dir, "pca_decoder.png"),
                cv2.cvtColor(pca_decoder, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.save_dir, "legend.png"),
                cv2.cvtColor(legend, cv2.COLOR_RGB2BGR))

    # summary canvas
    save_concat(
        [
            image_rgb,
            pca_patch,
            overlay_patch,
            pca_decoder,
            overlay_decoder,
        ],
        os.path.join(args.save_dir, "summary_compare.png")
    )

    print(f"Saved to: {args.save_dir}")


if __name__ == "__main__":
    main()