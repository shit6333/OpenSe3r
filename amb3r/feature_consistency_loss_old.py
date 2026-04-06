
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def _to_bthwc_points(points: torch.Tensor) -> torch.Tensor:
    """
    Convert points to shape (B, T, H, W, 3).

    Supported input layouts:
      - (B, T, H, W, 3)
      - (B, T, 3, H, W)
    """
    if points.ndim != 5:
        raise ValueError(f"points must be 5D, got shape={tuple(points.shape)}")

    if points.shape[-1] == 3:
        return points
    if points.shape[2] == 3:
        return points.permute(0, 1, 3, 4, 2).contiguous()

    raise ValueError(
        "Unsupported points layout. Expected (B,T,H,W,3) or (B,T,3,H,W), "
        f"got shape={tuple(points.shape)}"
    )


def _resize_points_and_mask(
    points: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    target_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resize 3D points and valid mask to the feature resolution.

    Args:
        points: (B, T, H0, W0, 3) or (B, T, 3, H0, W0)
        valid_mask: (B, T, H0, W0) or (B, T, 1, H0, W0) or None
        target_hw: (H, W)

    Returns:
        points_rs: (B, T, H, W, 3)
        valid_rs:  (B, T, H, W) bool
    """
    points = _to_bthwc_points(points).float()
    B, T, H0, W0, _ = points.shape
    H, W = target_hw

    points_btchw = points.permute(0, 1, 4, 2, 3).reshape(B * T, 3, H0, W0)
    points_rs = F.interpolate(
        points_btchw, size=(H, W), mode="bilinear", align_corners=False
    )
    points_rs = points_rs.reshape(B, T, 3, H, W).permute(0, 1, 3, 4, 2).contiguous()

    if valid_mask is None:
        valid_rs = torch.ones((B, T, H, W), dtype=torch.bool, device=points.device)
    else:
        if valid_mask.ndim == 5 and valid_mask.shape[2] == 1:
            valid_mask = valid_mask[:, :, 0]
        elif valid_mask.ndim != 4:
            raise ValueError(
                "valid_mask must be (B,T,H,W) or (B,T,1,H,W), "
                f"got shape={tuple(valid_mask.shape)}"
            )

        valid_rs = F.interpolate(
            valid_mask.float().reshape(B * T, 1, valid_mask.shape[-2], valid_mask.shape[-1]),
            size=(H, W),
            mode="nearest",
        )
        valid_rs = valid_rs.reshape(B, T, H, W) > 0.5

    finite_mask = torch.isfinite(points_rs).all(dim=-1)
    valid_rs = valid_rs & finite_mask
    return points_rs, valid_rs


class FeatureConsistencyLoss(nn.Module):
    """
    Generic confidence-weighted multi-view consistency loss.

    The idea is:
      1. Group pixels from different views using quantized 3D points.
      2. Build a confidence-weighted prototype per 3D group.
      3. Pull each feature toward the stop-gradient prototype with cosine loss.

    This can be used for both semantic features and instance features.

    Args:
        voxel_size: quantization size in 3D space. Smaller = stricter grouping.
        min_views: require at least this many unique views in a group.
        min_group_size: require at least this many total samples in a group.
        eps: numerical stability.
        detach_prototype: whether to stop gradient through the prototype.
        normalize_feats: whether to L2-normalize features before cosine loss.
    """

    def __init__(
        self,
        voxel_size: float = 0.05,
        min_views: int = 2,
        min_group_size: int = 2,
        eps: float = 1e-6,
        detach_prototype: bool = True,
        normalize_feats: bool = True,
    ) -> None:
        super().__init__()
        self.voxel_size = float(voxel_size)
        self.min_views = int(min_views)
        self.min_group_size = int(min_group_size)
        self.eps = float(eps)
        self.detach_prototype = bool(detach_prototype)
        self.normalize_feats = bool(normalize_feats)

    @torch.no_grad()
    def _make_group_keys(
        self,
        points: torch.Tensor,   # (N, 3)
        view_ids: torch.Tensor, # (N,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build voxel keys from 3D points.

        Returns:
            inverse: (N,) group index per point
            counts:  (G,) group sizes
        """
        vox = torch.round(points / self.voxel_size).to(torch.int64)  # (N, 3)
        # unique over rows
        _, inverse, counts = torch.unique(
            vox, dim=0, sorted=False, return_inverse=True, return_counts=True
        )
        return inverse, counts

    def forward(
        self,
        feat: torch.Tensor,
        points: torch.Tensor,
        conf: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat:  (B, T, C, H, W)
            points: (B, T, H0, W0, 3) or (B, T, 3, H0, W0)
            conf:  (B, T, 1, H, W) or (B, T, H, W) or None
            valid_mask: (B, T, H0, W0) or (B, T, 1, H0, W0) or None

        Returns:
            {
              "loss": scalar tensor,
              "num_groups": scalar tensor,
              "num_valid_groups": scalar tensor,
              "num_samples": scalar tensor,
            }
        """
        if feat.ndim != 5:
            raise ValueError(f"feat must be (B,T,C,H,W), got shape={tuple(feat.shape)}")

        B, T, C, H, W = feat.shape
        device = feat.device
        dtype = feat.dtype

        feat = feat.float()
        if self.normalize_feats:
            feat = F.normalize(feat, dim=2)

        points_rs, valid_rs = _resize_points_and_mask(points, valid_mask, target_hw=(H, W))

        if conf is None:
            conf_rs = torch.ones((B, T, 1, H, W), device=device, dtype=feat.dtype)
        else:
            if conf.ndim == 4:
                conf = conf.unsqueeze(2)  # (B,T,1,H,W)
            if conf.ndim != 5:
                raise ValueError(
                    "conf must be (B,T,1,H,W) or (B,T,H,W), "
                    f"got shape={tuple(conf.shape)}"
                )
            if conf.shape[-2:] != (H, W):
                conf_rs = F.interpolate(
                    conf.reshape(B * T, conf.shape[2], conf.shape[3], conf.shape[4]).float(),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(B, T, conf.shape[2], H, W)
            else:
                conf_rs = conf.float()

            if conf_rs.shape[2] != 1:
                raise ValueError(
                    f"conf channel dimension must be 1, got shape={tuple(conf_rs.shape)}"
                )

        # flatten
        feat_flat = feat.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)
        pts_flat = points_rs.reshape(B, T * H * W, 3)
        conf_flat = conf_rs.reshape(B, T * H * W).clamp_min(self.eps)
        valid_flat = valid_rs.reshape(B, T * H * W)

        # per-pixel view index, useful to require multi-view groups
        view_ids = (
            torch.arange(T, device=device)
            .view(1, T, 1, 1)
            .expand(B, T, H, W)
            .reshape(B, T * H * W)
        )

        total_loss = feat.new_zeros(())
        total_valid_groups = feat.new_zeros(())
        total_groups = feat.new_zeros(())
        total_samples = feat.new_zeros(())

        for b in range(B):
            vb = valid_flat[b]
            if vb.sum() < self.min_group_size:
                continue

            feat_b = feat_flat[b][vb]      # (N, C)
            pts_b = pts_flat[b][vb]        # (N, 3)
            conf_b = conf_flat[b][vb]      # (N,)
            views_b = view_ids[b][vb]      # (N,)

            inverse, counts = self._make_group_keys(pts_b, views_b)
            num_groups = counts.numel()
            total_groups = total_groups + feat.new_tensor(float(num_groups))

            if num_groups == 0:
                continue

            # count unique views per group
            # unique over (group_id, view_id), then bincount group_id
            pair = torch.stack([inverse, views_b.to(torch.int64)], dim=1)
            unique_pair = torch.unique(pair, dim=0, sorted=False)
            unique_view_counts = torch.bincount(
                unique_pair[:, 0], minlength=num_groups
            )

            valid_group_mask = (counts >= self.min_group_size) & (
                unique_view_counts >= self.min_views
            )

            if valid_group_mask.sum() == 0:
                continue

            total_valid_groups = total_valid_groups + valid_group_mask.sum().to(feat.dtype)

            # aggregate prototype = sum(w f) / sum(w)
            weighted_feat = feat_b * conf_b.unsqueeze(-1)
            proto_num = torch.zeros((num_groups, C), device=device, dtype=feat_b.dtype)
            proto_den = torch.zeros((num_groups, 1), device=device, dtype=feat_b.dtype)

            proto_num.index_add_(0, inverse, weighted_feat)
            proto_den.index_add_(0, inverse, conf_b.unsqueeze(-1))
            proto = proto_num / proto_den.clamp_min(self.eps)
            if self.normalize_feats:
                proto = F.normalize(proto, dim=-1)
            if self.detach_prototype:
                proto = proto.detach()

            valid_member_mask = valid_group_mask[inverse]
            if valid_member_mask.sum() == 0:
                continue

            member_feat = feat_b[valid_member_mask]
            member_proto = proto[inverse[valid_member_mask]]
            member_loss = 1.0 - (member_feat * member_proto).sum(dim=-1)

            total_loss = total_loss + member_loss.sum()
            total_samples = total_samples + member_loss.numel()

        if total_samples.item() == 0:
            loss = feat.new_zeros(())
        else:
            loss = total_loss / total_samples.clamp_min(1.0)

        return {
            "loss": loss.to(dtype=dtype),
            "num_groups": total_groups.to(dtype=dtype),
            "num_valid_groups": total_valid_groups.to(dtype=dtype),
            "num_samples": total_samples.to(dtype=dtype),
        }


def compute_semantic_consistency_loss(
    predictions: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    consistency_loss_fn: nn.Module,
    weight: float = 1.0,
    feat_key: str = "semantic_feat_expanded",
    conf_key: str = "semantic_conf",
    points_key: str = "pts3d",
    valid_mask_key: str = "valid_mask",
) -> Dict[str, torch.Tensor]:
    """
    Wrapper for semantic feature consistency.

    By default, this uses GT 3D points from batch["pts3d"] to build correspondences.
    """
    feat = predictions[feat_key].float()
    conf = predictions.get(conf_key, None)
    points = batch[points_key]
    valid_mask = batch.get(valid_mask_key, None)

    out = consistency_loss_fn(
        feat=feat,
        conf=conf,
        points=points,
        valid_mask=valid_mask,
    )
    return {
        "loss_semantic_consistency": weight * out["loss"],
        "loss_semantic_consistency_det": out["loss"].detach(),
        "sem_cons_num_groups": out["num_groups"].detach(),
        "sem_cons_num_valid_groups": out["num_valid_groups"].detach(),
        "sem_cons_num_samples": out["num_samples"].detach(),
    }


def compute_instance_consistency_loss(
    predictions: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    consistency_loss_fn: nn.Module,
    weight: float = 1.0,
    feat_key: str = "instance_feat",
    conf_key: str = "instance_conf",
    points_key: str = "pts3d",
    valid_mask_key: str = "valid_mask",
) -> Dict[str, torch.Tensor]:
    """
    Wrapper for instance feature consistency.

    By default, this uses GT 3D points from batch["pts3d"] to build correspondences.
    """
    feat = predictions[feat_key].float()
    conf = predictions.get(conf_key, None)
    points = batch[points_key]
    valid_mask = batch.get(valid_mask_key, None)

    out = consistency_loss_fn(
        feat=feat,
        conf=conf,
        points=points,
        valid_mask=valid_mask,
    )
    return {
        "loss_instance_consistency": weight * out["loss"],
        "loss_instance_consistency_det": out["loss"].detach(),
        "ins_cons_num_groups": out["num_groups"].detach(),
        "ins_cons_num_valid_groups": out["num_valid_groups"].detach(),
        "ins_cons_num_samples": out["num_samples"].detach(),
    }
