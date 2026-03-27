import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    SpatialSplat-style contrastive losses using SAM instance-id mask.

    Inputs:
      - feat:   [B, C, H, W]  (semantic C=512 or instance C=16, both OK)
      - inst_id:[B, H, W] long, values 0..N, where 0 = ignore/background, 1..N = instance id
    Assumption:
      - inst_id has already been resized to match feat's H,W outside this class.

    Usage:
      loss_fn = ContrastiveLoss(inter_margin=0.2, tau=0.1)
      intra = loss_fn.intra_loss(feat, inst_id)
      inter = loss_fn.inter_loss(feat, inst_id)
      total = intra + inter
    """

    def __init__(
        self,
        tau: float = 0.1,            # (optional) for NCE-style inter loss variant
        inter_margin: float = 0.2,   # for hinge-margin inter loss
        min_pixels_per_inst: int = 2,
        normalize_feats: bool = True,
        inter_mode: str = "hinge",   # "hinge" or "nce"
        eps: float = 1e-8,
    ):
        super().__init__()
        self.tau = float(tau)
        self.inter_margin = float(inter_margin)
        self.min_pixels_per_inst = int(min_pixels_per_inst)
        self.normalize_feats = bool(normalize_feats)
        self.inter_mode = str(inter_mode)
        self.eps = float(eps)

        if self.inter_mode not in ("hinge", "nce"):
            raise ValueError("inter_mode must be 'hinge' or 'nce'")

    def _prep(self, feat: torch.Tensor, inst_id: torch.Tensor):
        assert feat.dim() == 4, "feat must be [B,C,H,W]"
        assert inst_id.dim() == 3, "inst_id must be [B,H,W]"
        B, C, H, W = feat.shape
        assert inst_id.shape == (B, H, W), f"inst_id {inst_id.shape} must match feat spatial {(H,W)}"

        if self.normalize_feats:
            feat = F.normalize(feat, dim=1)

        return feat, inst_id.long()

    def intra_loss(self, feat: torch.Tensor, inst_id: torch.Tensor) -> torch.Tensor:
        """
        Intra-instance consistency (O(n)):
          - shuffle features within each instance id
          - pull each pixel feature toward another pixel feature in same instance via cosine
        Loss = mean(1 - cosine) over valid pixels (inst_id > 0)
        """
        feat, inst_id = self._prep(feat, inst_id)
        B, C, H, W = feat.shape

        losses = []
        for b in range(B):
            f = feat[b].view(C, -1).transpose(0, 1)   # [HW, C]
            m = inst_id[b].view(-1)                   # [HW]

            valid = m > 0
            if valid.sum() < 2:
                losses.append(f.new_tensor(0.0))
                continue

            ids = torch.unique(m[valid])
            f_star = f.clone()

            # shuffle within each instance
            for iid in ids.tolist():
                idx = (m == iid).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() < max(self.min_pixels_per_inst, 2):
                    continue
                perm = idx[torch.randperm(idx.numel(), device=idx.device)]
                f_star[idx] = f[perm]

            # cosine similarity (features already normalized if normalize_feats=True)
            sim = (f[valid] * f_star[valid]).sum(dim=1)
            losses.append((1.0 - sim).mean())

        return torch.stack(losses).mean()

    def inter_loss(self, feat: torch.Tensor, inst_id: torch.Tensor) -> torch.Tensor:
        """
        Inter-instance separation using instance prototypes.
        Two modes:

        (1) inter_mode="hinge" (simple, stable):
            - proto_k = mean(feat pixels in instance k)
            - penalize off-diagonal similarities above margin:
              inter = mean(ReLU(sim(proto_i, proto_j) - margin)) for i!=j

        (2) inter_mode="nce" (stronger contrast):
            - random split each instance pixels into two halves => proto_a_k, proto_b_k
            - InfoNCE: proto_a_k should match proto_b_k, negatives are other proto_b_j
        """
        feat, inst_id = self._prep(feat, inst_id)
        B, C, H, W = feat.shape

        losses = []
        for b in range(B):
            f = feat[b].view(C, -1).transpose(0, 1)  # [HW, C]
            m = inst_id[b].view(-1)                  # [HW]

            valid = m > 0
            if valid.sum() == 0:
                losses.append(f.new_tensor(0.0))
                continue

            ids = torch.unique(m[valid])

            if self.inter_mode == "hinge":
                protos = []
                for iid in ids.tolist():
                    idx = (m == iid).nonzero(as_tuple=False).squeeze(1)
                    if idx.numel() < self.min_pixels_per_inst:
                        continue
                    proto = f[idx].mean(dim=0)
                    if self.normalize_feats:
                        proto = F.normalize(proto, dim=0)
                    protos.append(proto)

                if len(protos) < 2:
                    losses.append(f.new_tensor(0.0))
                    continue

                P = torch.stack(protos, dim=0)            # [M, C]
                if not self.normalize_feats:
                    P = F.normalize(P, dim=1)

                sim = P @ P.t()                           # [M, M]
                eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
                offdiag = sim[~eye]                       # [M*(M-1)]
                inter = F.relu(offdiag - self.inter_margin).mean()
                losses.append(inter)

            else:  # "nce"
                proto_a = []
                proto_b = []
                for iid in ids.tolist():
                    idx = (m == iid).nonzero(as_tuple=False).squeeze(1)
                    if idx.numel() < max(self.min_pixels_per_inst, 2):
                        continue

                    perm = idx[torch.randperm(idx.numel(), device=idx.device)]
                    half = perm.numel() // 2
                    if half == 0:
                        continue
                    idx_a = perm[:half]
                    idx_b = perm[half:] if (perm.numel() - half) > 0 else perm[:half]

                    pa = f[idx_a].mean(dim=0)
                    pb = f[idx_b].mean(dim=0)

                    proto_a.append(pa)
                    proto_b.append(pb)

                if len(proto_a) < 2:
                    losses.append(f.new_tensor(0.0))
                    continue

                A = torch.stack(proto_a, dim=0)  # [M, C]
                Bp = torch.stack(proto_b, dim=0) # [M, C]
                A = F.normalize(A, dim=1)
                Bp = F.normalize(Bp, dim=1)

                logits = (A @ Bp.t()) / self.tau  # [M, M]
                target = torch.arange(logits.size(0), device=logits.device)
                losses.append(F.cross_entropy(logits, target))

        return torch.stack(losses).mean()
