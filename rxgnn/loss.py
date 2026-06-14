"""Multi-task loss: focal + metabolite auxiliary + contrastive."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class FocalLoss(nn.Module):
    """FL(p) = -alpha*(1-p)^gamma * log(p)"""
    def __init__(self, alpha=0.75, gamma=1.5):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce     = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        p_t     = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class PairContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, h_i, h_j, labels):
        dist  = F.pairwise_distance(h_i, h_j)
        toxic = labels.float()
        return (toxic * dist.pow(2) + (1 - toxic) * F.relu(self.margin - dist).pow(2)).mean()


class RxGNNLoss(nn.Module):
    """L = L_focal + lam_m*L_metab + lam_c*L_contrastive"""
    def __init__(self, focal_alpha=0.75, focal_gamma=1.5, lambda_metab=0.2, lambda_contrast=0.1):
        super().__init__()
        self.focal       = FocalLoss(focal_alpha, focal_gamma)
        self.metab_bce   = nn.BCEWithLogitsLoss()
        self.contrastive = PairContrastiveLoss()
        self.lam_m, self.lam_c = lambda_metab, lambda_contrast

    def forward(self, tox_logits, metab_logits, h_i, h_j, tox_labels, metab_labels):
        l_f = self.focal(tox_logits, tox_labels)
        l_m = self.metab_bce(metab_logits, metab_labels.float())
        l_c = self.contrastive(h_i, h_j, tox_labels)
        total = l_f + self.lam_m * l_m + self.lam_c * l_c
        return total, {"focal": l_f.item(), "metab": l_m.item(),
                       "contrastive": l_c.item(), "total": total.item()}