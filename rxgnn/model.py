"""RxGNN: reaction-aware heterogeneous GNN for TCM pair toxicity."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

RELATION_NAMES: list[str] = [
    "CYP3A4_inhibition",
    "CYP3A4_substrate_competition",
    "CYP2D6_inhibition",
    "shared_toxic_metabolite",
    "transporter_Pgp_competition",
]


class NodeProjection(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
    def forward(self, x):
        return self.net(x)


class RGCNBlock(nn.Module):
    """R-GCN layer with residual connection and layer norm."""
    def __init__(self, dim: int, n_rel: int, dropout: float) -> None:
        super().__init__()
        self.conv = RGCNConv(dim, dim, n_rel)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        h = self.conv(x, edge_index, edge_type)
        return F.gelu(self.drop(self.norm(h))) + x  # residual


class PairHead(nn.Module):
    """MLP: [h_i || h_j || edge_feat] -> scalar toxicity logit."""
    def __init__(self, hidden: int, edge_feat_dim: int = 5, dropout: float = 0.3) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2 + edge_feat_dim, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
    def forward(self, h_i, h_j, ef):
        return self.mlp(torch.cat([h_i, h_j, ef], dim=-1)).squeeze(-1)


class RxGNN(nn.Module):
    """
    Reaction-aware GNN for TCM synergy toxicity prediction.

    Parameters
    ----------
    in_dim   : input feature dim (16 for RDKit descriptors)
    hidden   : hidden embedding dim
    n_rel    : edge relation types (default 5)
    n_layers : R-GCN depth
    dropout  : dropout rate
    """
    def __init__(self, in_dim=16, hidden=128, n_rel=5, n_layers=3, dropout=0.3):
        super().__init__()
        self.proj  = NodeProjection(in_dim, hidden)
        self.rgcn  = nn.ModuleList([RGCNBlock(hidden, n_rel, dropout) for _ in range(n_layers)])
        self.metab = nn.Linear(hidden, 1)
        self.pair  = PairHead(hidden, dropout=dropout)

    def encode(self, x, edge_index, edge_type):
        h = self.proj(x)
        for layer in self.rgcn:
            h = layer(h, edge_index, edge_type)
        return h

    def forward(self, x, edge_index, edge_type, pair_idx, pair_ef):
        h = self.encode(x, edge_index, edge_type)
        h_i, h_j = h[pair_idx[:, 0]], h[pair_idx[:, 1]]
        return self.pair(h_i, h_j, pair_ef), self.metab(h).squeeze(-1)