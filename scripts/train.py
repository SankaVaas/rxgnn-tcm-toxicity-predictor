"""Train RxGNN.  Usage: python scripts/train.py --config configs/default.yaml"""
from __future__ import annotations
import argparse, yaml, torch
from rxgnn import RxGNN, RxGNNLoss
from rxgnn.utils import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RxGNN(
        in_dim=16,
        hidden=cfg["model"]["hidden_dim"],
        n_rel=cfg["model"]["num_relations"],
        n_layers=cfg["model"]["num_rgcn_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    print(f"Device: {device} | Params: {sum(p.numel() for p in model.parameters()):,}")
    # TODO: wire rxgnn.data loaders here


if __name__ == "__main__":
    main()