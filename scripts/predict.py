"""Predict toxicity for a herb pair given two SMILES strings.
Usage: python scripts/predict.py --smiles_a "CCO" --smiles_b "c1ccccc1" \
           --checkpoint checkpoints/best.pt
"""
from __future__ import annotations
import argparse, yaml, torch
from rxgnn import RxGNN
from rxgnn.data import mol_descriptors
from rxgnn.utils import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smiles_a",   required=True)
    p.add_argument("--smiles_b",   required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--threshold",  type=float, default=0.5)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = RxGNN(in_dim=16, hidden=cfg["model"]["hidden_dim"],
                  n_rel=cfg["model"]["num_relations"],
                  n_layers=cfg["model"]["num_rgcn_layers"], dropout=0.0)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    da = torch.tensor(mol_descriptors(args.smiles_a)).unsqueeze(0)
    db = torch.tensor(mol_descriptors(args.smiles_b)).unsqueeze(0)
    with torch.no_grad():
        ha, hb = model.proj(da), model.proj(db)
        prob = torch.sigmoid(model.pair(ha, hb, torch.zeros(1, 5))).item()
    print(f"Score: {prob:.4f}  ->  {'TOXIC' if prob >= args.threshold else 'SAFE'}")


if __name__ == "__main__":
    main()