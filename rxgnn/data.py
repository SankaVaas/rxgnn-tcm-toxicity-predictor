"""Graph construction from HERB / DrugBank / SuperCYP / TOXRIC."""
from __future__ import annotations
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch_geometric.data import HeteroData
from .model import RELATION_NAMES


def mol_descriptors(smiles: str) -> np.ndarray:
    """16-dim RDKit descriptor vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(16, dtype=np.float32)
    return np.array([
        Descriptors.MolWt(mol) / 600,
        Descriptors.MolLogP(mol) / 8,
        Descriptors.NumHDonors(mol) / 10,
        Descriptors.NumHAcceptors(mol) / 15,
        Descriptors.TPSA(mol) / 200,
        rdMolDescriptors.CalcNumRings(mol) / 8,
        rdMolDescriptors.CalcNumAromaticRings(mol) / 6,
        rdMolDescriptors.CalcNumRotatableBonds(mol) / 15,
        Descriptors.FractionCSP3(mol),
        Descriptors.BertzCT(mol) / 2000,
        Descriptors.HeavyAtomCount(mol) / 80,
        rdMolDescriptors.CalcNumHeterocycles(mol) / 6,
        float(rdMolDescriptors.CalcNumAmideBonds(mol)) / 3,
        Descriptors.RingCount(mol) / 8,
        Descriptors.MaxPartialCharge(mol),
        Descriptors.MinPartialCharge(mol),
    ], dtype=np.float32)


def build_hetero_graph(compound_descs, edges, device):
    """Build PyG HeteroData from descriptor array and typed edge list."""
    data = HeteroData()
    data["compound"].x = torch.tensor(compound_descs, dtype=torch.float32)
    buckets = {r: {"src": [], "dst": [], "attr": []} for r in range(len(RELATION_NAMES))}
    for src, dst, rel, attr in edges:
        buckets[rel]["src"].append(src)
        buckets[rel]["dst"].append(dst)
        buckets[rel]["attr"].append(attr)
    for rel_id, bkt in buckets.items():
        if not bkt["src"]:
            continue
        rn = RELATION_NAMES[rel_id]
        data["compound", rn, "compound"].edge_index = torch.tensor([bkt["src"], bkt["dst"]], dtype=torch.long)
        data["compound", rn, "compound"].edge_attr  = torch.tensor(bkt["attr"], dtype=torch.float32)
    return data.to(device)