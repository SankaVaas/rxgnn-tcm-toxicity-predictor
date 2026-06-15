"""Unit tests for rxgnn/data.py — run without real database files."""
import numpy as np
import pandas as pd
import pytest
import torch
from rxgnn.data import (
    mol_descriptors, is_valid_smiles, build_hetero_graph,
    _is_reactive_metabolite, merge_all_edges,
)
from rxgnn.model import RELATION_NAMES


# ── mol_descriptors ───────────────────────────────────────────────────────────

def test_mol_descriptors_valid():
    smiles = "CCO"   # ethanol
    desc = mol_descriptors(smiles)
    assert desc.shape == (16,)
    assert desc.dtype == np.float32
    assert not np.isnan(desc).any()


def test_mol_descriptors_invalid():
    desc = mol_descriptors("not_a_smiles!!!")
    assert desc.shape == (16,)
    assert (desc == 0).all()


def test_mol_descriptors_complex():
    berberine = "COc1ccc2cc3c(cc2c1OC)[N+](CC3)(C)Cc1ccc(cc1)OC"
    desc = mol_descriptors(berberine)
    assert desc.shape == (16,)
    assert desc[0] > 0   # MW should be > 0


# ── is_valid_smiles ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("smi,expected", [
    ("CCO",         True),
    ("c1ccccc1",    True),
    ("",            False),
    ("XXXX",        False),
    (None,          False),
    ("C(=O)([OH])O", True),
])
def test_is_valid_smiles(smi, expected):
    assert is_valid_smiles(smi) == expected


# ── reactive metabolite heuristic ─────────────────────────────────────────────

def test_reactive_epoxide():
    # ethylene oxide
    assert _is_reactive_metabolite("C1CO1") is True


def test_non_reactive():
    # ethanol
    assert _is_reactive_metabolite("CCO") is False


# ── build_hetero_graph ────────────────────────────────────────────────────────

def make_fake_compounds(n=10):
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CN", "CC#N",
              "c1ccc(O)cc1", "CCN", "CCC", "CCCO", "c1ccncc1"]
    descs  = np.random.rand(n, 16).astype(np.float32)
    return pd.DataFrame({"smiles": smiles[:n]}), descs


def test_build_hetero_graph_empty_edges():
    _, descs = make_fake_compounds(5)
    graph = build_hetero_graph(descs, [], torch.device("cpu"))
    assert graph["compound"].x.shape == (5, 16)


def test_build_hetero_graph_with_edges():
    _, descs = make_fake_compounds(8)
    edges = [
        (0, 1, 0, [0.5, 0.3, 1.0, 0.0, 0.0]),
        (2, 3, 1, [0.0, 0.6, 1.0, 0.0, 0.0]),
        (4, 5, 3, [0.0, 0.0, 0.0, 0.0, 1.0]),
    ]
    graph = build_hetero_graph(descs, edges, torch.device("cpu"))
    assert graph["compound"].x.shape == (8, 16)
    rel_name = RELATION_NAMES[0]
    assert ("compound", rel_name, "compound") in graph.edge_types


def test_merge_all_edges():
    cyp_edges   = [(0, 1, 0, [0.5, 0.3, 1.0, 0.0, 0.0])]
    metab_edges = [(2, 3, 3, [0.0, 0.0, 0.0, 0.0, 1.0])]
    ei, et = merge_all_edges(cyp_edges, metab_edges)
    assert ei.shape[0] == 2
    assert et.shape[0] == ei.shape[1]


# ── parser helpers (file-not-found raises) ────────────────────────────────────

def test_parse_herb_raises_not_found():
    from rxgnn.data import parse_herb_ingredients
    with pytest.raises(FileNotFoundError):
        parse_herb_ingredients("/nonexistent/path.txt")


def test_parse_supercyp_raises_not_found():
    from rxgnn.data import parse_supercyp
    with pytest.raises(FileNotFoundError):
        parse_supercyp("/nonexistent/path.csv")


def test_parse_toxric_raises_not_found():
    from rxgnn.data import parse_toxric
    with pytest.raises(FileNotFoundError):
        parse_toxric("/nonexistent/path.csv")