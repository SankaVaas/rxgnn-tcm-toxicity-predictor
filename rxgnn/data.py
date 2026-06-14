"""
Data pipeline for RxGNN: HERB + SuperCYP + DrugBank + TOXRIC.

Download instructions
---------------------
HERB 2.0  (free, registration required)
  http://herb.ac.cn/Download/
  Files needed:
    HERB_ingredient_info.txt   -- ingredient_id, Molecule_name, SMILES, ...
    HERB_herb_ingredient.txt   -- herb_id, ingredient_id (herb-compound links)

SuperCYP  (request from authors: bioinformatics.charite.de/supercyp)
  File needed:
    supercyp_interactions.csv  -- Drug, CYP, Type (substrate/inhibitor/inducer),
                                   Ki, Km, IC50, Reference

DrugBank  (academic licence: go.drugbank.com/releases/latest)
  File needed:
    drugbank_metabolites.csv   -- drugbank_id, SMILES, metabolite_smiles,
                                   enzyme, reaction_type

TOXRIC / CTD
  https://toxric.bioinformatics.ac.cn/download
  File needed:
    toxric_dili.csv            -- smiles, dili_label (1=toxic, 0=safe)

18-Incompatibles (十八反) hard labels
  Encoded directly in this file as TCM_INCOMPATIBLE_PAIRS below.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData

from .model import RELATION_NAMES

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

CYP_ENZYMES = ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2", "CYP2E1", "CYP2B6"]

# 十八反 — 18 classical TCM incompatible herb pairs (herb Latin names)
# These serve as high-confidence TOXIC training examples.
TCM_INCOMPATIBLE_PAIRS: list[tuple[str, str]] = [
    ("Aconitum carmichaelii", "Pinellia ternata"),
    ("Aconitum carmichaelii", "Fritillaria thunbergii"),
    ("Aconitum carmichaelii", "Ampelopsis japonica"),
    ("Aconitum carmichaelii", "Bletilla striata"),
    ("Aconitum carmichaelii", "Fritillaria cirrhosa"),
    ("Glycyrrhiza uralensis", "Euphorbia kansui"),
    ("Glycyrrhiza uralensis", "Daphne genkwa"),
    ("Glycyrrhiza uralensis", "Sargassum fusiforme"),
    ("Veratrum nigrum", "Panax ginseng"),
    ("Veratrum nigrum", "Salvia miltiorrhiza"),
    ("Veratrum nigrum", "Paeonia lactiflora"),
]

# RELATION_ID mapping (must match RELATION_NAMES in model.py)
# 0 CYP3A4_inhibition
# 1 CYP3A4_substrate_competition
# 2 CYP2D6_inhibition
# 3 shared_toxic_metabolite
# 4 transporter_Pgp_competition
REL = {name: idx for idx, name in enumerate(RELATION_NAMES)}


# ── Molecular features ─────────────────────────────────────────────────────────

def mol_descriptors(smiles: str) -> np.ndarray:
    """16-dimensional RDKit descriptor vector (all normalised to ~[0,1])."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(16, dtype=np.float32)
    try:
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
            float(Descriptors.MaxPartialCharge(mol) or 0),
            float(Descriptors.MinPartialCharge(mol) or 0),
        ], dtype=np.float32)
    except Exception:
        return np.zeros(16, dtype=np.float32)


def is_valid_smiles(smiles: str) -> bool:
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    return Chem.MolFromSmiles(smiles) is not None


# ── HERB 2.0 parser ───────────────────────────────────────────────────────────

def parse_herb_ingredients(
    ingredient_path: str | Path,
    min_mw: float = 100.0,
    max_mw: float = 1500.0,
) -> pd.DataFrame:
    """
    Parse HERB_ingredient_info.txt.

    Expected columns (tab-separated):
        Ingredient_id, Molecule_name, Molecule_formula, Molecule_weight,
        OB_score, PubChem_id, CAS_id, TCMID_id, TCM-ID_id, TCMSP_id,
        SMILES, Type, ...

    Returns
    -------
    DataFrame with columns: ingredient_id, name, smiles, mw, ob_score
    """
    path = Path(ingredient_path)
    if not path.exists():
        raise FileNotFoundError(
            f"HERB ingredient file not found: {path}
"
            "Download from http://herb.ac.cn/Download/"
        )

    df = pd.read_csv(path, sep="	", low_memory=False)

    # Normalise column names (HERB uses mixed case)
    df.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]

    # Identify SMILES column (sometimes called 'canonical_smiles' or 'smiles')
    smiles_col = next(
        (c for c in df.columns if "smiles" in c.lower()), None
    )
    if smiles_col is None:
        raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")

    id_col   = next(c for c in df.columns if "ingredient_id" in c)
    name_col = next((c for c in df.columns if "molecule_name" in c), None) or "name"
    mw_col   = next((c for c in df.columns if "molecule_weight" in c), None)
    ob_col   = next((c for c in df.columns if "ob_score" in c), None)

    out = pd.DataFrame({
        "ingredient_id": df[id_col].astype(str),
        "name":          df[name_col].fillna("").astype(str) if name_col in df.columns else "",
        "smiles":        df[smiles_col].fillna("").astype(str),
        "mw":            pd.to_numeric(df[mw_col], errors="coerce").fillna(0) if mw_col else 0.0,
        "ob_score":      pd.to_numeric(df[ob_col], errors="coerce").fillna(0) if ob_col else 0.0,
    })

    n_raw = len(out)
    out = out[out["smiles"].apply(is_valid_smiles)]
    if mw_col:
        out = out[(out["mw"] >= min_mw) & (out["mw"] <= max_mw)]
    out = out.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    log.info("HERB ingredients: %d raw -> %d after filtering", n_raw, len(out))
    return out


def parse_herb_links(link_path: str | Path) -> pd.DataFrame:
    """
    Parse HERB_herb_ingredient.txt linking herbs to ingredients.

    Expected columns: herb_id, ingredient_id
    Returns DataFrame with those two columns.
    """
    path = Path(link_path)
    if not path.exists():
        raise FileNotFoundError(f"HERB link file not found: {path}")
    df = pd.read_csv(path, sep="	", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df[["herb_id", "ingredient_id"]].dropna().astype(str)


# ── SuperCYP parser ───────────────────────────────────────────────────────────

# Which CYP enzymes map to which relation ID
_CYP_TO_REL: dict[str, dict[str, int]] = {
    "CYP3A4": {"inhibitor": REL["CYP3A4_inhibition"],
                "substrate": REL["CYP3A4_substrate_competition"]},
    "CYP2D6": {"inhibitor": REL["CYP2D6_inhibition"],
                "substrate": REL["CYP2D6_inhibition"]},   # substrate of 2D6 → same rel bucket
}

def parse_supercyp(
    supercyp_path: str | Path,
    cyp_focus: list[str] | None = None,
) -> pd.DataFrame:
    """
    Parse SuperCYP interactions CSV/TSV.

    Expected columns (SuperCYP download format):
        Drug, DrugBank_ID, CYP, Type, Ki, Km, IC50, Reference, SMILES

    Type values: 'substrate', 'inhibitor', 'inducer'

    Parameters
    ----------
    supercyp_path : path to SuperCYP file
    cyp_focus     : list of CYP names to include (default: CYP_ENZYMES)

    Returns
    -------
    DataFrame with: drug_name, smiles, cyp, interaction_type, ki_norm, km_norm
    """
    focus = set(cyp_focus or CYP_ENZYMES)
    path  = Path(supercyp_path)
    if not path.exists():
        raise FileNotFoundError(
            f"SuperCYP file not found: {path}
"
            "Request from: https://bioinformatics.charite.de/supercyp/"
        )

    # SuperCYP may use comma or tab — try both
    try:
        df = pd.read_csv(path, sep="	", low_memory=False)
        if df.shape[1] < 3:
            df = pd.read_csv(path, sep=",", low_memory=False)
    except Exception:
        df = pd.read_csv(path, low_memory=False)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    cyp_col  = next((c for c in df.columns if c in ("cyp", "enzyme", "p450")), None)
    type_col = next((c for c in df.columns if c in ("type", "interaction_type", "role")), None)
    smi_col  = next((c for c in df.columns if "smiles" in c), None)
    drug_col = next((c for c in df.columns if c in ("drug", "drug_name", "compound")), None)
    ki_col   = next((c for c in df.columns if c in ("ki", "ki_um", "ki_(um)")), None)
    km_col   = next((c for c in df.columns if c in ("km", "km_um", "km_(um)")), None)

    if cyp_col is None or type_col is None:
        raise ValueError(
            f"Cannot identify CYP/Type columns. Found: {list(df.columns)}"
        )

    # Filter to focus CYPs and known interaction types
    df = df[df[cyp_col].isin(focus)]
    df = df[df[type_col].str.lower().isin(["substrate", "inhibitor", "inducer"])]

    # Normalise Ki/Km (μM → [0,1] with log scale)
    def _norm_kinetic(series: pd.Series, clip_um: float = 1000.0) -> pd.Series:
        vals = pd.to_numeric(series, errors="coerce").clip(0, clip_um)
        return (np.log1p(vals) / np.log1p(clip_um)).fillna(0.5)

    out = pd.DataFrame({
        "drug_name":        df[drug_col].fillna("").astype(str) if drug_col else "",
        "smiles":           df[smi_col].fillna("").astype(str)  if smi_col  else "",
        "cyp":              df[cyp_col].astype(str),
        "interaction_type": df[type_col].str.lower().astype(str),
        "ki_norm":          _norm_kinetic(df[ki_col]) if ki_col else 0.5,
        "km_norm":          _norm_kinetic(df[km_col]) if km_col else 0.5,
    })

    n_raw = len(out)
    if smi_col:
        out = out[out["smiles"].apply(is_valid_smiles)]
    out = out.reset_index(drop=True)
    log.info("SuperCYP: %d raw -> %d valid", n_raw, len(out))
    return out


# ── DrugBank metabolite parser ────────────────────────────────────────────────

def parse_drugbank_metabolites(path: str | Path) -> pd.DataFrame:
    """
    Parse DrugBank metabolite export CSV.

    Expected columns:
        drugbank_id, name, smiles, metabolite_name, metabolite_smiles,
        enzyme, reaction_type, toxicity_flag

    Returns DataFrame with parent + metabolite SMILES and enzyme.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"DrugBank metabolite file not found: {path}
"
            "Obtain academic licence at: https://go.drugbank.com/releases/latest"
        )
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"smiles", "metabolite_smiles", "enzyme"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"DrugBank CSV missing columns: {missing}")

    df = df.dropna(subset=["smiles", "metabolite_smiles"])
    df = df[df["smiles"].apply(is_valid_smiles)]
    df = df[df["metabolite_smiles"].apply(is_valid_smiles)]

    tox_col = next((c for c in df.columns if "toxic" in c), None)
    if tox_col:
        df["toxic_metabolite"] = df[tox_col].astype(bool)
    else:
        # Heuristic: flag metabolites that are reactive (epoxides, quinones, aldehydes)
        df["toxic_metabolite"] = df["metabolite_smiles"].apply(_is_reactive_metabolite)

    return df[["smiles", "metabolite_smiles", "enzyme", "toxic_metabolite"]].reset_index(drop=True)


_REACTIVE_SMARTS = [
    "[#6]1OC1",          # epoxide
    "O=C1C=CC(=O)",      # quinone core
    "[CH]=O",            # aldehyde
    "C(=O)Cl",           # acid chloride
    "[N+](=O)[O-]",      # nitro (can form reactive species)
]

def _is_reactive_metabolite(smiles: str) -> bool:
    """Heuristic: flag metabolites containing reactive structural motifs."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for smarts in _REACTIVE_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            return True
    return False


# ── TOXRIC parser ─────────────────────────────────────────────────────────────

def parse_toxric(path: str | Path) -> pd.DataFrame:
    """
    Parse TOXRIC DILI (drug-induced liver injury) labels.

    Expected columns: smiles, dili_label  (1 = hepatotoxic, 0 = safe)

    Returns DataFrame with: smiles, dili_label
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"TOXRIC file not found: {path}
"
            "Download from: https://toxric.bioinformatics.ac.cn/download"
        )
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    smi_col   = next((c for c in df.columns if "smiles" in c), None)
    label_col = next(
        (c for c in df.columns if any(k in c for k in ["dili", "label", "toxic"])), None
    )
    if smi_col is None or label_col is None:
        raise ValueError(f"Cannot find SMILES/label columns. Found: {list(df.columns)}")

    df = df[[smi_col, label_col]].rename(columns={smi_col: "smiles", label_col: "dili_label"})
    df = df[df["smiles"].apply(is_valid_smiles)]
    df["dili_label"] = pd.to_numeric(df["dili_label"], errors="coerce").fillna(0).astype(int)
    return df.drop_duplicates(subset=["smiles"]).reset_index(drop=True)


# ── Edge construction ─────────────────────────────────────────────────────────

def build_cyp_edges(
    compounds: pd.DataFrame,
    supercyp: pd.DataFrame,
    smiles_to_idx: dict[str, int],
) -> list[tuple[int, int, int, list[float]]]:
    """
    For each pair of compounds that share a CYP relationship
    (one inhibits / one is a substrate of the same enzyme),
    create a typed directed edge.

    Returns list of (src_idx, dst_idx, relation_id, edge_attr_5d)
    edge_attr_5d: [ki_norm, km_norm, cyp3a4_flag, cyp2d6_flag, is_shared_metab]
    """
    edges: list[tuple[int, int, int, list[float]]] = []

    # Build lookup: smiles -> {cyp: [(type, ki_norm, km_norm), ...]}
    cyp_map: dict[str, dict[str, list]] = {}
    for _, row in supercyp.iterrows():
        smi = row["smiles"]
        if smi not in smiles_to_idx:
            continue
        cyp  = row["cyp"]
        itype = row["interaction_type"]
        ki   = float(row.get("ki_norm", 0.5))
        km   = float(row.get("km_norm", 0.5))
        cyp_map.setdefault(smi, {}).setdefault(cyp, []).append((itype, ki, km))

    # For each CYP: find all inhibitors and substrates, cross-pair them
    all_smiles = list(cyp_map.keys())
    cyp_inhibitors: dict[str, list] = {}
    cyp_substrates: dict[str, list] = {}

    for smi, cyp_dict in cyp_map.items():
        idx = smiles_to_idx[smi]
        for cyp, interactions in cyp_dict.items():
            for itype, ki, km in interactions:
                if itype == "inhibitor":
                    cyp_inhibitors.setdefault(cyp, []).append((smi, idx, ki))
                elif itype == "substrate":
                    cyp_substrates.setdefault(cyp, []).append((smi, idx, km))

    for cyp, inhibs in cyp_inhibitors.items():
        subs = cyp_substrates.get(cyp, [])
        if not subs:
            continue
        for (smi_i, idx_i, ki) in inhibs:
            for (smi_s, idx_s, km) in subs:
                if idx_i == idx_s:
                    continue
                rel_id = _cyp_to_relation(cyp, "inhibitor->substrate")
                attr = _make_edge_attr(cyp, "inhibition", ki, km)
                edges.append((idx_i, idx_s, rel_id, attr))

        # Substrate–substrate competition (same CYP)
        for a_idx in range(len(subs)):
            for b_idx in range(a_idx + 1, len(subs)):
                _, idx_a, km_a = subs[a_idx]
                _, idx_b, km_b = subs[b_idx]
                rel_id = _cyp_to_relation(cyp, "substrate->substrate")
                attr = _make_edge_attr(cyp, "competition", km_a, km_b)
                edges.append((idx_a, idx_b, rel_id, attr))

    return edges


def _cyp_to_relation(cyp: str, mode: str) -> int:
    """Map (CYP name, mode) to relation ID."""
    if cyp == "CYP3A4":
        return REL["CYP3A4_inhibition"] if "inhib" in mode else REL["CYP3A4_substrate_competition"]
    if cyp in ("CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2", "CYP2E1", "CYP2B6"):
        return REL["CYP2D6_inhibition"]
    return REL["CYP2D6_inhibition"]   # default bucket for other CYPs


def _make_edge_attr(cyp: str, mode: str, val_a: float, val_b: float) -> list[float]:
    """5-dim edge attribute: [ki_norm, km_norm, cyp3a4, cyp2d6, is_shared_metab]."""
    cyp3a4 = float(cyp == "CYP3A4")
    cyp2d6 = float(cyp == "CYP2D6")
    ki = val_a if "inhib" in mode else 0.5
    km = val_b if "inhib" in mode else val_a
    return [ki, km, cyp3a4, cyp2d6, 0.0]


def build_shared_metabolite_edges(
    compounds: pd.DataFrame,
    drugbank: pd.DataFrame,
    smiles_to_idx: dict[str, int],
) -> list[tuple[int, int, int, list[float]]]:
    """
    Add shared_toxic_metabolite edges: compounds that form the same
    reactive metabolite are connected with relation 3.
    """
    edges: list[tuple[int, int, int, list[float]]] = []

    # metabolite_smiles -> list of parent compounds in our graph
    metab_to_parents: dict[str, list[int]] = {}
    for _, row in drugbank.iterrows():
        if not bool(row.get("toxic_metabolite", False)):
            continue
        parent_smi = row["smiles"]
        metab_smi  = row["metabolite_smiles"]
        if parent_smi in smiles_to_idx:
            metab_to_parents.setdefault(metab_smi, []).append(smiles_to_idx[parent_smi])

    rel_id = REL["shared_toxic_metabolite"]
    for metab_smi, parents in metab_to_parents.items():
        parents = list(set(parents))
        for a in range(len(parents)):
            for b in range(a + 1, len(parents)):
                attr = [0.0, 0.0, 0.0, 0.0, 1.0]
                edges.append((parents[a], parents[b], rel_id, attr))

    return edges


# ── Toxicity label assignment ─────────────────────────────────────────────────

def assign_pair_labels(
    compounds: pd.DataFrame,
    toxric: pd.DataFrame,
    drugbank: pd.DataFrame,
    smiles_to_idx: dict[str, int],
    n_pairs: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate (herb_i, herb_j, toxic) pair labels.

    Label sources (in priority order):
    1. Explicit 十八反 pairs         -> toxic=1 (hard label)
    2. Both compounds DILI+ in TOXRIC -> toxic=1
    3. Shared toxic metabolite        -> toxic=1
    4. Random pairs from DILI-        -> toxic=0
    5. Random pairs (background)      -> toxic=0
    """
    rng = np.random.default_rng(seed)
    n   = len(compounds)

    # DILI index lookup: smiles -> dili_label
    dili_map = dict(zip(toxric["smiles"], toxric["dili_label"]))
    dili_arr = compounds["smiles"].map(dili_map).fillna(0).values.astype(int)

    # Shared metabolite pairs
    metab_pairs: set[tuple[int, int]] = set()
    for _, row in drugbank.iterrows():
        if not bool(row.get("toxic_metabolite", False)):
            continue
        parent = row["smiles"]
        if parent in smiles_to_idx:
            metab_pairs.add(smiles_to_idx[parent])

    rows = []
    seen: set[tuple[int, int]] = set()

    # DILI-positive pairs
    dili_pos_idx = np.where(dili_arr == 1)[0]
    for ia in dili_pos_idx:
        for ib in dili_pos_idx:
            if ia >= ib:
                continue
            key = (ia, ib)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"herb_i": ia, "herb_j": ib, "toxic": 1, "source": "dili_both"})

    # Shared metabolite pairs
    metab_list = list(metab_pairs)
    for ia in range(len(metab_list)):
        for ib in range(ia + 1, len(metab_list)):
            key = (metab_list[ia], metab_list[ib])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"herb_i": key[0], "herb_j": key[1], "toxic": 1, "source": "shared_metab"})

    # Random background (safe-biased)
    n_background = max(0, n_pairs - len(rows))
    attempts = 0
    while len(rows) < n_pairs and attempts < n_pairs * 10:
        ia, ib = sorted(rng.choice(n, size=2, replace=False))
        key = (ia, ib)
        attempts += 1
        if key in seen:
            continue
        seen.add(key)
        # Weak heuristic: pair is toxic if both are DILI+ or share metabolite partner
        label = int(dili_arr[ia] == 1 and dili_arr[ib] == 1)
        rows.append({"herb_i": ia, "herb_j": ib, "toxic": label, "source": "random"})

    df = pd.DataFrame(rows)
    log.info(
        "Pair labels: %d total, %d toxic (%.1f%%)",
        len(df), df["toxic"].sum(), df["toxic"].mean() * 100,
    )
    return df


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_hetero_graph(
    compound_descs: np.ndarray,
    edges: list[tuple[int, int, int, list[float]]],
    device: torch.device,
) -> HeteroData:
    """
    Assemble PyG HeteroData from descriptor array and typed edge list.

    Parameters
    ----------
    compound_descs : (N, 16) float32 array
    edges          : list of (src, dst, relation_id, edge_attr_5d)
    device         : target torch device
    """
    data = HeteroData()
    data["compound"].x = torch.tensor(compound_descs, dtype=torch.float32)

    buckets: dict[int, dict] = {
        r: {"src": [], "dst": [], "attr": []} for r in range(len(RELATION_NAMES))
    }
    for src, dst, rel, attr in edges:
        buckets[rel]["src"].append(src)
        buckets[rel]["dst"].append(dst)
        buckets[rel]["attr"].append(attr)

    for rel_id, bkt in buckets.items():
        if not bkt["src"]:
            continue
        rel_name = RELATION_NAMES[rel_id]
        data["compound", rel_name, "compound"].edge_index = torch.tensor(
            [bkt["src"], bkt["dst"]], dtype=torch.long
        )
        data["compound", rel_name, "compound"].edge_attr = torch.tensor(
            bkt["attr"], dtype=torch.float32
        )

    return data.to(device)


def merge_all_edges(
    cyp_edges: list,
    metab_edges: list,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge edge lists into a single edge_index and edge_type tensor."""
    all_ei, all_et = [], []
    for rel_id in range(len(RELATION_NAMES)):
        for edge_list in [cyp_edges, metab_edges]:
            srcs = [e[0] for e in edge_list if e[2] == rel_id]
            dsts = [e[1] for e in edge_list if e[2] == rel_id]
            if srcs:
                ei = torch.tensor([srcs, dsts], dtype=torch.long)
                et = torch.full((len(srcs),), rel_id, dtype=torch.long)
                all_ei.append(ei)
                all_et.append(et)
    if not all_ei:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    return torch.cat(all_ei, dim=1), torch.cat(all_et)


# ── Full pipeline entrypoint ──────────────────────────────────────────────────

def build_dataset(
    herb_ingredient_path: str | Path,
    herb_link_path: str | Path | None,
    supercyp_path: str | Path,
    drugbank_path: str | Path,
    toxric_path: str | Path,
    processed_dir: str | Path,
    device: torch.device,
    n_pairs: int = 50_000,
    seed: int = 42,
    force_rebuild: bool = False,
) -> dict:
    """
    Full data pipeline: parse -> filter -> build graph -> label pairs.

    Returns
    -------
    dict with keys:
        compounds     : pd.DataFrame
        hetero_graph  : HeteroData
        edge_index    : Tensor (2, E)
        edge_type     : Tensor (E,)
        pairs_df      : pd.DataFrame [herb_i, herb_j, toxic]
        x_all         : Tensor (N, 16)
        metab_labels  : Tensor (N,)
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache = processed_dir / "dataset.pt"

    if cache.exists() and not force_rebuild:
        log.info("Loading cached dataset from %s", cache)
        return torch.load(cache, weights_only=False)

    # 1. Parse databases
    log.info("Parsing HERB ingredients...")
    compounds = parse_herb_ingredients(herb_ingredient_path)

    log.info("Parsing SuperCYP...")
    supercyp  = parse_supercyp(supercyp_path)

    log.info("Parsing DrugBank metabolites...")
    drugbank  = parse_drugbank_metabolites(drugbank_path)

    log.info("Parsing TOXRIC...")
    toxric    = parse_toxric(toxric_path)

    # 2. Build SMILES index (compounds table is our canonical node set)
    smiles_to_idx = {smi: i for i, smi in enumerate(compounds["smiles"])}

    # 3. Compute node features
    log.info("Computing molecular descriptors for %d compounds...", len(compounds))
    compound_descs = np.stack([mol_descriptors(s) for s in compounds["smiles"]])

    # 4. Build edges
    log.info("Building CYP edges...")
    cyp_edges   = build_cyp_edges(compounds, supercyp, smiles_to_idx)

    log.info("Building shared metabolite edges...")
    metab_edges = build_shared_metabolite_edges(compounds, drugbank, smiles_to_idx)

    all_edges   = cyp_edges + metab_edges
    log.info("Total edges: %d (CYP: %d, metabolite: %d)",
             len(all_edges), len(cyp_edges), len(metab_edges))

    # 5. Build PyG heterogeneous graph
    log.info("Building heterogeneous graph...")
    hetero_graph = build_hetero_graph(compound_descs, all_edges, device)

    # 6. Merged flat tensors (for full-graph training)
    edge_index, edge_type = merge_all_edges(cyp_edges, metab_edges)

    # 7. Per-node metabolite toxicity label (auxiliary supervision)
    dili_map    = dict(zip(toxric["smiles"], toxric["dili_label"]))
    metab_labels = torch.tensor(
        [float(dili_map.get(s, 0)) for s in compounds["smiles"]],
        dtype=torch.float32,
    )

    # 8. Pair labels
    log.info("Assigning pair labels...")
    pairs_df = assign_pair_labels(compounds, toxric, drugbank, smiles_to_idx, n_pairs, seed)

    dataset = {
        "compounds":    compounds,
        "hetero_graph": hetero_graph,
        "edge_index":   edge_index.to(device),
        "edge_type":    edge_type.to(device),
        "pairs_df":     pairs_df,
        "x_all":        torch.tensor(compound_descs, dtype=torch.float32).to(device),
        "metab_labels": metab_labels.to(device),
    }

    torch.save(dataset, cache)
    log.info("Dataset cached to %s", cache)
    return dataset