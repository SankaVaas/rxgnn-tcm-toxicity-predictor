"""Utilities: seeding, metrics, checkpoint I/O."""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def compute_metrics(labels, probs):
    return {"auroc": roc_auc_score(labels, probs),
            "ap":    average_precision_score(labels, probs)}


def save_checkpoint(model, path, **meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **meta}, path)


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


CYP_ENZYMES = ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2", "CYP2E1", "CYP2B6"]