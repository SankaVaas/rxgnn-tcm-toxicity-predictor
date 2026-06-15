"""Train RxGNN on real data.

Usage
-----
# With real data files in place:
python scripts/train.py --config configs/default.yaml

# Dry-run (checks imports and model init only):
python scripts/train.py --config configs/default.yaml --dry_run
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from rxgnn import RxGNN, RxGNNLoss
from rxgnn.data import build_dataset
from rxgnn.utils import compute_metrics, save_checkpoint, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="configs/default.yaml")
    p.add_argument("--dry_run",  action="store_true", help="Verify setup without training")
    p.add_argument("--rebuild",  action="store_true", help="Force graph rebuild from raw files")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    if args.dry_run:
        log.info("Dry run: instantiating model only.")
        model = RxGNN(
            in_dim=16,
            hidden=cfg["model"]["hidden_dim"],
            n_rel=cfg["model"]["num_relations"],
            n_layers=cfg["model"]["num_rgcn_layers"],
            dropout=cfg["model"]["dropout"],
        )
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("Model OK — %d parameters", n)
        return

    # ── Load / build dataset ──────────────────────────────────────────────────
    dataset = build_dataset(
        herb_ingredient_path = cfg["data"]["herb_db_path"],
        herb_link_path       = cfg["data"].get("herb_link_path"),
        supercyp_path        = cfg["data"]["supercyp_path"],
        drugbank_path        = cfg["data"]["drugbank_path"],
        toxric_path          = cfg["data"]["toxric_path"],
        processed_dir        = cfg["data"]["processed_dir"],
        device               = device,
        force_rebuild        = args.rebuild,
    )

    x_all        = dataset["x_all"]
    edge_index   = dataset["edge_index"]
    edge_type    = dataset["edge_type"]
    metab_labels = dataset["metab_labels"]
    pairs_df     = dataset["pairs_df"]

    pair_idx  = torch.tensor(pairs_df[["herb_i", "herb_j"]].values, dtype=torch.long)
    tox_labels = torch.tensor(pairs_df["toxic"].values,             dtype=torch.float32)

    # Edge features for pairs (look up from graph edges, default 0)
    pair_ef = torch.zeros(len(pairs_df), 5)

    # Train / val / test split
    idx = np.arange(len(pairs_df))
    tr_idx, te_idx = train_test_split(
        idx, test_size=cfg["data"]["test_split"], random_state=cfg["training"]["seed"],
        stratify=pairs_df["toxic"].values,
    )
    tr_idx, va_idx = train_test_split(
        tr_idx, test_size=cfg["data"]["val_split"], random_state=cfg["training"]["seed"],
        stratify=pairs_df["toxic"].iloc[tr_idx].values,
    )
    log.info("Split — train: %d  val: %d  test: %d", len(tr_idx), len(va_idx), len(te_idx))

    def get_batch(indices):
        pi = pair_idx[indices].to(device)
        ef = pair_ef[indices].to(device)
        tl = tox_labels[indices].to(device)
        return pi, ef, tl

    # ── Model, optimiser, scheduler ───────────────────────────────────────────
    model = RxGNN(
        in_dim=16,
        hidden=cfg["model"]["hidden_dim"],
        n_rel=cfg["model"]["num_relations"],
        n_layers=cfg["model"]["num_rgcn_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    criterion = RxGNNLoss(
        focal_alpha     = cfg["loss"]["focal_alpha"],
        focal_gamma     = cfg["loss"]["focal_gamma"],
        lambda_metab    = cfg["loss"]["lambda_metab"],
        lambda_contrast = cfg["loss"]["lambda_contrast"],
    )

    optimizer = AdamW(model.parameters(),
                      lr=cfg["training"]["lr"],
                      weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"], eta_min=1e-6)

    EPOCHS     = cfg["training"]["epochs"]
    BATCH      = cfg["training"]["batch_size"]
    ckpt_dir   = Path(cfg["training"]["checkpoint_dir"])
    best_auroc = 0.0

    log.info("Training RxGNN for %d epochs...", EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        perm      = np.random.permutation(tr_idx)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, len(perm), BATCH):
            pi, ef, tl = get_batch(perm[start : start + BATCH])
            optimizer.zero_grad()

            h = model.encode(x_all, edge_index, edge_type)
            metab_logits = model.metab(h).squeeze(-1)
            h_i, h_j     = h[pi[:, 0]], h[pi[:, 1]]
            tox_logits   = model.pair(h_i, h_j, ef)

            loss, parts = criterion(tox_logits, metab_logits, h_i, h_j, tl, metab_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += parts["total"]
            n_batches  += 1

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            h = model.encode(x_all, edge_index, edge_type)
            metab_logits = model.metab(h).squeeze(-1)
            pi_v, ef_v, tl_v = get_batch(va_idx)
            h_iv, h_jv = h[pi_v[:, 0]], h[pi_v[:, 1]]
            tox_v = model.pair(h_iv, h_jv, ef_v)
            probs_v  = torch.sigmoid(tox_v).cpu().numpy()
            labels_v = tl_v.cpu().numpy()

        try:
            metrics = compute_metrics(labels_v, probs_v)
            auroc_v = metrics["auroc"]
        except ValueError:
            auroc_v = 0.0

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "Epoch %3d | loss %.4f | val AUROC %.4f",
                epoch, epoch_loss / max(n_batches, 1), auroc_v,
            )

        if auroc_v > best_auroc:
            best_auroc = auroc_v
            save_checkpoint(model, ckpt_dir / "best.pt",
                            epoch=epoch, val_auroc=auroc_v)

    log.info("Training complete. Best val AUROC: %.4f", best_auroc)
    log.info("Checkpoint saved to %s/best.pt", ckpt_dir)


if __name__ == "__main__":
    main()