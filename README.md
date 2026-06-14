# rxgnn-tcm-tox

**Synergy toxicity predictor for Traditional Chinese Medicine herbal pairs**  
via reaction-aware graph neural networks (RxGNN).

## Research gap

Existing TCM GNNs model compatibility (君臣佐使) or binding affinity.  
No published model encodes CYP metabolic transformation pathways as typed  
graph edges to predict *emergent* toxicity when herbs are co-administered.

**Key novelty:** metabolic reaction types (CYP3A4 inhibition, substrate  
competition, shared toxic metabolite formation) are first-class edge types  
in a heterogeneous compound interaction graph.

## Quick start

```bash
git clone https://github.com/yourname/rxgnn-tcm-tox
cd rxgnn-tcm-tox
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/train.py --config configs/default.yaml
```

## Project layout

```
rxgnn/          installable package (model, data, loss, utils)
data/raw/       HERB, DrugBank, SuperCYP, TOXRIC (not tracked by git)
data/processed/ cached PyG graphs (.pt)
notebooks/      Colab T4 notebook
scripts/        train.py, predict.py
configs/        YAML hyperparameter files
tests/          pytest suite
```

## Databases

| Source   | Content                        | Use                        |
|----------|--------------------------------|----------------------------|
| HERB     | TCM herb–molecule links        | Node set construction      |
| SymMap   | Symptom–target mapping         | Biological context         |
| DrugBank | Drug metabolite pathways       | Metabolite node generation |
| SuperCYP | CYP enzyme kinetics (Km, Ki)   | Typed edge labelling       |
| TOXRIC   | Compound–toxicity labels       | Training labels            |

## Edge relation types

| ID | Name |
|----|------|
| 0  | CYP3A4_inhibition |
| 1  | CYP3A4_substrate_competition |
| 2  | CYP2D6_inhibition |
| 3  | shared_toxic_metabolite |
| 4  | transporter_Pgp_competition |

## Citation

```bibtex
@misc{rxgnn2025,
  title = {Reaction-Aware GNN for TCM Synergy Toxicity Prediction},
  year  = {2025},
  note  = {Work in progress}
}
```