# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ICTC (Intra-class Connection based Triadic Closure) is a research implementation of bipartite link prediction methods using graph neural networks and matrix factorization. It implements and compares three approaches: ICTC (GAE+LGAE combination), BiSPM, and SRNMF.

Python 3.0+ required. No build system, test framework, or linter is configured.

## Running the Models

Before running, select a dataset by uncommenting the desired line in `ictc/config.py` (and commenting out others). Available datasets: gpcr, enzyme, ionchannel, malaria, drug, sw, nanet, movie100k.

```bash
# ICTC (runs GAE, LGAE, and combined ICTC)
python train.py

# BiSPM (Bipartite Spectral Perturbation Method)
python run_bispm.py

# SRNMF (set args.similarity in run_srnmf.py first: 'srnmf_cn', 'srnmf_jc', or 'srnmf_cpa')
python run_srnmf.py
```

## Dependencies

torch, networkx, scipy, scikit-learn, numpy, matplotlib, seaborn, nimfa, sparsesvd. No requirements.txt exists.

## Package Structure

```
ICTC/
├── data/                        # Datasets
├── ictc/                        # Main package
│   ├── __init__.py
│   ├── config.py                # Hyperparameters and dataset selection (module-level variables)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gae.py               # GAE, LGAE, VGAE classes + GraphConvSparse, glorot_init, dot_product_decode
│   │   ├── bispm.py             # getBrAndBtriangle(), getBiSPM() functions
│   │   └── srnmf.py             # getXY() function
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loading.py           # Data loaders: load_data, load_data_drug, load_data_citation, etc.
│   │   └── preprocessing.py     # Graph preprocessing, edge splitting, get_data()
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py           # Scoring: get_scores, get_precision, similarity scores (CN, JC, AA, CPA)
├── train.py                     # Entry point for ICTC
├── run_bispm.py                 # Entry point for BiSPM
├── run_srnmf.py                 # Entry point for SRNMF
├── data_analysis.py             # Standalone analysis script
├── CLAUDE.md
└── README.md
```

## Architecture

### Configuration (`ictc/config.py`)
All hyperparameters and dataset selection live here as module-level variables. Other modules import via `from ictc import config as args`. Key settings: `dataset`, `model1`/`model2`, `learning_rate`, `num_epoch`, `hidden1_dim`/`hidden2_dim`, `numexp` (number of experiment repetitions), `num_test` (test set ratio), `device` (CUDA device ID).

### Models (`ictc/models/gae.py`)
Three graph autoencoder variants, all using `GraphConvSparse` layers and `dot_product_decode` (sigmoid of Z @ Z^T):
- **VGAE**: Variational, samples from learned Gaussian (mean + logstd). Adds KL divergence to loss.
- **GAE**: Deterministic two-layer encoder (ReLU then linear).
- **LGAE**: Single linear layer encoder.

### Training Pipeline (`train.py`)
`run()` is the entry point (guarded by `if __name__ == '__main__'`). For each of `numexp` experiments:
1. Loads data and splits edges into train/val/test
2. Trains `model2` (GAE) and `model1` (LGAE) via `learn_train_adj()`
3. Combines predictions: `A_pred = (adj_norm @ A_lgae + transpose) / 2`
4. Evaluates with ROC-AUC, Average Precision, Precision@k
5. Reports mean +/- standard error across all experiments

State is managed through global variables shared between `run()` and `learn_train_adj()`.

### Data Pipeline
- **`ictc/data/loading.py`**: Loaders for bipartite edge lists (`load_data`), drug data (`load_data_drug`), citations (`load_data_citation`), and other formats. Datasets stored in `data/bipartite/`.
- **`ictc/data/preprocessing.py`**: `preprocess_graph()` does symmetric normalization (D^{-1/2} A D^{-1/2}). `mask_bipartite_perturbation_test_edges()` splits edges into train/val/test with negative sampling. `get_data()` orchestrates loading and splitting.

### Evaluation (`ictc/evaluation/metrics.py`)
- `get_scores()`: ROC-AUC and Average Precision
- `get_precision()`: Precision@k on top predictions
- Similarity scores for SRNMF: Common Neighbors, Jaccard, Adamic-Adar, CPA

### Alternative Methods
- **`ictc/models/bispm.py`**: SVD-based matrix completion with spectral perturbation.
- **`ictc/models/srnmf.py`**: NMF with similarity regularization using multiplicative update rules.
