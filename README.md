# ICTC

Implementation of **Bipartite Link Prediction by Intra-class Connection based Triadic Closure**.

ICTC is a bipartite link prediction method that combines Graph Autoencoders (GAE, LGAE) with triadic closure. This repository also includes **BiSPM** (Bipartite Spectral Perturbation Method) and **SRNMF** (Similarity-regularized NMF) for comparison.

**Requirements:** Python 3.x

## Setup

Before running, select a dataset in `ictc/config.py` by uncommenting the desired line (and commenting out others). Available datasets: `gpcr`, `enzyme`, `ionchannel`, `malaria`, `drug`, `sw`, `nanet`, `movie100k`.

## Running the Models

```bash
# ICTC (runs GAE, LGAE, and combined ICTC)
python train.py

# BiSPM (Bipartite Spectral Perturbation Method)
python run_bispm.py

# SRNMF — set similarity in run_srnmf.py first: 'srnmf_cn', 'srnmf_jc', or 'srnmf_cpa'
python run_srnmf.py
```

## Dependencies

`torch`, `networkx`, `scipy`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`, `nimfa`, `sparsesvd`

## Project Structure

```
ICTC/
├── ictc/                    # Main package
│   ├── config.py            # Dataset selection and hyperparameters
│   ├── models/
│   │   ├── gae.py           # GAE, LGAE, VGAE
│   │   ├── bispm.py         # BiSPM
│   │   └── srnmf.py         # SRNMF
│   ├── data/
│   │   ├── loading.py       # Data loaders
│   │   └── preprocessing.py # Graph preprocessing
│   └── evaluation/
│       └── metrics.py       # ROC-AUC, AP, Precision@k
├── train.py                 # Entry point for ICTC
├── run_bispm.py             # Entry point for BiSPM
├── run_srnmf.py             # Entry point for SRNMF
├── data_analysis.py         # Analysis script
└── data/                    # Datasets (bipartite edge lists, etc.)
```
