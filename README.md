# COUFGOT

Official implementation of:

**COUFGOT: A Coupled Unbalanced Filter Graph Optimal Transport Framework for Single-Cell Multi-omics Alignment and Cross-Modal Translation**

---

## Overview

COUFGOT is a unified optimal transport framework for single-cell multi-omics integration. The framework combines Unbalanced Optimal Transport (UOT), covariance-based structural modeling, and collaborative optimization to support:

* Vertical Alignment
* Diagonal Alignment
* Cross-modal Translation

The method is evaluated on multiple RNA–ATAC and RNA–ADT datasets from public single-cell multimodal benchmarks.

---

## Framework

<p align="center">
<img src="framework.png" width="85%">
</p>

---

## Repository Structure

```text
COUFGOT
│
├── model/
│
├── crossmo_main.py
├── diag_main.py
├── vertical_main_samp.py
├── train_test.py
│
├── data/
│   ├── Supplementary Table.xlsx
│   └── Dataset description
│
├── requirements.txt
└── README.md
```

---

## Installation

Create environment:

```bash
conda create -n coufgot python=3.10

conda activate coufgot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Datasets

The datasets used in this work are publicly available from the single-cell multimodal benchmark resource.

Dataset information and accession numbers are provided in:

```text
data/Supplementary Table.xlsx
```

Due to repository size limitations, processed datasets are provided separately.

### Dataset Directory

```text
data/
├── vertical/
├── diagonal/
└── crossmo/
```

---

## Running COUFGOT

### Vertical Alignment

```bash
python vertical_main_samp.py
```

### Diagonal Alignment

```bash
python diag_main.py
```

### Cross-modal Translation

```bash
python crossmo_main.py
```

---

## Evaluation Metrics

Alignment Tasks

* FOSCTTM
* ARI
* NMI

Cross-modal Translation Tasks

* MSE
* AUROC
* ARI
* NMI

---

## Citation

```bibtex
@inproceedings{COUFGOT2026,
  title={COUFGOT: A Coupled Unbalanced Filter Graph Optimal Transport Framework for Single-Cell Multi-omics Alignment and Cross-Modal Translation},
  author={Li, Weining},
  booktitle={APBC},
  year={2026}
}
```

---

## Contact

Weining Li

Xiamen University
