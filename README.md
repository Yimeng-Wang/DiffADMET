# DiffADMET

## About

DiffADMET is a multitask learning framework for molecular absorption, distribution, metabolism, excretion and toxicity prediction. It uses 2D and 3D molecular graphs, task-modality attention and directed transfer between related endpoints to learn from sparse classification and regression observations.

This repository contains the training code and processed graph data for 135 ADMET endpoints. The final prediction models are deployed on the DiffADMET platform (https://idrblab.org/diffadmet/)..

## Code Environment

Dependencies are specified in `environment.yml` for Linux with an NVIDIA GPU. Main versions:

| Package | Version |
| --- | --- |
| Python | 3.8.20 |
| PyTorch | 1.12.1 |
| CUDA toolkit | 11.3.1 |
| DGL (`dgl-cu113`) | 0.9.1.post1 |
| RDKit | 2021.03.1 |
| pandas | 2.0.3 |

## Installation

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate myenv
export DGLBACKEND=pytorch
```

## Data Description

The main dataset contains 179,810 molecular inputs and 471,034 observed endpoint labels. The separate external set contains 290 molecular inputs.

| Task type | Endpoints | Observed labels |
| --- | ---: | ---: |
| Classification | 108 | 381,231 |
| Regression | 27 | 89,803 |

Download all three data ZIPs from the [v1.0.0 release](https://github.com/Yimeng-Wang/DiffADMET/releases/tag/v1.0.0), then extract them in the repository root:

```bash
unzip DiffADMET-reproduction-data-part01.zip
unzip DiffADMET-reproduction-data-part02.zip
unzip DiffADMET-reproduction-data-part03.zip
```

The extracted files are located in `data/reproduction/`.

## Training

Model and optimizer settings are defined in `configs/classification.json` and `configs/regression.json`. Run classification and regression separately from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python -u src/train.py --mode classification --device cuda
CUDA_VISIBLE_DEVICES=0 python -u src/train.py --mode regression --device cuda
```

Each command runs the five predefined folds using the supplied BIN data. To run one existing fold, add `--folds 0`.

## Evaluation

At the end of each fold, the best validation checkpoint is reloaded and evaluated on the training, validation, test and external sets when available. Evaluation runs automatically as part of training.

Results are saved in `outputs/<mode>/fold_<n>/`. Use `--output-dir` to change the destination; existing fold directories are not overwritten.

| File | Content |
| --- | --- |
| `best.pth` | Best validation-epoch checkpoint |
| `normalizer.json` | Normalization parameters used by the fold |
| `run_manifest.json` | Run settings and source/input hashes |
| `history.json` | Training and validation metrics/losses by epoch |
| `<n><split>_evalue_c.csv` | Classification endpoint metrics |
| `<n><split>_evalue_r_denorm.csv` | Regression endpoint metrics on the original label scale |

`<split>` is `train`, `valid`, `test` or `external`. No per-sample prediction/probability/label/mask CSVs are generated. `R2_denorm` is squared Pearson correlation.

Console summaries show overall and fixed-group mean AUC or `R2_denorm`.

Data format: [BIN_DATA_FORMAT.md](docs/BIN_DATA_FORMAT.md). Reproduction notes: [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Third-party notices: [NOTICE.md](NOTICE.md).
