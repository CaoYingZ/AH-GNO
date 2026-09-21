# AH-GNO

Code accompanying the AH-GNO study on surrogate morphodynamic modelling with TELEMAC-2D/GAIA.

AH-GNO works directly on unstructured meshes. At each prediction point it uses recent histories of velocity, water depth and bed elevation (`U, V, H, Z`) to predict a bundle of bed-elevation increments (`ΔZ`). The model includes an area-weighted graph operator, a learned history selector and a learned write-back horizon for online coupling.

## Repository

```text
AH-GNO/
├── src/AH_GNO/                 model, preprocessing and loss functions
├── scripts/                    dataset building, training, evaluation and coupling
├── configs/                    case definitions used by the examples
├── examples/
│   ├── yen/                    Yen forcing files and split
│   ├── bump/                   Bump scenario table and split
│   └── waiho/                  notes for the field case
├── docs/
│   ├── DATA_PREPARATION.md
│   └── AREA_WEIGHTED_COUPLING.md
├── requirements.txt
├── pyproject.toml
└── CITATION.cff
```

## Requirements

The study used TELEMAC-MASCARET V8P4R0, Python 3.12 and PyTorch 2.5.1 on Ubuntu 22.04 with CUDA 12.4.

Install the Python dependencies and the local package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

TELEMAC-MASCARET is installed separately. Source the TELEMAC environment before using the SELAFIN reader or the TelApy coupling script.

## 1. Build a dataset

`scripts/build_dataset.py` converts paired TELEMAC-2D and GAIA result files into the pickle format used by the training code.

```bash
python scripts/build_dataset.py \
  --manifest examples/yen/dataset_manifest.example.csv \
  --history-k K \
  --bundle-b B \
  --base-step STEP \
  --output data/gno_dataset.pkl
```

The manifest assigns each simulated case to the training, validation or test split. Normalization statistics are calculated from the training cases only. See [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md) for the expected file layout and preprocessing details.

## 2. Train

```bash
python scripts/train.py \
  --dataset data/gno_dataset.pkl \
  --output models/best_model.pth
```

The training objective combines the area-weighted bed-change loss with the horizon-classification loss.

## 3. Evaluate

```bash
python scripts/evaluate.py \
  --dataset data/gno_dataset.pkl \
  --checkpoint models/best_model.pth \
  --split test \
  --output results/test_metrics.json
```

The evaluation script reports area-weighted RMSE, MAE, relative L2 error and R² in physical units. Optional CSV outputs provide results by prediction horizon and by case.

## 4. Run TELEMAC coupling

```bash
python scripts/run_telemac_coupling.py \
  --telemac-root /path/to/telemac/v8p4r0 \
  --cas /path/to/t2d_case.cas \
  --checkpoint models/best_model.pth \
  --dataset data/gno_dataset.pkl
```

The coupling script advances TELEMAC-2D, gathers the required state history, evaluates AH-GNO, writes the accepted bed change back to TELEMAC and continues the run. The area-weighted coupling used here is described in [docs/AREA_WEIGHTED_COUPLING.md](docs/AREA_WEIGHTED_COUPLING.md).

## Example cases

| Case | Material included here |
|---|---|
| Yen 180° bend | 20 forcing files, scenario table and train/validation/test split |
| Bump | 8 discharge/water-level combinations and split |
| Waiho River | case notes only; geometry and bathymetry are not redistributed |

The Yen and Bump examples use the corresponding TELEMAC-MASCARET/GAIA benchmark cases as their starting point. Generated `.slf` files are not stored in this repository.

## Data availability

Large simulation outputs, processed pickle datasets and model checkpoints are not tracked by default. The Waiho River mesh, bathymetry and associated boundary data are not included because of data-use restrictions.

## Python package name

The project is **AH-GNO**. Python identifiers cannot contain a hyphen, so imports use `AH_GNO`:

```python
from AH_GNO import LearnedHorizonGNO
```

## Citation

Citation metadata are in [CITATION.cff](CITATION.cff). The journal reference and archived software DOI can be added when they are available.

## License

No software license has been added yet. Licensing terms will be set after the institutional review is complete.
