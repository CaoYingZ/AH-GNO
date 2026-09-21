# AH-GNO

**Adaptive-Horizon Graph Neural Operator for coupled surrogate morphodynamic modelling**

AH-GNO is a graph neural operator developed for morphodynamic bed-evolution prediction on native unstructured TELEMAC meshes. The model learns bundled bed-elevation increments and adaptively determines both the effective historical context and the reliable write-back horizon used during online coupling.

The released implementation follows the area-weighted formulation described in the manuscript. Nodal control area is used consistently in graph aggregation, temporal global descriptors, normalization, and loss evaluation so that the learned operator is less sensitive to variations in mesh-node density.

## Overview

AH-GNO takes the recent history of

- streamwise velocity `U`,
- transverse velocity `V`,
- water depth `H`, and
- bed elevation `Z`

as node-wise input features and predicts a bundle of future bed-elevation increments `ΔZ`.

The workflow contains three adaptive components:

1. **Area-weighted graph operator** — performs local message aggregation using nodal control area as the quadrature measure.
2. **Adaptive history selector** — learns how much recent temporal information contributes to the current prediction.
3. **Adaptive write-back horizon** — estimates how many predicted increments can be safely written back before the next AH-GNO evaluation.

For online morphodynamic acceleration, AH-GNO is coupled to TELEMAC-2D through TelApy. TELEMAC advances the hydrodynamics, AH-GNO predicts bed updates, and the accepted bed-elevation increments are written back to the TELEMAC state.

## Repository structure

```text
AH-GNO/
├── src/
│   └── AH_GNO/
│       ├── __init__.py
│       ├── model.py
│       ├── preprocessing.py
│       └── losses.py
│
├── scripts/
│   ├── build_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_telemac_coupling.py
│
├── configs/
│   ├── yen.yaml
│   ├── bump.yaml
│   └── waiho_template.yaml
│
├── examples/
│   ├── yen/
│   │   ├── hydrographs/
│   │   ├── scenarios.csv
│   │   └── README.md
│   ├── bump/
│   │   ├── scenarios.csv
│   │   └── README.md
│   └── waiho/
│       └── README.md
│
├── docs/
│   ├── DATA_PREPARATION.md
│   └── AREA_WEIGHTED_COUPLING.md
│
├── CITATION.cff
├── pyproject.toml
└── requirements.txt
```

## Installation

The manuscript experiments were developed with:

- TELEMAC-MASCARET V8P4R0
- Python 3.12
- PyTorch 2.5.1
- Ubuntu 22.04
- CUDA 12.4

Create a Python environment and install the released AH-GNO package:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

TELEMAC-MASCARET is not installed through `pip`. For online coupling, install TELEMAC separately and source the corresponding TELEMAC environment before running the coupling script.

## Training

Training starts from a preprocessed AH-GNO dataset containing the native mesh coordinates, nodal control areas, normalized node features, bundled `ΔZ` targets, and the scenario-level train/validation/test split.

```bash
python scripts/train.py \
  --dataset /path/to/gno_dataset.pkl \
  --output models/best_model.pth
```

The public training script implements the area-weighted Huber loss and the learned horizon objective used by AH-GNO.

The full research dataset binaries are not included in the repository. Public benchmark forcing files and scenario definitions are provided under `examples/`. A reproducible SELAFIN-to-AH-GNO dataset builder is provided in `scripts/build_dataset.py`; see [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md) for the expected manifest and preprocessing workflow.

## Evaluation

Evaluate a trained AH-GNO checkpoint on the train, validation, or test split with area-weighted physical-space metrics:

```bash
python scripts/evaluate.py \
  --dataset /path/to/gno_dataset.pkl \
  --checkpoint /path/to/best_model.pth \
  --split test \
  --output results/test_metrics.json \
  --per-horizon-csv results/test_per_horizon.csv \
  --per-case-csv results/test_per_case.csv
```

The evaluator reports area-weighted **RMSE**, **MAE**, **relative L2 error**, and **R²** after converting predictions back to physical bed-elevation units. It reports both per-step `ΔZ` metrics and cumulative bed-change metrics, and also summarizes the learned effective write-back horizon. Training activity weights are not reused for evaluation; nodal control area is the spatial weighting measure.

## TELEMAC online coupling

The compact coupling entry point is:

```bash
python scripts/run_telemac_coupling.py \
  --telemac-root /path/to/telemac/v8p4r0 \
  --cas /path/to/t2d_case.cas \
  --checkpoint /path/to/best_model.pth \
  --dataset /path/to/gno_dataset.pkl
```

During coupling, the script:

1. advances TELEMAC-2D to the next AH-GNO evaluation time;
2. collects the required `U, V, H, Z` history;
3. normalizes the node-wise state;
4. evaluates AH-GNO using `node_area`;
5. determines the learned effective write-back horizon;
6. accumulates the accepted `ΔZ` bundle;
7. updates TELEMAC bed elevation and water depth; and
8. continues the hydrodynamic simulation.

Further details on the area-weighted formulation are provided in [`docs/AREA_WEIGHTED_COUPLING.md`](docs/AREA_WEIGHTED_COUPLING.md).

## Cases used in the manuscript

| Case | Type | Public material in this repository |
|---|---|---|
| Yen 180° bend | TELEMAC-MASCARET/GAIA benchmark | 20 forcing hydrographs, scenario table, and scenario-level split |
| Bump | TELEMAC-MASCARET/GAIA benchmark | 8 discharge/water-level scenarios and split |
| Waiho River | Field-scale river case | workflow description only; restricted geometry and bathymetry are not redistributed |

### Yen benchmark

The 20 Yen forcing scenarios are stored in `examples/yen/hydrographs/`, with their roles summarized in `examples/yen/scenarios.csv`.

The scenario-level split is:

- **Training:** Y01, Y02, Y03, Y04, Y05, Y06, Y09, Y10, Y12, Y14, Y17, Y19
- **Validation:** Y07, Y13, Y16, Y20
- **Test:** Y08, Y11, Y15, Y18

### Bump benchmark

The eight Bump scenarios are summarized in `examples/bump/scenarios.csv`. Six scenarios are used for training, one for validation, and one for testing.

### Waiho River

The Waiho River mesh, bathymetry, and associated boundary-condition files are subject to data-use restrictions and are therefore not distributed in this repository. The same AH-GNO architecture and coupling workflow are used for the field-scale case.

## Data and model availability

This repository intentionally separates source code from large or restricted research data.

The following are not tracked:

- generated TELEMAC/GAIA `.slf` result files;
- processed `.pkl` datasets;
- model checkpoints unless explicitly approved for public release;
- Waiho mesh, bathymetry, DEM/topographic data, and restricted boundary files.

The public benchmark forcing definitions are included so that the study-specific Yen and Bump scenarios can be reconstructed from the corresponding official TELEMAC-MASCARET examples.

## Naming convention

The scientific method and repository are named **AH-GNO**.

Because Python module names cannot contain a hyphen, the importable package uses the identifier `AH_GNO`:

```python
from AH_GNO import LearnedHorizonGNO
```

## Citation

If you use AH-GNO in academic work, please cite the associated manuscript. Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

The journal citation and archival software DOI will be added once available.

## License

No open-source license is currently attached to this repository. Reuse and redistribution terms will be added after institutional licensing review.
