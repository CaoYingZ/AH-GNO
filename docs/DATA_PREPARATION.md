# Preparing training data

`scripts/build_dataset.py` converts paired TELEMAC-2D and GAIA result files into the dataset used by the training script.

Each row in the input manifest represents one simulated case:

```csv
case_id,role,hydro_file,gaia_file
Y01,train,/path/to/case_01_hy.slf,/path/to/case_01_ga.slf
Y07,val,/path/to/case_07_hy.slf,/path/to/case_07_ga.slf
Y08,test,/path/to/case_08_hy.slf,/path/to/case_08_ga.slf
```

The TELEMAC and GAIA files for a case must use the same mesh and output times.

For each case, the script reads `U`, `V` and `H` from the TELEMAC-2D result and bed elevation `Z` from the GAIA result. It then computes nodal control areas, forms rolling history windows, builds future `ΔZ` targets, and applies normalization based only on the training cases.

Example:

```bash
python scripts/build_dataset.py \
  --manifest examples/yen/dataset_manifest.example.csv \
  --history-k 7 \
  --bundle-b 10 \
  --base-step 2 \
  --output data/gno_dataset.pkl
```

`--history-k` and `--bundle-b` should match the model configuration used for the run being reproduced. `--base-step` sets the sampling interval relative to the saved SELAFIN records.

The output pickle contains the mesh coordinates, nodal areas, split samples, normalization statistics and the metadata required by `scripts/train.py`.

Processed datasets are not committed to the repository.
