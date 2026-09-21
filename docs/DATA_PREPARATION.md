# Data preparation

AH-GNO training data are constructed from paired TELEMAC-2D and GAIA result files on the same native unstructured mesh.

The public builder is provided in `scripts/build_dataset.py`. It implements the data representation used by the manuscript:

- historical node states `[U, V, H, Z]`;
- future bed-elevation increments `ΔZ`;
- triangular nodal control areas;
- training-only area-weighted normalization statistics;
- scenario-level train/validation/test separation;
- activity weighting for morphodynamically active nodes.

## Input manifest

Create a CSV with one row per simulated scenario:

```csv
case_id,role,hydro_file,gaia_file
Y01,train,/path/to/case_01_hy.slf,/path/to/case_01_ga.slf
Y07,val,/path/to/case_07_hy.slf,/path/to/case_07_ga.slf
Y08,test,/path/to/case_08_hy.slf,/path/to/case_08_ga.slf
```

The `role` field must be `train`, `val`/`validation`, or `test`.

The HYDRO and GAIA files for a scenario must use the same mesh and output-time grid. The builder reads `U`, `V`, and `H` from the TELEMAC-2D result and, by default, `Z` from the GAIA `BOTTOM` field.

## Build a dataset

Example:

```bash
python scripts/build_dataset.py \
  --manifest examples/yen/dataset_manifest.example.csv \
  --history-k 7 \
  --bundle-b 10 \
  --base-step 2 \
  --window-stride 1 \
  --output data/yen_gno_dataset.pkl
```

`--base-step` controls the temporal sampling interval relative to the saved SELAFIN records. `--window-stride` controls the movement of the rolling training window.

For the Yen experiments, the archived run metadata available with the project show a 2-minute base interval with `K=7` and `B=10` for one AREAWEIGHTED v6 configuration. Treat these values as an example configuration unless they are confirmed against the final manuscript checkpoint.

## Normalization

Input and target normalization statistics are calculated **only from training scenarios**. Means and standard deviations are area-weighted using the nodal control areas of the native triangular mesh. Validation and test samples are transformed using the training statistics.

## Activity weights

The public builder uses

```text
w = 1 + gain * clip(activity / scale, 0, 1)
```

where `activity` is the maximum absolute physical `ΔZ` over the target bundle at a node. The default gain is 4. If `--activity-scale` is not specified, the training-set physical `ΔZ` standard deviation is used as a reproducible fallback.

For exact reconstruction of a previously trained checkpoint, pass the activity scale stored with that training run rather than relying on the fallback.

## Output

The resulting pickle contains the fields expected by `scripts/train.py`, including:

- `node_pos`
- `node_area`
- `elements`
- `train_samples`, `val_samples`, `test_samples`
- `normalizer_state`
- `history_k`, `bundle_b`, `base_step`, `base_dt_minutes`
- `scenario_split`

Large processed pickle datasets are intentionally excluded from version control.
