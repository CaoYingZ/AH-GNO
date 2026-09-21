# Coupling code update: area-weighted inference

The previous coupling script did not match the latest training notebook. The following changes are required for checkpoint-consistent inference:

1. **Graph integral**
   - Old: `sum_j K_ij f_j`
   - New: `sum_j A_j K_ij f_j / sum_j A_j`
   - `A_j` is the nodal control area stored in `gno_dataset.pkl` / `mesh_info.pkl`.

2. **History selector descriptor**
   - Old: `lifted_stack[-1].mean(dim=0)`
   - New: `sum_i A_i h_i / sum_i A_i`.

3. **Horizon-head descriptor**
   - Old: `h.mean(dim=0)`
   - New: `sum_i A_i h_i / sum_i A_i`.

4. **Model forward signature**
   - Old: `model(pos, f)`
   - New: `model(pos, f, node_area)`.

5. **Inference data**
   - `node_area` is loaded with the same mesh as `node_pos` and checked against TELEMAC `NPOIN`.

The standalone patched file supplied with this repository draft preserves the original research script structure and timing diagnostics while applying these changes.

## Write-back setting

The old research script still contains `USE_LEARNED_BUNDLE_STEPS = False`. That switch is intentionally not changed in the standalone compatibility patch because changing it also changes the numerical experiment. For the manuscript AH-GNO workflow, use the learned `effective_b` (set the switch to `True`). The compact public script in `scripts/run_telemac_coupling.py` uses the learned horizon by default.
