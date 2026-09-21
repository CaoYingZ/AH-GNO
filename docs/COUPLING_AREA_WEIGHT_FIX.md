# Coupling code update: area-weighted inference

Earlier development versions of the coupling script did not match the latest area-weighted training formulation. Manuscript-consistent inference requires the following changes:

1. **Graph integral**
   - Old: `sum_j K_ij f_j`
   - New: `sum_j A_j K_ij f_j / sum_j A_j`
   - `A_j` is the nodal control area stored with the processed mesh/dataset.

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

The public implementation in `src/ahgno/model.py` and `scripts/run_telemac_coupling.py` already follows this area-weighted formulation.

## Write-back setting

A development compatibility script may still expose a switch between fixed and learned bundle lengths for ablation or debugging. For the manuscript AH-GNO workflow, use the learned `effective_b`. The compact public script in `scripts/run_telemac_coupling.py` uses the learned horizon by default.
