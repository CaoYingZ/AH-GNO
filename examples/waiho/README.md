# Waiho River case

The Waiho River case is used in the manuscript to evaluate AH-GNO in a field-scale morphodynamic setting with an irregular river geometry and a substantially larger computational domain than the benchmark cases.

The same AH-GNO architecture and area-weighted formulation used for the Yen and Bump benchmarks are applied to the Waiho case. Model inputs consist of the hydrodynamic and bed-state variables `U`, `V`, `H`, and `Z` defined on the native unstructured TELEMAC mesh.

## Data availability

The Waiho River mesh, bathymetry, and associated boundary-condition files are not distributed with this repository because they are subject to data-use restrictions. The source code and model workflow are identical to those used for the public benchmark cases.

Users can apply the released AH-GNO implementation to their own TELEMAC river models by providing the corresponding mesh, hydrodynamic fields, bed elevation, and reference morphodynamic results required for training or evaluation.
