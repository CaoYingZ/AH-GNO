# Yen 180° bend example

This directory documents how to reproduce the public benchmark setup without redistributing the full TELEMAC-MASCARET example directory.

1. Obtain the official GAIA Yen benchmark from your TELEMAC-MASCARET installation.
2. Apply the scenario-specific hydrographs and parameter changes used in the paper.
3. Run TELEMAC-2D/GAIA to generate the reference fields.
4. Convert those outputs to `gno_dataset.pkl` using the same area-weighted preprocessing as the paper.

Recommended public contents here are the scenario hydrographs, a scenario table, and a short description of changes relative to the official benchmark. Do not commit large generated `.slf` results.
