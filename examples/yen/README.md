# Yen 180° bend

The Yen case uses the standard TELEMAC-MASCARET/GAIA 180° bend benchmark. This folder contains the forcing files used to build the 20 scenarios in the study.

## Split

- **Train:** Y01, Y02, Y03, Y04, Y05, Y06, Y09, Y10, Y12, Y14, Y17, Y19
- **Validation:** Y07, Y13, Y16, Y20
- **Test:** Y08, Y11, Y15, Y18

The split is made by scenario, not by time window.

## Files

`hydrographs/` contains the QSL forcing files. `scenarios.csv` gives the scenario names and split. `dataset_manifest.example.csv` shows the file list expected by `scripts/build_dataset.py`.

The original benchmark geometry and generated TELEMAC/GAIA result files are not copied into this repository.
