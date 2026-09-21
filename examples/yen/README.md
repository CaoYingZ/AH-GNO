# Yen 180° bend benchmark

This directory contains the 20 study-specific inflow/free-surface forcing files used with the official TELEMAC-MASCARET/GAIA Yen benchmark. The original TELEMAC benchmark geometry and generated `.slf` outputs are not redistributed here.

## Scenario split

- **Training:** Y01, Y02, Y03, Y04, Y05, Y06, Y09, Y10, Y12, Y14, Y17, Y19
- **Validation:** Y07, Y13, Y16, Y20
- **Test:** Y08, Y11, Y15, Y18

The split is performed at the **scenario level** (12 train / 4 validation / 4 test), so time windows from the same forcing scenario never appear in more than one subset.

## Files

- `hydrographs/`: cleaned TELEMAC-compatible `.qsl` files. Numerical values are unchanged from the research inputs; only filenames, comments, and whitespace were standardized for release.
- `scenarios.csv`: machine-readable scenario summary and split.

Each `.qsl` file keeps the TELEMAC format:

```text
T       Q(1)     SL(2)
s       m3/s     m
...
```

where `T` is time, `Q(1)` is prescribed discharge, and `SL(2)` is the prescribed free-surface level.

The separate `Y08_DoublePeak_sensitivity` file used during sensitivity testing is not part of the 20-scenario training/validation/test dataset and is therefore not included here.
