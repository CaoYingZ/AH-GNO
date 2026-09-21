# Bump benchmark example

This directory documents the eight Bump scenarios used in the AH-GNO study. The scenarios are derived from the official TELEMAC-MASCARET/GAIA Bump benchmark and vary the upstream discharge and downstream free-surface level.

| Scenario | Upstream Q (m3/s) | Downstream SL (m) | Role |
|---|---:|---:|---|
| B01_Base | 1000 | 10.0 | Train |
| B02_LowQ | 850 | 10.0 | Train |
| B03_HighQ | 1150 | 10.0 | Train |
| B04_LowZ | 1000 | 9.6 | Train |
| B05_HighZ | 1000 | 10.4 | Train |
| B06_LowQ_LowZ | 850 | 9.6 | Validation |
| B07_HighQ_HighZ | 1150 | 10.4 | Train |
| B08_HighQ_LowZ | 1150 | 9.6 | Test |

The split is therefore **6 training / 1 validation / 1 test** at the scenario level.

Use the official Bump benchmark from your TELEMAC-MASCARET installation as the base case. Only the study-specific scenario definition is distributed here; generated TELEMAC/GAIA `.slf` outputs are intentionally not tracked.
