# Waiho River application

The Waiho River geometry, bathymetry, and computational mesh used by the study are not included in this repository because they may be subject to data-use or confidentiality restrictions.

The same AH-GNO implementation is used for this case. To apply the code to an authorized river model, provide:

- node coordinates and triangular connectivity;
- TELEMAC hydrodynamic fields `U`, `V`, `H`, and bed elevation `Z`;
- GAIA reference bed elevations for supervised training;
- the corresponding boundary and forcing files that you are permitted to use.

Before making this repository public, confirm with the data owner and your institution whether trained Waiho checkpoints may also be redistributed.
