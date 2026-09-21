# Pre-publication checklist

- Keep the GitHub repository private until the patent/IP review is complete.
- Remove all absolute server paths, user names, temporary output folders, and credentials.
- Do not commit Waiho mesh, coordinates, bathymetry, DEMs, boundary files, or restricted derived data.
- Do not commit large generated `.slf`/`.pkl` result archives.
- Confirm the license with the supervisor/institution before adding a LICENSE file.
- Verify that the public coupling script uses `node_area` in the graph operator, K descriptor, and horizon descriptor.
- Verify that the paper-release run uses learned `effective_b` rather than a fixed bundle, except for explicitly labeled ablation tests.
- Tag the manuscript-matching version (for example `v1.0.0`) and archive that release in a DOI-granting repository after approval.
