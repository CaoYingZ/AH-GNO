# Release maintenance checklist

- Remove all absolute server paths, user names, temporary output folders, and credentials before each release.
- Do not commit Waiho mesh, coordinates, bathymetry, DEMs, boundary files, or restricted derived data.
- Do not commit large generated `.slf`/`.pkl` result archives.
- Keep model checkpoints out of version control unless they are intentionally approved for public release.
- Confirm the software license with the supervisor/institution before adding a LICENSE file.
- Verify that the public coupling script uses `node_area` in the graph operator, K descriptor, and horizon descriptor.
- Verify that the manuscript-matching run uses learned `effective_b` rather than a fixed bundle, except for explicitly labeled ablation tests.
- Tag the manuscript-matching version (for example `v1.0.0`) and archive that release in a DOI-granting repository after approval.
- Recheck README and CITATION metadata after any GitHub username or repository-name change.
