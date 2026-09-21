# Area-weighted AH-GNO coupling

AH-GNO uses nodal control area consistently during inference so that the discrete graph operator more closely represents an integral operator on an unstructured mesh.

## Graph integral

For query node (i), the area-weighted graph aggregation is

[
h_i^{(l+1)}
=
h_i^{(l)}
+
\sigma\left(
\frac{
\sum_{j\in\mathcal{N}(i)}
A_j K(s_i-s_j)\odot h_j^{(l)}
}{
\sum_{j\in\mathcal{N}(i)} A_j
}
\right),
]

where (A_j) is the nodal control area associated with neighbor node (j).

In the released implementation, this weighting is handled inside the graph integral transform and the same `node_area` array is passed to every GNO layer.

## Area-weighted global descriptors

Nodal control area is also used when constructing the global descriptors for the adaptive temporal modules.

For both the history selector and the horizon head, the global feature is computed as

[
\bar{h}
=
\frac{\sum_i A_i h_i}{\sum_i A_i}.
]

This replaces an ordinary node-wise mean and reduces sensitivity to spatial variations in mesh density.

## Online TELEMAC coupling

The public coupling workflow performs the following steps:

1. advance TELEMAC-2D to the next AH-GNO evaluation time;
2. collect the required history of `U`, `V`, `H`, and `Z`;
3. normalize the state variables using the training statistics;
4. evaluate AH-GNO with `node_area`;
5. use the learned effective write-back horizon `effective_b`;
6. accumulate the predicted bed-elevation increments over the accepted horizon;
7. update TELEMAC bed elevation and water depth;
8. continue the hydrodynamic simulation.

The implementation used for the public release is provided in `scripts/run_telemac_coupling.py`.
