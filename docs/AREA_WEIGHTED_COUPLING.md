# Area-weighted coupling

AH-GNO uses nodal control area in the graph aggregation and in the global features used by the temporal modules.

For node (i), the graph update is based on

```text
sum_j A_j K(i,j) h_j
--------------------
      sum_j A_j
```

over the neighbours inside the search radius. Here `A_j` is the control area associated with node `j`.

The same area weighting is used when the model forms the global feature for the history selector and the write-back horizon head:

```text
sum_i A_i h_i
-------------
   sum_i A_i
```

This keeps the aggregation tied to represented area rather than to the number of mesh nodes.

## TELEMAC loop

`scripts/run_telemac_coupling.py` follows the online loop used by the model:

1. advance TELEMAC-2D;
2. collect the required `U, V, H, Z` history;
3. normalize the state;
4. run AH-GNO with `node_area`;
5. take the learned `effective_b`;
6. sum the accepted bed increments;
7. write the updated bed and water depth back to TELEMAC;
8. continue the simulation.
