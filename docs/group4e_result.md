# Group 4E — Mechanism Analysis Results

## 1. Aggregate optimization diagnostics

The aggregate mechanism screen did not reveal a simple relationship
between the Group-4D stress-response gap G and:

- shared-gradient cosine;
- frequency of negative gradient cosine;
- MTM-to-DT shared-gradient norm ratio;
- final clean DT probe error;
- final clean MTM reconstruction error.

The strongly negative-G runs therefore cannot be explained by a simple
increase in gradient conflict, stronger auxiliary-gradient pressure, or
larger final clean-probe degradation.

## 2. Temporal optimization diagnostics

The 10K-binned analysis likewise did not identify a stable temporal
signature separating strongly negative-G runs from positive-G runs.

Some early intervals showed slightly greater relative MTM pressure or
gradient conflict for the negative-G group, but these differences were
not persistent and frequently reversed at later stages.

Therefore neither aggregate nor time-resolved first-order optimization
diagnostics provide a common explanation for the heterogeneous
Group-4D stress response.

## 3. Parameter-divergence result

Final poison-vs-clean parameter displacement strongly reflected the
perturbation level rho, but did not explain the Group-4D stress-response
gap G.

Across the 12 runs, Pearson correlations between G and final relative-L2
parameter displacement were weak:

- shared embeddings: -0.1502
- whole DT: -0.1634
- bridges: -0.1461
- MTM branch: -0.1398

Runs with rho=0.05 generally showed substantially larger parameter
displacement than runs with rho=0.01.

However, matched runs with the same rho and training seed could exhibit
nearly identical displacement magnitudes while producing substantially
different G values.

Therefore the magnitude of parameter drift is insufficient to explain
the heterogeneous behavioral response.

The next mechanism analysis should examine the direction and localization
of the learned parameter changes rather than only their global norm.
