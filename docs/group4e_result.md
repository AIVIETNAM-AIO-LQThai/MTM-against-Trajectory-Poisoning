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

## 4. Parameter-displacement direction

Canonical and s2_overlap_r0 perturbations with the same rho and seed
often moved the final DT+MTM model in substantially different
parameter-space directions despite similar displacement magnitudes.

Observed canonical-vs-S2 displacement cosines were approximately:

- shared embeddings: 0.897 to 0.968
- whole DT: 0.587 to 0.703
- bridges: 0.752 to 0.876
- MTM branch: 0.688 to 0.757

However, directional disagreement did not explain the behavioral
difference between canonical and S2.

Correlations between absolute G difference and 1-cosine were weak:

- shared embeddings: +0.0479
- whole DT: +0.0781
- bridges: +0.1044
- MTM branch: +0.1416

Therefore neither global displacement magnitude nor global displacement
direction provides a sufficient explanation for the heterogeneous
Group-4D behavioral response.

The next analysis moves from model-level geometry to data-level
localization: exactly which transitions and trajectory regions differ
between the clean and poisoned datasets.
