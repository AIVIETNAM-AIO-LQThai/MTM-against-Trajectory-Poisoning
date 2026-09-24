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

## 5. Raw poison localization

The frozen trajectory perturbations modify observations and actions only.

No changes were detected in rewards, terminals, or timeouts. Therefore
the Group-4D stress response is not caused by direct reward or RTG
corruption.

At rho=0.01:

- canonical modifies exactly 10,000 transitions per seed and touches
  922-935 of 1,190 completed trajectories (~77.9% on average);
- s2_overlap_r0 modifies 9,997-10,000 realized transitions and touches
  857-874 trajectories (~72.8% on average).

At rho=0.05:

- canonical modifies 50,000 transitions and touches 1,186-1,189
  trajectories (~99.8%);
- s2_overlap_r0 modifies 49,947-49,998 realized transitions and touches
  1,180-1,189 trajectories (~99.5%).

The perturbation magnitudes are similar across conditions:

- action maximum absolute change is approximately 0.05;
- observation maximum absolute change is approximately 0.5.

Thus gross perturbation count and magnitude do not explain the strongly
different Group-4D behavioral responses. The next analysis examines the
sequential organization of poisoned transitions and their exposure in
DT and MTM training windows.

## 6. Sequential poison and window exposure

S2 perturbations are more temporally clustered than canonical
perturbations.

At rho=0.01:

- canonical poison-run mean is approximately 5.1 transitions;
- S2 poison-run mean is approximately 6.0 transitions;
- canonical DT-context exposure is approximately 4.3%;
- S2 DT-context exposure is approximately 3.9%;
- canonical MTM-window exposure is approximately 1.58%;
- S2 MTM-window exposure is approximately 1.50%.

At rho=0.05:

- canonical poison-run mean is approximately 5.25 transitions;
- S2 poison-run mean is approximately 6.95 transitions;
- canonical DT-context exposure is approximately 19.8%;
- S2 DT-context exposure is approximately 16.7%;
- canonical MTM-window exposure is approximately 7.7%;
- S2 MTM-window exposure is approximately 7.1%.

Thus S2 concentrates modified transitions into longer contiguous runs
while exposing fewer distinct DT and MTM windows.

However, these structural exposure quantities are nearly constant across
seeds within a condition/rho while Group-4D G varies strongly and can
change sign.

Therefore perturbation clustering and total window-exposure breadth are
insufficient to explain the seed-dependent behavioral response.

The next analysis examines which trajectories and state/action regions
are selected, rather than how many windows they affect.
