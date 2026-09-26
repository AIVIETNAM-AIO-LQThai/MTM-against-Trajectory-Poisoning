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

## 7. Poison selection localization

The aggregate behavioral region occupied by selected transitions is
similar across attack seeds within each condition/rho.

Mean trajectory return, trajectory length, relative trajectory position,
clean reward, action norm, and observation norm vary only modestly
between seeds.

Attack-seed overlap reveals a more informative pattern.

At rho=0.01:

- canonical seed1 vs seed2 transition Jaccard = 0.7805;
- S2 seed1 vs seed2 transition Jaccard = 0.7840.

Despite this high overlap, their Group-4D responses differ strongly:

- canonical: seed1 G=-10.7822 vs seed2 G=+3.3489;
- S2: seed1 G=+1.0972 vs seed2 G=-6.7478.

At rho=0.05, seed1 vs seed2 overlap is even higher:

- canonical transition Jaccard = 0.9279;
- S2 transition Jaccard = 0.9589.

Therefore the identity and gross behavioral region of selected
transitions are insufficient to explain the seed-dependent response.

The next analysis compares the actual observation/action perturbation
vectors on shared selected transitions.

## 8. Perturbation-vector direction

On transitions modified by two attack artifacts, the actual observation
and action perturbation vectors are nearly orthogonal.

Within-condition attack-seed comparisons produced mean/global
perturbation cosines generally close to zero for both observations and
actions.

Canonical-vs-S2 comparisons on the same rho/seed also produced only
small positive cosines.

However, disagreement in perturbation direction was only weakly related
to the corresponding behavioral difference |dG|.

Within-condition descriptive correlations between |dG| and directional
or normalized-distance metrics were approximately 0.23-0.28.

Canonical-vs-S2 correlations were small and negative, approximately
-0.08 to -0.18.

Therefore neither selected-transition identity nor raw perturbation-vector
geometry sufficiently explains Group-4D behavioral heterogeneity.

The next analysis is model-aware: measure the response of frozen clean
DT and clean DT+MTM policies to the exact poisoned contexts.

## 9. Frozen clean-policy sensitivity

Frozen clean vanilla-DT and clean DT+MTM policy branches were evaluated
on paired clean and poisoned causal contexts without retraining.

A systematic distinction emerged between direct endpoint corruption and
history-only corruption.

For all 12 frozen perturbation artifacts:

- A_direct = sensitivity_joint - sensitivity_DT was negative;
- A_history was positive.

Direct endpoint effects:

- A_direct ranged approximately from -0.112 to -0.089.

History-only effects:

- A_history ranged approximately from +0.0062 to +0.0115.

Thus the clean DT+MTM policy was consistently less locally sensitive
than vanilla DT when the current transition itself was perturbed, but
consistently more sensitive when the current endpoint remained clean
and perturbations occurred only earlier in its causal context.

The aggregate sensitivity difference remained negative because the
direct effect was substantially larger in magnitude.

Correlations with Group-4D G were weak:

- all contexts: corr(G,A) = -0.0934;
- direct endpoints: corr(G,A) = -0.0203;
- history-only: corr(G,A) = -0.2628.

Therefore frozen-policy local sensitivity does not explain the
seed-dependent Group-4D return response.

However, the sign-consistent history-only effect suggests that MTM
training changes how the final causal policy uses preceding trajectory
context.

The next analysis tests whether this additional history sensitivity
depends on the temporal distance between a poisoned token and the clean
prediction endpoint.

## 10. Historical-distance sensitivity

The excess history sensitivity of the clean DT+MTM policy persists
throughout the entire causal DT context.

Mean A = sensitivity_joint - sensitivity_DT by nearest poisoned-token
distance was:

- distance 1-3:  +0.018905
- distance 4-7:  +0.007533
- distance 8-12: +0.006121
- distance 13-19:+0.005262

A was positive for all 12 artifacts in every distance bin.

Thus DT+MTM's excess sensitivity is strongest for recently corrupted
history, but remains detectable even when the nearest corrupted token is
13-19 transitions behind the clean prediction endpoint.

The near-history effect is approximately 3.6 times the far-history
effect.

The fraction of contexts for which joint sensitivity exceeded vanilla
DT sensitivity was also above 0.60 in every bin and reached approximately
0.71 in the farthest bin.

Correlations between A and Group-4D G remained weak to moderate:

- 1-3:   -0.1114
- 4-7:   -0.3455
- 8-12:  -0.2020
- 13-19: -0.0382

Therefore this establishes a consistent representation/policy effect of
MTM training, but does not by itself explain the heterogeneous Group-4D
return response.

The next analysis decomposes the history effect into observation
perturbations versus action perturbations.

## 11. Historical modality sensitivity

The excess historical sensitivity of DT+MTM is almost entirely carried
by observation/state perturbations.

Across all 12 frozen artifacts:

- state-only:
  mean A = +0.008445;
  median A = +0.008044;
  A > 0 in 12/12 artifacts;

- action-only:
  mean A = +0.000023;
  median A = +0.000016;
  A > 0 in 8/12 artifacts;

- state+action:
  mean A = +0.008429;
  median A = +0.008035;
  A > 0 in 12/12 artifacts.

The state-only and combined effects are nearly identical, while
historical action corruption contributes essentially no excess
DT+MTM sensitivity.

Therefore the consistent history effect identified previously is a
state/observation-channel phenomenon.

Its correlation with Group-4D G remains weak
(corr(G,A_state) = -0.2601), so this does not explain the complete
seed-dependent return response.

The next analysis localizes the state effect within the policy:
whether amplification is already present in the shared DT state
embedding or emerges later through causal temporal processing.

## 12. State-representation localization

DT+MTM does not amplify observation perturbations at the input
representation.

Across all 12 artifacts:

Raw state embedding:
- mean B = -5.581857;
- joint/DT sensitivity ratio = 0.5703;
- B > 0 in 0/12 artifacts.

Pre-Transformer state token after timestep addition and LayerNorm:
- mean B = -0.238488;
- joint/DT sensitivity ratio = 0.9396;
- B > 0 in 0/12 artifacts.

Thus MTM training substantially contracts the immediate perturbation in
the shared state embedding and still slightly contracts it at the
pre-Transformer token level.

This contrasts with the previous history-only policy result, where
state corruption caused consistently greater final action sensitivity
in DT+MTM.

Therefore the excess historical sensitivity is not generated by the
shared state input projection. It emerges downstream, after causal
temporal mixing inside the Decision Transformer backbone.

The next analysis traces this amplification layer by layer through the
three causal GPT-2 blocks.

## 13. Layerwise historical-state propagation

The excess DT+MTM sensitivity to corrupted historical states emerges
inside the causal Transformer rather than at its input.

For history-only state corruption:

- transformer input:
  mean C = 0 exactly;
  the current endpoint is clean and therefore has identical input tokens;

- block 1:
  mean C = +0.003759;
  C > 0 in 9/12 artifacts;

- block 2:
  mean C = +0.104363;
  C > 0 in 12/12 artifacts;

- block 3:
  mean C = +0.102596;
  C > 0 in 12/12 artifacts;

- final action prediction:
  mean C = +0.008391;
  C > 0 in 12/12 artifacts.

Thus the clearest amplification appears between block 1 and block 2
and persists through the remaining causal backbone into the action
prediction.

Combined with the previous representation audit, the evidence indicates:

1. MTM training contracts direct observation perturbations in the shared
   state embedding;

2. historical observation corruption nevertheless produces greater
   endpoint-policy sensitivity;

3. this reversal is generated during causal temporal processing,
   becoming systematic by Transformer block 2.

Layerwise C remains only weakly correlated with Group-4D G, so this
mechanism explains a consistent representational effect of MTM training
but not the large seed-dependent return variation.

The final localization analysis tests whether the block-2 effect is
associated with changed causal attention routing toward corrupted
historical state tokens.

## 14. Historical-state attention routing

The final Group-4E probe examined whether the layerwise amplification of
historical state corruption was associated with altered causal attention
routing toward corrupted historical state positions.

Mean joint-minus-DT perturbation-induced attention rerouting was:

- block 1:
  mean D = -0.00032680;
  D > 0 in 3/12 artifacts;

- block 2:
  mean D = +0.00136793;
  D > 0 in 10/12 artifacts;

- block 3:
  mean D = -0.00019637;
  D > 0 in 4/12 artifacts.

Thus block 2 is the only Transformer layer showing a clear tendency for
the DT+MTM model to reroute more endpoint-state attention toward
corrupted historical state positions after perturbation.

This aligns with the independent layerwise representation result, where
joint-minus-DT hidden-state sensitivity became strongly and
systematically positive by block 2.

However:

- the block-2 attention effect is not positive in every artifact;
- its magnitude is small;
- corr(G,D_block2) = -0.1190;
- attention weights alone are not a causal decomposition of the full
  Transformer computation.

Therefore the supported interpretation is that altered block-2
attention routing is consistent with, and may contribute to, the
observed temporal amplification. The evidence does not establish that
attention rerouting alone causes the amplification or the downstream
Group-4D return differences.

## Group 4E final conclusion

Group 4E investigated why masked trajectory modeling changes the
response of a causal Decision Transformer to frozen trajectory
perturbations.

Several simple explanations were not supported:

1. Aggregate gradient conflict, gradient-ratio changes, and clean-probe
   losses did not track the stress-response gap G.

2. Global parameter displacement magnitude strongly tracked rho but not
   behavioral response.

3. Canonical and S2 models often moved in different parameter-space
   directions, but directional disagreement did not explain differences
   in G.

4. Poison count, trajectory coverage, temporal clustering, and DT/MTM
   window exposure differed structurally between conditions but were
   nearly invariant across seeds relative to the large variation in G.

5. Selected dataset regions and raw perturbation-vector geometry were
   insufficient to explain G.

6. Frozen clean-policy sensitivity also did not explain the
   seed-dependent return response.

Despite this, Group 4E identified a consistent model-level effect of
joint MTM training:

- direct endpoint corruption produces LESS action sensitivity in
  DT+MTM than vanilla DT;

- history-only corruption produces MORE action sensitivity in DT+MTM
  than vanilla DT;

- the excess historical sensitivity is positive across the entire
  20-step causal context and is strongest for recent corrupted history;

- the effect is almost entirely attributable to corrupted observations
  / states rather than historical actions;

- MTM does NOT amplify these perturbations at the shared state embedding
  or pre-Transformer token level; it actually contracts them there;

- the excess sensitivity emerges during causal temporal processing,
  becoming systematic by Transformer block 2;

- block 2 is also the only layer showing a consistent tendency toward
  increased attention routing to corrupted historical state positions.

The resulting mechanism supported by the experiments is therefore:

    MTM auxiliary training changes how the causal DT backbone
    integrates historical state information. Although the shared
    state representation itself becomes less locally sensitive,
    perturbations occurring in earlier states propagate more strongly
    to the current policy representation during deeper causal temporal
    processing, with the clearest amplification emerging around the
    second Transformer block.

This is a consistent architectural effect, not a defense result.

It does not explain the large seed-dependent Group-4D stress-response
gap, and the frozen Group2 perturbations did not establish a successful
vanilla-DT attack under the predeclared Group4C criterion.

Therefore Group 4E is closed without further post-hoc mechanism search.
