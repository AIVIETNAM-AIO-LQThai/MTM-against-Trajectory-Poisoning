# Group 4E — Perturbation-vector direction analysis

## Question

Different attack seeds can perturb highly overlapping transition sets
while producing substantially different Group-4D stress responses.

This analysis tests whether their actual state/action perturbation
vectors differ in direction on those shared transitions.

## Comparisons

A. Within each condition/rho:
   compare attack-seed pairs 0-1, 0-2, and 1-2.

B. For each rho/seed:
   compare canonical vs s2_overlap_r0.

Only transitions modified by both members of a comparison are used for
directional comparisons.

## Metrics

Separately for observation and action perturbations:

1. number of overlapping modified transitions;
2. mean row-wise perturbation cosine;
3. median row-wise perturbation cosine;
4. flattened/global perturbation cosine;
5. element-wise sign agreement;
6. normalized direct perturbation-vector distance.

The behavioral quantity paired with each comparison is absolute
difference in Group-4D G.

This is exploratory mechanism analysis.
