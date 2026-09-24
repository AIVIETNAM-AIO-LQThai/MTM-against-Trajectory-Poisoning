# Group 4E — Parameter-displacement direction analysis

## Question

For the same training seed and the same rho, canonical and
s2_overlap_r0 perturbations often produce similar parameter-displacement
magnitudes but substantially different behavioral stress responses G.

This analysis tests whether they move the trained model in different
parameter-space directions.

## Comparison

For each rho in {0.01, 0.05} and seed in {0,1,2}:

delta_canonical =
    theta_canonical - theta_clean

delta_s2 =
    theta_s2_overlap_r0 - theta_clean

using the same-seed clean DT+MTM checkpoint at update 100000.

For each parameter group compute:

- cosine(delta_canonical, delta_s2)
- angle in degrees
- norm of each displacement
- direct distance between canonical and s2 models
- direct distance normalized by the mean displacement norm

Parameter groups:

1. shared DT state/action embeddings
2. non-shared DT backbone
3. complete DT policy
4. state/action bridges
5. MTM branch

## Interpretation

Cosine near +1:
the two perturbations move parameters in similar directions.

Cosine near 0:
their learned changes are approximately orthogonal.

Cosine below 0:
their learned changes oppose one another.

This is exploratory analysis selected after observing Group-4D results.
Associations with G are descriptive and do not establish causality.
