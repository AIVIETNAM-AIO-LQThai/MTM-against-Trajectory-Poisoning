# Group 4E — Poison selection localization

## Question

Aggregate poison count, perturbation magnitude, parameter geometry, and
sequence-window exposure do not explain the seed-dependent Group-4D
stress-response gap.

This analysis asks whether different attack seeds perturb different
behavioral regions of the clean offline dataset.

## Metrics

For every poisoned artifact, characterize changed transitions using the
clean dataset:

1. clean trajectory return;
2. trajectory length;
3. relative temporal position within trajectory;
4. distance from trajectory end;
5. clean instantaneous reward;
6. clean action L2 norm;
7. clean observation L2 norm.

Also compute:

8. transition-index Jaccard overlap between attack seeds within the same
   condition/rho;
9. affected-trajectory Jaccard overlap between attack seeds.

All quantities are descriptive and computed from the frozen artifacts.
