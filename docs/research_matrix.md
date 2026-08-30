# Research Matrix

## Development
- Active dataset: `walker2d-medium-v2`
- Active method: vanilla DT
- Poison rate: 0%
- Development seeds: 0, 1, 2
- MTM: disabled
- CSDPC: disabled

## Main paper matrix, later
Datasets: Walker2d / Hopper / HalfCheetah medium-v2.
Methods: BC / CQL / DT / RDT / DT + MTM.
Core poison rates: 0% / 1% / 5%.
Final evaluation: five seeds after development is validated.

## Experiment order
1. Freeze development configuration.
2. Reproduce clean DT and pass Gate A.
3. Implement CSDPC independently.
4. Validate CSDPC with CQL.
5. Test CSDPC transfer to DT.
6. Add BC transfer baseline.
7. Reproduce MTM independently.
8. Build DT + MTM.
9. Check clean DT + MTM performance.
10. First defense experiment.
11. Add RDT.
12. Regularization/capacity controls.
13. Mechanistic Walker2d analyses.
14. Scale to Hopper and HalfCheetah.
15. Final main matrix.
16. MTM-specific ablations.
17. Attack-length × mask-length.
18. Harder/adaptive attacks.
19. Dataset-quality generalization.
