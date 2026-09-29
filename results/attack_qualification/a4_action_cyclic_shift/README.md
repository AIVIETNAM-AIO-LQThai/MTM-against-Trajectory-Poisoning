# A4 — High-Return Action Cyclic-Shift Qualification

Status: **FAIL**

The attack specification and qualification gate were frozen before victim outcomes
were observed.

## Clean controls

Reused frozen A2 clean controls:

- seed 0: 70.978705
- seed 1: 60.075498
- seed 2: 68.590957
- mean: 66.548386

## Qualification result

- mean degradation: -4.101633
- median degradation: -5.390493
- required mean degradation: +3.327419
- positive cells: 2/9

Model-seed mean degradation:

- seed 0: +2.992177
- seed 1: -10.988096
- seed 2: -4.308981

Attack-seed mean degradation:

- seed 20: -5.071628
- seed 21: -5.422301
- seed 22: -1.810971

Final decision:

**A4 ATTACK QUALIFICATION: FAIL**

This action cyclic-shift attack must not be used for DT vs DT+MTM vs RDT
defense comparison.
