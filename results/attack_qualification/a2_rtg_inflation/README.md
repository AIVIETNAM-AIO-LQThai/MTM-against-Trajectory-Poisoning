# A2 — RTG Inflation Attack Qualification

Status: **FAIL**

This result is frozen. The attack specification and qualification gate must not
be modified retrospectively based on these results.

## Clean bridge

- Clean seed 0: 70.978705
- Clean seed 1: 60.075498
- Clean seed 2: 68.590957
- Clean mean: 66.548386
- Required clean floor: 62.828449
- Result: PASS

## Attack qualification

- Mean degradation: -2.508318
- Median degradation: -1.034089
- Required mean degradation: +3.327419
- Positive degradation cells: 4/9

Model-seed mean degradations:
- seed 0: -0.057151
- seed 1: -7.027526
- seed 2: -0.440277

Attack-seed mean degradations:
- attack 10: -2.456805
- attack 11: -3.266728
- attack 12: -1.801420

Final decision: **A2 ATTACK QUALIFICATION FAIL**

Consequently, this attack must not be used for a DT vs DT+MTM vs RDT
defense comparison.
