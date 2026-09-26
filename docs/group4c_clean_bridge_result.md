# Group 4C Clean Bridge Result

The fresh Group 4C vanilla-DT clean controls were trained with the
same Group 4C runner and numerical environment used for poisoned runs.

Evaluation contract:
- Walker2d-v3
- target return: 5000
- 100 episodes per training seed
- evaluation seed base: 30000

Results:

| Training seed | Normalized return mean |
| --- | ---: |
| 0 | 73.716021030119 |
| 1 | 65.341193981272 |
| 2 | 67.012623942427 |

Three-seed mean: 68.689946317939
Three-seed standard deviation: 3.618884707390

Frozen Group 4C compatibility floor:
62.82844891348723

Margin above floor:
+5.861497404452

GROUP 4C CLEAN BRIDGE: PASS

This result was frozen before any Group 4C poisoned-DT outcome was examined.
