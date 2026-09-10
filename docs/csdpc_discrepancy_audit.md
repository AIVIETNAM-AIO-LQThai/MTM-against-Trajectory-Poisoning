# CSDPC Mechanism / Fidelity Discrepancy Audit

## Status

The canonical k=8 CSDPC reproduction remains frozen.

Canonical Gate B is INCONCLUSIVE.

The predeclared k=6 and k=10 cluster-count sensitivities also failed
to produce a strong CQL degradation effect.

No canonical setting will be changed in response to these results.

## Why a mechanism audit is required

The current reproduction passes artifact-integrity and reproducibility
checks, but downstream CQL degradation is much weaker than reported by
the CSDPC source paper.

The next question is therefore upstream of the learner:

    Does the generated poisoned dataset actually collapse
    sequence-level decision-pattern coverage?

This audit uses no CQL, DT, BC, or MTM performance to select an
alternative attack configuration.

## Known source / reproduction differences

### 1. Decision-pattern deduplication

The source paper reports that deduplication reduces the number of
distinct Walker2D decision patterns by nearly 80%.

The current canonical reproduction produces an approximately 42--44%
reduction across attack seeds.

This is treated as a descriptive fidelity discrepancy, not as a
post-hoc acceptance threshold.

### 2. Perturbation candidate generation

The source attack generates multiple bounded poisoned candidates and
chooses a candidate associated with a more frequent decision pattern.

The publication does not sufficiently specify the candidate count or
candidate sampling distribution.

Canonical reproduction choice:

    num_candidates = 100
    independent seeded Uniform bounded perturbations

This choice remains frozen.

### 3. Poison-budget and overlap semantics

The source defines poisoning through low-frequency decision patterns
selected according to rho, but does not operationally specify all
transition/window overlap accounting.

Canonical reproduction choices:

    budget = floor(rho * N transitions)
    complete windows only
    selected windows may not overlap

These choices remain frozen.

### 4. Action clipping

The publication specifies an L-infinity relative perturbation bound.

Canonical reproduction additionally clips Walker2d actions to [-1, 1].

This remains a frozen reproduction choice.

### 5. Feature preprocessing

Raw state-action decision units are the canonical source setting.

The publication does not operationally specify an additional
state/action scaling procedure before clustering.

Canonical reproduction uses no feature scaling.

### 6. Cluster count

Walker2D canonical cluster count is k=8.

Predeclared k=6 and k=10 sensitivity experiments did not recover a
strong attack effect.

Cluster count alone therefore does not explain the current discrepancy.

## Mechanism-audit questions

The audit will measure:

1. How much deduplication reduces clean distinct pattern count.
2. How often a selected poisoned window actually changes pattern.
3. Conditional on changing pattern, how often frequency increases.
4. How many modified transition cluster assignments actually change.
5. How many targeted rare pattern types disappear completely.
6. How much global occurrence mass of targeted patterns is removed.
7. Whether overall clean-to-poison pattern diversity decreases.
8. How many pattern types disappear and how many new types are created.
9. How many candidate windows are rejected because of overlap.
10. How frequently modified actions end at an action bound.

## Scientific rule

The purpose is diagnosis, not optimization.

A mechanism sensitivity may be proposed only after these canonical
measurements are recorded.

Any such sensitivity must be separately named and must never replace
the canonical Gate-B result retrospectively.

## Deduplication-order source-fidelity diagnostic

A separately frozen clean-data diagnostic compared:

- D0_CANONICAL:
  construct length-5 original-position windows, then remove
  consecutive repeated cluster labels within each window.

- D3_TRAJECTORY_DEDUP_BEFORE_WINDOW:
  remove consecutive repeated cluster labels over each completed
  trajectory first, then construct length-5 windows over the
  compressed label sequence.

Both variants reused the exact same canonical raw-data k=8 KMeans
clustering for each attack seed. No poisoned dataset or learner
performance was used.

Observed three-seed means:

D0_CANONICAL:
- completed-label fraction retained: 100.00%
- raw distinct patterns: 11108.67
- diagnostic distinct patterns: 6281.00
- distinct-pattern reduction: 43.46%

D3_TRAJECTORY_DEDUP_BEFORE_WINDOW:
- completed-label fraction retained: 27.53%
- raw distinct patterns: 11108.67
- diagnostic distinct patterns: 8884.67
- distinct-pattern reduction: 20.02%

D3 removes approximately 72.47% of repeated trajectory labels, but
this must not be conflated with the source paper's reported nearly
80% reduction in the number of distinct decision patterns.

The D3 interpretation therefore does not explain the source/reproduction
deduplication discrepancy.

The publication's method description also more directly supports the
canonical window-then-deduplicate interpretation: a sequence is
extracted first, converted to decision-unit labels, and consecutive
repeated units are then merged to form a decision pattern.

Status:
DEDUP_ORDER_DIAGNOSTIC_NO_EXPLANATION

## Sequence-enumeration source-fidelity diagnostic

A separately frozen clean-data diagnostic compared the canonical
overlapping sequence enumeration against non-overlapping length-5
enumeration.

All conditions reused the same canonical raw-data k=8 KMeans
clustering within each attack seed. Deduplication remained
window-then-deduplicate. No poisoned dataset or learner performance
was used.

Observed three-seed means:

D0_OVERLAPPING_STRIDE_1:
- window count: 995235
- raw distinct patterns: 11108.67
- deduplicated distinct patterns: 6281.00
- distinct-pattern reduction: 43.46%
- dedup-affected window fraction: 94.61%
- average deduplicated pattern length: 2.0935

D4_NONOVERLAPPING_STRIDE_5_OFFSET_0:
- window count: 199800
- raw distinct patterns: 6031.67
- deduplicated distinct patterns: 3347.33
- distinct-pattern reduction: 44.50%
- dedup-affected window fraction: 94.47%
- average deduplicated pattern length: 2.0988

The complete predeclared stride-5 offset sensitivity produced:

- offset 0: 44.50%
- offset 1: 44.74%
- offset 2: 44.60%
- offset 3: 44.79%
- offset 4: 44.73%

The range across all predeclared offsets was only 0.29 percentage
points.

Therefore sequence stride and trajectory-relative offset do not explain
the discrepancy between the canonical reproduction and the source
paper's descriptive nearly-80% distinct-pattern reduction.

No offset is selected as preferred based on these results.

Status:
SEQUENCE_ENUMERATION_DIAGNOSTIC_NO_EXPLANATION

## Deduplication-metric reconciliation

A final clean-data-only diagnostic evaluated multiple possible
deduplication statistics using the exact canonical k=8, raw-data,
length-5, stride-1, window-then-deduplicate pipeline.

No attack variant was introduced and no learner performance was used.

Three-seed means were:

- canonical distinct-type reduction: 43.46%
- window-instance changed fraction: 94.61%
- distinct raw-sequence-type changed fraction: 64.22%
- total label-token reduction: 58.13%
- average deduplicated pattern length: 2.0935
- full-length pattern occurrence fraction: 5.39%
- full-length distinct-pattern fraction: 63.27%
- non-equivalent total-window denominator reduction: 99.37%

The 99.37% quantity compares the number of unique deduplicated pattern
types against the total number of window occurrences and therefore does
not use equivalent pre/post distinct-pattern denominators. It cannot be
interpreted as a reduction in the number of distinct decision patterns.

The 94.61% window-instance quantity similarly measures how often
deduplication changes an individual window, rather than reduction in the
number of distinct pattern types.

None of the predeclared semantically distinct metrics reconciles the
canonical 43.46% distinct-type reduction with the source paper's
descriptive nearly-80% distinct-pattern reduction.

Together with the previously frozen source-fidelity diagnostics:

- literal length-6 interpretation: no explanation;
- per-dimension z-score preprocessing: no explanation;
- trajectory-level deduplication before windowing: no explanation;
- non-overlapping stride-5 enumeration and all offsets: no explanation;
- k=6/k=10 cluster sensitivity: no strong learner degradation recovery.

The source-fidelity discrepancy therefore remains unresolved.

This is not evidence that any alternative metric should replace the
canonical one. No additional attack interpretation will be selected
post hoc solely to reproduce the source statistic.

Status:
SOURCE_FIDELITY_DISCREPANCY_UNRESOLVED